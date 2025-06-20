import os
import json
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm
import logging
from torch.amp import GradScaler, autocast
import argparse

# Importar desde el módulo de utilidades de preparación de datasets
from dataset_utils import (
    get_dataset_file_list,
    GLOBAL_CLASSES_MAP,
    BASE_DATA_DIR_DEFAULT,
    OUTPUT_LIST_DIR_DEFAULT
)

# ----- CONFIGURACIÓN -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Semilla para Reproducibilidad ---
RANDOM_SEED = 23

# --- Selección del Dataset de Entrenamiento ---
TRAIN_DATASET_NAME = "rwf2000"  # Opciones: "rwf2000", "rlvs"

CLASSES = GLOBAL_CLASSES_MAP

# --- Parámetros Específicos de TSM ---
NUM_FRAMES_TO_SAMPLE_TSM = 8 # TSM suele usar menos frames por segmento
FRAME_STEP_TSM = 8           # Ajustar según la configuración original de TSM
IMG_CROP_SIZE_TSM = 224
IMG_RESIZE_DIM_TSM = 256     # Dimensión a la que se redimensiona antes del recorte
N_SEGMENTS = NUM_FRAMES_TO_SAMPLE_TSM # Para TSM, N_SEGMENTS es el número de frames muestreados

# --- Hiperparámetros de Entrenamiento ---
BATCH_SIZE = 16 
LR = 1e-4 
WEIGHT_DECAY = 1e-4
EPOCHS = 30
USE_AMP = True

# --- Checkpoint Preentrenado de TSM ---
# Asegúrate de que esta ruta sea correcta y el archivo exista.
PRETRAINED_CHECKPOINT_PATH = "TSM/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth" 

# --- Rutas de Salida y Directorios de Listas de Archivos ---
OUTPUT_DIR_BASE = f"tsm_r50_outputs_seed_{RANDOM_SEED}"
FILE_LIST_DIR = OUTPUT_LIST_DIR_DEFAULT
BASE_DATA_DIR = BASE_DATA_DIR_DEFAULT

# LOG_DIR_TENSORBOARD, METRICS_JSON_PATH, BEST_MODEL_PATH se definirán en main()

# --- Dispositivo y Eficiencia ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type # Para autocast
NUM_DATA_WORKERS = 2 if os.name == 'posix' else 0

# --- Control de Flujo ---
PERFORM_TRAINING = True
PERFORM_CROSS_INFERENCE = True

# --- Constantes de Normalización (Kinetics) ---
KINETICS_MEAN_LIST = [0.485, 0.456, 0.406]
KINETICS_STD_LIST = [0.229, 0.224, 0.225]

TRIALS_FPS = 50

# Funcion para asignar semilla
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # Para multi-GPU
        # Para eproducibilidad en cuDNN (afectan el rendimiento: quedan desactivadas)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 
    logging.info(f"Semilla fijada a: {seed_value}")

# Funcion para inicializar workers del Dataloader
def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# ----- TSM MÓDULO Y MODELO (Sin cambios respecto al original) -----
class TemporalShift(nn.Module):
    def __init__(self, n_segment, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            logging.debug('=> Using in-place shift...') # Cambiado a debug

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        # No es necesario .contiguous() aquí si la siguiente vista lo maneja
        return out.view(nt, c, h, w)


def make_temporal_shift(net, n_segment, n_div=8, places='blockres', temporal_pool=False):
    if temporal_pool: # No se usa en la config actual, pero se mantiene por completitud
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    if places == 'blockres':
        def make_block_temporal(stage, this_n_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if isinstance(b, models.resnet.Bottleneck):
                    # Guardar conv1 original
                    conv1_original = b.conv1
                    # Crear nuevo conv1 con TemporalShift
                    # Asegurar que inplace=False si no se maneja cuidadosamente la memoria
                    b.conv1 = nn.Sequential(TemporalShift(this_n_segment, n_div, inplace=False), conv1_original)
            return nn.Sequential(*blocks)
        
        # Aplicar a cada capa de ResNet
        for i in range(1, 5): # layer1, layer2, layer3, layer4
            layer_name = f'layer{i}'
            original_layer = getattr(net, layer_name)
            shifted_layer = make_block_temporal(original_layer, n_segment_list[i-1])
            setattr(net, layer_name, shifted_layer)
    else:
        raise NotImplementedError(f"Unsupported places: {places}")


class TSM_ResNet50(nn.Module):
    def __init__(self, num_classes, n_segment):
        super(TSM_ResNet50, self).__init__()
        self.n_segment = n_segment
        self.num_classes = num_classes

        logging.info(f"Initializing TSM with ResNet50 backbone, num_segments={n_segment}")
        base_model = models.resnet50(weights=None) # Cargar sin pesos de ImageNet por defecto
        
        # Aplicar TSM a las capas de ResNet
        make_temporal_shift(base_model, n_segment)

        # Quitar la capa FC original de ResNet
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # Usar avgpool adaptativo
        
        num_fc_inputs = base_model.fc.in_features
        self.fc = nn.Linear(num_fc_inputs, num_classes)

    def forward(self, x): # x shape: (N, C, T, H, W)
        # Permutar a (N, T, C, H, W) y luego a (N*T, C, H, W) para ResNet2D
        n, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous() # N, T, C, H, W
        x = x.view(n * t, c, h, w) # N*T, C, H, W
        
        out = self.features(x)
        out = self.avgpool(out) # N*T, num_features, 1, 1
        out = out.view(n, t, -1) # N, T, num_features
        out = out.mean(dim=1) # N, num_features (promedio sobre la dimensión temporal)
        out = self.fc(out) # N, num_classes
        return out
    
def load_tsm_model_custom(num_model_classes, n_segment_model=N_SEGMENTS, checkpoint_path=None):
    # Initialize your model structure first
    model = TSM_ResNet50(num_classes=num_model_classes, n_segment=n_segment_model)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint '{checkpoint_path}' no encontrado o no proporcionado. "
                        "El modelo se inicializará desde cero (o ResNet50 sin pesos).")
        return model

    logging.info(f"Cargando checkpoint desde: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Get the state_dict, handling potential nesting
        if 'state_dict' in checkpoint:
            source_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            source_state_dict = checkpoint
        else:
            logging.error("El checkpoint no es un diccionario o no contiene 'state_dict'.")
            return model
            
        remapped_state_dict = {}
        model_state_keys = set(model.state_dict().keys()) # Keys expected by your current model

        for src_key, src_value in source_state_dict.items():
            trg_key = src_key

            # Remove "module." prefix if present (from DataParallel saving)
            if trg_key.startswith('module.'):
                trg_key = trg_key[len('module.'):]

            # --- Start Specific Remapping for TSM_ResNet50 structure ---
            
            # 1. Remap backbone keys from 'base_model.LAYER.BLOCK.PART' to 'features.INDEX.BLOCK.PART'
            if trg_key.startswith('base_model.'):
                temp_key = trg_key[len('base_model.'):]
                
                # Top-level ResNet layers to features indices:
                # conv1 -> features[0]
                # bn1 -> features[1]
                # layer1 -> features[4]
                # layer2 -> features[5]
                # layer3 -> features[6]
                # layer4 -> features[7]
                
                if temp_key.startswith('conv1.'):
                    trg_key = 'features.0.' + temp_key[len('conv1.'):]
                elif temp_key.startswith('bn1.'):
                    trg_key = 'features.1.' + temp_key[len('bn1.'):]
                elif temp_key.startswith('layer1.'):
                    trg_key = 'features.4.' + temp_key[len('layer1.'):]
                elif temp_key.startswith('layer2.'):
                    trg_key = 'features.5.' + temp_key[len('layer2.'):]
                elif temp_key.startswith('layer3.'):
                    trg_key = 'features.6.' + temp_key[len('layer3.'):]
                elif temp_key.startswith('layer4.'):
                    trg_key = 'features.7.' + temp_key[len('layer4.'):]
                else:
                    # If it's not one of the above, it might be an unexpected key from base_model
                    # or a key we don't need to remap further here (like fc if it was base_model.fc)
                    pass # Keep temp_key as is for now, further processing might apply

            # 2. Handle '.net.' for the first conv in Bottlenecks and map to '.1'
            # e.g., 'features.4.0.conv1.net.weight' (after step 1) -> 'features.4.0.conv1.1.weight'
            # This assumes your make_temporal_shift puts the original conv at index 1 of an nn.Sequential
            parts = trg_key.split('.')
            # Look for patterns like features.INDEX.BLOCK_IDX.conv1.net.PARAMS
            if len(parts) > 4 and parts[-2] == 'net' and parts[-3].startswith('conv1'):
                 # Check if TemporalShift module is parameter-free, if so, original conv is at index 1
                if not any(p.requires_grad for p in model.features[int(parts[1])][int(parts[2])].conv1[0].parameters()): # Check if TemporalShift (at index 0) is parameter-free
                    parts[-2] = '1' # Replace 'net' with '1'
                    trg_key = '.'.join(parts)
                else:
                    logging.warning(f"TemporalShift en {'.'.join(parts[:-2])} podría tener parámetros. "
                                    "La reasignación de '.net' a '.1' podría ser incorrecta.")
            
            # 3. Handle classification layer: checkpoint 'new_fc' to model 'fc'
            if trg_key.startswith('new_fc.'):
                if src_value.shape[0] != num_model_classes:
                    logging.info(f"Omitiendo capa FC del checkpoint '{src_key}' debido a la diferencia de clases "
                                 f"(checkpoint: {src_value.shape[0]}, modelo: {num_model_classes}). "
                                 "La capa FC del modelo se entrenará desde cero.")
                    continue # Skip this key, do not add to remapped_state_dict
                else:
                    # If classes match (unlikely for Kinetics to custom), map it
                    trg_key = trg_key.replace('new_fc.', 'fc.')
            
            # 4. Skip loading 'num_batches_tracked' for BatchNorm layers
            if 'num_batches_tracked' in trg_key:
                continue

            # Only add if the target key actually exists in the current model
            if trg_key in model_state_keys:
                remapped_state_dict[trg_key] = src_value
            elif src_key == trg_key : # If no remapping occurred for this key
                 if not (trg_key.startswith("fc.") and src_key.startswith("new_fc.")): # Allow fc to be missing if new_fc was skipped
                    logging.debug(f"Clave del checkpoint '{src_key}' no mapeada y no encontrada en el modelo.")
            # else: # A remapping occurred, but the target key is still not in the model (should be rare with this logic)
                # logging.debug(f"Clave del checkpoint '{src_key}' remapeada a '{trg_key}', pero no encontrada en el modelo.")


        # Load the remapped state dict
        # It's crucial to only load weights for layers that exist and match.
        msg = model.load_state_dict(remapped_state_dict, strict=False)
        logging.info(f"Mensaje de carga de state_dict (después del remapeo detallado): {msg}")

        if msg.missing_keys:
            # Expected missing keys: fc.weight, fc.bias IF they were intentionally skipped.
            # Also, if TemporalShift has learnable params and we didn't map them.
            # My current TemporalShift is parameter-free.
            is_fc_missing = all(k.startswith('fc.') for k in msg.missing_keys)
            if not (len(msg.missing_keys) <= 2 and is_fc_missing): # Allow up to 2 fc keys to be missing
                 logging.warning(f"Claves FALTANTES (después del remapeo detallado) no esperadas o adicionales: {msg.missing_keys}")
        if msg.unexpected_keys:
            # This should be empty if remapping was perfect for all relevant layers.
            logging.warning(f"Claves INESPERADAS (después del remapeo detallado): {msg.unexpected_keys}")
        
        num_loaded_layers = len(remapped_state_dict)
        total_model_layers = len(model.state_dict())
        logging.info(f"Se cargaron {num_loaded_layers} tensores de parámetros en el modelo (de {total_model_layers} capas totales en el modelo).")
        if num_loaded_layers < total_model_layers / 2 and num_loaded_layers > 0 : # Heuristic
            logging.warning("Parece que se cargó menos de la mitad de las capas del modelo. Verifica el remapeo.")
        elif num_loaded_layers == 0 and len(source_state_dict) > 0:
            logging.error("No se cargó NINGUNA capa. El remapeo de claves falló por completo.")


    except Exception as e:
        logging.error(f"Error EXCEPCIONAL cargando y remapeando el checkpoint desde {checkpoint_path}: {e}", exc_info=True)
        logging.warning("El modelo continuará con inicialización aleatoria.")
        
    return model

# ----- PROCESAMIENTO DE VÍDEO Y DATASET (Específico de TSM) -----
def process_video_tsm(path, num_frames_to_sample, frame_step, resize_dim, crop_size, is_train=True):
    """Procesa un vídeo para TSM. Devuelve una lista de frames NumPy (H, W, C)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.warning(f"TSM Error: No se pudo abrir el vídeo: {path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logging.warning(f"TSM Error: El vídeo no tiene fotogramas o es inválido: {path}")
        return None

    # Lógica de muestreo de frames (similar a la original de TSM)
    selectable_frames_range = total_frames - (num_frames_to_sample - 1) * frame_step
    if selectable_frames_range > 0:
        if is_train:
            start_index_in_original_video = random.randint(0, selectable_frames_range - 1)
        else: # Muestreo central para validación/test
            start_index_in_original_video = selectable_frames_range // 2
        frame_indices = [start_index_in_original_video + i * frame_step for i in range(num_frames_to_sample)]
    else: # No hay suficientes frames, ajustar
        available_indices = list(range(0, total_frames, max(1, frame_step)))
        if not available_indices: available_indices = list(range(total_frames))
        if not available_indices:
            cap.release(); logging.warning(f"TSM: No hay frames disponibles para muestrear en {path}"); return None
        
        frame_indices = available_indices[:num_frames_to_sample]
        while len(frame_indices) < num_frames_to_sample and available_indices: # Relleno
            frame_indices.append(available_indices[-1])
    
    if not frame_indices and num_frames_to_sample > 0: # Si aún no hay índices y se esperaban
        cap.release(); logging.warning(f"TSM: Fallo en la selección de índices de frames para {path}"); return None

    frames_processed_list = []
    for frame_idx_orig in frame_indices:
        frame_idx = min(int(frame_idx_orig), total_frames - 1)
        frame_idx = max(0, frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"TSM: No se pudo leer el fotograma {frame_idx} de {path}. Usando fotograma negro.")
            frame = np.zeros((resize_dim, resize_dim, 3), dtype=np.uint8) # Usar resize_dim para el dummy
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_dim, resize_dim)) # Redimensionar primero

        # Recorte (aleatorio o central) + volteo horizontal
        if is_train:
            # Aplico volteo horizontal con prob 50%
            if random.random() < 0.5: # 50% de probabilidad
                frame = cv2.flip(frame, 1) # 1 para volteo horizontal

            top = random.randint(0, resize_dim - crop_size)
            left = random.randint(0, resize_dim - crop_size)
        else:
            top = (resize_dim - crop_size) // 2
            left = (resize_dim - crop_size) // 2
        frame = frame[top:top+crop_size, left:left+crop_size, :] # Aplicar recorte
        frames_processed_list.append(frame)
    cap.release()

    if len(frames_processed_list) != num_frames_to_sample and num_frames_to_sample > 0:
        logging.error(f"TSM Error de procesamiento: se esperaban {num_frames_to_sample} fotogramas, se obtuvieron {len(frames_processed_list)} para {path}")
        return None
    return frames_processed_list


# Transformaciones de TSM (aplicadas en VideoListDatasetTSM)
mean_tensor_tsm = torch.tensor(KINETICS_MEAN_LIST, dtype=torch.float32).view(1, 3, 1, 1) # (1,C,1,1) para broadcast sobre T
std_tensor_tsm = torch.tensor(KINETICS_STD_LIST, dtype=torch.float32).view(1, 3, 1, 1)  # (1,C,1,1)

tsm_custom_transforms = Compose(
    [
        Lambda(lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32)), # x: list of (H,W,C) frames -> (T,H,W,C) tensor
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: x.permute(0, 3, 1, 2)), # (T,C,H,W)
        Lambda(lambda x: (x - mean_tensor_tsm.to(x.device)) / std_tensor_tsm.to(x.device)), # Normalizar
        Lambda(lambda x: x.permute(1, 0, 2, 3))  # (C,T,H,W) - forma final para el modelo TSM
    ]
)

class VideoListDatasetTSM(Dataset):
    def __init__(self, file_list_data, num_frames, frame_step, resize_dim, crop_size, transforms, is_train, dataset_name_log=""):
        self.file_list_data = file_list_data
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resize_dim = resize_dim
        self.crop_size = crop_size
        self.transforms = transforms
        self.is_train = is_train
        self.dataset_name_log = dataset_name_log

        if not self.file_list_data:
            logging.warning(f"VideoListDatasetTSM inicializado con lista de archivos vacía para {self.dataset_name_log}")

    def __len__(self):
        return len(self.file_list_data)

    def __getitem__(self, idx):
        if idx >= len(self.file_list_data):
            raise IndexError(f"Índice {idx} fuera de rango para dataset {self.dataset_name_log}")

        item_data = self.file_list_data[idx]
        video_path = item_data['path']
        label = item_data['label']

        # Obtener lista de frames NumPy (H,W,C)
        frames_np_list = process_video_tsm(
            video_path, self.num_frames, self.frame_step, self.resize_dim, self.crop_size, self.is_train
        )

        if frames_np_list is None or (self.num_frames > 0 and len(frames_np_list) != self.num_frames):
            logging.warning(f"TSM: Fallo al procesar vídeo {video_path}. Devolviendo tensor dummy y etiqueta -1.")
            dummy_tensor = torch.zeros((3, self.num_frames if self.num_frames > 0 else 0, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)
        
        # Aplicar transformaciones (incluye to_tensor, normalización, permutación)
        try:
            clip_tensor = self.transforms(frames_np_list) # Espera lista de (H,W,C)
        except Exception as e:
            logging.error(f"TSM: Error aplicando transformaciones a {video_path}: {e}")
            dummy_tensor = torch.zeros((3, self.num_frames if self.num_frames > 0 else 0, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)
            
        return clip_tensor, torch.tensor(label, dtype=torch.long)


# ----- FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN (Adaptadas para TSM y etiquetas -1) -----
def train_epoch_tsm(model, loader, criterion, optimizer, device, use_amp_flag, scaler): # Renombrada para claridad
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_valid_samples = 0
    
    progress_bar = tqdm(loader, desc="Train TSM", leave=False)
    for x_batch, y_batch in progress_bar: # x_batch ya es (N, C, T, H, W)
        valid_indices = y_batch != -1
        if not valid_indices.any(): continue
        
        x = x_batch[valid_indices].to(device, non_blocking=True)
        y = y_batch[valid_indices].to(device, non_blocking=True)
        if x.size(0) == 0: continue
            
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
            outputs = model(x) # TSM_ResNet50.forward espera (N,C,T,H,W)
            loss = criterion(outputs, y) # outputs ya es (N, num_classes)
        
        if use_amp_flag and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        _, preds = torch.max(outputs, 1)
        current_batch_valid_samples = x.size(0)
        running_loss += loss.item() * current_batch_valid_samples
        running_corrects += torch.sum(preds == y.data)
        total_valid_samples += current_batch_valid_samples
        
        if current_batch_valid_samples > 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(torch.sum(preds == y.data).item() / current_batch_valid_samples):.4f}")
        if device.type == 'cuda': torch.cuda.empty_cache()

    epoch_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    epoch_acc = (running_corrects.double() / total_valid_samples if total_valid_samples > 0 else torch.tensor(0.0)).item()
    return epoch_loss, epoch_acc


def evaluate_tsm(model, loader, criterion, device, use_amp_flag, pos_label_value=1, num_classes_eval=len(CLASSES)): # Renombrada
    model.eval()
    running_loss = 0.0
    total_valid_samples = 0
    all_preds_list, all_labels_list = [], []
    
    progress_bar = tqdm(loader, desc="Eval TSM", leave=False)
    with torch.no_grad():
        for x_batch, y_batch in progress_bar:
            valid_indices = y_batch != -1
            if not valid_indices.any(): continue

            x = x_batch[valid_indices].to(device, non_blocking=True)
            y_true = y_batch[valid_indices].to(device, non_blocking=True)
            if x.size(0) == 0: continue
            
            with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
                outputs = model(x)
                loss = criterion(outputs, y_true)
            
            _, preds = torch.max(outputs, 1)
            current_batch_valid_samples = x.size(0)
            running_loss += loss.item() * current_batch_valid_samples
            total_valid_samples += current_batch_valid_samples
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(y_true.cpu().numpy())
            
    if total_valid_samples == 0:
        logging.warning("TSM Eval: No se procesaron muestras válidas.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    epoch_loss = running_loss / total_valid_samples
    acc = accuracy_score(all_labels_list, all_preds_list)
    # Lógica de pos_label y avg_method similar a I3D
    avg_method = 'binary' if num_classes_eval == 2 else 'macro'
    unique_labels_in_data = np.unique(all_labels_list + all_preds_list)
    valid_pos_label = None
    if avg_method == 'binary':
        if pos_label_value in unique_labels_in_data: valid_pos_label = pos_label_value
        elif len(unique_labels_in_data) > 0:
            if len(unique_labels_in_data) == 2 and 0 in unique_labels_in_data:
                other_labels = [l for l in unique_labels_in_data if l != 0]
                if other_labels: valid_pos_label = other_labels[0]
            elif 1 in unique_labels_in_data: valid_pos_label = 1
            elif unique_labels_in_data: valid_pos_label = sorted(list(unique_labels_in_data))[0]
    if avg_method == 'binary' and valid_pos_label is None and len(unique_labels_in_data) > 1:
        avg_method = 'macro'

    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    cm = confusion_matrix(all_labels_list, all_preds_list, labels=list(range(num_classes_eval))).tolist()
    return epoch_loss, acc, precision, recall, f1, cm

# ----- MEDICIÓN DE FPS Y GUARDADO DE MÉTRICAS -----
def measure_inference_fps_tsm(model, device, n_segments, crop_size, trials=TRIALS_FPS): # Parámetros TSM
    # TSM espera (N, C, T, H, W)
    dummy_input_shape = (1, 3, n_segments if n_segments > 0 else 1, crop_size, crop_size)
    if n_segments == 0: logging.warning("FPS TSM medido con T=1 para n_segments=0.")
    
    dummy_input = torch.randn(dummy_input_shape, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input) # Warm-up
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(trials): _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return trials / total_time if total_time > 0 else 0.0

# ----- Guardado de Metricas -----
def save_or_update_json(new_data: dict, json_path: str):
    """
    Lee un archivo JSON existente, actualiza su contenido con los nuevos datos
    y lo guarda de nuevo. Si el archivo no existe, lo crea.
    """
    # Asegurarse de que el directorio de salida exista
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    existing_data = {}
    # Leer el contenido actual del archivo si ya existe y no está vacío
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"El archivo JSON {json_path} estaba corrupto. Se va a sobrescribir.")

    # Actualizar el diccionario existente con los nuevos datos.
    # El método .update() fusiona los diccionarios. Las claves nuevas se añaden
    # y las claves existentes en 'existing_data' se actualizan con los valores de 'new_data'.
    existing_data.update(new_data)

    # Escribir el diccionario completo de vuelta al archivo
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        logging.info(f"JSON actualizado en '{json_path}' con las claves: {list(new_data.keys())}")
    except Exception as e:
        logging.error(f"No se pudo guardar el JSON en '{json_path}': {e}")
'''    
# save_metrics_to_json (puede ser la misma que en I3D, ya es genérica)
def save_metrics_to_json(metrics_dict, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        def convert_to_native_types(item): # Función auxiliar recursiva
            if isinstance(item, list): return [convert_to_native_types(i) for i in item]
            elif isinstance(item, dict): return {k: convert_to_native_types(v) for k, v in item.items()}
            elif isinstance(item, (np.integer, np.int_)): return int(item)
            elif isinstance(item, (np.floating, np.float64)): return float(item)
            elif isinstance(item, np.ndarray): return item.tolist()
            return item
        cleaned_metrics_dict = convert_to_native_types(metrics_dict)
        existing_data = {}
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            with open(json_path, 'r') as f:
                try: existing_data = json.load(f)
                except json.JSONDecodeError: logging.warning(f"JSON {json_path} corrupto. Se sobrescribirá parcialmente.")
        for key, value in cleaned_metrics_dict.items(): # Fusionar inteligentemente
            if key in existing_data and isinstance(existing_data[key], dict) and isinstance(value, dict):
                existing_data[key].update(value)
            else: existing_data[key] = value
        with open(json_path, 'w') as f: json.dump(existing_data, f, indent=4)
        logging.info(f"Métricas guardadas/actualizadas en {json_path}")
    except Exception as e: logging.error(f"Error al guardar métricas en JSON {json_path}: {e}")
'''
# ----- PARSER DE ARGUMENTOS -----
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script de entrenamiento y evaluación para ViViT.")
    parser.add_argument(
        "train_dataset_name_arg", # Nombre del argumento
        type=str,
        choices=["rwf2000", "rlvs"],
        help="Nombre del dataset a usar para el entrenamiento (opciones: 'rwf2000', 'rlvs')."
    )
    return parser.parse_args()

# ----- FUNCIÓN PRINCIPAL -----
def main():
    args = parse_arguments()
    TRAIN_DATASET_NAME_FROM_ARG = args.train_dataset_name_arg # Usar el argumento parseado    
    if TRAIN_DATASET_NAME_FROM_ARG in ["rlvs","rwf2000"]: 
        TRAIN_DATASET_NAME = TRAIN_DATASET_NAME_FROM_ARG #Si el arg recibido es valido se asigna, si no queda por defecto rwf-2000
    
    set_seed(RANDOM_SEED)
    
    # Crear generador para DataLoader
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    
    dataset_name_for_history = f"{TRAIN_DATASET_NAME}_TSM_R50" # Para nombres de archivo
    current_output_dir = os.path.join(OUTPUT_DIR_BASE, f"trained_on_{TRAIN_DATASET_NAME}")
    
    log_dir_tensorboard = os.path.join(current_output_dir, "logs_tensorboard_tsm")
    metrics_json_path = os.path.join(current_output_dir, f"train_metrics_tsm_{TRAIN_DATASET_NAME.lower()}.json") 
    best_model_path = os.path.join(current_output_dir, f"tsm_{TRAIN_DATASET_NAME.lower()}_best_model.pth")

    os.makedirs(current_output_dir, exist_ok=True)
    os.makedirs(log_dir_tensorboard, exist_ok=True)

    # writer = SummaryWriter(log_dir_tensorboard) # Descomentar si se usa TensorBoard

    # Cargar modelo TSM
    model = load_tsm_model_custom(
        num_model_classes=len(CLASSES),
        n_segment_model=N_SEGMENTS,
        checkpoint_path=PRETRAINED_CHECKPOINT_PATH
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Ajustar parámetros para fine-tuning de TSM (ej. solo la capa FC o algunas capas del backbone)
    # Por ahora, entrenamos todos los parámetros.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # T_max es el número total de épocas. El scheduler ajustará el LR en cada época.
    # eta_min=0 significa que el LR llegará a 0 al final de las EPOCHS.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    use_amp_for_training = USE_AMP and DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp_for_training)

    if PERFORM_TRAINING:
        logging.info(f"Iniciando entrenamiento del modelo TSM en {TRAIN_DATASET_NAME}...")
        logging.info(f"Device: {DEVICE.type}")
        train_file_list = get_dataset_file_list(
            dataset_name=TRAIN_DATASET_NAME, split_name="train",
            base_data_dir=BASE_DATA_DIR, output_list_dir=FILE_LIST_DIR
        )
        val_file_list = get_dataset_file_list(
            dataset_name=TRAIN_DATASET_NAME, split_name="val",
            base_data_dir=BASE_DATA_DIR, output_list_dir=FILE_LIST_DIR
        )

        if not train_file_list:
            logging.error(f"No se pudo cargar la lista de archivos de entrenamiento para {TRAIN_DATASET_NAME}. Abortando.")
            # writer.close(); # Si se usa TensorBoard
            return
        
        train_dataset = VideoListDatasetTSM(
            train_file_list, N_SEGMENTS, FRAME_STEP_TSM, IMG_RESIZE_DIM_TSM, IMG_CROP_SIZE_TSM, 
            tsm_custom_transforms, is_train=True, dataset_name_log=f"{TRAIN_DATASET_NAME} Train TSM"
        )
        val_dataset = VideoListDatasetTSM(
            val_file_list, N_SEGMENTS, FRAME_STEP_TSM, IMG_RESIZE_DIM_TSM, IMG_CROP_SIZE_TSM, 
            tsm_custom_transforms, is_train=False, dataset_name_log=f"{TRAIN_DATASET_NAME} Val TSM"
        ) if val_file_list else None
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker) if val_dataset and len(val_dataset) > 0 else None

        history = {
            'dataset_trained_on': TRAIN_DATASET_NAME, 'model_name': 'TSM_R50',
            'hyperparameters': {'lr': LR, 'batch_size': BATCH_SIZE, 'epochs_config': EPOCHS, 
                                'n_segments': N_SEGMENTS, 'frame_step': FRAME_STEP_TSM, 
                                'resize_dim': IMG_RESIZE_DIM_TSM, 'crop_size': IMG_CROP_SIZE_TSM,
                                'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY,
                                'pretrained_checkpoint': os.path.basename(PRETRAINED_CHECKPOINT_PATH) if PRETRAINED_CHECKPOINT_PATH else "None"},
            'epochs_run': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_cm': [],
            'epoch_time_seconds': []
        }
        best_val_f1 = 0.0

        logging.info(f"Config de entrenamiento TSM: Dataset={TRAIN_DATASET_NAME}, N_SEGMENTS={N_SEGMENTS}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}")

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            logging.info(f"--- Época {epoch}/{EPOCHS} (Entrenando TSM en {TRAIN_DATASET_NAME}) ---")
            
            train_loss_val, train_acc_val = train_epoch_tsm(model, train_loader, criterion, optimizer, DEVICE, use_amp_for_training, scaler)
            history['epochs_run'].append(epoch); history['train_loss'].append(train_loss_val); history['train_acc'].append(train_acc_val)
            # writer.add_scalar(f'Loss/train_tsm_{TRAIN_DATASET_NAME}', train_loss_val, epoch) # Descomentar si se usa TensorBoard
            logging.info(f"Época {epoch} Train TSM ({TRAIN_DATASET_NAME}): Pérdida={train_loss_val:.4f}, Acc={train_acc_val:.4f}")

            val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = 0.0,0.0,0.0,0.0,0.0,[]
            if val_loader:
                val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = evaluate_tsm(
                    model, val_loader, criterion, DEVICE, use_amp_for_training, 
                    pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
                )
                # writer.add_scalar(f'F1/val_tsm_{TRAIN_DATASET_NAME}', val_f1_val, epoch) # Descomentar si se usa TensorBoard
                logging.info(f"Época {epoch} Val TSM ({TRAIN_DATASET_NAME}): Pérdida={val_loss_val:.4f}, Acc={val_acc_val:.4f}, F1={val_f1_val:.4f}")
            
            history['val_loss'].append(val_loss_val if val_loader else None); history['val_acc'].append(val_acc_val if val_loader else None); 
            history['val_precision'].append(val_prec_val if val_loader else None); history['val_recall'].append(val_rec_val if val_loader else None); 
            history['val_f1'].append(val_f1_val if val_loader else None); history['val_cm'].append(val_cm_val if val_loader else None)

            epoch_duration_val = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(epoch_duration_val)
            logging.info(f"Época {epoch} TSM ({TRAIN_DATASET_NAME}) completada en {epoch_duration_val:.2f}s")
            save_or_update_json(history, metrics_json_path)
            
            # Actualiza la tasa de aprendizaje para la siguiente época
            scheduler.step()
            # Opcional: registrar el nuevo LR para verificar que está funcionando
            logging.info(f"LR para la época {epoch + 1}: {scheduler.get_last_lr()[0]:.7f}")
            
            if val_loader and val_f1_val > best_val_f1:
                best_val_f1 = val_f1_val
                torch.save(model.state_dict(), best_model_path) # Guardar solo state_dict para TSM es común
                logging.info(f"  Mejor F1 en validación TSM ({TRAIN_DATASET_NAME}): {best_val_f1:.4f}. Modelo guardado en {best_model_path}")
            elif not val_loader and epoch == EPOCHS:
                 torch.save(model.state_dict(), best_model_path)
                 logging.info(f"Entrenamiento TSM sin validación ({TRAIN_DATASET_NAME}). Modelo de época {epoch} guardado.")
        
        logging.info(f"Entrenamiento TSM en {TRAIN_DATASET_NAME} completado.")
        if val_loader: logging.info(f"Mejor F1 en validación TSM ({TRAIN_DATASET_NAME}) final: {best_val_f1:.4f}.")
    else:
        logging.info("Entrenamiento TSM omitido (PERFORM_TRAINING=False).")

    # Cargar el mejor modelo para análisis
    model_for_analysis = load_tsm_model_custom(len(CLASSES), N_SEGMENTS, checkpoint_path=None).to(DEVICE) # Crear estructura
    model_loaded_for_analysis = False
    if os.path.exists(best_model_path):
        logging.info(f"Cargando modelo TSM desde {best_model_path} para análisis/inferencia.")
        model_for_analysis.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model_for_analysis.eval()
        model_loaded_for_analysis = True
    else:
        logging.warning(f"No se encontró el archivo de modelo TSM {best_model_path}.")
        if PERFORM_TRAINING and 'model' in locals() and isinstance(model, nn.Module):
            model_for_analysis = model; model_for_analysis.eval(); model_loaded_for_analysis = True
            logging.info("Usando modelo TSM de la última época de entrenamiento para análisis.")
        else: logging.error("TSM: No hay modelo entrenado y no se encontró checkpoint.")


    if model_loaded_for_analysis:
        logging.info("Calculando estadísticas de rendimiento del modelo TSM cargado...")
        fps = measure_inference_fps_tsm(model_for_analysis, DEVICE, N_SEGMENTS, IMG_CROP_SIZE_TSM)
        params_count = parameter_count(model_for_analysis).get('', 0)
        gflops = -1.0
        try:
            dummy_flops_shape = (1, 3, N_SEGMENTS if N_SEGMENTS > 0 else 1, IMG_CROP_SIZE_TSM, IMG_CROP_SIZE_TSM)
            dummy_input_flops = torch.randn(dummy_flops_shape, device=DEVICE)
            if hasattr(model_for_analysis, '_modules') and model_for_analysis._modules:
                 flops_analyzer = FlopCountAnalysis(model_for_analysis, dummy_input_flops)
                 gflops = flops_analyzer.total() / 1e9
            else: logging.warning("Modelo TSM de análisis vacío.")
        except Exception as e: logging.error(f"No se pudieron calcular los FLOPs para TSM: {e}")
        logging.info(f"Modelo TSM Cargado - FPS: {fps:.2f}, Params: {params_count/1e6:.2f}M, GFLOPs: {gflops:.2f}G")
        performance_stats_data = {'performance_stats': { 'fps': float(fps), 'parameters': int(params_count), 'gflops': float(gflops) }}
        save_or_update_json(performance_stats_data, metrics_json_path)
        # writer.add_scalar('Performance/FPS_tsm_final', fps) # Descomentar si se usa TensorBoard
    else:
        logging.warning("Análisis de rendimiento TSM omitido.")

    if PERFORM_CROSS_INFERENCE and model_loaded_for_analysis:
        logging.info(f"\nComenzando inferencia cruzada con el modelo TSM entrenado en {TRAIN_DATASET_NAME}...")
        datasets_for_cross_inference = []
        if TRAIN_DATASET_NAME == "rwf2000": datasets_for_cross_inference = ["rlvs", "hockey"]
        elif TRAIN_DATASET_NAME == "rlvs": datasets_for_cross_inference = ["rwf2000", "hockey"]
        
        for inference_ds_name in datasets_for_cross_inference:
            logging.info(f"--- Inferencia TSM en: {inference_ds_name} ---")
            cross_inf_list = get_dataset_file_list(inference_ds_name, "all", BASE_DATA_DIR, FILE_LIST_DIR)
            if not cross_inf_list:
                logging.warning(f"No se pudo cargar lista de {inference_ds_name} para inferencia TSM. Omitiendo."); continue
            
            cross_inf_dataset = VideoListDatasetTSM(
                cross_inf_list, N_SEGMENTS, FRAME_STEP_TSM, IMG_RESIZE_DIM_TSM, IMG_CROP_SIZE_TSM,
                tsm_custom_transforms, is_train=False, dataset_name_log=f"{inference_ds_name} Cross-Inf TSM"
            )
            if len(cross_inf_dataset) == 0:
                logging.warning(f"Dataset de inferencia TSM {inference_ds_name} vacío. Omitiendo."); continue
            
            cross_inf_loader = DataLoader(cross_inf_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker)
            inf_loss, inf_acc, inf_prec, inf_rec, inf_f1, inf_cm = evaluate_tsm(
                model_for_analysis, cross_inf_loader, criterion, DEVICE, use_amp_for_training, # AMP puede usarse en eval
                pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
            )
            logging.info(f"Resultados Inferencia TSM en {inference_ds_name} (entrenado en {TRAIN_DATASET_NAME}):")
            logging.info(f"  Loss: {inf_loss:.4f}, Acc: {inf_acc:.4f}, F1 (Violence): {inf_f1:.4f}")
            
            current_cross_metrics = {f'cross_inference_tsm_on_{inference_ds_name}': {
                'model_trained_on': TRAIN_DATASET_NAME, 'evaluated_on': f"{inference_ds_name}_full",
                'loss': inf_loss, 'accuracy': inf_acc, 'precision_violence': inf_prec, 
                'recall_violence': inf_rec, 'f1_score_violence': inf_f1, 'confusion_matrix': inf_cm
            }}
            save_or_update_json(current_cross_metrics, metrics_json_path)
            # writer.add_scalar(f'F1/cross_eval_tsm_{inference_ds_name}_Violence', inf_f1) # Descomentar si se usa TensorBoard

    # if 'writer' in locals(): writer.close() # Si se usa TensorBoard
    logging.info("Proceso TSM completado.")

if __name__ == '__main__':
    # Configurar el directorio de salida base si se ejecuta directamente
    # Esto es solo para el caso de ejecución directa, normalmente se configuraría arriba.
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    main()
