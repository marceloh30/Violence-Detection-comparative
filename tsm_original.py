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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm
import logging
from torch.amp import GradScaler, autocast

# ----- CONFIGURACIÓN (adaptada de SlowFast y I3D) -----
BASE_DATA_DIR = "assets"
RWF_2000_SUBDIR = "RWF-2000"
HOCKEY_FIGHTS_SUBDIR = "HockeyFights"
RLVS_SUBDIR = "RealLifeViolenceDataset"

CLASSES = {"Fight": 1, "NonFight": 0}
SPLITS = {"train": "train", "val": "val"}

NUM_FRAMES_TO_SAMPLE = 8
FRAME_STEP = 8
IMG_CROP_SIZE = 224
IMG_RESIZE_DIM = 256

N_SEGMENTS = NUM_FRAMES_TO_SAMPLE

BATCH_SIZE = 2
LR = 1e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 10
NUM_CLASSES_MODEL = len(CLASSES)
USE_AMP = True

OUTPUT_DIR = "tsm_r50_finetuned_outputs" # Updated output directory
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "train_metrics_tsm_finetuned.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_tsm_finetuned.pth")

# Path to your Kinetics-400 pretrained checkpoint
PRETRAINED_CHECKPOINT_PATH = "TSM/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth" # Added this line

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KINETICS_MEAN_LIST = [0.485, 0.456, 0.406]
KINETICS_STD_LIST = [0.229, 0.224, 0.225]

TRIALS_FPS = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- TSM MÓDULO Y MODELO (remains the same) -----

class TemporalShift(nn.Module):
    def __init__(self, n_segment, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            logging.info('=> Using in-place shift...')

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        if self.inplace:
            return out.contiguous().view(nt, c, h, w)
        else:
            return out.contiguous().view(nt, c, h, w)

def make_temporal_shift(net, n_segment, n_div=8, places='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    if places == 'block':
        def make_block_temporal(stage, this_n_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                blocks[i] = nn.Sequential(b, TemporalShift(this_n_segment, n_div))
            return nn.Sequential(*blocks)
        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    elif places == 'blockres':
        def make_block_temporal(stage, this_n_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if isinstance(b, models.resnet.Bottleneck):
                    # Store original conv1
                    conv1_original = b.conv1
                    # Create new conv1 with TemporalShift
                    b.conv1 = nn.Sequential(TemporalShift(this_n_segment, n_div, inplace=False), conv1_original) # Make sure inplace is False if not handled carefully
            return nn.Sequential(*blocks)
        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(f"Unsupported places: {places}")

class TSM_ResNet50(nn.Module):
    def __init__(self, num_classes, n_segment): # Removed pretrained_kinetics argument
        super(TSM_ResNet50, self).__init__()
        self.n_segment = n_segment
        self.num_classes = num_classes

        logging.info(f"Initializing TSM with ResNet50 backbone, num_segments={n_segment}")
        # We will load weights externally, so initialize ResNet50 without default ImageNet weights for clarity
        base_model = models.resnet50(weights=None) # Changed this
        
        make_temporal_shift(base_model, n_segment)

        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        num_fc_inputs = base_model.fc.in_features
        # Initialize the new FC layer for the target number of classes.
        # The pretrained FC layer from Kinetics (400 classes) will be ignored if checkpoint loading handles it.
        self.fc = nn.Linear(num_fc_inputs, num_classes)


    def forward(self, x):
        n, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(n * t, c, h, w)
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(n, t, -1)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

# ----- FUNCIONES AUXILIARES Y CLASES (remain largely the same) -----
def process_video_cv2_frames(video_path, num_frames_to_sample, frame_step, resize_dim, crop_size, is_train=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Error: No se pudo abrir el vídeo: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logging.warning(f"Error: El vídeo no tiene fotogramas o es inválido: {video_path}")
        return None
    selectable_frames_range = total_frames - (num_frames_to_sample - 1) * frame_step
    if selectable_frames_range > 0:
        if is_train:
            start_index_in_original_video = random.randint(0, selectable_frames_range - 1)
        else:
            start_index_in_original_video = selectable_frames_range // 2
        frame_indices = [start_index_in_original_video + i * frame_step for i in range(num_frames_to_sample)]
    else:
        available_indices = list(range(0, total_frames, frame_step))
        if not available_indices: available_indices = list(range(total_frames))
        if not available_indices:
            cap.release()
            logging.warning(f"No hay suficientes fotogramas para muestrear y el relleno falló para: {video_path}")
            return None
        frame_indices = available_indices[:num_frames_to_sample]
        while len(frame_indices) < num_frames_to_sample:
            frame_indices.append(available_indices[-1])
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"No se pudo leer el fotograma {int(frame_idx)} de {video_path}. Usando fotograma negro.")
            frame = np.zeros((resize_dim, resize_dim, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_dim, resize_dim))
        if is_train:
            top = random.randint(0, resize_dim - crop_size)
            left = random.randint(0, resize_dim - crop_size)
        else:
            top = (resize_dim - crop_size) // 2
            left = (resize_dim - crop_size) // 2
        frame = frame[top:top+crop_size, left:left+crop_size, :]
        frames.append(frame)
    cap.release()
    if len(frames) != num_frames_to_sample:
        logging.error(f"Error de procesamiento de fotogramas: se esperaban {num_frames_to_sample} fotogramas, se obtuvieron {len(frames)} para {video_path}")
        return None
    return frames

mean_tensor = torch.tensor(KINETICS_MEAN_LIST, dtype=torch.float32).view(1, 3, 1, 1)
std_tensor = torch.tensor(KINETICS_STD_LIST, dtype=torch.float32).view(1, 3, 1, 1)

tsm_transforms = Compose(
    [
        Lambda(lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32)),
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: x.permute(0, 3, 1, 2)),
        Lambda(lambda x: (x - mean_tensor) / std_tensor),
        Lambda(lambda x: x.permute(1, 0, 2, 3))
    ]
)

class VideoDatasetTSM(Dataset):
    def __init__(self, video_files, labels, transform_pipeline, num_frames, frame_step, resize_dim, crop_size, is_train=True, dataset_name=""):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform_pipeline
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resize_dim = resize_dim
        self.crop_size = crop_size
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames_list_hwc_rgb = process_video_cv2_frames(
            video_path, self.num_frames, self.frame_step, self.resize_dim, self.crop_size, self.is_train
        )
        if frames_list_hwc_rgb is None:
            dummy_tensor = torch.zeros((3, self.num_frames, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1)
        try:
            frames_tensor = self.transform(frames_list_hwc_rgb)
        except Exception as e:
            logging.error(f"Error aplicando transformaciones a {video_path}: {e}")
            dummy_tensor = torch.zeros((3, self.num_frames, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

def load_dataset_paths_and_labels(base_dir, split_folder_name, class_mapping, dataset_name_for_log=""):
    video_paths, labels = [], []
    split_path = os.path.join(base_dir, split_folder_name)
    logging.info(f"Cargando datos de {dataset_name_for_log} desde: {split_path}")
    for class_name, label_id in class_mapping.items():
        class_folder = os.path.join(split_path, class_name)
        if not os.path.isdir(class_folder):
            logging.warning(f"Carpeta de clase no encontrada: {class_folder}")
            continue
        video_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]
        logging.info(f"  Encontrados {len(video_files)} vídeos en {class_folder} para la clase '{class_name}'")
        for vf in video_files:
            video_paths.append(os.path.join(class_folder, vf))
            labels.append(label_id)
    if not video_paths:
        logging.warning(f"No se encontraron vídeos para {dataset_name_for_log} en {split_path}.")
    return video_paths, labels

def get_rwf2000_data(split_type):
    dataset_dir = os.path.join(BASE_DATA_DIR, RWF_2000_SUBDIR)
    return load_dataset_paths_and_labels(dataset_dir, SPLITS[split_type], CLASSES, "RWF-2000")
def get_hockey_data(split_type):
    dataset_dir = os.path.join(BASE_DATA_DIR, HOCKEY_FIGHTS_SUBDIR)
    if not os.path.exists(dataset_dir): return [], []
    return load_dataset_paths_and_labels(dataset_dir, SPLITS[split_type], CLASSES, "HockeyFights")
def get_rlvs_data(split_type):
    dataset_dir = os.path.join(BASE_DATA_DIR, RLVS_SUBDIR)
    if not os.path.exists(dataset_dir): return [], []
    return load_dataset_paths_and_labels(dataset_dir, SPLITS[split_type], CLASSES, "RLVS")

# REVISED FUNCTION TO LOAD PRETRAINED WEIGHTS WITH DETAILED KEY REMAPPING
def load_tsm_model_custom(num_model_classes=NUM_CLASSES_MODEL, checkpoint_path=None):
    # Initialize your model structure first
    model = TSM_ResNet50(num_classes=num_model_classes, n_segment=N_SEGMENTS)
    
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


# ----- ENTRENAMIENTO Y EVALUACIÓN (remain the same) -----
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp_flag=False):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples_processed = 0
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in progress_bar:
        valid_indices = labels != -1
        if not valid_indices.any(): continue
        inputs = inputs[valid_indices].to(device, non_blocking=True)
        labels = labels[valid_indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if inputs.size(0) == 0: continue
        with autocast(device_type=device.type, enabled=use_amp_flag):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if use_amp_flag and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _, preds = torch.max(outputs, 1)
        current_batch_size = inputs.size(0)
        running_loss += loss.item() * current_batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples_processed += current_batch_size
        progress_bar.set_postfix(loss=loss.item(), acc_batch=(torch.sum(preds == labels.data).item() / current_batch_size if current_batch_size > 0 else 0))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    epoch_loss = running_loss / total_samples_processed if total_samples_processed > 0 else 0
    epoch_acc_tensor = running_corrects.double() / total_samples_processed if total_samples_processed > 0 else torch.tensor(0.0)
    epoch_acc = epoch_acc_tensor.item() if isinstance(epoch_acc_tensor, torch.Tensor) else float(epoch_acc_tensor)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, use_amp_flag=False):
    model.eval()
    running_loss = 0.0
    total_samples_processed = 0
    all_preds_list, all_labels_list = [], []
    progress_bar = tqdm(loader, desc="Eval", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            valid_indices = labels != -1
            if not valid_indices.any(): continue
            inputs = inputs[valid_indices].to(device, non_blocking=True)
            labels = labels[valid_indices].to(device, non_blocking=True)
            if inputs.size(0) == 0: continue
            with autocast(device_type=device.type, enabled=use_amp_flag):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            current_batch_size = inputs.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples_processed += current_batch_size
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
    if total_samples_processed == 0:
        logging.warning("No valid samples processed during evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_loss = running_loss / total_samples_processed
    if not all_labels_list or not all_preds_list:
        logging.warning("Evaluation lists are empty, cannot compute metrics.")
        return epoch_loss, 0.0, 0.0, 0.0, 0.0
    acc = accuracy_score(all_labels_list, all_preds_list)
    unique_labels = np.unique(all_labels_list)
    # unique_preds = np.unique(all_preds_list) # Not used directly here but good for debugging
    avg_method = 'binary' if len(CLASSES) == 2 and len(unique_labels) > 1 else 'macro'

    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    return float(epoch_loss), float(acc), float(precision), float(recall), float(f1)

def measure_inference_fps_tsm(model, device, trials=TRIALS_FPS):
    dummy_input = torch.randn(1, 3, N_SEGMENTS, IMG_CROP_SIZE, IMG_CROP_SIZE, device=device)
    model.eval()
    for _ in range(10): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(trials): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return trials / total_time if total_time > 0 else 0.0

def save_training_metrics(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for key, values in history.items():
        if isinstance(values, list):
            history[key] = [float(v) if isinstance(v, (np.float32, np.float64, np.int64, np.int32, torch.Tensor)) else v for v in values]
        elif isinstance(values, dict):
             for sub_key, sub_val in values.items():
                 if isinstance(sub_val, (np.float32, np.float64, np.int64, np.int32, torch.Tensor)):
                     values[sub_key] = float(sub_val)
    with open(path, 'w') as f: json.dump(history, f, indent=4)
    logging.info(f"Métricas de entrenamiento guardadas en {path}")

# ----- MAIN LOOP (adaptado de SlowFast) -----
def main_train_loop(dataset_name="RWF-2000"):
    logging.info(f"Usando dispositivo: {DEVICE}")
    logging.info(f"Dataset seleccionado para entrenamiento: {dataset_name}")
    logging.info(f"Precisión Mixta Automática (AMP) habilitada: {USE_AMP}")
    logging.info(f"Modelo: TSM con ResNet50, N_SEGMENTS={N_SEGMENTS}")
    logging.info(f"Intentando cargar checkpoint preentrenado desde: {PRETRAINED_CHECKPOINT_PATH if PRETRAINED_CHECKPOINT_PATH else 'Ninguno'}")


    if dataset_name == "RWF-2000":
        train_video_paths, train_labels = get_rwf2000_data("train")
        val_video_paths, val_labels = get_rwf2000_data("val")
    elif dataset_name == "HockeyFights":
        train_video_paths, train_labels = get_hockey_data("train")
        val_video_paths, val_labels = get_hockey_data("val")
    elif dataset_name == "RLVS":
        train_video_paths, train_labels = get_rlvs_data("train")
        val_video_paths, val_labels = get_rlvs_data("val")
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")

    if not train_video_paths:
        logging.error(f"No se encontraron vídeos de entrenamiento para {dataset_name}. Saliendo.")
        return
    if not val_video_paths:
        logging.warning(f"No se encontraron vídeos de validación para {dataset_name}.")

    train_dataset = VideoDatasetTSM(train_video_paths, train_labels, tsm_transforms,
                                     N_SEGMENTS, FRAME_STEP, IMG_RESIZE_DIM, IMG_CROP_SIZE,
                                     is_train=True, dataset_name=dataset_name + "-train")
    num_data_workers = 2 if os.name == 'posix' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_data_workers, pin_memory=True)

    val_loader = None
    if val_video_paths:
        val_dataset = VideoDatasetTSM(val_video_paths, val_labels, tsm_transforms,
                                       N_SEGMENTS, FRAME_STEP, IMG_RESIZE_DIM, IMG_CROP_SIZE,
                                       is_train=False, dataset_name=dataset_name + "-val")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=num_data_workers, pin_memory=True)

    # Pass the checkpoint path to the model loading function
    model = load_tsm_model_custom(num_model_classes=NUM_CLASSES_MODEL, checkpoint_path=PRETRAINED_CHECKPOINT_PATH).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=USE_AMP and DEVICE.type == 'cuda')
    use_amp_for_training = USE_AMP and DEVICE.type == 'cuda'

    history = {'dataset': dataset_name, 'epochs_run': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
               'epoch_time_seconds': []}
    best_val_f1 = 0.0

    logging.info(f"Iniciando fine-tuning de {EPOCHS} épocas en {dataset_name}...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        logging.info(f"--- Época {epoch}/{EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, use_amp_for_training)
        logging.info(f"Época {epoch} Train: Pérdida={train_loss:.4f}, Acc={train_acc:.4f}")
        history['epochs_run'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        if val_loader and (val_loader.dataset is not None and len(val_loader.dataset) > 0):
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, DEVICE, use_amp_for_training)
            logging.info(f"Época {epoch} Val: Pérdida={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}")
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['val_precision'].append(val_prec); history['val_recall'].append(val_rec)
            history['val_f1'].append(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                logging.info(f"  Mejor F1 en validación: {best_val_f1:.4f}. Modelo guardado.")
        else:
            history['val_loss'].append(None); history['val_acc'].append(None)
            history['val_precision'].append(None); history['val_recall'].append(None)
            history['val_f1'].append(None)
            if epoch == EPOCHS:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                logging.info(f"Sin validación. Modelo de época {epoch} guardado.")

        epoch_duration = time.time() - epoch_start_time
        history['epoch_time_seconds'].append(float(epoch_duration))
        logging.info(f"Época {epoch} completada en {epoch_duration:.2f}s")
        save_training_metrics(history, METRICS_JSON_PATH)

    logging.info("Fine-tuning completado.")
    if val_loader and (val_loader.dataset is not None and len(val_loader.dataset) > 0) : logging.info(f"Mejor F1 en validación: {best_val_f1:.4f}.")
    logging.info(f"Mejor modelo guardado en: {BEST_MODEL_PATH}")
    logging.info(f"Métricas de entrenamiento guardadas en: {METRICS_JSON_PATH}")

    if os.path.exists(BEST_MODEL_PATH):
        logging.info(f"Cargando mejor modelo desde {BEST_MODEL_PATH} para análisis final.")
        model_final = load_tsm_model_custom(num_model_classes=NUM_CLASSES_MODEL, checkpoint_path=None) # Create new instance
        model_final.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model_final = model_final.to(DEVICE)
        model_final.eval()
    else:
        logging.warning("No se encontró el mejor modelo. Usando el modelo de la última época para análisis.")
        model_final = model
    model_final.eval()

    fps = measure_inference_fps_tsm(model_final, DEVICE)
    params_count = parameter_count(model_final)['']
    dummy_input_flops = torch.randn(1, 3, N_SEGMENTS, IMG_CROP_SIZE, IMG_CROP_SIZE, device=DEVICE)
    gflops = -1.0
    try:
        if hasattr(model_final, '_modules') and model_final._modules:
            flops_analyzer = FlopCountAnalysis(model_final, dummy_input_flops)
            gflops = flops_analyzer.total() / 1e9
        else:
            logging.warning("El modelo parece estar vacío o no es compatible con FlopCountAnalysis.")
    except Exception as e:
        logging.error(f"No se pudieron calcular los FLOPs: {e}.")

    logging.info(f"FPS de Inferencia del Modelo: {fps:.2f}")
    logging.info(f"Parámetros del Modelo: {params_count:,}")
    logging.info(f"GFLOPs del Modelo: {gflops:.2f}G")

    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, 'r') as f: final_metrics = json.load(f)
            final_metrics['performance_stats'] = {'fps': float(fps), 'parameters': int(params_count), 'gflops': float(gflops)}
            save_training_metrics(final_metrics, METRICS_JSON_PATH)
        except Exception as e:
            logging.error(f"Error al actualizar métricas con estadísticas de rendimiento: {e}")

if __name__ == '__main__':
    dataset_to_train = "RWF-2000"
    # Ensure PRETRAINED_CHECKPOINT_PATH is correctly set at the top of the script
    if not PRETRAINED_CHECKPOINT_PATH or not os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        logging.warning(f"El checkpoint preentrenado '{PRETRAINED_CHECKPOINT_PATH}' no fue encontrado. "
                        "El modelo se entrenará desde cero o con pesos base de ResNet si se configuró así.")
    main_train_loop(dataset_name=dataset_to_train)