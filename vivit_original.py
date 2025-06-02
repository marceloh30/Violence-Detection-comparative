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
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fvcore.nn import FlopCountAnalysis, parameter_count # Para análisis, aunque puede ser complicado con HF
from tqdm import tqdm
import logging
from torch.amp import GradScaler, autocast

# Importar desde el módulo de utilidades de preparación de datasets
from dataset_utils import (
    get_dataset_file_list,
    GLOBAL_CLASSES_MAP,
    BASE_DATA_DIR_DEFAULT,
    OUTPUT_LIST_DIR_DEFAULT
)

# ----- CONFIGURACIÓN -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Selección del Dataset de Entrenamiento ---
TRAIN_DATASET_NAME = "rwf2000"  # Opciones: "rwf2000", "rlvs"

CLASSES = GLOBAL_CLASSES_MAP

# --- Semilla para Reproducibilidad ---
RANDOM_SEED = 23

# --- Parámetros Específicos de ViViT ---
MODEL_NAME_VIVIT = "google/vivit-b-16x2-kinetics400" # Modelo base de Hugging Face
NUM_FRAMES_TO_SAMPLE_VIVIT = 32
FRAME_STEP_VIVIT = 4
IMG_RESIZE_DIM_AUG_VIVIT = 256 # Dimensión a la que se redimensiona antes del recorte aleatorio/central
VIDEO_IMAGE_SIZE_VIVIT = 224  # Tamaño final del frame/recorte para el modelo

# --- Hiperparámetros de Entrenamiento ---
BATCH_SIZE = 2 # Mantener bajo por memoria con ViViT
LR = 2e-5 # Learning rate más bajo, común para fine-tuning de Transformers
WEIGHT_DECAY = 1e-2 
EPOCHS = 5
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True # Para ahorrar memoria con ViViT
LOAD_CHECKPOINT_IF_EXISTS = True

# --- Rutas de Salida y Directorios de Listas de Archivos ---
OUTPUT_DIR_BASE = f"vivit_outputs_seed_{RANDOM_SEED}" 
FILE_LIST_DIR = OUTPUT_LIST_DIR_DEFAULT 
BASE_DATA_DIR = BASE_DATA_DIR_DEFAULT   
# Rutas específicas (log_dir_tensorboard, metrics_json_path, etc.) se definirán en main()

# --- Dispositivo y Eficiencia ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type 
NUM_DATA_WORKERS = 2 if os.name == 'posix' else 0

# --- Control de Flujo ---
PERFORM_TRAINING = True 
PERFORM_CROSS_INFERENCE = True 

TRIALS_FPS = 30 # ViViT puede ser más pesado para medir FPS


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
    

# ----- INICIALIZACIÓN DEL PROCESADOR ViViT (Global) -----
# Es importante que el procesador esté disponible para la función process_video_vivit
try:
    vivit_processor = VivitImageProcessor.from_pretrained(MODEL_NAME_VIVIT)
    # No realizar redimensionamiento/recorte en el procesador, se hará manualmente
    vivit_processor.do_resize = False
    vivit_processor.do_center_crop = False 
    # El tamaño se maneja en process_video_vivit antes de pasar al procesador
except Exception as e:
    logging.error(f"Fallo al cargar VivitImageProcessor desde '{MODEL_NAME_VIVIT}': {e}")
    logging.error("Asegúrate de tener conexión a internet o el modelo/procesador cacheado.")
    vivit_processor = None # Manejar este caso en main si es None

# ----- MODELO ViViT -----
def load_vivit_model(num_model_classes, pretrained_model_name=MODEL_NAME_VIVIT, image_size=VIDEO_IMAGE_SIZE_VIVIT):
    logging.info(f"Cargando modelo ViViT '{pretrained_model_name}' para {num_model_classes} clases, image_size={image_size}.")
    config = VivitConfig.from_pretrained(
        pretrained_model_name, 
        num_labels=num_model_classes, 
        image_size=image_size, # Asegurar que el config del modelo coincida
        ignore_mismatched_sizes=True # Permite cambiar el cabezal
    )
    
    try:
        # Cargar pesos preentrenados para la base ViViT, luego adjuntar un nuevo cabezal
        base_vivit_model = VivitForVideoClassification.from_pretrained(pretrained_model_name, ignore_mismatched_sizes=True)
        model = VivitForVideoClassification(config) # Nuevo modelo con el cabezal correcto
        model.vivit.load_state_dict(base_vivit_model.vivit.state_dict()) # Transferir pesos de la base
        logging.info(f"Pesos de la base ViViT transferidos desde '{pretrained_model_name}'.")
    except Exception as e:
        logging.warning(f"No se pudieron cargar los pesos preentrenados para {pretrained_model_name} debido a: {e}. Inicializando desde cero con la configuración.")
        model = VivitForVideoClassification(config)

    if USE_GRADIENT_CHECKPOINTING:
        try:
            model.vivit.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing habilitado para ViViT.")
        except Exception as e:
            logging.warning(f"No se pudo habilitar gradient checkpointing para ViViT: {e}")
    return model

# ----- PROCESAMIENTO DE VÍDEO Y DATASET -----
def process_video_vivit(path, num_frames_to_sample, frame_step, 
                        resize_dim_aug, final_image_size, 
                        image_processor_vivit, is_train=True):
    """Procesa un vídeo para ViViT. Devuelve un tensor pixel_values (T, C, H, W) o None."""
    if image_processor_vivit is None:
        logging.error("VivitImageProcessor no está disponible en process_video_vivit.")
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): logging.warning(f"ViViT Error: No se pudo abrir {path}"); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: cap.release(); logging.warning(f"ViViT Error: Vídeo inválido {path}"); return None

    # Lógica de muestreo de frames (similar a otros modelos)
    selectable_frames_range = total_frames - (num_frames_to_sample - 1) * frame_step
    if selectable_frames_range > 0:
        start_idx = random.randint(0, selectable_frames_range - 1) if is_train else selectable_frames_range // 2
        frame_indices = [start_idx + i * frame_step for i in range(num_frames_to_sample)]
    else:
        available_indices = list(range(0, total_frames, max(1, frame_step)))
        if not available_indices: available_indices = list(range(total_frames))
        if not available_indices: cap.release(); logging.warning(f"ViViT: No frames {path}"); return None
        frame_indices = available_indices[:num_frames_to_sample]
        while len(frame_indices) < num_frames_to_sample and available_indices:
            frame_indices.append(available_indices[-1])
    
    if not frame_indices and num_frames_to_sample > 0:
        cap.release(); logging.warning(f"ViViT: Fallo selección índices {path}"); return None

    raw_sampled_frames_rgb = []
    for frame_idx_orig in frame_indices:
        frame_idx = min(int(frame_idx_orig), total_frames - 1); frame_idx = max(0, frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"ViViT: No se leyó frame {frame_idx} de {path}. Usando negro.")
            # Usar resize_dim_aug para el dummy, ya que es el paso previo al recorte
            frame_rgb = np.zeros((resize_dim_aug, resize_dim_aug, 3), dtype=np.uint8)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_sampled_frames_rgb.append(frame_rgb)
    cap.release()

    if len(raw_sampled_frames_rgb) != num_frames_to_sample and num_frames_to_sample > 0:
        logging.error(f"ViViT Error: {len(raw_sampled_frames_rgb)} frames, se esperaban {num_frames_to_sample} para {path}")
        return None
    
    if num_frames_to_sample == 0: # Devolver forma correcta para 0 frames
        # T, C, H, W -> 0, 3, H, W (H y W son final_image_size)
        return torch.empty((0, 3, final_image_size, final_image_size), dtype=torch.float)


    # Aplicar redimensionamiento y recorte a cada frame
    augmented_frames_list = []
    for frame_np_rgb in raw_sampled_frames_rgb:
        try:
            # Redimensionar a una dimensión mayor para aumentación de datos (recorte aleatorio/central)
            resized_frame = cv2.resize(frame_np_rgb, (resize_dim_aug, resize_dim_aug), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logging.error(f"ViViT: Error redimensionando frame de {path}: {e}"); return None

        h_res, w_res, _ = resized_frame.shape
        
        if is_train: # Recorte aleatorio
            top = random.randint(0, h_res - final_image_size)
            left = random.randint(0, w_res - final_image_size)
        else: # Recorte central
            top = (h_res - final_image_size) // 2
            left = (w_res - final_image_size) // 2
        
        cropped_frame = resized_frame[top:top + final_image_size, left:left + final_image_size, :]
        augmented_frames_list.append(cropped_frame) # Lista de frames (H,W,C) ya en el tamaño final

    # Usar el procesador de ViViT
    try:
        # El procesador espera una lista de imágenes NumPy (H,W,C) o un tensor (N,H,W,C)
        # y devuelve un dict con 'pixel_values'
        inputs = image_processor_vivit(images=augmented_frames_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"] # Debería ser (T, C, H, W) si images es lista de T frames
        
        # El procesador puede añadir una dimensión de lote si solo se le pasa una lista para un vídeo.
        # Queremos (T, C, H, W) para el Dataset.
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1: # (1, T, C, H, W)
            pixel_values = pixel_values.squeeze(0)
        
        if pixel_values.shape[0] != num_frames_to_sample: # Verificar número de frames
             logging.error(f"ViViT: Discrepancia de frames tras procesador para {path}. Esperados {num_frames_to_sample}, obtenidos {pixel_values.shape[0]}")
             return None
        return pixel_values # (T, C, H, W)

    except Exception as e:
        logging.error(f"ViViT: Error procesando frames con ViViT processor para {path}: {e}"); return None


class VideoListDatasetVivit(Dataset):
    def __init__(self, file_list_data, num_frames, frame_step, 
                 resize_dim_aug, final_image_size, 
                 vivit_img_processor, is_train, dataset_name_log=""):
        self.file_list_data = file_list_data
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resize_dim_aug = resize_dim_aug
        self.final_image_size = final_image_size
        self.vivit_img_processor = vivit_img_processor # Pasar el procesador global
        self.is_train = is_train
        self.dataset_name_log = dataset_name_log

    def __len__(self):
        return len(self.file_list_data)

    def __getitem__(self, idx):
        item_data = self.file_list_data[idx]
        video_path, label = item_data['path'], item_data['label']

        pixel_values_tensor = process_video_vivit(
            video_path, self.num_frames, self.frame_step, 
            self.resize_dim_aug, self.final_image_size,
            self.vivit_img_processor, self.is_train
        )

        if pixel_values_tensor is None:
            logging.warning(f"ViViT: Fallo al procesar vídeo {video_path}. Devolviendo dummy.")
            # Dummy tensor (T, C, H, W)
            # Si num_frames es 0, T debe ser 0.
            t_dim = self.num_frames if self.num_frames > 0 else 0
            dummy_pixel_values = torch.zeros((t_dim, 3, self.final_image_size, self.final_image_size), dtype=torch.float)
            return dummy_pixel_values, torch.tensor(-1, dtype=torch.long)
            
        return pixel_values_tensor, torch.tensor(label, dtype=torch.long)

# ----- FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN -----
def train_epoch_vivit(model, loader, criterion, optimizer, device, use_amp_flag, scaler):
    model.train()
    running_loss, running_corrects, total_valid_samples = 0.0, 0, 0
    progress_bar = tqdm(loader, desc="Train ViViT", leave=False)

    for pixel_values_batch, labels_batch in progress_bar: # pixel_values_batch: (N, T, C, H, W)
        valid_indices = labels_batch != -1
        if not valid_indices.any(): continue

        pixel_values = pixel_values_batch[valid_indices].to(device, non_blocking=True)
        labels = labels_batch[valid_indices].to(device, non_blocking=True)
        if pixel_values.size(0) == 0: continue

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
            outputs = model(pixel_values=pixel_values).logits # ViViT devuelve un dict
            loss = criterion(outputs, labels)
        
        if use_amp_flag and scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
            
        _, preds = torch.max(outputs, 1)
        current_valid_samples = pixel_values.size(0)
        running_loss += loss.item() * current_valid_samples
        running_corrects += torch.sum(preds == labels.data)
        total_valid_samples += current_valid_samples
        
        if current_valid_samples > 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(torch.sum(preds == labels.data).item() / current_valid_samples):.4f}")
        if device.type == 'cuda': torch.cuda.empty_cache()

    epoch_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    epoch_acc = (running_corrects.double() / total_valid_samples if total_valid_samples > 0 else torch.tensor(0.0)).item()
    return epoch_loss, epoch_acc

def evaluate_vivit(model, loader, criterion, device, use_amp_flag, pos_label_value=1, num_classes_eval=len(CLASSES)):
    model.eval()
    running_loss, total_valid_samples = 0.0, 0
    all_preds_list, all_labels_list = [], []
    progress_bar = tqdm(loader, desc="Eval ViViT", leave=False)

    with torch.no_grad():
        for pixel_values_batch, labels_batch in progress_bar:
            valid_indices = labels_batch != -1
            if not valid_indices.any(): continue

            pixel_values = pixel_values_batch[valid_indices].to(device, non_blocking=True)
            labels_true = labels_batch[valid_indices].to(device, non_blocking=True)
            if pixel_values.size(0) == 0: continue

            with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
                outputs = model(pixel_values=pixel_values).logits
                loss = criterion(outputs, labels_true)
            
            _, preds = torch.max(outputs, 1)
            current_valid_samples = pixel_values.size(0)
            running_loss += loss.item() * current_valid_samples
            total_valid_samples += current_valid_samples
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels_true.cpu().numpy())
            
    if total_valid_samples == 0:
        logging.warning("ViViT Eval: No se procesaron muestras válidas.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    epoch_loss = running_loss / total_valid_samples
    # Lógica de métricas (acc, prec, rec, f1, cm) igual que en otros modelos
    acc = accuracy_score(all_labels_list, all_preds_list)
    avg_method = 'binary' if num_classes_eval == 2 else 'macro'
    unique_labels_in_data = np.unique(all_labels_list + all_preds_list)
    valid_pos_label = None
    if avg_method == 'binary': # Lógica de valid_pos_label copiada
        if pos_label_value in unique_labels_in_data: valid_pos_label = pos_label_value
        elif len(unique_labels_in_data) > 0:
            if len(unique_labels_in_data) == 2 and 0 in unique_labels_in_data:
                other_labels = [l for l in unique_labels_in_data if l != 0]; valid_pos_label = other_labels[0] if other_labels else None
            elif 1 in unique_labels_in_data: valid_pos_label = 1
            elif unique_labels_in_data: valid_pos_label = sorted(list(unique_labels_in_data))[0]
    if avg_method == 'binary' and valid_pos_label is None and len(unique_labels_in_data) > 1: avg_method = 'macro'
    
    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method=='binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    cm = confusion_matrix(all_labels_list, all_preds_list, labels=list(range(num_classes_eval))).tolist()
    return epoch_loss, acc, precision, recall, f1, cm

# ----- WRAPPER PARA ViViT PARA ANÁLISIS DE FLOPs -----
class VivitModelWrapperForFlops(nn.Module):
    def __init__(self, vivit_model_instance): # Renombrado para evitar conflicto con nombre de variable
        super().__init__()
        self.vivit_model_instance = vivit_model_instance # Renombrado para evitar conflicto
    def forward(self, pixel_values):
        return self.vivit_model_instance(pixel_values=pixel_values).logits

# ----- MEDICIÓN DE FPS Y GUARDADO DE MÉTRICAS -----
def measure_inference_fps_vivit(model, device, num_frames, image_size, trials=TRIALS_FPS):
    model.eval()
    # ViViT espera (N, T, C, H, W)
    actual_num_frames = num_frames if num_frames > 0 else 1
    dummy_input_shape = (1, actual_num_frames, 3, image_size, image_size)
    if num_frames == 0: logging.warning("FPS ViViT medido con T=1 para num_frames=0.")
        
    dummy_pixel_values = torch.randn(dummy_input_shape, device=device)
    with torch.no_grad():
        for _ in range(10): _ = model(pixel_values=dummy_pixel_values) # Warm-up
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(trials): _ = model(pixel_values=dummy_pixel_values)
        if device.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return trials / total_time if total_time > 0 else 0.0

# save_metrics_to_json (puede ser la misma que en otros scripts)
def save_metrics_to_json(metrics_dict, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        def convert_to_native_types(item):
            if isinstance(item, list): return [convert_to_native_types(i) for i in item]
            elif isinstance(item, dict): return {k: convert_to_native_types(v) for k, v in item.items()}
            elif isinstance(item, (np.integer, np.int_)): return int(item)
            elif isinstance(item, (np.floating, np.float_)): return float(item)
            elif isinstance(item, np.ndarray): return item.tolist()
            return item
        cleaned_metrics_dict = convert_to_native_types(metrics_dict)
        existing_data = {}
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            with open(json_path, 'r') as f:
                try: existing_data = json.load(f)
                except json.JSONDecodeError: logging.warning(f"JSON {json_path} corrupto.")
        for key, value in cleaned_metrics_dict.items():
            if key in existing_data and isinstance(existing_data[key], dict) and isinstance(value, dict):
                existing_data[key].update(value)
            else: existing_data[key] = value
        with open(json_path, 'w') as f: json.dump(existing_data, f, indent=4)
        logging.info(f"Métricas guardadas/actualizadas en {json_path}")
    except Exception as e: logging.error(f"Error al guardar métricas en JSON {json_path}: {e}")

# ----- FUNCIÓN PRINCIPAL -----
def main():
    set_seed(RANDOM_SEED) #Para reproducibilidad
    if vivit_processor is None:
        logging.error("VivitImageProcessor no se pudo inicializar. Saliendo del script ViViT.")
        return

    dataset_name_for_history = f"{TRAIN_DATASET_NAME}_ViViT"
    current_output_dir = os.path.join(OUTPUT_DIR_BASE, f"trained_on_{TRAIN_DATASET_NAME}")
    
    log_dir_tensorboard = os.path.join(current_output_dir, "logs_tensorboard_vivit")
    metrics_json_path = os.path.join(current_output_dir, f"train_metrics_vivit_{TRAIN_DATASET_NAME.lower()}.json") 
    best_model_path = os.path.join(current_output_dir, f"vivit_{TRAIN_DATASET_NAME.lower()}_best_model.pth")
    # Guardar también config y processor con el modelo
    vivit_config_save_dir = os.path.join(current_output_dir, f"vivit_{TRAIN_DATASET_NAME.lower()}_config_processor")


    os.makedirs(current_output_dir, exist_ok=True)
    os.makedirs(log_dir_tensorboard, exist_ok=True)
    os.makedirs(vivit_config_save_dir, exist_ok=True) # Directorio para config/processor
    # writer = SummaryWriter(log_dir_tensorboard) # Descomentar para TensorBoard

    model = load_vivit_model(
        num_model_classes=len(CLASSES), 
        pretrained_model_name=MODEL_NAME_VIVIT,
        image_size=VIDEO_IMAGE_SIZE_VIVIT
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp_for_training = USE_AMP and DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp_for_training)
    
    start_epoch_train = 1
    if PERFORM_TRAINING:
        logging.info(f"Iniciando entrenamiento del modelo ViViT en {TRAIN_DATASET_NAME}...")
        
        if LOAD_CHECKPOINT_IF_EXISTS and os.path.exists(best_model_path):
            logging.info(f"Cargando checkpoint ViViT desde {best_model_path}")
            try:
                # Para ViViT, es mejor cargar el state_dict en un modelo ya configurado.
                # La config y processor se cargarían por separado si se guardaron.
                checkpoint = torch.load(best_model_path, map_location=DEVICE)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and use_amp_for_training: scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    start_epoch_train = checkpoint.get('epoch', 0) + 1
                    best_val_f1_loaded = checkpoint.get('best_val_f1', 0.0)
                    logging.info(f"Checkpoint ViViT cargado. Reanudando desde época {start_epoch_train}, mejor F1 anterior: {best_val_f1_loaded:.4f}")
                else: # Si solo es el state_dict del modelo
                    model.load_state_dict(checkpoint)
                    logging.info("Checkpoint ViViT (solo state_dict del modelo) cargado. Reanudando desde época 1.")
                # No se cargan config/processor aquí, se asume que el modelo actual es compatible.
            except Exception as e:
                logging.error(f"Error al cargar checkpoint ViViT: {e}. Iniciando desde cero.")
                start_epoch_train = 1 # Resetear si falla la carga

        train_file_list = get_dataset_file_list(TRAIN_DATASET_NAME, "train", BASE_DATA_DIR, FILE_LIST_DIR)
        val_file_list = get_dataset_file_list(TRAIN_DATASET_NAME, "val", BASE_DATA_DIR, FILE_LIST_DIR)

        if not train_file_list:
            logging.error(f"No se pudo cargar lista de entrenamiento para {TRAIN_DATASET_NAME}. Abortando."); return
        
        train_dataset = VideoListDatasetVivit(
            train_file_list, NUM_FRAMES_TO_SAMPLE_VIVIT, FRAME_STEP_VIVIT, 
            IMG_RESIZE_DIM_AUG_VIVIT, VIDEO_IMAGE_SIZE_VIVIT, vivit_processor,
            is_train=True, dataset_name_log=f"{TRAIN_DATASET_NAME} Train ViViT"
        )
        val_dataset = VideoListDatasetVivit(
            val_file_list, NUM_FRAMES_TO_SAMPLE_VIVIT, FRAME_STEP_VIVIT,
            IMG_RESIZE_DIM_AUG_VIVIT, VIDEO_IMAGE_SIZE_VIVIT, vivit_processor,
            is_train=False, dataset_name_log=f"{TRAIN_DATASET_NAME} Val ViViT"
        ) if val_file_list else None
        
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True) if val_dataset and len(val_dataset) > 0 else None

        history = {} # Lógica de carga/inicialización de historial similar a SlowFast
        if os.path.exists(metrics_json_path) and LOAD_CHECKPOINT_IF_EXISTS and start_epoch_train > 1:
            try:
                with open(metrics_json_path, 'r') as f: history = json.load(f)
                for key_hist in ['epochs_run', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_cm', 'epoch_time_seconds']:
                    if key_hist not in history: history[key_hist] = []
            except Exception as e: logging.warning(f"No se pudo cargar historial ViViT: {e}. Creando nuevo."); history = {}
        
        if not history:
            history = {
                'dataset_trained_on': TRAIN_DATASET_NAME, 'model_name': MODEL_NAME_VIVIT,
                'hyperparameters': {'lr': LR, 'batch_size': BATCH_SIZE, 'epochs_config': EPOCHS, 
                                    'num_frames': NUM_FRAMES_TO_SAMPLE_VIVIT, 'frame_step': FRAME_STEP_VIVIT,
                                    'resize_dim_aug': IMG_RESIZE_DIM_AUG_VIVIT, 'video_image_size': VIDEO_IMAGE_SIZE_VIVIT,
                                    'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY, 
                                    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING},
                'epochs_run': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_cm': [],
                'epoch_time_seconds': []
            }
        best_val_f1 = max(history.get('val_f1', [0.0])) if history.get('val_f1') else 0.0

        logging.info(f"Config entrenamiento ViViT: Dataset={TRAIN_DATASET_NAME}, Frames={NUM_FRAMES_TO_SAMPLE_VIVIT}, EPOCHS={EPOCHS}")

        for epoch in range(start_epoch_train, EPOCHS + 1):
            epoch_start_time = time.time() # ... (bucle de entrenamiento similar a otros modelos) ...
            logging.info(f"--- Época {epoch}/{EPOCHS} (Entrenando ViViT en {TRAIN_DATASET_NAME}) ---")
            train_loss_val, train_acc_val = train_epoch_vivit(model, train_loader, criterion, optimizer, DEVICE, use_amp_for_training, scaler)
            history['epochs_run'].append(epoch); history['train_loss'].append(train_loss_val); history['train_acc'].append(train_acc_val)
            logging.info(f"Época {epoch} Train ViViT ({TRAIN_DATASET_NAME}): Pérdida={train_loss_val:.4f}, Acc={train_acc_val:.4f}")

            val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = 0.0,0.0,0.0,0.0,0.0,[]
            if val_loader:
                val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = evaluate_vivit(
                    model, val_loader, criterion, DEVICE, use_amp_for_training, 
                    pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
                )
                logging.info(f"Época {epoch} Val ViViT ({TRAIN_DATASET_NAME}): Pérdida={val_loss_val:.4f}, Acc={val_acc_val:.4f}, F1={val_f1_val:.4f}")
            history['val_loss'].append(val_loss_val if val_loader else None); # ... (resto de appends a history)
            history['val_acc'].append(val_acc_val if val_loader else None); history['val_precision'].append(val_prec_val if val_loader else None); 
            history['val_recall'].append(val_rec_val if val_loader else None); history['val_f1'].append(val_f1_val if val_loader else None); 
            history['val_cm'].append(val_cm_val if val_loader else None)

            epoch_duration_val = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(epoch_duration_val)
            logging.info(f"Época {epoch} ViViT ({TRAIN_DATASET_NAME}) completada en {epoch_duration_val:.2f}s")
            save_metrics_to_json(history, metrics_json_path)

            if val_loader and val_f1_val > best_val_f1:
                best_val_f1 = val_f1_val
                checkpoint_to_save = { # Guardar más info para ViViT si es necesario
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scaler_state_dict': scaler.state_dict() if use_amp_for_training else None,
                    'best_val_f1': best_val_f1,
                    'vivit_config': model.config.to_dict() # Guardar config del modelo
                }
                torch.save(checkpoint_to_save, best_model_path)
                # Guardar config y processor por separado también es buena práctica para HF
                model.config.save_pretrained(vivit_config_save_dir)
                vivit_processor.save_pretrained(vivit_config_save_dir)
                logging.info(f"  Mejor F1 Val ViViT ({TRAIN_DATASET_NAME}): {best_val_f1:.4f}. Checkpoint, config y processor guardados.")
            elif not val_loader and epoch == EPOCHS:
                 torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'vivit_config': model.config.to_dict()}, best_model_path)
                 model.config.save_pretrained(vivit_config_save_dir)
                 vivit_processor.save_pretrained(vivit_config_save_dir)
                 logging.info(f"Entrenamiento ViViT sin validación. Modelo de época {epoch}, config y processor guardados.")
        
        logging.info(f"Entrenamiento ViViT en {TRAIN_DATASET_NAME} completado.")
        if val_loader: logging.info(f"Mejor F1 Val ViViT ({TRAIN_DATASET_NAME}) final: {best_val_f1:.4f}.")
    else:
        logging.info("Entrenamiento ViViT omitido (PERFORM_TRAINING=False).")

    # --- ANÁLISIS FINAL DEL MODELO ---
    model_loaded_for_analysis = False; model_for_analysis = None
    if os.path.exists(best_model_path) and os.path.isdir(vivit_config_save_dir):
        logging.info(f"Cargando modelo ViViT desde {best_model_path} y config desde {vivit_config_save_dir} para análisis.")
        try:
            loaded_config = VivitConfig.from_pretrained(vivit_config_save_dir)
            model_for_analysis = VivitForVideoClassification(loaded_config).to(DEVICE)
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
            model_for_analysis.load_state_dict(state_dict_to_load)
            model_for_analysis.eval(); model_loaded_for_analysis = True
        except Exception as e: logging.error(f"Error al cargar ViViT desde ckpt/config: {e}. Intentando fallback."); model_for_analysis = None
    if not model_loaded_for_analysis and os.path.exists(best_model_path): # Fallback
        logging.info(f"Intentando cargar solo state_dict de ViViT desde {best_model_path}.")
        model_for_analysis = load_vivit_model(len(CLASSES), MODEL_NAME_VIVIT, VIDEO_IMAGE_SIZE_VIVIT).to(DEVICE)
        try:
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model_for_analysis.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model_for_analysis.eval(); model_loaded_for_analysis = True
        except Exception as e_fallback: logging.error(f"Fallo al cargar state_dict de ViViT: {e_fallback}"); model_for_analysis = None
    if not model_loaded_for_analysis:
        logging.warning(f"No se encontró/cargó modelo ViViT en {best_model_path}.")
        if PERFORM_TRAINING and 'model' in locals() and isinstance(model, nn.Module):
            model_for_analysis = model; model_for_analysis.eval(); model_loaded_for_analysis = True
            logging.info("Usando modelo ViViT de la última época para análisis.")
        else: logging.error("ViViT: No hay modelo para análisis.")

    if model_loaded_for_analysis:
        logging.info("Calculando estadísticas de rendimiento del modelo ViViT cargado...")
        fps = measure_inference_fps_vivit(model_for_analysis, DEVICE, NUM_FRAMES_TO_SAMPLE_VIVIT, VIDEO_IMAGE_SIZE_VIVIT, trials=TRIALS_FPS)
        try: params_count = parameter_count(model_for_analysis)['']
        except Exception as e: logging.error(f"Error con fvcore.parameter_count para ViViT: {e}. Sumando manualmente."); params_count = sum(p.numel() for p in model_for_analysis.parameters())
        gflops = -1.0 
        try:
            actual_t_flops = max(1, NUM_FRAMES_TO_SAMPLE_VIVIT)
            dummy_input_flops = torch.randn(1, actual_t_flops, 3, VIDEO_IMAGE_SIZE_VIVIT, VIDEO_IMAGE_SIZE_VIVIT, device=DEVICE)
            wrapped_model_for_flops = VivitModelWrapperForFlops(model_for_analysis).to(DEVICE).eval()
            logging.info("Intentando calcular GFLOPs para ViViT con el wrapper...")
            flops_analyzer = FlopCountAnalysis(wrapped_model_for_flops, (dummy_input_flops,))
            gflops = flops_analyzer.total() / 1e9
            logging.info(f"GFLOPs para ViViT calculados (aprox.): {gflops:.2f}G")
        except NotImplementedError as nie: logging.warning(f"FLOPs ViViT (NotImplementedError): {nie}. Reportando -1.0.")
        except Exception as e: logging.error(f"Error general FLOPs ViViT: {e}", exc_info=False); logging.warning("FLOPs ViViT falló. Reportando -1.0.")
        logging.info(f"Modelo ViViT Cargado - FPS: {fps:.2f}, Params: {params_count/1e6:.2f}M, GFLOPs: {gflops:.2f}G")
        if os.path.exists(metrics_json_path) or not PERFORM_TRAINING: # Guardar métricas de rendimiento
            try:
                final_metrics = {}; # ... (lógica de carga y guardado de final_metrics como en otros scripts)
                if os.path.exists(metrics_json_path):
                    with open(metrics_json_path, 'r') as f: final_metrics = json.load(f)
                final_metrics['hyperparameters'] = { 'model_name': MODEL_NAME_VIVIT, 'num_frames': NUM_FRAMES_TO_SAMPLE_VIVIT, # ...
                                                     'frame_step': FRAME_STEP_VIVIT, 'video_image_size': VIDEO_IMAGE_SIZE_VIVIT,
                                                     'batch_size': BATCH_SIZE, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                                                     'epochs_config': EPOCHS, 'use_amp': USE_AMP,
                                                     'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING }
                final_metrics['performance_stats'] = { 'fps': float(fps), 'parameters': int(params_count), 'gflops': float(gflops) }
                save_metrics_to_json(final_metrics, metrics_json_path)
            except Exception as e: logging.error(f"Error actualizando métricas ViViT: {e}")
    else: logging.warning("Análisis de rendimiento ViViT omitido.")

    if PERFORM_CROSS_INFERENCE and model_loaded_for_analysis:
        logging.info(f"\nComenzando inferencia cruzada con el modelo ViViT entrenado en {TRAIN_DATASET_NAME}...")
        datasets_for_cross_inference = []
        if TRAIN_DATASET_NAME == "rwf2000": datasets_for_cross_inference = ["rlvs", "hockey"]
        elif TRAIN_DATASET_NAME == "rlvs": datasets_for_cross_inference = ["rwf2000", "hockey"]
        
        for inference_ds_name in datasets_for_cross_inference:
            logging.info(f"--- Inferencia ViViT en: {inference_ds_name} ---")
            cross_inf_list = get_dataset_file_list(inference_ds_name, "all", BASE_DATA_DIR, FILE_LIST_DIR)
            if not cross_inf_list:
                logging.warning(f"No se pudo cargar lista de {inference_ds_name} para inferencia ViViT. Omitiendo."); continue
            
            cross_inf_dataset = VideoListDatasetVivit(
                cross_inf_list, NUM_FRAMES_TO_SAMPLE_VIVIT, FRAME_STEP_VIVIT, 
                IMG_RESIZE_DIM_AUG_VIVIT, VIDEO_IMAGE_SIZE_VIVIT, vivit_processor,
                is_train=False, dataset_name_log=f"{inference_ds_name} Cross-Inf ViViT"
            )
            if len(cross_inf_dataset) == 0:
                logging.warning(f"Dataset de inferencia ViViT {inference_ds_name} vacío. Omitiendo."); continue
            
            cross_inf_loader = DataLoader(cross_inf_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True)
            inf_loss, inf_acc, inf_prec, inf_rec, inf_f1, inf_cm = evaluate_vivit(
                model_for_analysis, cross_inf_loader, criterion, DEVICE, use_amp_for_training, 
                pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
            )
            logging.info(f"Resultados Inferencia ViViT en {inference_ds_name} (entrenado en {TRAIN_DATASET_NAME}):")
            logging.info(f"  Loss: {inf_loss:.4f}, Acc: {inf_acc:.4f}, F1 (Violence): {inf_f1:.4f}")
            
            cross_inf_metrics_path = os.path.join(current_output_dir, f"cross_inf_vivit_on_{inference_ds_name}_trained_{TRAIN_DATASET_NAME.lower()}.json")
            current_cross_metrics = {f'cross_inference_vivit_on_{inference_ds_name}': {
                'model_trained_on': TRAIN_DATASET_NAME, 'evaluated_on': f"{inference_ds_name}_full",
                'loss': inf_loss, 'accuracy': inf_acc, 'precision_violence': inf_prec, 
                'recall_violence': inf_rec, 'f1_score_violence': inf_f1, 'confusion_matrix': inf_cm
            }}
            save_metrics_to_json(current_cross_metrics, cross_inf_metrics_path)

    # if 'writer' in locals(): writer.close() # Si se usa TensorBoard
    logging.info("Proceso ViViT completado.")

if __name__ == '__main__':
    main()
