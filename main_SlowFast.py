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

# --- Parámetros Específicos de SlowFast ---
NUM_FRAMES_TO_SAMPLE_SF = 32 
FRAME_STEP_SF = 4            
IMG_CROP_SIZE_SF = 224       # Tamaño final del frame después de cualquier recorte
IMG_RESIZE_DIM_SF = 256      # Dimensión a la que se redimensiona antes del recorte (si crop_size < resize_dim)

ALPHA_SLOWFAST = 4           
NUM_FRAMES_FAST_PATHWAY = NUM_FRAMES_TO_SAMPLE_SF
NUM_FRAMES_SLOW_PATHWAY = NUM_FRAMES_FAST_PATHWAY // ALPHA_SLOWFAST

# --- Hiperparámetros de Entrenamiento ---
BATCH_SIZE = 16 
LR = 1e-5 
WEIGHT_DECAY = 2e-5
EPOCHS = 30 
USE_AMP = True 
LOAD_CHECKPOINT_IF_EXISTS = True 

# --- Rutas de Salida y Directorios de Listas de Archivos ---
OUTPUT_DIR_BASE = f"slowfast_r50_outputs_seed_{RANDOM_SEED}"
FILE_LIST_DIR = OUTPUT_LIST_DIR_DEFAULT 
BASE_DATA_DIR = BASE_DATA_DIR_DEFAULT   

# --- Dispositivo y Eficiencia ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type 
NUM_DATA_WORKERS = 2 if os.name == 'posix' else 0

# --- Control de Flujo ---
PERFORM_TRAINING = True 
PERFORM_CROSS_INFERENCE = True 

# --- Constantes de Normalización (Kinetics) ---
KINETICS_MEAN_LIST = [0.45, 0.45, 0.45] 
KINETICS_STD_LIST = [0.225, 0.225, 0.225]
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

# ----- CLASES Y FUNCIONES ESPECÍFICAS DE SLOWFAST -----
class PackPathwayCustom(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor): # frames: (C, T, H, W)
        fast_pathway = frames
        num_frames_for_slow = max(1, frames.shape[1] // self.alpha) if frames.shape[1] > 0 else 0
        
        if frames.shape[1] == 0: 
             return [frames.clone(), frames.clone()]

        slow_pathway_indices = torch.linspace(
            0, frames.shape[1] - 1, num_frames_for_slow 
        ).long()
        
        if num_frames_for_slow > 0 and slow_pathway_indices.numel() == 0 and frames.shape[1] > 0:
            slow_pathway_indices = torch.tensor([0], dtype=torch.long, device=frames.device)
        elif num_frames_for_slow == 0 : 
             return [frames.clone()[:,0:0,:,:], fast_pathway.clone()] 

        slow_pathway = torch.index_select(frames, 1, slow_pathway_indices.to(frames.device))
        return [slow_pathway, fast_pathway]

mean_tensor_sf = torch.tensor(KINETICS_MEAN_LIST, dtype=torch.float32).view(3, 1, 1, 1) 
std_tensor_sf = torch.tensor(KINETICS_STD_LIST, dtype=torch.float32).view(3, 1, 1, 1)  

slowfast_custom_transforms = Compose( 
    [
        Lambda(lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32) if len(x) > 0 else torch.empty((0, IMG_CROP_SIZE_SF, IMG_CROP_SIZE_SF, 3), dtype=torch.float32)),
        Lambda(lambda x: x / 255.0 if x.numel() > 0 else x), 
        Lambda(lambda x: x.permute(3, 0, 1, 2) if x.numel() > 0 and x.ndim == 4 else torch.empty((3, 0, IMG_CROP_SIZE_SF, IMG_CROP_SIZE_SF), dtype=x.dtype if hasattr(x, 'dtype') else torch.float32)),
        Lambda(lambda x: (x - mean_tensor_sf.to(x.device)) / std_tensor_sf.to(x.device) if x.numel() > 0 and x.shape[1]>0 else x), 
        PackPathwayCustom(alpha=ALPHA_SLOWFAST) 
    ]
)

def load_slowfast_model_custom(num_model_classes, pretrained=True):
    logging.info(f"Cargando modelo SlowFast_R50 (pretrained={pretrained}) para {num_model_classes} clases.")
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        original_head_in_features = model.blocks[6].proj.in_features
        model.blocks[6].proj = nn.Linear(original_head_in_features, num_model_classes)
    except Exception as e:
        logging.error(f"Error al cargar el modelo SlowFast desde torch.hub: {e}"); raise
    return model

# ----- PROCESAMIENTO DE VÍDEO Y DATASET -----
def process_video_slowfast(path, num_frames_to_sample, frame_step, resize_dim, crop_size, is_train=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): logging.warning(f"SF Error: No se pudo abrir {path}"); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: cap.release(); logging.warning(f"SF Error: Vídeo inválido {path}"); return None

    if num_frames_to_sample == 0:
        cap.release(); return [] 

    selectable_frames_range = total_frames - (num_frames_to_sample - 1) * frame_step
    if selectable_frames_range > 0:
        start_idx = random.randint(0, selectable_frames_range - 1) if is_train else selectable_frames_range // 2
        frame_indices = [start_idx + i * frame_step for i in range(num_frames_to_sample)]
    else:
        available_indices = list(range(0, total_frames, max(1, frame_step)))
        if not available_indices: available_indices = list(range(total_frames))
        if not available_indices: cap.release(); logging.warning(f"SF: No frames {path}"); return None
        frame_indices = available_indices[:num_frames_to_sample]
        while len(frame_indices) < num_frames_to_sample and available_indices:
            frame_indices.append(available_indices[-1])
    
    if not frame_indices: 
        cap.release(); logging.warning(f"SF: Fallo selección índices {path}"); return None

    frames_list = []
    for frame_idx_orig in frame_indices:
        frame_idx = min(int(frame_idx_orig), total_frames - 1); frame_idx = max(0, frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logging.warning(f"SF: No se leyó frame {frame_idx} de {path}. Usando negro.")
            frame_processed = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        else:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_processed.shape[0] != resize_dim or frame_processed.shape[1] != resize_dim:
                frame_processed = cv2.resize(frame_processed, (resize_dim, resize_dim))
            if resize_dim > crop_size: 
                if is_train:
                    # Aplico volteo horizontal con prob 50%
                    if random.random() < 0.5: # 50% de probabilidad
                        frame_processed = cv2.flip(frame_processed, 1) # 1 para volteo horizontal

                    top = random.randint(0, resize_dim - crop_size)
                    left = random.randint(0, resize_dim - crop_size)
                else:
                    top = (resize_dim - crop_size) // 2
                    left = (resize_dim - crop_size) // 2
                frame_processed = frame_processed[top:top+crop_size, left:left+crop_size, :]
            elif resize_dim < crop_size: 
                if random.random() < 0.5: # 50% de probabilidad
                    frame_processed = cv2.flip(frame_processed, 1) # 1 para volteo horizontal
                    
                frame_processed = cv2.resize(frame_processed, (crop_size, crop_size))
        frames_list.append(frame_processed)
    cap.release()

    if len(frames_list) != num_frames_to_sample :
        logging.error(f"SF Error: {len(frames_list)} frames, se esperaban {num_frames_to_sample} para {path}")
        return None 
    return frames_list

class VideoListDatasetSlowFast(Dataset):
    def __init__(self, file_list_data, num_frames_fast, frame_step, resize_dim, crop_size, transforms, is_train, dataset_name_log=""):
        self.file_list_data = file_list_data
        self.num_frames_fast = num_frames_fast 
        self.num_frames_slow = max(1, num_frames_fast // ALPHA_SLOWFAST) if num_frames_fast > 0 else 0
        self.frame_step = frame_step
        self.resize_dim = resize_dim
        self.crop_size = crop_size 
        self.transforms = transforms
        self.is_train = is_train
        self.dataset_name_log = dataset_name_log

    def __len__(self):
        return len(self.file_list_data)

    def _get_dummy_data(self):
        t_slow = self.num_frames_slow
        t_fast = self.num_frames_fast
        dummy_s = torch.zeros((3, t_slow, self.crop_size, self.crop_size), dtype=torch.float32)
        dummy_f = torch.zeros((3, t_fast, self.crop_size, self.crop_size), dtype=torch.float32)
        return [dummy_s, dummy_f], torch.tensor(-1, dtype=torch.long)

    def __getitem__(self, idx):
        item_data = self.file_list_data[idx]
        video_path, label = item_data['path'], item_data['label']

        if self.num_frames_fast == 0: 
            return self._get_dummy_data()

        frames_np_list = process_video_slowfast(
            video_path, self.num_frames_fast, self.frame_step, self.resize_dim, self.crop_size, self.is_train
        )

        if frames_np_list is None or len(frames_np_list) != self.num_frames_fast :
            logging.warning(f"SF: Fallo al procesar vídeo {video_path} o discrepancia de frames. Devolviendo dummy.")
            return self._get_dummy_data()
        
        try:
            packed_frames_list = self.transforms(frames_np_list) 
        except Exception as e:
            logging.error(f"SF: Error aplicando transformaciones a {video_path}: {e}", exc_info=True) 
            return self._get_dummy_data()
            
        return packed_frames_list, torch.tensor(label, dtype=torch.long)

def train_epoch_slowfast(model, loader, criterion, optimizer, device, use_amp_flag, scaler):
    model.train()
    running_loss, running_corrects, total_valid_samples = 0.0, 0, 0
    progress_bar = tqdm(loader, desc="Train SlowFast", leave=False)

    for inputs_list, labels_batch in progress_bar: 
        valid_indices = labels_batch != -1
        if not valid_indices.any(): continue

        inputs_on_device = [tensor[valid_indices].to(device, non_blocking=True) for tensor in inputs_list]
        labels = labels_batch[valid_indices].to(device, non_blocking=True)
        
        if inputs_on_device[0].size(0) == 0: continue 

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
            outputs = model(inputs_on_device) 
            loss = criterion(outputs, labels)
        
        if use_amp_flag and scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
            
        _, preds = torch.max(outputs, 1)
        current_valid_samples = inputs_on_device[0].size(0)
        running_loss += loss.item() * current_valid_samples
        running_corrects += torch.sum(preds == labels.data)
        total_valid_samples += current_valid_samples
        
        if current_valid_samples > 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(torch.sum(preds == labels.data).item() / current_valid_samples):.4f}")
        if device.type == 'cuda': torch.cuda.empty_cache()

    epoch_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    epoch_acc = (running_corrects.double() / total_valid_samples if total_valid_samples > 0 else torch.tensor(0.0)).item()
    return epoch_loss, epoch_acc

def evaluate_slowfast(model, loader, criterion, device, use_amp_flag, pos_label_value=1, num_classes_eval=len(CLASSES)):
    model.eval()
    running_loss, total_valid_samples = 0.0, 0
    all_preds_list, all_labels_list = [], []
    progress_bar = tqdm(loader, desc="Eval SlowFast", leave=False)

    with torch.no_grad():
        for inputs_list, labels_batch in progress_bar:
            valid_indices = labels_batch != -1
            if not valid_indices.any(): continue

            inputs_on_device = [tensor[valid_indices].to(device, non_blocking=True) for tensor in inputs_list]
            labels_true = labels_batch[valid_indices].to(device, non_blocking=True)
            if inputs_on_device[0].size(0) == 0: continue

            with autocast(device_type=DEVICE_TYPE, enabled=use_amp_flag):
                outputs = model(inputs_on_device)
                loss = criterion(outputs, labels_true)
            
            _, preds = torch.max(outputs, 1)
            current_valid_samples = inputs_on_device[0].size(0)
            running_loss += loss.item() * current_valid_samples
            total_valid_samples += current_valid_samples
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels_true.cpu().numpy())
            
    if total_valid_samples == 0:
        logging.warning("SF Eval: No se procesaron muestras válidas.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    epoch_loss = running_loss / total_valid_samples
    acc = accuracy_score(all_labels_list, all_preds_list)
    avg_method = 'binary' if num_classes_eval == 2 else 'macro'
    unique_labels_in_data = np.unique(all_labels_list + all_preds_list)
    valid_pos_label = None
    if avg_method == 'binary':
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

def measure_inference_fps_slowfast(model, device, num_frames_fast, num_frames_slow, crop_size, trials=TRIALS_FPS):
    model.eval()
    # Warm-up: Regenerate dummy input for each iteration
    for _ in range(10): 
        dummy_slow_iter = torch.randn(1, 3, num_frames_slow, crop_size, crop_size, device=device)
        dummy_fast_iter = torch.randn(1, 3, num_frames_fast, crop_size, crop_size, device=device)
        dummy_input_iter = [dummy_slow_iter, dummy_fast_iter]
        _ = model(dummy_input_iter) 
        
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(trials):
        # Regenerate dummy input for each timing iteration
        dummy_slow_iter = torch.randn(1, 3, num_frames_slow, crop_size, crop_size, device=device)
        dummy_fast_iter = torch.randn(1, 3, num_frames_fast, crop_size, crop_size, device=device)
        dummy_input_iter = [dummy_slow_iter, dummy_fast_iter]
        _ = model(dummy_input_iter)
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
def save_metrics_to_json(metrics_dict, json_path): 
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        def convert_to_native_types(item):
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
                except json.JSONDecodeError: logging.warning(f"JSON {json_path} corrupto.")
        for key, value in cleaned_metrics_dict.items():
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
    
    dataset_name_for_history = f"{TRAIN_DATASET_NAME}_SlowFast_R50"
    current_output_dir = os.path.join(OUTPUT_DIR_BASE, f"trained_on_{TRAIN_DATASET_NAME}")
    
    log_dir_tensorboard = os.path.join(current_output_dir, "logs_tensorboard_slowfast") 
    metrics_json_path = os.path.join(current_output_dir, f"train_metrics_slowfast_{TRAIN_DATASET_NAME.lower()}.json") 
    best_model_path = os.path.join(current_output_dir, f"slowfast_{TRAIN_DATASET_NAME.lower()}_best_model.pth")

    os.makedirs(current_output_dir, exist_ok=True)
    # os.makedirs(log_dir_tensorboard, exist_ok=True) 

    model = load_slowfast_model_custom(num_model_classes=len(CLASSES), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # T_max es el número total de épocas. El scheduler ajustará el LR en cada época.
    # eta_min=0 significa que el LR llegará a 0 al final de las EPOCHS.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    use_amp_for_training = USE_AMP and DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp_for_training)
    
    start_epoch_train = 1
    if PERFORM_TRAINING:
        logging.info(f"Iniciando entrenamiento del modelo SlowFast en {TRAIN_DATASET_NAME}...")
        logging.info(f"Device: {DEVICE.type}")
        if LOAD_CHECKPOINT_IF_EXISTS and os.path.exists(best_model_path):
            logging.info(f"Cargando checkpoint SF desde {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and use_amp_for_training : scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch_train = checkpoint.get('epoch', 0) + 1
            best_val_f1_loaded = checkpoint.get('best_val_f1', 0.0)
            logging.info(f"Checkpoint SF cargado. Reanudando desde época {start_epoch_train}, mejor F1 anterior: {best_val_f1_loaded:.4f}")
        
        train_file_list = get_dataset_file_list(TRAIN_DATASET_NAME, "train", BASE_DATA_DIR, FILE_LIST_DIR)
        val_file_list = get_dataset_file_list(TRAIN_DATASET_NAME, "val", BASE_DATA_DIR, FILE_LIST_DIR)

        if not train_file_list:
            logging.error(f"No se pudo cargar lista de entrenamiento para {TRAIN_DATASET_NAME}. Abortando."); return
        
        train_dataset = VideoListDatasetSlowFast(
            train_file_list, NUM_FRAMES_FAST_PATHWAY, FRAME_STEP_SF, IMG_RESIZE_DIM_SF, IMG_CROP_SIZE_SF,
            slowfast_custom_transforms, is_train=True, dataset_name_log=f"{TRAIN_DATASET_NAME} Train SF"
        )
        val_dataset = VideoListDatasetSlowFast(
            val_file_list, NUM_FRAMES_FAST_PATHWAY, FRAME_STEP_SF, IMG_RESIZE_DIM_SF, IMG_CROP_SIZE_SF,
            slowfast_custom_transforms, is_train=False, dataset_name_log=f"{TRAIN_DATASET_NAME} Val SF"
        ) if val_file_list else None
        
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker) if val_dataset and len(val_dataset) > 0 else None

        history = {}
        if os.path.exists(metrics_json_path) and LOAD_CHECKPOINT_IF_EXISTS and start_epoch_train > 1:
            try:
                with open(metrics_json_path, 'r') as f: history = json.load(f)
                for key_hist in ['epochs_run', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_cm', 'epoch_time_seconds']:
                    if key_hist not in history: history[key_hist] = []
            except Exception as e: logging.warning(f"No se pudo cargar historial SF: {e}. Creando nuevo."); history = {}
        
        if not history:
            history = { 
                'dataset_trained_on': TRAIN_DATASET_NAME, 'model_name': 'SlowFast_R50',
                'hyperparameters': {'lr': LR, 'batch_size': BATCH_SIZE, 'epochs_config': EPOCHS, 
                                    'num_frames_fast': NUM_FRAMES_FAST_PATHWAY, 'alpha_slowfast': ALPHA_SLOWFAST,
                                    'frame_step': FRAME_STEP_SF, 'resize_dim': IMG_RESIZE_DIM_SF, 'crop_size': IMG_CROP_SIZE_SF,
                                    'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY},
                'epochs_run': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_cm': [],
                'epoch_time_seconds': []
            }
        best_val_f1 = 0.0
        valid_f1_scores = [f1 for f1 in history.get('val_f1', []) if f1 is not None]
        if valid_f1_scores:
            best_val_f1 = max(valid_f1_scores)

        logging.info(f"Config entrenamiento SF: Dataset={TRAIN_DATASET_NAME}, FramesFast={NUM_FRAMES_FAST_PATHWAY}, Alpha={ALPHA_SLOWFAST}, EPOCHS={EPOCHS}")

        for epoch in range(start_epoch_train, EPOCHS + 1): 
            epoch_start_time = time.time()
            logging.info(f"--- Época {epoch}/{EPOCHS} (Entrenando SlowFast en {TRAIN_DATASET_NAME}) ---")
            train_loss_val, train_acc_val = train_epoch_slowfast(model, train_loader, criterion, optimizer, DEVICE, use_amp_for_training, scaler)
            history['epochs_run'].append(epoch); history['train_loss'].append(train_loss_val); history['train_acc'].append(train_acc_val)
            logging.info(f"Época {epoch} Train SF ({TRAIN_DATASET_NAME}): Pérdida={train_loss_val:.4f}, Acc={train_acc_val:.4f}")

            val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = 0.0,0.0,0.0,0.0,0.0,[]
            if val_loader:
                val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = evaluate_slowfast(
                    model, val_loader, criterion, DEVICE, use_amp_for_training, 
                    pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
                )
                logging.info(f"Época {epoch} Val SF ({TRAIN_DATASET_NAME}): Pérdida={val_loss_val:.4f}, Acc={val_acc_val:.4f}, F1={val_f1_val:.4f}")
            history['val_loss'].append(val_loss_val if val_loader else None); history['val_acc'].append(val_acc_val if val_loader else None); 
            history['val_precision'].append(val_prec_val if val_loader else None); history['val_recall'].append(val_rec_val if val_loader else None); 
            history['val_f1'].append(val_f1_val if val_loader else None); history['val_cm'].append(val_cm_val if val_loader else None)
            epoch_duration_val = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(epoch_duration_val)
            logging.info(f"Época {epoch} SF ({TRAIN_DATASET_NAME}) completada en {epoch_duration_val:.2f}s")
            save_or_update_json(history, metrics_json_path)
            
            # Actualiza la tasa de aprendizaje para la siguiente época
            scheduler.step()
            # Opcional: registrar el nuevo LR para verificar que está funcionando
            logging.info(f"LR para la época {epoch + 1}: {scheduler.get_last_lr()[0]:.7f}")
            

            if val_loader and val_f1_val > best_val_f1: 
                best_val_f1 = val_f1_val
                checkpoint_data = { 'epoch': epoch, 'model_state_dict': model.state_dict(), 
                                   'optimizer_state_dict': optimizer.state_dict(), 
                                   'scaler_state_dict': scaler.state_dict() if use_amp_for_training else None,
                                   'best_val_f1': best_val_f1}
                torch.save(checkpoint_data, best_model_path)
                logging.info(f"  Mejor F1 Val SF ({TRAIN_DATASET_NAME}): {best_val_f1:.4f}. Checkpoint guardado.")
            elif not val_loader and epoch == EPOCHS:
                 torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, best_model_path)
                 logging.info(f"Entrenamiento SF sin validación. Modelo de época {epoch} guardado.")
        logging.info(f"Entrenamiento SF en {TRAIN_DATASET_NAME} completado.")
        if val_loader: logging.info(f"Mejor F1 Val SF ({TRAIN_DATASET_NAME}) final: {best_val_f1:.4f}.")
    else:
        logging.info("Entrenamiento SF omitido (PERFORM_TRAINING=False).")

    # --- ANÁLISIS FINAL DEL MODELO (CARGA MEJORADO) ---
    model_for_analysis = None 
    model_loaded_for_analysis = False
    if os.path.exists(best_model_path): 
        logging.info(f"Cargando modelo SF desde {best_model_path} para análisis/inferencia.")
        # Cargar con pretrained=True para la misma base que el modelo de entrenamiento
        model_for_analysis = load_slowfast_model_custom(len(CLASSES), pretrained=True).to(DEVICE) 
        
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        # Obtener el state_dict del modelo del checkpoint
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint) # Maneja ambos formatos
        
        try:
            model_for_analysis.load_state_dict(state_dict_to_load)
            logging.info("State_dict del checkpoint cargado en model_for_analysis.")
            model_for_analysis.eval()
            model_loaded_for_analysis = True
        except RuntimeError as e:
            logging.error(f"Error al cargar state_dict en model_for_analysis: {e}")
            logging.warning("Intentando cargar con strict=False.")
            try:
                model_for_analysis.load_state_dict(state_dict_to_load, strict=False)
                logging.info("State_dict cargado con strict=False.")
                model_for_analysis.eval()
                model_loaded_for_analysis = True
            except Exception as e_strict_false:
                logging.error(f"Falló la carga incluso con strict=False: {e_strict_false}")
                model_for_analysis = None # Indicar que la carga falló
    else: 
        logging.warning(f"No se encontró modelo SF en {best_model_path}.")
        if PERFORM_TRAINING and 'model' in locals() and isinstance(model, nn.Module):
            # 'model' es la instancia que se usó (y posiblemente se modificó) durante el entrenamiento
            model_for_analysis = model # Reutilizar el modelo de entrenamiento
            model_for_analysis.eval() 
            model_loaded_for_analysis = True
            logging.info("Usando modelo SF de la última época de entrenamiento para análisis (mejor modelo no encontrado o no se guardó).")
        else: 
            logging.error("SF: No hay modelo entrenado en memoria y no se encontró checkpoint. No se puede realizar análisis.")

    if model_loaded_for_analysis: 
        logging.info("Calculando estadísticas de rendimiento del modelo SF cargado...")
        fps = measure_inference_fps_slowfast(model_for_analysis, DEVICE, NUM_FRAMES_FAST_PATHWAY, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE_SF)
        params_count = parameter_count(model_for_analysis).get('', 0)
        gflops = -1.0
        try:
            dummy_s_shape = (1, 3, max(1,NUM_FRAMES_SLOW_PATHWAY), IMG_CROP_SIZE_SF, IMG_CROP_SIZE_SF)
            dummy_f_shape = (1, 3, max(1,NUM_FRAMES_FAST_PATHWAY), IMG_CROP_SIZE_SF, IMG_CROP_SIZE_SF)
            dummy_input_flops = [torch.randn(dummy_s_shape, device=DEVICE), torch.randn(dummy_f_shape, device=DEVICE)]
            if hasattr(model_for_analysis, '_modules') and model_for_analysis._modules:
                 flops_analyzer = FlopCountAnalysis(model_for_analysis, dummy_input_flops)
                 gflops = flops_analyzer.total() / 1e9
            else: logging.warning("Modelo SF de análisis vacío.")
        except Exception as e: logging.error(f"No se pudieron calcular los FLOPs para SF: {e}")
        logging.info(f"Modelo SF Cargado - FPS: {fps:.2f}, Params: {params_count/1e6:.2f}M, GFLOPs: {gflops:.2f}G")
        performance_stats_data = {'performance_stats': { 'fps': float(fps), 'parameters': int(params_count), 'gflops': float(gflops) }}
        save_or_update_json(performance_stats_data, metrics_json_path)
    else:
        logging.warning("Análisis de rendimiento SF omitido porque el modelo no se pudo cargar.")

    if PERFORM_CROSS_INFERENCE and model_loaded_for_analysis: 
        logging.info(f"\nComenzando inferencia cruzada con el modelo SF entrenado en {TRAIN_DATASET_NAME}...")
        datasets_for_cross_inference = []
        if TRAIN_DATASET_NAME == "rwf2000": datasets_for_cross_inference = ["rlvs", "hockey"]
        elif TRAIN_DATASET_NAME == "rlvs": datasets_for_cross_inference = ["rwf2000", "hockey"]
        
        for inference_ds_name in datasets_for_cross_inference:
            logging.info(f"--- Inferencia SF en: {inference_ds_name} ---")
            cross_inf_list = get_dataset_file_list(inference_ds_name, "all", BASE_DATA_DIR, FILE_LIST_DIR)
            if not cross_inf_list: logging.warning(f"No se pudo cargar lista de {inference_ds_name}. Omitiendo."); continue
            
            cross_inf_dataset = VideoListDatasetSlowFast(
                cross_inf_list, NUM_FRAMES_FAST_PATHWAY, FRAME_STEP_SF, IMG_RESIZE_DIM_SF, IMG_CROP_SIZE_SF,
                slowfast_custom_transforms, is_train=False, dataset_name_log=f"{inference_ds_name} Cross-Inf SF"
            )
            if len(cross_inf_dataset) == 0: logging.warning(f"Dataset de inferencia SF {inference_ds_name} vacío. Omitiendo."); continue
            
            cross_inf_loader = DataLoader(cross_inf_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker)
            inf_loss, inf_acc, inf_prec, inf_rec, inf_f1, inf_cm = evaluate_slowfast(
                model_for_analysis, cross_inf_loader, criterion, DEVICE, use_amp_for_training, 
                pos_label_value=CLASSES.get("Violence", 1), num_classes_eval=len(CLASSES)
            )
            logging.info(f"Resultados Inferencia SF en {inference_ds_name} (entrenado en {TRAIN_DATASET_NAME}):")
            logging.info(f"  Loss: {inf_loss:.4f}, Acc: {inf_acc:.4f}, F1 (Violence): {inf_f1:.4f}")
            
            current_cross_metrics = {f'cross_inference_sf_on_{inference_ds_name}': { 
                'model_trained_on': TRAIN_DATASET_NAME, 'evaluated_on': f"{inference_ds_name}_full",
                'loss': inf_loss, 'accuracy': inf_acc, 'precision_violence': inf_prec, 
                'recall_violence': inf_rec, 'f1_score_violence': inf_f1, 'confusion_matrix': inf_cm
            }}
            save_or_update_json(current_cross_metrics, metrics_json_path)

    logging.info("Proceso SlowFast completado.")

if __name__ == '__main__':
    main()
