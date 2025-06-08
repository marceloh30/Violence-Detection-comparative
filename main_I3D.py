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
from pytorchvideo.models.hub import i3d_r50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fvcore.nn import FlopCountAnalysis, parameter_count
from tensorboardX import SummaryWriter 
from tqdm import tqdm
import logging
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

RANDOM_SEED = 23

# --- Selección del Dataset de Entrenamiento ---
# Cambiar a "rlvs" para entrenar con RLVS
TRAIN_DATASET_NAME = "rwf2000"  # Opciones: "rwf2000", "rlvs"

# Usar las clases globales importadas
CLASSES = GLOBAL_CLASSES_MAP 
# DATASET_NAME_FOR_HISTORY se establecerá dinámicamente en main()

# --- Parámetros de Procesamiento de Vídeo y Modelo (específicos de I3D) ---
NUM_FRAMES = 32
FRAME_STEP = 4
IMG_CROP_SIZE = 224
IMG_RESIZE_DIM = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) 
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1) 
# --- Hiperparámetros de Entrenamiento ---
BATCH_SIZE = 2 
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10 

# --- Rutas de Salida y Directorios de Listas de Archivos ---
# El nombre del directorio de salida principal ahora incluirá el dataset de entrenamiento
OUTPUT_DIR_BASE = f"i3d_r50_outputs_seed_{RANDOM_SEED}" 
FILE_LIST_DIR = OUTPUT_LIST_DIR_DEFAULT 
BASE_DATA_DIR = BASE_DATA_DIR_DEFAULT   

# LOG_DIR_TENSORBOARD, METRICS_JSON_PATH, BEST_MODEL_PATH, etc., se definirán en main()
# para incluir el nombre del dataset de entrenamiento.

# --- Dispositivo y Eficiencia ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type 
USE_AMP_TRAINING = (DEVICE.type == 'cuda') 
NUM_DATA_WORKERS = 2 if os.name == 'posix' else 0

# --- Control de Flujo ---
PERFORM_TRAINING = True 
PERFORM_CROSS_INFERENCE = True # Control general para todas las inferencias cruzadas

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv') 
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

# ----- FUNCIÓN DE PROCESAMIENTO DE VÍDEO (Específica de I3D) -----
def process_video(path, num_frames_to_sample=NUM_FRAMES, resize_dim=IMG_RESIZE_DIM, crop_size=IMG_CROP_SIZE, sampling_step=FRAME_STEP, is_train=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.warning(f"Error: No se pudo abrir el vídeo: {path}")
        return None
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        logging.warning(f"Error: El vídeo no tiene fotogramas o es inválido: {path}")
        return None
        
    indices_a_muestrear = []
    selectable_frames_range = total_video_frames - (num_frames_to_sample - 1) * sampling_step
    
    if selectable_frames_range > 0:
        if is_train: 
            start_index = random.randint(0, selectable_frames_range - 1)
        else: 
            start_index = selectable_frames_range // 2
        indices_a_muestrear = [start_index + i * sampling_step for i in range(num_frames_to_sample)]
    else: 
        effective_step = max(1, total_video_frames // num_frames_to_sample if num_frames_to_sample > 0 else 1)
        available_indices = list(range(0, total_video_frames, effective_step))
        if not available_indices: available_indices = list(range(total_video_frames))
        if not available_indices:
            cap.release()
            logging.warning(f"No hay suficientes fotogramas para muestrear (falló relleno inicial) para: {path}")
            return None
        indices_a_muestrear = available_indices[:num_frames_to_sample]
        while len(indices_a_muestrear) < num_frames_to_sample and available_indices:
            indices_a_muestrear.append(available_indices[-1])
        if not indices_a_muestrear and num_frames_to_sample > 0 : 
             cap.release()
             logging.warning(f"Fallo catastrófico en la selección de frames para: {path}")
             return None
        elif num_frames_to_sample == 0: 
            cap.release()
            return torch.empty((3, 0, crop_size, crop_size), dtype=torch.float32)

    frames_procesados = []
    for frame_num_orig_idx in indices_a_muestrear:
        frame_num = min(int(frame_num_orig_idx), total_video_frames - 1)
        frame_num = max(0, frame_num) 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret: 
            logging.warning(f"No se pudo leer el fotograma {frame_num} de {path}. Usando fotograma negro.")
            frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != resize_dim or frame.shape[1] != resize_dim:
                frame = cv2.resize(frame, (resize_dim, resize_dim))
            if resize_dim > crop_size: 
                if is_train:
                    top = random.randint(0, resize_dim - crop_size)
                    left = random.randint(0, resize_dim - crop_size)
                else:
                    top = (resize_dim - crop_size) // 2
                    left = (resize_dim - crop_size) // 2
                frame = frame[top:top+crop_size, left:left+crop_size, :]
            elif resize_dim < crop_size: 
                 frame = cv2.resize(frame, (crop_size, crop_size))
        frames_procesados.append(frame)
    cap.release()
    
    if len(frames_procesados) != num_frames_to_sample:
        logging.warning(f"Se obtuvieron {len(frames_procesados)} frames, se esperaban {num_frames_to_sample} para {path}. Rellenando...")
        while len(frames_procesados) < num_frames_to_sample and frames_procesados: 
            frames_procesados.append(frames_procesados[-1].copy()) 
        if len(frames_procesados) != num_frames_to_sample and num_frames_to_sample > 0:
            logging.error(f"Error de procesamiento final: se esperaban {num_frames_to_sample} fotogramas, se obtuvieron {len(frames_procesados)} para {path} después del relleno.")
            return None 
        elif num_frames_to_sample == 0 and not frames_procesados: 
            return torch.empty((3, 0, crop_size, crop_size), dtype=torch.float32)

    if num_frames_to_sample == 0:
        return torch.empty((3, 0, crop_size, crop_size), dtype=torch.float32)

    clip = np.stack(frames_procesados) 
    clip_tensor = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0 
    clip_tensor = clip_tensor.permute(1, 0, 2, 3) 
    
    global MEAN, STD 
    clip_tensor = (clip_tensor - MEAN) / STD
    return clip_tensor

# ----- CLASE DATASET PERSONALIZADA -----
class VideoListDataset(Dataset):
    def __init__(self, file_list_data, num_frames, resize_dim, crop_size, frame_step, is_train, dataset_name_log=""):
        self.file_list_data = file_list_data
        self.num_frames = num_frames
        self.resize_dim = resize_dim
        self.crop_size = crop_size
        self.frame_step = frame_step
        self.is_train = is_train
        self.dataset_name_log = dataset_name_log

        if not self.file_list_data:
            logging.warning(f"VideoListDataset inicializado con una lista de archivos vacía para {self.dataset_name_log}")

    def __len__(self):
        return len(self.file_list_data)

    def __getitem__(self, idx):
        if idx >= len(self.file_list_data):
            raise IndexError(f"Índice {idx} fuera de rango para dataset {self.dataset_name_log} con tamaño {len(self.file_list_data)}")

        item_data = self.file_list_data[idx]
        video_path = item_data['path']
        label = item_data['label']

        clip_tensor = process_video(
            video_path, self.num_frames, self.resize_dim, self.crop_size, self.frame_step, self.is_train
        )

        if clip_tensor is None:
            logging.warning(f"Fallo al procesar vídeo {video_path} en VideoListDataset. Devolviendo tensor dummy y etiqueta -1.")
            dummy_clip = torch.zeros((3, self.num_frames if self.num_frames > 0 else 0, self.resize_dim, self.resize_dim), dtype=torch.float32)
            return dummy_clip, torch.tensor(-1, dtype=torch.long) 
        
        return clip_tensor, torch.tensor(label, dtype=torch.long)

# ----- FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN -----
def train_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_valid_samples = 0 
    
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for x_batch, y_batch in progress_bar:
        valid_indices = y_batch != -1
        if not valid_indices.any(): 
            continue
        
        x = x_batch[valid_indices].to(device, non_blocking=True)
        y = y_batch[valid_indices].to(device, non_blocking=True)

        if x.size(0) == 0: 
            continue
            
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=use_amp):
            out = model(x) 
            out_flat = out.view(out.size(0), -1) 
            loss = criterion(out_flat, y)
        
        if use_amp and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        _, preds = torch.max(out_flat, 1)
        current_batch_valid_samples = x.size(0)
        running_loss += loss.item() * current_batch_valid_samples
        running_corrects += torch.sum(preds == y.data)
        total_valid_samples += current_batch_valid_samples
        
        if current_batch_valid_samples > 0:
            progress_bar.set_postfix(loss_batch=f"{loss.item():.4f}", acc_batch=f"{(torch.sum(preds == y.data).item() / current_batch_valid_samples):.4f}")
        else:
            progress_bar.set_postfix(loss_batch="N/A", acc_batch="N/A")

        if device.type == 'cuda': torch.cuda.empty_cache()

    epoch_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    epoch_acc_tensor = running_corrects.double() / total_valid_samples if total_valid_samples > 0 else torch.tensor(0.0)
    epoch_acc = epoch_acc_tensor.item() if isinstance(epoch_acc_tensor, torch.Tensor) else float(epoch_acc_tensor)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, use_amp, pos_label_value=1, num_classes_eval=len(CLASSES)):
    model.eval()
    running_loss = 0.0 
    running_corrects = 0 
    total_valid_samples = 0    
    all_preds_list, all_labels_list = [], []
    
    progress_bar = tqdm(loader, desc="Eval", leave=False)
    with torch.no_grad():
        for x_batch, y_batch in progress_bar:
            valid_indices = y_batch != -1
            if not valid_indices.any():
                continue
            
            x = x_batch[valid_indices].to(device, non_blocking=True)
            y_true = y_batch[valid_indices].to(device, non_blocking=True)

            if x.size(0) == 0:
                continue
            
            with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=use_amp):
                out = model(x) 
                out_flat = out.view(out.size(0), -1)
                loss = criterion(out_flat, y_true) 
            
            _, preds = torch.max(out_flat, 1)
            current_batch_valid_samples = x.size(0)
            running_loss += loss.item() * current_batch_valid_samples
            running_corrects += torch.sum(preds == y_true.data)
            total_valid_samples += current_batch_valid_samples

            all_preds_list.extend(preds.cpu().numpy()) 
            all_labels_list.extend(y_true.cpu().numpy())
            
    if total_valid_samples == 0:
        logging.warning("No se procesaron muestras válidas durante la evaluación.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, [] 

    epoch_val_loss = running_loss / total_valid_samples
    acc = accuracy_score(all_labels_list, all_preds_list)
    
    avg_method = 'binary' if num_classes_eval == 2 else 'macro'
    unique_labels_in_data = np.unique(all_labels_list + all_preds_list)
    valid_pos_label = None
    if avg_method == 'binary':
        if pos_label_value in unique_labels_in_data:
            valid_pos_label = pos_label_value
        elif len(unique_labels_in_data) > 0 : 
            if len(unique_labels_in_data) == 2 and 0 in unique_labels_in_data: # Asume que la otra clase es la positiva
                other_labels = [l for l in unique_labels_in_data if l != 0]
                if other_labels: valid_pos_label = other_labels[0]
            elif 1 in unique_labels_in_data: # Fallback a 1 si está presente
                 valid_pos_label = 1
            elif unique_labels_in_data: # Fallback a la primera etiqueta si no se encuentra 0 o 1
                 valid_pos_label = sorted(list(unique_labels_in_data))[0]

    if avg_method == 'binary' and valid_pos_label is None and len(unique_labels_in_data) > 1 : 
        logging.warning(f"No se pudo determinar una etiqueta positiva válida para métricas binarias (pos_label_value={pos_label_value}, etiquetas presentes={unique_labels_in_data}). Usando 'macro'.")
        avg_method = 'macro'

    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method == 'binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method == 'binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, pos_label=valid_pos_label if avg_method == 'binary' else None, zero_division=0, labels=list(range(num_classes_eval)))
    cm = confusion_matrix(all_labels_list, all_preds_list, labels=list(range(num_classes_eval))).tolist()
    
    return epoch_val_loss, acc, precision, recall, f1, cm

# ----- MEDICIÓN DE FPS Y GUARDADO DE MÉTRICAS -----
def measure_inference_fps(model_to_measure, device_to_use, clip_s=NUM_FRAMES, img_s=IMG_RESIZE_DIM, num_trials=TRIALS_FPS):
    dummy_input_shape = (1, 3, clip_s if clip_s > 0 else 1, img_s, img_s) 
    if clip_s == 0: 
        logging.warning("FPS medido con T=1 para num_frames=0. La inferencia real puede variar.")

    dummy = torch.randn(dummy_input_shape, device=device_to_use)
    model_to_measure.eval() 
    with torch.no_grad():
        for _ in range(10): _ = model_to_measure(dummy) 
        
        if device_to_use.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_trials): _ = model_to_measure(dummy)
        if device_to_use.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return num_trials / total_time if total_time > 0 else 0.0

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
                except json.JSONDecodeError: logging.warning(f"JSON {json_path} corrupto. Se sobrescribirá.")
        
        # Fusionar inteligentemente para no perder datos de ejecuciones anteriores si el archivo ya existe
        for key, value in cleaned_metrics_dict.items():
            if key in existing_data and isinstance(existing_data[key], dict) and isinstance(value, dict):
                existing_data[key].update(value)
            else:
                existing_data[key] = value
        
        with open(json_path, 'w') as f: json.dump(existing_data, f, indent=4)
        logging.info(f"Métricas guardadas/actualizadas en {json_path}")
    except Exception as e: logging.error(f"Error al guardar métricas en JSON {json_path}: {e}")

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
    
    # --- Determinar nombres de archivo y directorios dinámicamente ---
    dataset_name_for_history = f"{TRAIN_DATASET_NAME}_I3D"
    current_output_dir = os.path.join(OUTPUT_DIR_BASE, f"trained_on_{TRAIN_DATASET_NAME}")
    
    log_dir_tensorboard = os.path.join(current_output_dir, "logs_tensorboard")
    metrics_json_path = os.path.join(current_output_dir, f"train_metrics_i3d_{TRAIN_DATASET_NAME.lower()}.json") 
    best_model_path = os.path.join(current_output_dir, f"i3d_{TRAIN_DATASET_NAME.lower()}_best_model.pth")

    os.makedirs(current_output_dir, exist_ok=True)
    os.makedirs(log_dir_tensorboard, exist_ok=True)

    writer = SummaryWriter(log_dir_tensorboard) 
    
    model = i3d_r50(pretrained=True)
    final_projection_layer = model.blocks[-1].proj
    num_target_classes = len(CLASSES) 
    if isinstance(final_projection_layer, nn.Conv3d):
        model.blocks[-1].proj = nn.Conv3d(final_projection_layer.in_channels, num_target_classes, final_projection_layer.kernel_size, final_projection_layer.stride)
    elif isinstance(final_projection_layer, nn.Linear):
        model.blocks[-1].proj = nn.Linear(final_projection_layer.in_features, num_target_classes)
    else:
        raise TypeError(f"Capa final del modelo inesperada: {type(final_projection_layer)}")
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_TRAINING)

    if PERFORM_TRAINING:
        logging.info(f"Iniciando entrenamiento del modelo I3D en {TRAIN_DATASET_NAME}...")
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
            logging.error(f"No se pudo cargar la lista de archivos de entrenamiento para {TRAIN_DATASET_NAME}. Abortando entrenamiento.")
            writer.close()
            return
        
        train_dataset = VideoListDataset(train_file_list, NUM_FRAMES, IMG_RESIZE_DIM, IMG_CROP_SIZE, FRAME_STEP, is_train=True, dataset_name_log=f"{TRAIN_DATASET_NAME} Train")
        val_dataset = VideoListDataset(val_file_list, NUM_FRAMES, IMG_RESIZE_DIM, IMG_CROP_SIZE, FRAME_STEP, is_train=False, dataset_name_log=f"{TRAIN_DATASET_NAME} Val") if val_file_list else None
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker) if val_dataset and len(val_dataset) > 0 else None

        history = {
            'dataset_trained_on': TRAIN_DATASET_NAME, 'model_name': 'I3D_R50',
            'hyperparameters': {'lr': LR, 'batch_size': BATCH_SIZE, 'epochs_config': EPOCHS, 
                                'num_frames': NUM_FRAMES, 'frame_step': FRAME_STEP, 'img_size': IMG_RESIZE_DIM,
                                'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY},
            'epochs_run': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_cm': [],
            'epoch_time_seconds': []
        }
        best_val_f1 = 0.0

        logging.info(f"Config de entrenamiento: Dataset={TRAIN_DATASET_NAME}, NUM_FRAMES={NUM_FRAMES}, FRAME_STEP={FRAME_STEP}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}")

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            logging.info(f"--- Época {epoch}/{EPOCHS} (Entrenando en {TRAIN_DATASET_NAME}) ---")
            
            train_loss_val, train_acc_val = train_epoch(model, train_loader, criterion, optimizer, DEVICE, USE_AMP_TRAINING, scaler)
            history['epochs_run'].append(epoch)
            history['train_loss'].append(train_loss_val)
            history['train_acc'].append(train_acc_val)
            writer.add_scalar(f'Loss/train_{TRAIN_DATASET_NAME}', train_loss_val, epoch)
            writer.add_scalar(f'Accuracy/train_{TRAIN_DATASET_NAME}', train_acc_val, epoch)
            logging.info(f"Época {epoch} Train ({TRAIN_DATASET_NAME}): Pérdida={train_loss_val:.4f}, Acc={train_acc_val:.4f}")

            val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = 0.0, 0.0, 0.0, 0.0, 0.0, []
            if val_loader: 
                val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val, val_cm_val = evaluate(
                    model, val_loader, criterion, DEVICE, USE_AMP_TRAINING, 
                    pos_label_value=CLASSES.get("Violence", 1), 
                    num_classes_eval=len(CLASSES)
                )
                writer.add_scalar(f'Loss/val_{TRAIN_DATASET_NAME}', val_loss_val, epoch)
                writer.add_scalar(f'Accuracy/val_{TRAIN_DATASET_NAME}', val_acc_val, epoch)
                writer.add_scalar(f'Precision/val_{TRAIN_DATASET_NAME}', val_prec_val, epoch)
                writer.add_scalar(f'Recall/val_{TRAIN_DATASET_NAME}', val_rec_val, epoch)
                writer.add_scalar(f'F1/val_{TRAIN_DATASET_NAME}', val_f1_val, epoch)
                logging.info(f"Época {epoch} Val ({TRAIN_DATASET_NAME}): Pérdida={val_loss_val:.4f}, Acc={val_acc_val:.4f}, Prec={val_prec_val:.4f}, Rec={val_rec_val:.4f}, F1={val_f1_val:.4f}")
            
            history['val_loss'].append(val_loss_val if val_loader else None) 
            history['val_acc'].append(val_acc_val if val_loader else None)
            history['val_precision'].append(val_prec_val if val_loader else None)
            history['val_recall'].append(val_rec_val if val_loader else None)
            history['val_f1'].append(val_f1_val if val_loader else None)
            history['val_cm'].append(val_cm_val if val_loader else None)

            epoch_duration_val = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(epoch_duration_val)
            writer.add_scalar(f'Time/epoch_{TRAIN_DATASET_NAME}', epoch_duration_val, epoch)
            logging.info(f"Época {epoch} ({TRAIN_DATASET_NAME}) completada en {epoch_duration_val:.2f}s")
            save_metrics_to_json(history, metrics_json_path)

            if val_loader and val_f1_val > best_val_f1: 
                best_val_f1 = val_f1_val
                checkpoint_data = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_f1': best_val_f1, 
                    'scaler_state_dict': scaler.state_dict() if USE_AMP_TRAINING else None
                }
                torch.save(checkpoint_data, best_model_path)
                logging.info(f"  Mejor F1 en validación ({TRAIN_DATASET_NAME}): {best_val_f1:.4f}. Checkpoint guardado en {best_model_path}")
            elif not val_loader and epoch == EPOCHS: 
                 torch.save({'model_state_dict': model.state_dict()}, best_model_path)
                 logging.info(f"Entrenamiento sin validación ({TRAIN_DATASET_NAME}). Modelo de época {epoch} guardado en {best_model_path}")
        
        logging.info(f"Entrenamiento en {TRAIN_DATASET_NAME} completado.")
        if val_loader: logging.info(f"Mejor F1 en validación ({TRAIN_DATASET_NAME}) final: {best_val_f1:.4f}.")
    else:
        logging.info("Entrenamiento omitido según configuración (PERFORM_TRAINING=False).")

    model_for_analysis = i3d_r50(pretrained=False) 
    final_proj_analysis = model_for_analysis.blocks[-1].proj
    if isinstance(final_proj_analysis, nn.Conv3d):
        model_for_analysis.blocks[-1].proj = nn.Conv3d(final_proj_analysis.in_channels, num_target_classes, final_proj_analysis.kernel_size, final_proj_analysis.stride)
    elif isinstance(final_proj_analysis, nn.Linear):
        model_for_analysis.blocks[-1].proj = nn.Linear(final_proj_analysis.in_features, num_target_classes)
    
    model_loaded_for_analysis = False
    if os.path.exists(best_model_path):
        logging.info(f"Cargando modelo desde {best_model_path} para análisis/inferencia.")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint: 
            model_for_analysis.load_state_dict(checkpoint['model_state_dict'])
        else: 
            model_for_analysis.load_state_dict(checkpoint)
        model_for_analysis.to(DEVICE)
        model_for_analysis.eval()
        model_loaded_for_analysis = True
    else: 
        logging.warning(f"No se encontró el archivo de modelo {best_model_path}.")
        if PERFORM_TRAINING and 'model' in locals() and isinstance(model, nn.Module): 
            model_for_analysis = model 
            model_for_analysis.eval()
            model_loaded_for_analysis = True
            logging.info("Usando el modelo de la última época de entrenamiento para análisis (mejor modelo no encontrado).")
        else:
            logging.error("No hay modelo entrenado en memoria y no se encontró checkpoint. No se puede proceder con análisis o inferencia.")


    if model_loaded_for_analysis:
        logging.info("Calculando estadísticas de rendimiento del modelo cargado...")
        fps = measure_inference_fps(model_for_analysis, DEVICE)
        params_count = parameter_count(model_for_analysis).get('', 0)
        gflops = -1.0
        try:
            dummy_input_flops_shape = (1, 3, NUM_FRAMES if NUM_FRAMES >0 else 1, IMG_RESIZE_DIM, IMG_RESIZE_DIM)
            dummy_input_flops = torch.randn(dummy_input_flops_shape, device=DEVICE)
            if hasattr(model_for_analysis, '_modules') and model_for_analysis._modules:
                 flops_analyzer = FlopCountAnalysis(model_for_analysis, dummy_input_flops)
                 gflops = flops_analyzer.total() / 1e9
            else: logging.warning("Modelo de análisis vacío o no compatible con FlopCountAnalysis.")
        except Exception as e: logging.error(f"No se pudieron calcular los FLOPs: {e}")
        logging.info(f"Modelo Cargado - Inference FPS: {fps:.2f}, Params: {params_count/1e6:.2f}M, GFLOPs: {gflops:.2f}G")
        performance_stats_data = {'performance_stats': { 'fps': float(fps), 'parameters': int(params_count), 'gflops': float(gflops) }}
        save_metrics_to_json(performance_stats_data, metrics_json_path) 
        writer.add_scalar('Performance/FPS_final_model', fps); writer.add_scalar('Performance/Params_M_final_model', params_count / 1e6); writer.add_scalar('Performance/FLOPs_G_final_model', gflops)
    else:
        logging.warning("Análisis de rendimiento final omitido porque no se pudo cargar un modelo.")

    if PERFORM_CROSS_INFERENCE and model_loaded_for_analysis:
        logging.info(f"\nComenzando inferencia cruzada con el modelo I3D entrenado en {TRAIN_DATASET_NAME}...")
        
        datasets_for_cross_inference = []
        if TRAIN_DATASET_NAME == "rwf2000":
            datasets_for_cross_inference = ["rlvs", "hockey"]
        elif TRAIN_DATASET_NAME == "rlvs":
            datasets_for_cross_inference = ["rwf2000", "hockey"]
        else:
            logging.warning(f"Dataset de entrenamiento '{TRAIN_DATASET_NAME}' no configurado para inferencia cruzada específica. Se omitirá.")

        for inference_dataset_name in datasets_for_cross_inference:
            logging.info(f"--- Inferencia en: {inference_dataset_name} ---")
            
            cross_inference_file_list = get_dataset_file_list(
                dataset_name=inference_dataset_name, split_name="all",
                base_data_dir=BASE_DATA_DIR, output_list_dir=FILE_LIST_DIR
            )

            if not cross_inference_file_list:
                logging.warning(f"No se pudo cargar la lista de archivos de {inference_dataset_name} para la inferencia cruzada. Omitiendo.")
                continue
            
            current_inference_dataset = VideoListDataset(
                cross_inference_file_list, NUM_FRAMES, IMG_RESIZE_DIM, IMG_CROP_SIZE, FRAME_STEP, is_train=False, 
                dataset_name_log=f"{inference_dataset_name} Cross-Inference"
            )
            if len(current_inference_dataset) == 0:
                logging.warning(f"Dataset de inferencia {inference_dataset_name} está vacío después de cargar la lista. Omitiendo.")
                continue

            current_inference_loader = DataLoader(
                current_inference_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                num_workers=NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=seed_worker
            )
            
            inf_loss, inf_acc, inf_prec, inf_rec, inf_f1, inf_cm = evaluate(
                model_for_analysis, current_inference_loader, criterion, DEVICE, USE_AMP_TRAINING,
                pos_label_value=CLASSES.get("Violence", 1), 
                num_classes_eval=len(CLASSES)
            )
            
            logging.info(f"Resultados de Inferencia en {inference_dataset_name} (modelo entrenado en {TRAIN_DATASET_NAME}):")
            logging.info(f"  Loss: {inf_loss:.4f}, Accuracy: {inf_acc:.4f}")
            logging.info(f"  Precision (Violence): {inf_prec:.4f}, Recall (Violence): {inf_rec:.4f}, F1-Score (Violence): {inf_f1:.4f}")
            logging.info(f"  Matriz de Confusión:\n{np.array(inf_cm)}")

            cross_inference_metrics_output_path = os.path.join(
                current_output_dir, 
                f"cross_inference_on_{inference_dataset_name}_trained_with_{TRAIN_DATASET_NAME.lower()}.json"
            )
            current_cross_metrics = {
                f'cross_inference_on_{inference_dataset_name}': {
                    'model_trained_on': TRAIN_DATASET_NAME, 
                    'evaluated_on': f"{inference_dataset_name}_full_dataset",
                    'loss': inf_loss, 'accuracy': inf_acc, 
                    'precision_violence': inf_prec, 'recall_violence': inf_rec, 
                    'f1_score_violence': inf_f1, 'confusion_matrix': inf_cm
                }}
            save_metrics_to_json(current_cross_metrics, cross_inference_metrics_output_path)
            # También podrías añadir estas métricas al JSON principal de entrenamiento si lo deseas,
            # bajo una clave general 'all_cross_inferences' por ejemplo.
            # save_metrics_to_json(current_cross_metrics, metrics_json_path)


            writer.add_scalar(f'Loss/cross_eval_{inference_dataset_name}', inf_loss)
            writer.add_scalar(f'Accuracy/cross_eval_{inference_dataset_name}', inf_acc)
            writer.add_scalar(f'F1/cross_eval_{inference_dataset_name}_Violence', inf_f1)
            
    elif PERFORM_CROSS_INFERENCE and not model_loaded_for_analysis:
        logging.warning("Se solicitó inferencia cruzada, pero no hay un modelo cargado/disponible.")

    writer.close()
    logging.info("Proceso completado.")

if __name__ == '__main__':
    main()
