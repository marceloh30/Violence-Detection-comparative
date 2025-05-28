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
# API actualizada para Precisión Mixta Automática (AMP)
from torch.amp import GradScaler, autocast # Actualizado desde torch.cuda.amp

# ----- CONFIGURACIÓN (adaptada de tus scripts I3D y ViViT) -----
BASE_DATA_DIR = "assets"
RWF_2000_SUBDIR = "RWF-2000"
HOCKEY_FIGHTS_SUBDIR = "HockeyFights"
RLVS_SUBDIR = "RealLifeViolenceDataset"

CLASSES = {"Fight": 1, "NonFight": 0}
SPLITS = {"train": "train", "val": "val"}

NUM_FRAMES_TO_SAMPLE = 32
FRAME_STEP = 4
IMG_CROP_SIZE = 224
IMG_RESIZE_DIM = IMG_CROP_SIZE 

ALPHA_SLOWFAST = 4
NUM_FRAMES_FAST_PATHWAY = NUM_FRAMES_TO_SAMPLE
NUM_FRAMES_SLOW_PATHWAY = NUM_FRAMES_FAST_PATHWAY // ALPHA_SLOWFAST

# AJUSTES DE HIPERPARÁMETROS Y EFICIENCIA
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10
NUM_CLASSES_MODEL = len(CLASSES)
USE_AMP = True 
LOAD_CHECKPOINT_IF_EXISTS = True # Nueva opción para cargar checkpoint

OUTPUT_DIR = "slowfast_r50_outputs_workaround_amp" 
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "train_metrics_slowfast_workaround_amp.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_slowfast_workaround_amp.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type # Para autocast (ej. 'cuda' o 'cpu')

KINETICS_MEAN_LIST = [0.45, 0.45, 0.45] 
KINETICS_STD_LIST = [0.225, 0.225, 0.225]

TRIALS_FPS = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- FUNCIONES AUXILIARES Y CLASES -----

def process_video_cv2_frames(video_path, num_frames_to_sample, frame_step, resize_dim, is_train=True):
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
        if not available_indices: 
             available_indices = list(range(total_frames))
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
        frames.append(frame)
    cap.release()
    if len(frames) != num_frames_to_sample:
        logging.error(f"Error de procesamiento de fotogramas: se esperaban {num_frames_to_sample} fotogramas, se obtuvieron {len(frames)} para {video_path}")
        return None
    return frames

class PackPathwayCustom(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames, 1, 
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long().to(frames.device)
        )
        return [slow_pathway, fast_pathway]

mean_tensor = torch.tensor(KINETICS_MEAN_LIST, dtype=torch.float32).view(3, 1, 1, 1)
std_tensor = torch.tensor(KINETICS_STD_LIST, dtype=torch.float32).view(3, 1, 1, 1)

slowfast_transforms_workaround = Compose(
    [
        Lambda(lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32)),
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: x.permute(3, 0, 1, 2)),
        Lambda(lambda x: (x - mean_tensor.to(x.device)) / std_tensor.to(x.device)), # Mover mean/std al dispositivo del tensor x
        PackPathwayCustom(alpha=ALPHA_SLOWFAST)
    ]
)

class VideoDatasetSlowFast(Dataset):
    def __init__(self, video_files, labels, transform_pipeline, num_frames, frame_step, resize_dim, is_train=True, dataset_name=""):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform_pipeline 
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.resize_dim = resize_dim 
        self.is_train = is_train
        self.dataset_name = dataset_name
    def __len__(self):
        return len(self.video_files)
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames_list_hwc_rgb = process_video_cv2_frames(
            video_path, self.num_frames, self.frame_step, self.resize_dim, self.is_train
        )
        if frames_list_hwc_rgb is None:
            dummy_fast = torch.zeros((3, NUM_FRAMES_FAST_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE), dtype=torch.float32)
            dummy_slow = torch.zeros((3, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE), dtype=torch.float32)
            return [dummy_slow, dummy_fast], torch.tensor(-1) 
        try:
            packed_frames = self.transform(frames_list_hwc_rgb) 
        except Exception as e:
            logging.error(f"Error aplicando transformaciones (workaround) a {video_path}: {e}")
            dummy_fast = torch.zeros((3, NUM_FRAMES_FAST_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE), dtype=torch.float32)
            dummy_slow = torch.zeros((3, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE), dtype=torch.float32)
            return [dummy_slow, dummy_fast], torch.tensor(-1)
        return packed_frames, torch.tensor(label, dtype=torch.long)

video_train_transform = slowfast_transforms_workaround
video_val_transform = slowfast_transforms_workaround

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
    logging.warning("Usando carga de datos placeholder para HockeyFights.")
    dataset_dir = os.path.join(BASE_DATA_DIR, HOCKEY_FIGHTS_SUBDIR)
    if not os.path.exists(dataset_dir): return [], []
    return load_dataset_paths_and_labels(dataset_dir, SPLITS[split_type], CLASSES, "HockeyFights")
def get_rlvs_data(split_type):
    logging.warning("Usando carga de datos placeholder para RLVS.")
    dataset_dir = os.path.join(BASE_DATA_DIR, RLVS_SUBDIR)
    if not os.path.exists(dataset_dir): return [], []
    return load_dataset_paths_and_labels(dataset_dir, SPLITS[split_type], CLASSES, "RLVS")

def load_slowfast_model_custom(num_model_classes=NUM_CLASSES_MODEL, pretrained=True):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
    original_head_in_features = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Linear(original_head_in_features, num_model_classes)
    return model

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp_flag, device_type_for_amp):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples_processed = 0
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for inputs_list, labels in progress_bar:
        valid_indices = labels != -1
        if not valid_indices.any(): continue
        inputs_on_device = [inp[valid_indices].to(device, non_blocking=True) for inp in inputs_list]
        labels = labels[valid_indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True) 
        if inputs_on_device[0].size(0) == 0: continue

        with autocast(device_type=device_type_for_amp, enabled=use_amp_flag): 
            outputs = model(inputs_on_device) 
            loss = criterion(outputs, labels)
        
        if use_amp_flag and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        _, preds = torch.max(outputs, 1)
        current_batch_size = inputs_on_device[0].size(0)
        running_loss += loss.item() * current_batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples_processed += current_batch_size
        progress_bar.set_postfix(loss=loss.item(), acc_batch=(torch.sum(preds == labels.data).item() / current_batch_size))
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples_processed if total_samples_processed > 0 else 0
    epoch_acc_tensor = running_corrects.double() / total_samples_processed if total_samples_processed > 0 else torch.tensor(0.0)
    epoch_acc = epoch_acc_tensor.item() if isinstance(epoch_acc_tensor, torch.Tensor) else float(epoch_acc_tensor)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, use_amp_flag, device_type_for_amp): 
    model.eval()
    running_loss = 0.0
    total_samples_processed = 0
    all_preds_list, all_labels_list = [], []
    progress_bar = tqdm(loader, desc="Eval", leave=False)
    with torch.no_grad():
        for inputs_list, labels in progress_bar:
            valid_indices = labels != -1
            if not valid_indices.any(): continue
            inputs_on_device = [inp[valid_indices].to(device, non_blocking=True) for inp in inputs_list]
            labels = labels[valid_indices].to(device, non_blocking=True)
            if inputs_on_device[0].size(0) == 0: continue
            
            with autocast(device_type=device_type_for_amp, enabled=use_amp_flag): 
                outputs = model(inputs_on_device)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            current_batch_size = inputs_on_device[0].size(0)
            running_loss += loss.item() * current_batch_size
            total_samples_processed += current_batch_size
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
    if total_samples_processed == 0:
        logging.warning("No valid samples processed during evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_loss = running_loss / total_samples_processed
    acc = accuracy_score(all_preds_list, all_preds_list)
    precision = precision_score(all_labels_list, all_preds_list, average='binary', zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average='binary', zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average='binary', zero_division=0)
    return float(epoch_loss), float(acc), float(precision), float(recall), float(f1)

def measure_inference_fps_slowfast(model, device, trials=TRIALS_FPS):
    model.eval()
    # Warm-up: Regenerate dummy input for each iteration
    for _ in range(10): 
        dummy_slow_iter = torch.randn(1, 3, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=device)
        dummy_fast_iter = torch.randn(1, 3, NUM_FRAMES_FAST_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=device)
        dummy_input_iter = [dummy_slow_iter, dummy_fast_iter]
        _ = model(dummy_input_iter) 
        
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(trials):
        # Regenerate dummy input for each timing iteration
        dummy_slow_iter = torch.randn(1, 3, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=device)
        dummy_fast_iter = torch.randn(1, 3, NUM_FRAMES_FAST_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=device)
        dummy_input_iter = [dummy_slow_iter, dummy_fast_iter]
        _ = model(dummy_input_iter)
    if device.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return trials / total_time if total_time > 0 else 0.0

def save_training_metrics(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(history, f, indent=4)
    logging.info(f"Métricas de entrenamiento guardadas en {path}")

def main_train_loop(dataset_name="RWF-2000"):
    logging.info(f"Usando dispositivo: {DEVICE} (Tipo: {DEVICE_TYPE})")
    logging.info(f"Dataset seleccionado para entrenamiento: {dataset_name}")
    
    use_amp_for_training = USE_AMP and DEVICE_TYPE == 'cuda' 
    logging.info(f"Precisión Mixta Automática (AMP) para entrenamiento: {use_amp_for_training}")

    model_for_training_or_analysis = None 
    perform_training = True 

    if LOAD_CHECKPOINT_IF_EXISTS and os.path.exists(BEST_MODEL_PATH):
        logging.info(f"LOAD_CHECKPOINT_IF_EXISTS es True y se encontró checkpoint en: {BEST_MODEL_PATH}")
        logging.info("Se omitirá el entrenamiento. El modelo se cargará para análisis.")
        perform_training = False
        model_for_training_or_analysis = load_slowfast_model_custom(num_model_classes=NUM_CLASSES_MODEL, pretrained=True) 
        checkpoint_data_loaded = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint_data_loaded:
            model_for_training_or_analysis.load_state_dict(checkpoint_data_loaded['model_state_dict'])
        else:
            model_for_training_or_analysis.load_state_dict(checkpoint_data_loaded)
        model_for_training_or_analysis = model_for_training_or_analysis.to(DEVICE)
        logging.info("Modelo de checkpoint cargado para análisis.")
    else:
        logging.info("No se encontró checkpoint o LOAD_CHECKPOINT_IF_EXISTS es False. Se procederá con el entrenamiento.")
        model_for_training_or_analysis = load_slowfast_model_custom(num_model_classes=NUM_CLASSES_MODEL, pretrained=True)
        model_for_training_or_analysis = model_for_training_or_analysis.to(DEVICE)

    if perform_training and model_for_training_or_analysis is not None:
        if dataset_name == "RWF-2000": train_video_paths, train_labels = get_rwf2000_data("train"); val_video_paths, val_labels = get_rwf2000_data("val")
        elif dataset_name == "HockeyFights": train_video_paths, train_labels = get_hockey_data("train"); val_video_paths, val_labels = get_hockey_data("val")
        elif dataset_name == "RLVS": train_video_paths, train_labels = get_rlvs_data("train"); val_video_paths, val_labels = get_rlvs_data("val")
        else: raise ValueError(f"Dataset no soportado: {dataset_name}")

        if not train_video_paths: logging.error(f"No se encontraron vídeos de entrenamiento para {dataset_name}. Saliendo."); return
        if not val_video_paths: logging.warning(f"No se encontraron vídeos de validación para {dataset_name}.")

        train_dataset = VideoDatasetSlowFast(train_video_paths, train_labels, video_train_transform, NUM_FRAMES_FAST_PATHWAY, FRAME_STEP, IMG_RESIZE_DIM, is_train=True, dataset_name=dataset_name+"-train")
        num_data_workers = 2 if os.name == 'posix' else 0 
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_data_workers, pin_memory=True)
        
        val_loader = None
        if val_video_paths:
            val_dataset = VideoDatasetSlowFast(val_video_paths, val_labels, video_val_transform, NUM_FRAMES_FAST_PATHWAY, FRAME_STEP, IMG_RESIZE_DIM, is_train=False, dataset_name=dataset_name+"-val")
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_data_workers, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_for_training_or_analysis.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler() if use_amp_for_training else None

        history = {'dataset': dataset_name, 'epochs_run': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'epoch_time_seconds': []}
        best_val_f1 = 0.0
        start_epoch = 1 

        logging.info(f"Iniciando bucle de entrenamiento desde la época {start_epoch} hasta {EPOCHS} en {dataset_name}...")
        for epoch in range(start_epoch, EPOCHS + 1): 
            epoch_start_time = time.time()
            logging.info(f"--- Época {epoch}/{EPOCHS} ---")
            train_loss, train_acc = train_epoch(model_for_training_or_analysis, train_loader, criterion, optimizer, DEVICE, scaler, use_amp_for_training, DEVICE_TYPE)
            logging.info(f"Época {epoch} Train: Pérdida={train_loss:.4f}, Acc={train_acc:.4f}")
            history['epochs_run'].append(epoch); history['train_loss'].append(train_loss) ; history['train_acc'].append(train_acc)

            if val_loader and len(val_loader.dataset) > 0 :
                val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model_for_training_or_analysis, val_loader, criterion, DEVICE, use_amp_for_training, DEVICE_TYPE)
                logging.info(f"Época {epoch} Val: Pérdida={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}")
                history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_precision'].append(val_prec); history['val_recall'].append(val_rec); history['val_f1'].append(val_f1)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model_for_training_or_analysis.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_val_f1': best_val_f1, 
                        'scaler_state_dict': scaler.state_dict() if scaler else None
                    }
                    torch.save(checkpoint_data, BEST_MODEL_PATH)
                    logging.info(f"  Mejor F1 en validación: {best_val_f1:.4f}. Modelo guardado en {BEST_MODEL_PATH}")
            else: 
                 history['val_loss'].append(None); history['val_acc'].append(None); history['val_precision'].append(None); history['val_recall'].append(None); history['val_f1'].append(None)
                 if epoch == EPOCHS: 
                    torch.save({'model_state_dict': model_for_training_or_analysis.state_dict()}, BEST_MODEL_PATH)
                    logging.info(f"Sin validación. Modelo de época {epoch} guardado en {BEST_MODEL_PATH}")
            
            epoch_duration = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(float(epoch_duration))
            logging.info(f"Época {epoch} completada en {epoch_duration:.2f}s")
            save_training_metrics(history, METRICS_JSON_PATH)

        logging.info("Entrenamiento completado.")
        if val_loader and len(val_loader.dataset) > 0: logging.info(f"Mejor F1 en validación final: {best_val_f1:.4f}.")
        logging.info(f"Mejor modelo (o último si no hay validación) guardado en: {BEST_MODEL_PATH}")
        logging.info(f"Métricas de entrenamiento guardadas en: {METRICS_JSON_PATH}")

    # --- ANÁLISIS FINAL DEL MODELO ---
    final_analysis_model = None
    if os.path.exists(BEST_MODEL_PATH):
        logging.info(f"Cargando modelo desde {BEST_MODEL_PATH} para análisis final.")
        final_analysis_model = load_slowfast_model_custom(num_model_classes=NUM_CLASSES_MODEL, pretrained=True) 
        
        checkpoint_data_loaded = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint_data_loaded:
            final_analysis_model.load_state_dict(checkpoint_data_loaded['model_state_dict'])
        else: 
            final_analysis_model.load_state_dict(checkpoint_data_loaded) 

        final_analysis_model = final_analysis_model.to(DEVICE)
        final_analysis_model.eval()
    else:
        # Si el entrenamiento se omitió pero el checkpoint no existe (caso improbable si LOAD_CHECKPOINT_IF_EXISTS es True y se esperaba omitir)
        # o si el entrenamiento se realizó pero no se guardó ningún modelo (ej. sin validación y no es la última época).
        if perform_training and model_for_training_or_analysis is not None:
            logging.warning(f"No se encontró el archivo {BEST_MODEL_PATH}. Usando el modelo en memoria del entrenamiento para análisis.")
            final_analysis_model = model_for_training_or_analysis
            final_analysis_model.eval() # Asegurar modo evaluación
        else:
            logging.warning(f"No se encontró modelo en {BEST_MODEL_PATH} y no hay modelo en memoria del entrenamiento. El análisis se omitirá.")
            final_analysis_model = None


    if final_analysis_model: 
        fps = measure_inference_fps_slowfast(final_analysis_model, DEVICE)
        params_count = parameter_count(final_analysis_model)['']
        
        dummy_slow_flops = torch.randn(1, 3, NUM_FRAMES_SLOW_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=DEVICE)
        dummy_fast_flops = torch.randn(1, 3, NUM_FRAMES_FAST_PATHWAY, IMG_CROP_SIZE, IMG_CROP_SIZE, device=DEVICE)
        dummy_input_flops = [dummy_slow_flops, dummy_fast_flops]
        gflops = -1.0
        try:
            if hasattr(final_analysis_model, '_modules') and final_analysis_model._modules:
                flops_analyzer = FlopCountAnalysis(final_analysis_model, dummy_input_flops)
                gflops = flops_analyzer.total() / 1e9
            else:
                logging.warning("El modelo de análisis parece estar vacío o no es compatible con FlopCountAnalysis.")
        except Exception as e:
            logging.error(f"No se pudieron calcular los FLOPs: {e}.")
            
        logging.info(f"FPS de Inferencia del Modelo: {fps:.2f}")
        logging.info(f"Parámetros del Modelo: {params_count:,}")
        logging.info(f"GFLOPs del Modelo: {gflops:.2f}G")

        if os.path.exists(METRICS_JSON_PATH) or not perform_training:
            try:
                final_metrics = {}
                if os.path.exists(METRICS_JSON_PATH): 
                    with open(METRICS_JSON_PATH, 'r') as f: 
                        final_metrics = json.load(f)
                
                if 'performance_stats' not in final_metrics:
                    final_metrics['performance_stats'] = {}
                
                final_metrics['performance_stats']['fps'] = float(fps)
                final_metrics['performance_stats']['parameters'] = int(params_count)
                final_metrics['performance_stats']['gflops'] = float(gflops)

                save_training_metrics(final_metrics, METRICS_JSON_PATH)
            except Exception as e:
                logging.error(f"Error al actualizar métricas con estadísticas de rendimiento: {e}")
    else:
        logging.warning("Análisis final omitido porque no se pudo cargar o encontrar un modelo.")


if __name__ == '__main__':
    dataset_to_train = "RWF-2000" 
    main_train_loop(dataset_name=dataset_to_train)
