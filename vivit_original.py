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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm
import logging
from torch.amp import GradScaler, autocast

# ----- CONFIGURACIÓN -----
BASE_DATA_DIR = "assets"
RWF_2000_SUBDIR = "RWF-2000"
HOCKEY_FIGHTS_SUBDIR = "HockeyFights" # Placeholder for future use
RLVS_SUBDIR = "RealLifeViolenceDataset" # Placeholder for future use

CLASSES = {"Fight": 1, "NonFight": 0}
SPLITS = {"train": "train", "val": "val"} # Assuming 'val' corresponds to 'validation'

# ViViT specific parameters
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_FRAMES_TO_SAMPLE = 32
FRAME_STEP = 4            
IMG_RESIZE_DIM_AUG = 256 # Dimensión a la que se redimensiona antes del recorte
VIDEO_IMAGE_SIZE = 224  # Tamaño final del recorte (ya lo tienes, es el crop_size)

# Training hyperparameters
BATCH_SIZE = 2
LR = 5e-5 # From original ViViT script
WEIGHT_DECAY = 1e-2 
EPOCHS = 5 
NUM_CLASSES_MODEL = len(CLASSES)

# Efficiency and Output
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True # From original ViViT
OUTPUT_DIR = "vivit_r2000_finetuned_outputs_v2"
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "train_metrics_vivit_finetuned.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_vivit_finetuned.pth")
LOAD_CHECKPOINT_IF_EXISTS = False # Set to True to skip training if best_model exists

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type

TRIALS_FPS = 50
NUM_WORKERS = 4 if os.name == 'posix' else 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- MODELO Y PROCESADOR ViViT -----
# Initialize processor globally as it's needed by the dataset
try:
    vivit_processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
    vivit_processor.do_center_crop = False #Se hacen manuales para aumentacion de datos
    vivit_processor.do_resize = False #Se hacen manuales para aumentacion de datos
    #vivit_processor.size = {"height": VIDEO_IMAGE_SIZE, "width": VIDEO_IMAGE_SIZE}
except Exception as e:
    logging.error(f"Failed to load ViViT processor: {e}. Ensure you have internet or cached model.")
    vivit_processor = None

def load_vivit_model(num_model_classes=NUM_CLASSES_MODEL, pretrained_model_name=MODEL_NAME):
    logging.info(f"Initializing ViViT model '{pretrained_model_name}' for {num_model_classes} classes.")
    config = VivitConfig.from_pretrained(pretrained_model_name, num_labels=num_model_classes, image_size=VIDEO_IMAGE_SIZE)
    
    # Load pre-trained weights for the base ViViT model, then attach a new head
    # This is a common way to do transfer learning with Hugging Face models.
    # We first load the full pre-trained model to get its base.
    try:
        base_vivit_model = VivitForVideoClassification.from_pretrained(pretrained_model_name, ignore_mismatched_sizes=True)
        model = VivitForVideoClassification(config) # New model with correct head
        # Transfer weights from the base ViViT part
        model.vivit.load_state_dict(base_vivit_model.vivit.state_dict())
    except Exception as e:
        logging.warning(f"Could not load pretrained weights for {pretrained_model_name} due to {e}. Initializing from scratch.")
        model = VivitForVideoClassification(config)


    if USE_GRADIENT_CHECKPOINTING:
        model.vivit.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled for ViViT.")
    return model

# ----- FUNCIONES AUXILIARES Y CLASES DE DATOS -----

def process_video_for_vivit(video_path, num_frames_to_sample, frame_step, image_processor, is_train=True): # Añadido is_train
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Error: No se pudo abrir el vídeo: {video_path}")
        return None
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_in_video <= 0:
        cap.release()
        logging.warning(f"Error: El vídeo no tiene fotogramas o es inválido: {video_path}")
        return None

    # Lógica de selección de frames (similar a tu original, buscando un clip continuo)
    # Puedes mantener tu lógica original de selección de 'start_frame_actual' y 'frame_step' ajustado
    raw_sampled_frames = [] # Lista para los frames leídos directamente del vídeo
    
    # Tu lógica original de selección de frames (ajustada ligeramente para claridad)
    effective_frame_step = frame_step
    if total_frames_in_video < num_frames_to_sample: # Muy corto, tomar todos y rellenar
        start_frame_actual = 0
        effective_frame_step = 1 
    elif total_frames_in_video < num_frames_to_sample * frame_step : # No suficientes para step, tomar desde inicio con step mínimo
        start_frame_actual = 0
        effective_frame_step = max(1, total_frames_in_video // num_frames_to_sample if num_frames_to_sample > 0 else 1)
    else: # Suficientes frames, tomar un clip (ej. centrado o aleatorio si es train)
        # Para un clip continuo y centrado (eval) o inicio aleatorio (train)
        # Este es un ejemplo de muestreo de clip continuo, tu script original ya tenía una lógica similar
        if is_train: # Para entrenamiento, podrías variar el inicio del clip
             max_start_offset = total_frames_in_video - (num_frames_to_sample -1) * effective_frame_step -1
             start_frame_actual = random.randint(0, max_start_offset) if max_start_offset >=0 else 0
        else: # Para validación, un clip centrado o desde el inicio
             start_frame_actual = (total_frames_in_video - (num_frames_to_sample -1) * effective_frame_step) // 2
             start_frame_actual = max(0, start_frame_actual)


    for i in range(num_frames_to_sample):
        frame_idx_to_read = start_frame_actual + i * effective_frame_step
        if frame_idx_to_read >= total_frames_in_video:
            break 
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx_to_read))
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"No se pudo leer el fotograma {int(frame_idx_to_read)} de {video_path}. Saltando.")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_sampled_frames.append(frame_rgb)
    cap.release()

    if not raw_sampled_frames:
        logging.warning(f"No se extrajeron fotogramas de {video_path}")
        return None

    # Padding (repetir último frame)
    while len(raw_sampled_frames) < num_frames_to_sample and raw_sampled_frames:
        raw_sampled_frames.append(raw_sampled_frames[-1].copy())
    
    if len(raw_sampled_frames) != num_frames_to_sample:
        logging.error(f"Fallo de padding: se esperaban {num_frames_to_sample}, se obtuvieron {len(raw_sampled_frames)} para {video_path}")
        return None

    # Aplicar redimensionamiento y recorte (aleatorio o central) a cada frame
    augmented_frames_list = []
    for frame_np_rgb in raw_sampled_frames:
        try:
            resized_frame = cv2.resize(frame_np_rgb, (IMG_RESIZE_DIM_AUG, IMG_RESIZE_DIM_AUG), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logging.error(f"Error al redimensionar frame de {video_path}: {e}")
            return None # Fallo en un frame, abortar para este vídeo

        h_res, w_res, _ = resized_frame.shape
        h_crop, w_crop = VIDEO_IMAGE_SIZE, VIDEO_IMAGE_SIZE # Esta es tu IMG_CROP_SIZE_VIVIT

        if is_train: # Recorte aleatorio
            top = random.randint(0, h_res - h_crop)
            left = random.randint(0, w_res - w_crop)
        else: # Recorte central
            top = (h_res - h_crop) // 2
            left = (w_res - w_crop) // 2
        
        cropped_frame = resized_frame[top:top + h_crop, left:left + w_crop, :]
        augmented_frames_list.append(cropped_frame)

    # Pasar la lista de frames ya aumentados (ej. 224x224) al procesador
    try:
        inputs = image_processor(images=augmented_frames_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"] 
        
        # Verificar si el procesador añadió una dimensión de lote para un solo vídeo
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            # Si es (1, T, C, H, W), quitar el batch dim para que DataLoader lo cree
            pixel_values = pixel_values.squeeze(0) 
            logging.debug(f"Shape de pixel_values tras squeeze(0) en process_video: {pixel_values.shape}")
        elif pixel_values.ndim != 4:
            logging.error(f"Forma inesperada de pixel_values ({pixel_values.shape}) tras el procesador para {video_path}")
            return None
        # Ahora pixel_values debería ser (T, C, H, W)
        return pixel_values
    except Exception as e:
        logging.error(f"Error procesando frames con ViViT processor para {video_path}: {e}")
        return None

class VideoDatasetVivit(Dataset):
    def __init__(self, video_files, labels, image_processor, num_frames, frame_step, is_train=True, dataset_name=""): # Añadido is_train
        self.video_files = video_files
        self.labels = labels
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.is_train = is_train # Guardar el flag
        self.dataset_name = dataset_name

        if self.image_processor is None:
            raise ValueError("VivitImageProcessor no está inicializado.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        processed_pixel_values = process_video_for_vivit(
            video_path, self.num_frames, self.frame_step, self.image_processor, self.is_train # Pasar self.is_train
        )

        if processed_pixel_values is None:
            logging.warning(f"Procesamiento fallido para {video_path} en dataset {self.dataset_name}, devolviendo tensor dummy.")
            # Dummy tensor con forma (T, C, H, W)
            dummy_tensor = torch.zeros((self.num_frames, 3, VIDEO_IMAGE_SIZE, VIDEO_IMAGE_SIZE), dtype=torch.float32)
            return dummy_tensor, torch.tensor(-1) 

        return processed_pixel_values, torch.tensor(label, dtype=torch.long)


def load_dataset_paths_and_labels(base_dir, split_folder_name, class_mapping, dataset_name_for_log=""):
    video_paths, labels = [], []
    split_path = os.path.join(base_dir, split_folder_name)
    logging.info(f"Cargando datos de {dataset_name_for_log} desde: {split_path}")
    for class_name, label_id in class_mapping.items():
        class_folder = os.path.join(split_path, class_name)
        if not os.path.isdir(class_folder):
            logging.warning(f"Carpeta de clase no encontrada: {class_folder}")
            continue
        # Adjusted to common video formats
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

# ----- ENTRENAMIENTO Y EVALUACIÓN -----
def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp_flag, device_type_for_amp):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples_processed = 0
    progress_bar = tqdm(loader, desc="Train", leave=False)

    for pixel_values, labels in progress_bar:
        valid_indices = labels != -1
        if not valid_indices.any():
            continue
        
        pixel_values = pixel_values[valid_indices].to(device, non_blocking=True)
        labels = labels[valid_indices].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        if pixel_values.size(0) == 0:
            continue

        with autocast(device_type=device_type_for_amp, enabled=use_amp_flag):
            outputs = model(pixel_values=pixel_values).logits # ViViT output is a dict
            loss = criterion(outputs, labels)
        
        if use_amp_flag and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        _, preds = torch.max(outputs, 1)
        current_batch_size = pixel_values.size(0)
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

def evaluate(model, loader, criterion, device, use_amp_flag, device_type_for_amp):
    model.eval()
    running_loss = 0.0
    total_samples_processed = 0
    all_preds_list, all_labels_list = [], []
    progress_bar = tqdm(loader, desc="Eval", leave=False)

    with torch.no_grad():
        for pixel_values, labels in progress_bar:
            valid_indices = labels != -1
            if not valid_indices.any():
                continue

            pixel_values = pixel_values[valid_indices].to(device, non_blocking=True)
            labels = labels[valid_indices].to(device, non_blocking=True)

            if pixel_values.size(0) == 0:
                continue
            
            with autocast(device_type=device_type_for_amp, enabled=use_amp_flag):
                outputs = model(pixel_values=pixel_values).logits
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            current_batch_size = pixel_values.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples_processed += current_batch_size
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    if total_samples_processed == 0:
        logging.warning("No valid samples processed during evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0 # loss, acc, prec, rec, f1

    epoch_loss = running_loss / total_samples_processed
    if not all_labels_list or not all_preds_list:
        logging.warning("Evaluation lists are empty, cannot compute metrics.")
        return epoch_loss, 0.0, 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels_list, all_preds_list)
    # Ensure there are positive class predictions for binary classification metrics
    avg_method = 'binary' if len(CLASSES) == 2 and len(np.unique(all_labels_list)) > 1 else 'macro'
    
    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, zero_division=0)
    
    return float(epoch_loss), float(acc), float(precision), float(recall), float(f1)

# ----- MEDICIÓN DE RENDIMIENTO Y GUARDADO -----
def measure_inference_fps_vivit(model, device, num_frames, image_size, trials=TRIALS_FPS):
    # ViViT expects pixel_values of shape (batch_size, num_frames, num_channels, height, width)
    dummy_input = torch.randn(1, num_frames, 3, image_size, image_size, device=device)
    model.eval()
    
    # Warm-up
    for _ in range(10):
        _ = model(pixel_values=dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(trials):
        _ = model(pixel_values=dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    return trials / total_time if total_time > 0 else 0.0

def save_training_metrics(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy types to native Python types for JSON serialization
    for key, values in history.items():
        if isinstance(values, list):
            history[key] = [float(v) if isinstance(v, (np.float32, np.float64, np.int64, np.int32, torch.Tensor)) else v for v in values]
        elif isinstance(values, dict): # For performance_stats
            for sub_key, sub_val in values.items():
                 if isinstance(sub_val, (np.float32, np.float64, np.int64, np.int32, torch.Tensor)):
                     values[sub_key] = float(sub_val)
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"Métricas de entrenamiento guardadas en {path}")

# ----- BUCLE PRINCIPAL DE ENTRENAMIENTO -----
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
        model_for_training_or_analysis = load_vivit_model(num_model_classes=NUM_CLASSES_MODEL)
        try:
            model_for_training_or_analysis.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
            logging.info(f"Pesos del modelo cargados desde {BEST_MODEL_PATH}")
        except Exception as e:
            logging.error(f"Error al cargar el estado del modelo desde {BEST_MODEL_PATH}: {e}. Se procederá con un modelo nuevo si el entrenamiento está activo.")
            if not perform_training: # If we intended to load but failed, and not training, then error out.
                 logging.error("Fallo al cargar el modelo para análisis y el entrenamiento está desactivado. Saliendo.")
                 return
            model_for_training_or_analysis = load_vivit_model(num_model_classes=NUM_CLASSES_MODEL) # re-init

        model_for_training_or_analysis = model_for_training_or_analysis.to(DEVICE)
    else:
        if LOAD_CHECKPOINT_IF_EXISTS:
            logging.info(f"LOAD_CHECKPOINT_IF_EXISTS es True pero no se encontró checkpoint en {BEST_MODEL_PATH}.")
        logging.info("Se procederá con el entrenamiento (o carga de preentrenado de Hugging Face).")
        model_for_training_or_analysis = load_vivit_model(num_model_classes=NUM_CLASSES_MODEL)
        model_for_training_or_analysis = model_for_training_or_analysis.to(DEVICE)

    if perform_training and model_for_training_or_analysis is not None:
        if dataset_name == "RWF-2000":
            train_video_paths, train_labels = get_rwf2000_data("train")
            val_video_paths, val_labels = get_rwf2000_data("val")
        # Add other dataset loaders here if needed
        # elif dataset_name == "HockeyFights": ...
        else:
            raise ValueError(f"Dataset no soportado: {dataset_name}")

        if not train_video_paths:
            logging.error(f"No se encontraron vídeos de entrenamiento para {dataset_name}. Saliendo.")
            return
        if not val_video_paths:
            logging.warning(f"No se encontraron vídeos de validación para {dataset_name}.")


        train_dataset = VideoDatasetVivit(train_video_paths, train_labels, vivit_processor,
                                          NUM_FRAMES_TO_SAMPLE, FRAME_STEP, 
                                          is_train=True, # Para el dataset de entrenamiento
                                          dataset_name=dataset_name + "-train")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        
        val_loader = None
        if val_video_paths:
            val_dataset = VideoDatasetVivit(val_video_paths, val_labels, vivit_processor,
                                            NUM_FRAMES_TO_SAMPLE, FRAME_STEP,
                                            is_train=False, # Para el dataset de validación
                                            dataset_name=dataset_name + "-val")
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_for_training_or_analysis.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler(enabled=use_amp_for_training)

        history = {'dataset': dataset_name, 'model_name': MODEL_NAME, 
                   'epochs_run': [], 'train_loss': [], 'train_acc': [],
                   'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
                   'epoch_time_seconds': []}
        best_val_f1 = 0.0

        logging.info(f"Iniciando fine-tuning de {EPOCHS} épocas en {dataset_name}...")
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            logging.info(f"--- Época {epoch}/{EPOCHS} ---")
            
            train_loss, train_acc = train_epoch(model_for_training_or_analysis, train_loader, criterion, optimizer, DEVICE, scaler, use_amp_for_training, DEVICE_TYPE)
            logging.info(f"Época {epoch} Train: Pérdida={train_loss:.4f}, Acc={train_acc:.4f}")
            history['epochs_run'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader and (val_loader.dataset is not None and len(val_loader.dataset) > 0):
                val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model_for_training_or_analysis, val_loader, criterion, DEVICE, use_amp_for_training, DEVICE_TYPE)
                logging.info(f"Época {epoch} Val: Pérdida={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}")
                history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
                history['val_precision'].append(val_prec); history['val_recall'].append(val_rec)
                history['val_f1'].append(val_f1)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model_for_training_or_analysis.state_dict(), BEST_MODEL_PATH)
                    # Save processor and config with the best model for easier reloading
                    model_for_training_or_analysis.config.save_pretrained(OUTPUT_DIR)
                    vivit_processor.save_pretrained(OUTPUT_DIR)
                    logging.info(f"  Mejor F1 en validación: {best_val_f1:.4f}. Modelo, config y processor guardados en {OUTPUT_DIR}.")
            else: # No validation loader
                history['val_loss'].append(None); history['val_acc'].append(None)
                history['val_precision'].append(None); history['val_recall'].append(None)
                history['val_f1'].append(None)
                if epoch == EPOCHS: # Save last model if no validation
                    torch.save(model_for_training_or_analysis.state_dict(), BEST_MODEL_PATH)
                    model_for_training_or_analysis.config.save_pretrained(OUTPUT_DIR)
                    vivit_processor.save_pretrained(OUTPUT_DIR)
                    logging.info(f"Sin validación. Modelo de época {epoch}, config y processor guardados en {OUTPUT_DIR}.")

            epoch_duration = time.time() - epoch_start_time
            history['epoch_time_seconds'].append(float(epoch_duration))
            logging.info(f"Época {epoch} completada en {epoch_duration:.2f}s")
            save_training_metrics(history, METRICS_JSON_PATH)

        logging.info("Fine-tuning completado.")
        if val_loader and (val_loader.dataset is not None and len(val_loader.dataset) > 0):
            logging.info(f"Mejor F1 en validación: {best_val_f1:.4f}.")
        logging.info(f"Mejor modelo guardado en: {BEST_MODEL_PATH}")
        logging.info(f"Métricas de entrenamiento guardadas en: {METRICS_JSON_PATH}")
        logging.info(f"Config y processor guardados en: {OUTPUT_DIR}")


    # --- ANÁLISIS FINAL DEL MODELO ---
    final_analysis_model = None
    if os.path.exists(BEST_MODEL_PATH):
        logging.info(f"Cargando mejor modelo desde {BEST_MODEL_PATH} para análisis final.")
        # For analysis, load the model structure first, then the state_dict
        final_analysis_model = load_vivit_model(num_model_classes=NUM_CLASSES_MODEL)
        try:
            final_analysis_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            logging.error(f"Error al cargar state_dict para análisis final desde {BEST_MODEL_PATH}: {e}")
            final_analysis_model = None # Ensure it's None if loading failed
        
        if final_analysis_model:
            final_analysis_model = final_analysis_model.to(DEVICE)
            final_analysis_model.eval()
    else:
        if model_for_training_or_analysis is not None and not perform_training: # Loaded for analysis but path was wrong
             logging.warning(f"Se esperaba un modelo en {BEST_MODEL_PATH} para análisis pero no se encontró. Usando el modelo cargado en memoria si existe.")
             final_analysis_model = model_for_training_or_analysis
        elif model_for_training_or_analysis is not None and perform_training: # Training happened, use that model
            logging.warning(f"No se encontró {BEST_MODEL_PATH} (quizás no hubo mejora o no validación). Usando el modelo de la última época para análisis.")
            final_analysis_model = model_for_training_or_analysis # Already on device
        else:
            logging.error(f"No se encontró el mejor modelo en {BEST_MODEL_PATH} y no hay modelo en memoria. No se puede realizar análisis final.")
            final_analysis_model = None

    if final_analysis_model:
        final_analysis_model.eval() # Ensure eval mode

        fps = measure_inference_fps_vivit(final_analysis_model, DEVICE, NUM_FRAMES_TO_SAMPLE, VIDEO_IMAGE_SIZE, trials=TRIALS_FPS)
        
        params_count = 0
        try: # fvcore parameter_count expects a PyTorch nn.Module
            params_count = parameter_count(final_analysis_model)['']
        except Exception as e:
            logging.error(f"No se pudieron contar los parámetros con fvcore: {e}. Intentando con PyTorch sum.")
            params_count = sum(p.numel() for p in final_analysis_model.parameters() if p.requires_grad)


        # For GFLOPs, ViViT expects 'pixel_values' as a keyword argument or in a dict.
        # FlopCountAnalysis typically works best with models whose forward accepts tensors directly.
        # We might need to wrap the model or pass input carefully.
        # Model forward: forward(self, pixel_values=None, head_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None)
        # The input to FlopCountAnalysis should be a tuple/list of args for model.forward()
        dummy_input_flops = torch.randn(1, NUM_FRAMES_TO_SAMPLE, 3, VIDEO_IMAGE_SIZE, VIDEO_IMAGE_SIZE, device=DEVICE)
        
        gflops = -1.0
        try:
            # We need to pass pixel_values as the first positional argument if the model's forward can take it like that,
            # or FlopCountAnalysis might not pick it up correctly.
            # Let's try passing it as a tuple: (pixel_values_tensor,)
            # Or, if FlopCountAnalysis supports kwargs, {'pixel_values': dummy_input_flops}
            # Most robust way is to ensure the model's forward can accept it as *args
            # For HuggingFace models, it's often better to analyze the base model (e.g., model.vivit)
            # then add classifier FLOPs manually, or use a dedicated tool if fvcore struggles.
            
            # Attempt with the full model; FlopCountAnalysis expects args, not kwargs for forward.
            # We'll pass it as the first argument.
            flops_analyzer = FlopCountAnalysis(final_analysis_model, (dummy_input_flops,))
            gflops = flops_analyzer.total() / 1e9
        except Exception as e:
            logging.error(f"No se pudieron calcular los FLOPs con fvcore para ViVitForVideoClassification: {e}.")
            logging.info("Esto puede ocurrir con modelos de Transformers que esperan argumentos nombrados o tienen estructuras complejas no fácilmente analizables por fvcore.")
            logging.info("Se puede intentar analizar `model.vivit` y añadir los FLOPs del clasificador manualmente si es necesario.")

        logging.info(f"FPS de Inferencia del Modelo: {fps:.2f}")
        logging.info(f"Parámetros del Modelo: {params_count:,}")
        logging.info(f"GFLOPs del Modelo: {gflops:.2f}G (puede ser impreciso o -1.0 si falla el cálculo)")

        if os.path.exists(METRICS_JSON_PATH) or not perform_training:
            try:
                final_metrics = {}
                if os.path.exists(METRICS_JSON_PATH):
                    with open(METRICS_JSON_PATH, 'r') as f:
                        final_metrics = json.load(f)
                
                final_metrics['hyperparameters'] = {
                    'model_name': MODEL_NAME,
                    'num_frames': NUM_FRAMES_TO_SAMPLE,
                    'frame_step': FRAME_STEP,
                    'video_image_size': VIDEO_IMAGE_SIZE,
                    'batch_size': BATCH_SIZE,
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'epochs_config': EPOCHS,
                    'use_amp': USE_AMP,
                    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING
                }
                final_metrics['performance_stats'] = {
                    'fps': float(fps),
                    'parameters': int(params_count),
                    'gflops': float(gflops)
                }
                save_training_metrics(final_metrics, METRICS_JSON_PATH)
            except Exception as e:
                logging.error(f"Error al actualizar métricas con estadísticas de rendimiento e hiperparámetros: {e}")
    else:
        logging.warning("Análisis final omitido porque no se pudo cargar o encontrar un modelo.")


if __name__ == '__main__':
    if vivit_processor is None:
        logging.error("El procesador ViViT no se pudo inicializar. Saliendo del script.")
    else:
        dataset_to_train = "RWF-2000" # O el dataset que desees entrenar
        main_train_loop(dataset_name=dataset_to_train)