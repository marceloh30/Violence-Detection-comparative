""""
Script modular para entrenamiento, evaluación e inferencia de I3D R50 en clasificación de vídeos violentos vs no violentos.
Incluye:
 - Extracción de frames con OpenCV (función process_video)
 - Creación de datasets con cargar_dataset (devuelve TensorDataset)
 - Funciones de entrenamiento y evaluación (train_epoch, evaluate)
 - Logging de métricas en JSON (formato similar a SlowFast) y TensorBoardX
 - Análisis de tiempo, FPS, parámetros y FLOPs
 - Sección de Inferencia en dataset adicional (ej. RLVS)
Requisitos: pip install torch torchvision pytorchvideo opencv-python fvcore scikit-learn tensorboardX tqdm matplotlib
"""
import os
import json
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pytorchvideo.models.hub import i3d_r50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fvcore.nn import FlopCountAnalysis, parameter_count
from tensorboardX import SummaryWriter # Opcional: puedes cambiar a torch.utils.tensorboard
from tqdm import tqdm
import logging # Añadido para un logging más estructurado

# ----- CONFIGURACIÓN -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_PATH = "assets/RWF-2000"
SPLITS = {"train": "train", "val": "val"}
CLASSES = {"Fight": 1, "NonFight": 0}
DATASET_NAME_FOR_HISTORY = "RWF-2000_I3D" # Nombre del dataset para el JSON

NUM_FRAMES = 32
FRAME_STEP = 4
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10 # Manteniendo 10 épocas como en tu I3D original
IMG_SIZE = 224

OUTPUT_DIR = "i3d_r50_outputs_rwf2000_metrics_aligned" # Directorio de salida
LOG_DIR_TENSORBOARD = os.path.join(OUTPUT_DIR, "logs_tensorboard") # Carpeta para logs de TensorBoard
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "train_metrics_i3d.json") # Ruta para el JSON de métricas
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "i3d_best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type # Para autocast ('cuda' o 'cpu')

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
TRIALS_FPS = 100
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

USE_AMP_TRAINING = (DEVICE.type == 'cuda') # Habilitar AMP para entrenamiento si es GPU

# Configuración para inferencia en RLVS (mantener si es necesario)
USE_RLVS_INFERENCE = False # Cambiar a True para ejecutar inferencia en RLVS
RLVS_INFERENCE_PATH = "assets/RLVS"
RLVS_CLASSES = {"Violence": 1, "NonViolence": 0}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR_TENSORBOARD, exist_ok=True)

# ----- FUNCIONES UTILES -----
# (process_video, cargar_dataset como las tenías, adaptadas para logging)
def process_video(path, num_frames_to_sample=NUM_FRAMES, img_target_size=IMG_SIZE, sampling_step=FRAME_STEP):
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
        start_index = random.randint(0, selectable_frames_range - 1)
        indices_a_muestrear = [start_index + i * sampling_step for i in range(num_frames_to_sample)]
    else: 
        available_indices = list(range(0, total_video_frames, max(1, sampling_step))) # Asegurar que sampling_step > 0
        if not available_indices: available_indices = list(range(total_video_frames))
        if not available_indices:
            cap.release()
            logging.warning(f"No hay suficientes fotogramas para muestrear y el relleno falló para: {path}")
            return None
        indices_a_muestrear = available_indices[:num_frames_to_sample]
        while len(indices_a_muestrear) < num_frames_to_sample:
            indices_a_muestrear.append(available_indices[-1])
        
    frames_procesados = []
    for frame_num in indices_a_muestrear:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        if not ret: 
            logging.warning(f"No se pudo leer el fotograma {int(frame_num)} de {path}. Usando fotograma negro.")
            frame = np.zeros((img_target_size, img_target_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_target_size, img_target_size))
        frames_procesados.append(frame)
    cap.release()
    
    if len(frames_procesados) != num_frames_to_sample:
        logging.error(f"Error de procesamiento: se esperaban {num_frames_to_sample} fotogramas, se obtuvieron {len(frames_procesados)} para {path}")
        return None # Devolver None si el número de frames no es el esperado

    clip = np.stack(frames_procesados)
    clip = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0
    clip = clip.permute(1, 0, 2, 3)
    clip = (clip - MEAN) / STD
    return clip

def cargar_dataset(split_name, base_dataset_path, split_folders_map, class_labels_map):
    pixels, labels = [], []
    logging.info(f"Cargando split: {split_name} desde {base_dataset_path}...")
    folder_name_for_split = split_folders_map[split_name]
    
    for cls_name, lbl in class_labels_map.items():
        folder_path = os.path.join(base_dataset_path, folder_name_for_split, cls_name)
        if not os.path.isdir(folder_path):
            logging.warning(f"Carpeta no encontrada {folder_path}")
            continue
        
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(VIDEO_EXTENSIONS)]
        logging.info(f"  Encontrados {len(video_files)} vídeos en {folder_path}")
        for fname in tqdm(video_files, desc=f"Cargando {cls_name} ({split_name})"):
            path = os.path.join(folder_path, fname)
            clip = process_video(path)
            if clip is not None: # Solo añadir si el clip se procesó correctamente
                pixels.append(clip)
                labels.append(lbl)
            else:
                logging.warning(f"Clip omitido debido a error de procesamiento: {path}")
                
    if not pixels: # Si no se cargó ningún vídeo válido para este split
        logging.error(f"No se procesaron vídeos válidos para split={split_name}. Revisa BASE_PATH y la estructura de carpetas.")
        return None # Devolver None si no hay datos
        
    x = torch.stack(pixels)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)

# Modificar train_epoch y evaluate para devolver val_loss y usar AMP
def train_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for x, y in progress_bar:
        if x is None: continue # Omitir si process_video devolvió None y se propagó
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=use_amp):
            out = model(x) 
            out_flat = out.view(out.size(0), -1) 
            loss = criterion(out_flat, y)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        _, preds = torch.max(out_flat, 1)
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y.data)
        total_samples += x.size(0)
        progress_bar.set_postfix(loss_batch=f"{loss.item():.4f}")
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = running_corrects.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc

def evaluate(model, loader, criterion, device, use_amp): # Añadido criterion
    model.eval()
    running_loss = 0.0 # Para val_loss
    running_corrects = 0 # Para val_acc
    total_samples = 0    # Para val_loss y val_acc
    all_preds_list, all_labels_list = [], []
    
    progress_bar = tqdm(loader, desc="Eval", leave=False)
    with torch.no_grad():
        for x, y in progress_bar:
            if x is None: continue
            x, y_true = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=use_amp):
                out = model(x) 
                out_flat = out.view(out.size(0), -1)
                loss = criterion(out_flat, y_true) # Calcular loss de validación
            
            _, preds = torch.max(out_flat, 1)
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds == y_true.data)
            total_samples += x.size(0)

            all_preds_list.extend(preds.cpu().numpy()) 
            all_labels_list.extend(y_true.cpu().numpy())
            
    if total_samples == 0:
        logging.warning("No se procesaron muestras válidas durante la evaluación.")
        return 0.0, 0.0, 0.0, 0.0, 0.0 # val_loss, acc, prec, rec, f1

    epoch_val_loss = running_loss / total_samples
    acc = accuracy_score(all_labels_list, all_preds_list)
    # Usar CLASSES["Fight"] como pos_label asumiendo que es 1 y es la clase de interés
    precision = precision_score(all_labels_list, all_preds_list, labels=list(CLASSES.values()), average='binary', pos_label=CLASSES.get("Fight", 1), zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, labels=list(CLASSES.values()), average='binary', pos_label=CLASSES.get("Fight", 1), zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, labels=list(CLASSES.values()), average='binary', pos_label=CLASSES.get("Fight", 1), zero_division=0)
    
    return epoch_val_loss, acc, precision, recall, f1


def measure_inference_fps(model_to_measure, device_to_use, clip_s=NUM_FRAMES, img_s=IMG_SIZE, num_trials=TRIALS_FPS):
    dummy = torch.randn(1, 3, clip_s, img_s, img_s, device=device_to_use)
    model_to_measure.eval() 
    for _ in range(10): _ = model_to_measure(dummy) # Warm-up
        
    if device_to_use.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials): _ = model_to_measure(dummy)
    if device_to_use.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_time
    return num_trials / total_time if total_time > 0 else 0.0

def save_metrics_to_json(metrics_dict, json_path): # Renombrada para claridad
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        # Si el archivo ya existe y tiene contenido, cargarlo para actualizarlo
        existing_data = {}
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            with open(json_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"El archivo JSON {json_path} existe pero está corrupto. Se sobrescribirá.")
        
        # Actualizar o fusionar con los nuevos datos
        # Para métricas de época, se espera que metrics_dict sea el historial completo
        # Para performance_stats, se espera que metrics_dict sea solo esa parte
        if 'performance_stats' in metrics_dict: # Si estamos guardando solo performance_stats
            if 'performance_stats' not in existing_data:
                existing_data['performance_stats'] = {}
            existing_data['performance_stats'].update(metrics_dict['performance_stats'])
            data_to_save = existing_data
        else: # Guardando el historial completo de épocas
            data_to_save = metrics_dict
            # Si había performance_stats previamente, mantenerlas
            if 'performance_stats' in existing_data:
                data_to_save['performance_stats'] = existing_data['performance_stats']


        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logging.info(f"Métricas guardadas/actualizadas en {json_path}")
    except Exception as e:
        logging.error(f"Error al guardar métricas en JSON {json_path}: {e}")


# ----- FUNCIÓN PRINCIPAL -----
def main():
    writer = SummaryWriter(LOG_DIR_TENSORBOARD) # TensorBoard writer

    train_dataset = cargar_dataset('train', BASE_PATH, SPLITS, CLASSES)
    val_dataset   = cargar_dataset('val', BASE_PATH, SPLITS, CLASSES)

    if train_dataset is None:
        logging.error("No se pudo cargar el dataset de entrenamiento. Abortando.")
        return
    
    num_data_workers = 2 if os.name == 'posix' else 0 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_data_workers, pin_memory=True)
    # Solo crear val_loader si val_dataset se cargó correctamente
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_data_workers, pin_memory=True) if val_dataset else None

    model = i3d_r50(pretrained=True)
    final_projection_layer = model.blocks[-1].proj
    if isinstance(final_projection_layer, nn.Conv3d):
        model.blocks[-1].proj = nn.Conv3d(final_projection_layer.in_channels, len(CLASSES), final_projection_layer.kernel_size, final_projection_layer.stride)
    elif isinstance(final_projection_layer, nn.Linear):
        model.blocks[-1].proj = nn.Linear(final_projection_layer.in_features, len(CLASSES))
    else:
        raise TypeError(f"Capa final del modelo inesperada: {type(final_projection_layer)}")
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device=DEVICE_TYPE, enabled=USE_AMP_TRAINING)

    # Inicializar historial con el formato deseado
    history = {
        'dataset': DATASET_NAME_FOR_HISTORY,
        'epochs_run': [],
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'epoch_time_seconds': []
    }
    best_val_f1 = 0.0

    logging.info(f"Iniciando entrenamiento en {DEVICE} para {DATASET_NAME_FOR_HISTORY}...")
    logging.info(f"Config: NUM_FRAMES={NUM_FRAMES}, FRAME_STEP={FRAME_STEP}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        logging.info(f"--- Época {epoch}/{EPOCHS} ---")
        
        train_loss_val, train_acc_val = train_epoch(model, train_loader, criterion, optimizer, DEVICE, USE_AMP_TRAINING, scaler)
        history['epochs_run'].append(epoch)
        history['train_loss'].append(train_loss_val)
        history['train_acc'].append(train_acc_val)
        writer.add_scalar('Loss/train', train_loss_val, epoch)
        writer.add_scalar('Accuracy/train', train_acc_val, epoch)
        logging.info(f"Época {epoch} Train: Pérdida={train_loss_val:.4f}, Acc={train_acc_val:.4f}")

        val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val = 0.0, 0.0, 0.0, 0.0, 0.0
        if val_loader:
            val_loss_val, val_acc_val, val_prec_val, val_rec_val, val_f1_val = evaluate(model, val_loader, criterion, DEVICE, USE_AMP_TRAINING)
            writer.add_scalar('Loss/val', val_loss_val, epoch)
            writer.add_scalar('Accuracy/val', val_acc_val, epoch)
            writer.add_scalar('Precision/val', val_prec_val, epoch)
            writer.add_scalar('Recall/val', val_rec_val, epoch)
            writer.add_scalar('F1/val', val_f1_val, epoch)
            logging.info(f"Época {epoch} Val: Pérdida={val_loss_val:.4f}, Acc={val_acc_val:.4f}, Prec={val_prec_val:.4f}, Rec={val_rec_val:.4f}, F1={val_f1_val:.4f}")
        
        history['val_loss'].append(val_loss_val if val_loader else None)
        history['val_acc'].append(val_acc_val if val_loader else None)
        history['val_precision'].append(val_prec_val if val_loader else None)
        history['val_recall'].append(val_rec_val if val_loader else None)
        history['val_f1'].append(val_f1_val if val_loader else None)
        
        epoch_duration_val = time.time() - epoch_start_time
        history['epoch_time_seconds'].append(epoch_duration_val)
        writer.add_scalar('Time/epoch', epoch_duration_val, epoch)
        logging.info(f"Época {epoch} completada en {epoch_duration_val:.2f}s")

        # Guardar métricas al final de cada época
        save_metrics_to_json(history, METRICS_JSON_PATH)

        if val_loader and val_f1_val > best_val_f1:
            best_val_f1 = val_f1_val
            # Guardar checkpoint completo (similar a SlowFast)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'best_val_f1': best_val_f1, 
                'scaler_state_dict': scaler.state_dict() if USE_AMP_TRAINING else None
            }
            torch.save(checkpoint_data, BEST_MODEL_PATH)
            logging.info(f"  Mejor F1 en validación: {best_val_f1:.4f}. Checkpoint guardado en {BEST_MODEL_PATH}")
        elif not val_loader and epoch == EPOCHS: # Si no hay validación, guardar al final
             torch.save({'model_state_dict': model.state_dict()}, BEST_MODEL_PATH)
             logging.info(f"Entrenamiento sin validación. Modelo de época {epoch} guardado en {BEST_MODEL_PATH}")


    logging.info("Entrenamiento completado.")
    if val_loader: logging.info(f"Mejor F1 en validación final: {best_val_f1:.4f}.")
    logging.info(f"Modelo final (mejor o último) guardado en: {BEST_MODEL_PATH}")
    logging.info(f"Historial de métricas guardado en: {METRICS_JSON_PATH}")

    # ----- Cargar el mejor modelo para análisis final -----
    model_for_final_analysis = i3d_r50(pretrained=False) 
    final_proj_analysis = model_for_final_analysis.blocks[-1].proj
    if isinstance(final_proj_analysis, nn.Conv3d):
        model_for_final_analysis.blocks[-1].proj = nn.Conv3d(final_proj_analysis.in_channels, len(CLASSES), final_proj_analysis.kernel_size, final_proj_analysis.stride)
    elif isinstance(final_proj_analysis, nn.Linear):
        model_for_final_analysis.blocks[-1].proj = nn.Linear(final_proj_analysis.in_features, len(CLASSES))
    
    model_loaded_for_analysis = False
    if os.path.exists(BEST_MODEL_PATH):
        logging.info(f"Cargando modelo desde {BEST_MODEL_PATH} para análisis final.")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint: # Si es un checkpoint completo
            model_for_final_analysis.load_state_dict(checkpoint['model_state_dict'])
        else: # Si solo se guardó el state_dict del modelo
            model_for_final_analysis.load_state_dict(checkpoint)
        model_for_final_analysis.to(DEVICE)
        model_for_final_analysis.eval()
        model_loaded_for_analysis = True
    else:
        logging.warning(f"No se encontró el archivo {BEST_MODEL_PATH}. El análisis de rendimiento (FPS, etc.) se omitirá o usará el modelo en memoria si es el caso.")
        # Si el entrenamiento acaba de ocurrir y no se guardó un "best_model" (ej. sin val),
        # 'model' aún está en memoria.
        if 'model' in locals() and isinstance(model, nn.Module):
            model_for_final_analysis = model # Usar el modelo de la última época
            model_for_final_analysis.eval()
            model_loaded_for_analysis = True
            logging.info("Usando el modelo de la última época de entrenamiento para análisis.")


    if model_loaded_for_analysis:
        logging.info("Calculando estadísticas de rendimiento del modelo cargado...")
        fps = measure_inference_fps(model_for_final_analysis, DEVICE)
        params_count = parameter_count(model_for_final_analysis).get('', 0) # Usar get con default
        
        gflops = -1.0 # Default si falla el cálculo
        try:
            dummy_input_flops = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE, device=DEVICE)
            # Comprobar si el modelo tiene módulos antes de FlopCountAnalysis
            if hasattr(model_for_final_analysis, '_modules') and model_for_final_analysis._modules:
                 flops_analyzer = FlopCountAnalysis(model_for_final_analysis, dummy_input_flops)
                 gflops = flops_analyzer.total() / 1e9
            else:
                logging.warning("El modelo de análisis parece estar vacío o no es compatible con FlopCountAnalysis.")
        except Exception as e:
            logging.error(f"No se pudieron calcular los FLOPs: {e}")
            
        logging.info(f"Modelo Cargado - Inference FPS: {fps:.2f}, Params: {params_count/1e6:.2f}M, GFLOPs: {gflops:.2f}G")
        
        # Cargar historial, añadir performance_stats y guardar
        performance_stats_data = {
            'performance_stats': {
                'fps': float(fps),
                'parameters': int(params_count),
                'gflops': float(gflops)
            }
        }
        save_metrics_to_json(performance_stats_data, METRICS_JSON_PATH) # Esto fusionará performance_stats

        writer.add_scalar('Performance/FPS_final_model', fps)
        writer.add_scalar('Performance/Params_M_final_model', params_count / 1e6)
        writer.add_scalar('Performance/FLOPs_G_final_model', gflops)
    else:
        logging.warning("Análisis de rendimiento final omitido porque no se pudo cargar un modelo.")

    # ----- INFERENCIA EN DATASET RLVS (O CUALQUIER OTRO) -----
    # (La lógica de inferencia RLVS que te proporcioné antes iría aquí)
    # ... Asegúrate de usar model_for_final_analysis si está cargado y es válido ...
    if USE_RLVS_INFERENCE and model_loaded_for_analysis:
        logging.info(f"\nComenzando inferencia en el dataset RLVS: '{RLVS_INFERENCE_PATH}'...")
        rlvs_results_list = []
        all_rlvs_true_labels, all_rlvs_predicted_labels = [], []

        for class_name_rlvs, label_rlvs in tqdm(RLVS_CLASSES.items(), desc="Procesando carpetas de RLVS"):
            folder_path_rlvs = os.path.join(RLVS_INFERENCE_PATH, class_name_rlvs)
            if not os.path.isdir(folder_path_rlvs):
                logging.warning(f"  Carpeta de inferencia RLVS '{class_name_rlvs}' no encontrada en {folder_path_rlvs}")
                continue
            
            video_files_rlvs = [f for f in os.listdir(folder_path_rlvs) if f.lower().endswith(VIDEO_EXTENSIONS)]
            for video_filename_rlvs in tqdm(video_files_rlvs, desc=f"  Videos en {class_name_rlvs}", leave=False):
                full_video_path_rlvs = os.path.join(folder_path_rlvs, video_filename_rlvs)
                clip_tensor = process_video(full_video_path_rlvs) 
                
                if clip_tensor is None:
                    logging.warning(f"    RLVS: Clip omitido (error de procesamiento): {full_video_path_rlvs}")
                    continue
                
                clip_batch_tensor = clip_tensor.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP_TRAINING):
                        outputs_rlvs = model_for_final_analysis(clip_batch_tensor)
                        logits_rlvs = outputs_rlvs.view(outputs_rlvs.size(0), -1)
                    pred_id_rlvs = logits_rlvs.argmax(-1).item()
                
                all_rlvs_true_labels.append(label_rlvs)
                all_rlvs_predicted_labels.append(pred_id_rlvs)
                rlvs_results_list.append({
                    "file": full_video_path_rlvs, 
                    "predicted_label_id": pred_id_rlvs, 
                    "true_label_id": label_rlvs
                })
        
        if all_rlvs_true_labels:
            rlvs_pos_label_val = 1 if 1 in RLVS_CLASSES.values() else list(RLVS_CLASSES.values())[0]
            rlvs_acc = accuracy_score(all_rlvs_true_labels, all_rlvs_predicted_labels)
            rlvs_prec = precision_score(all_rlvs_true_labels, all_rlvs_predicted_labels, labels=list(RLVS_CLASSES.values()), average='binary', pos_label=rlvs_pos_label_val, zero_division=0)
            rlvs_rec = recall_score(all_rlvs_true_labels, all_rlvs_predicted_labels, labels=list(RLVS_CLASSES.values()), average='binary', pos_label=rlvs_pos_label_val, zero_division=0)
            rlvs_f1 = f1_score(all_rlvs_true_labels, all_rlvs_predicted_labels, labels=list(RLVS_CLASSES.values()), average='binary', pos_label=rlvs_pos_label_val, zero_division=0)
            rlvs_cm = confusion_matrix(all_rlvs_true_labels, all_rlvs_predicted_labels, labels=list(RLVS_CLASSES.values())).tolist()
            
            logging.info("\n--- Resultados de Inferencia en Dataset RLVS ---")
            logging.info(f"Accuracy: {rlvs_acc:.4f}")
            logging.info(f"Precision (clase {rlvs_pos_label_val}): {rlvs_prec:.4f}")
            logging.info(f"Recall (clase {rlvs_pos_label_val}): {rlvs_rec:.4f}")
            logging.info(f"F1-Score (clase {rlvs_pos_label_val}): {rlvs_f1:.4f}")
            logging.info(f"Matriz de Confusión RLVS:\n{rlvs_cm}")

            rlvs_final_stats = {
                'dataset_name': "RLVS_Inference",
                'accuracy': rlvs_acc, 'precision': rlvs_prec, 'recall': rlvs_rec, 'f1': rlvs_f1, 
                'confusion_matrix': rlvs_cm, 'resultados_detallados_rlvs': rlvs_results_list
            }
            rlvs_metrics_json_path = os.path.join(OUTPUT_DIR, 'inference_metrics_rlvs.json')
            save_metrics_to_json(rlvs_final_stats, rlvs_metrics_json_path) # Usar la función renombrada
        else:
            logging.info("No se procesaron vídeos para inferencia en RLVS o no se encontraron etiquetas.")
    elif USE_RLVS_INFERENCE and not model_loaded_for_analysis:
        logging.warning("Se solicitó inferencia en RLVS, pero no hay un modelo cargado/disponible.")

    writer.close()

if __name__ == '__main__':
    main()