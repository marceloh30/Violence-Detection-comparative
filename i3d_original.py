""""
Script modular para entrenamiento, evaluación e inferencia de I3D R50 en clasificación de vídeos violentos vs no violentos.
Incluye:
 - Extracción de frames con OpenCV (función process_video)
 - Creación de datasets con cargar_dataset (devuelve TensorDataset)
 - Funciones de entrenamiento y evaluación (train_epoch, evaluate)
 - Logging de métricas en JSON y TensorBoardX
 - Análisis de tiempo, FPS, parámetros y FLOPs
 - Visualización de Grad-CAM
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fvcore.nn import FlopCountAnalysis, parameter_count
from tensorboardX import SummaryWriter
from tqdm import tqdm

# ----- CONFIGURACIÓN -----
BASE_PATH = "assets/RWF-2000"  # Asegúrate que esta ruta sea correcta
SPLITS = {"train": "train", "val": "val"}
CLASSES = {"Fight": 1, "NonFight": 0} # "Fight" es la clase positiva por defecto para métricas binarias
NUM_FRAMES = 16
IMG_SIZE = 224
NUM_FRAMES = 32                # Originalmente estaba en 16 y sin salto entre frames (batch_size =4 ) probar y ver
FRAME_STEP = 4                 # Saltos entre frames
BATCH_SIZE = 2                 # Reducir batch size
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10
OUTPUT_DIR = "i3d_r50_outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
METRICS_JSON = os.path.join(OUTPUT_DIR, "train_metrics.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalización manual de canales para tensores [C,T,H,W] en CPU
# Se aplican en process_video que devuelve tensores en CPU
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
TRIALS_FPS = 100  # número de iteraciones para medir FPS

# ----- FUNCIONES UTILES -----
def process_video(path, num_frames=NUM_FRAMES, img_size=IMG_SIZE, step=FRAME_STEP):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el vídeo: {path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"Error: El vídeo no tiene frames o es inválido: {path}")
        return None
        
    # generar índices de frames equidistantes
    if total >= num_frames * step:
        start = random.randint(0, total - num_frames * step)
        idxs = [start + i * step for i in range(num_frames)]
    else: # Pad si el vídeo es corto
        idxs = list(range(total)) + [total-1] * (num_frames - total)
        
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret: # Si no se puede leer un frame, usar uno negro
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    
    # convertir a tensor [C,T,H,W]
    clip = np.stack(frames)  # [T,H,W,C]
    clip = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
    clip = clip.permute(1, 0, 2, 3)  # [C,T,H,W]
    
    # normalización manual (en CPU)
    clip = (clip - MEAN) / STD
    return clip


def cargar_dataset(split):
    pixels, labels = [], []
    print(f"Cargando split: {split}...")
    for cls_name, lbl in CLASSES.items():
        folder = os.path.join(BASE_PATH, SPLITS[split], cls_name)
        if not os.path.isdir(folder):
            print(f"Advertencia: Carpeta no encontrada {folder}")
            continue
        video_files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"  Encontrados {len(video_files)} vídeos en {folder}")
        for fname in tqdm(video_files, desc=f"[cargar_dataset()] Procesando {cls_name} ({split})"):
            path = os.path.join(folder, fname)
            clip = process_video(path)
            if clip is not None:
                pixels.append(clip)
                labels.append(lbl)
                
    if not pixels:
        raise ValueError(f"No se procesaron vídeos para split={split}. Revisa BASE_PATH y la estructura de carpetas.")
        
    x = torch.stack(pixels)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, accs = [], []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x) # Output: (B, num_classes, 1, 1, 1) para I3D con head por defecto
        out = out.view(out.size(0), -1) # Aplanar a (B, num_classes)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        preds = out.argmax(1)
        losses.append(loss.item())
        accs.append((preds == y).float().mean().item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y_true = x.to(device), y.to(device) # Renombrar y para claridad
            out = model(x) # Output: (B, num_classes, 1, 1, 1)
            out = out.view(out.size(0), -1) # Aplanar a (B, num_classes)
            p = out.argmax(1)
            all_preds.extend(p.cpu().tolist()) # Usar extend para listas
            all_labels.extend(y_true.cpu().tolist())
            
    return accuracy_score(all_labels, all_preds), \
           precision_score(all_labels, all_preds, zero_division=0), \
           recall_score(all_labels, all_preds, zero_division=0), \
           f1_score(all_labels, all_preds, zero_division=0)


def measure_inference_fps(model, device, clip_size=NUM_FRAMES, trials=TRIALS_FPS):
    dummy = torch.randn(1, 3, clip_size, IMG_SIZE, IMG_SIZE, device=device)
    model.eval() # Asegurar modo evaluación
    # warm-up
    for _ in range(10):
        _ = model(dummy)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(trials):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.time() - start
    return trials / total_time


def save_metrics(history, path=METRICS_JSON):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    # Para reproducibilidad (opcional, descomentar si es necesario)
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # if DEVICE.type == 'cuda':
    #     torch.cuda.manual_seed_all(42)
    #     # Puede afectar al rendimiento, usar con cautela
    #     # torch.backends.cudnn.deterministic = True 
    #     # torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # cargar datasets
    train_ds = cargar_dataset('train')
    val_ds   = cargar_dataset('val')
    
    # Ajustar num_workers según tu sistema. 0 si hay problemas con multiprocessing.
    num_data_workers = 2 if os.name == 'posix' else 0 # Ejemplo: 2 para Linux/Mac, 0 para Windows

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_data_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_data_workers, pin_memory=True)

    # cargar modelo
    model = i3d_r50(pretrained=True)
    # Identificar la capa de proyección final en la cabeza del modelo
    # model.blocks[-1] es típicamente ResNetBasicHead
    # model.blocks[-1].proj es la capa de clasificación final dentro de esa cabeza
    final_projection_layer = model.blocks[-1].proj

    if isinstance(final_projection_layer, nn.Conv3d):
        print("Capa final detectada: nn.Conv3d. Reemplazando...")
        original_kernel_size = final_projection_layer.kernel_size
        original_stride = final_projection_layer.stride
        model.blocks[-1].proj = nn.Conv3d(
            in_channels=final_projection_layer.in_channels,
            out_channels=len(CLASSES),
            kernel_size=original_kernel_size, # Mantener kernel original (usualmente (1,1,1))
            stride=original_stride # Mantener stride original (usualmente (1,1,1))
        )
    elif isinstance(final_projection_layer, nn.Linear):
        print("Capa final detectada: nn.Linear. Reemplazando...")
        model.blocks[-1].proj = nn.Linear(
            in_features=final_projection_layer.in_features,
            out_features=len(CLASSES)
        )
    else:
        raise TypeError(
            f"La capa final del modelo (model.blocks[-1].proj) es de un tipo inesperado: {type(final_projection_layer)}"
        )
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []}

    best_val_f1 = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    print(f"Iniciando entrenamiento en {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        tloss, tacc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vacc, vprec, vrec, vf1 = evaluate(model, val_loader, DEVICE)
        duration = time.time() - start_time
        
        history['epoch'].append(epoch)
        history['train_loss'].append(tloss)
        history['train_acc'].append(tacc)
        history['val_acc'].append(vacc)
        history['val_prec'].append(vprec)
        history['val_rec'].append(vrec)
        history['val_f1'].append(vf1)
        
        writer.add_scalar('Loss/train', tloss, epoch)
        writer.add_scalar('Accuracy/train', tacc, epoch)
        writer.add_scalar('Accuracy/val', vacc, epoch)
        writer.add_scalar('Precision/val', vprec, epoch)
        writer.add_scalar('Recall/val', vrec, epoch)
        writer.add_scalar('F1/val', vf1, epoch)
        writer.add_scalar('Time/epoch', duration, epoch)
        
        print(f"Epoch {epoch}/{EPOCHS}: loss={tloss:.4f}, t_acc={tacc:.4f} | v_acc={vacc:.4f}, v_prec={vprec:.4f}, v_rec={vrec:.4f}, v_f1={vf1:.4f} | time={duration:.1f}s")

        if vf1 > best_val_f1:
            best_val_f1 = vf1
            torch.save(model.state_dict(), best_model_path)
            print(f"  Mejor F1 de validación: {best_val_f1:.4f}. Guardando modelo en {best_model_path}")

    save_metrics(history, METRICS_JSON)
    print(f"Entrenamiento completado. Métricas guardadas en {METRICS_JSON}")
    print(f"Mejor modelo guardado en {best_model_path} con F1 de validación: {best_val_f1:.4f}")


    # Cargar el mejor modelo para análisis y Grad-CAM
    if os.path.exists(best_model_path):
        print(f"Cargando el mejor modelo desde {best_model_path} para análisis final.")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("No se encontró el mejor modelo guardado, usando el último modelo para análisis.")

    model.eval() # Asegurar modo evaluación

    # análisis coste
    fps = measure_inference_fps(model, DEVICE)
    params = parameter_count(model)['']
    dummy_input_flops = torch.randn(1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE, device=DEVICE)
    flops = FlopCountAnalysis(model, dummy_input_flops).total() / 1e9 # GFLOPs
    
    print(f"Inference FPS: {fps:.1f}, Params: {params:,}, FLOPs: {flops:.2f}G")
    writer.add_scalar('Inference/FPS', fps)
    writer.add_scalar('Cost/Params_M', params / 1e6) # Parámetros en Millones
    writer.add_scalar('Cost/FLOPs_G', flops)
    writer.close()

    

if __name__ == '__main__':
    main()