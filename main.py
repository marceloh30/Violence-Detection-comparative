import os
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import numpy as np

# ----- MACROS Y CONFIGURACIÓN -----
BASE_PATH = "TFM/assets"            # Ruta base de tus carpetas
SUBFOLDERS = {"Violence": 1, "NonViolence": 0}
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_FRAMES = 32                  # Número de frames a extraer
FRAME_STEP = 4                   # Saltos entre frames
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
USE_INFERENCE = True            # Desactiva inferencia si vas a entrenar

# ----- CONFIGURAR MODELO Y PROCESADOR -----
# Definimos configuración binaria
NUM_LABELS = 2  # Violence vs NonViolence
config = VivitConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
processor = VivitImageProcessor.from_pretrained(MODEL_NAME)

# Cargamos backbone preentrenado y reclasificador manualmente
temp_model = VivitForVideoClassification.from_pretrained(MODEL_NAME)
model = VivitForVideoClassification(config)
# Transferimos pesos del backbone (vivit) al nuevo modelo
model.vivit.load_state_dict(temp_model.vivit.state_dict())
# La cabeza de clasificación (model.classifier) ya tiene forma (2, hidden)
# y está inicializada aleatoriamente.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device es: ",device)

# Envía el modelo al dispositivo (CPU/GPU)
model.to(device)

# ----- FUNCIONES DE PREPROCESADO -----
def process_video(video_path, num_frames=NUM_FRAMES, frame_step=FRAME_STEP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while len(frames) < num_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        idx += frame_step
    cap.release()
    if not frames: return None
    while len(frames) < num_frames: frames.append(frames[-1].copy())
    frames = frames[:num_frames]
    return processor(frames, return_tensors="pt")["pixel_values"][0]

# ----- CARGAR Y PREPROCESAR DATASET -----
pixel_list, label_list = [], []
for folder, lbl in tqdm(SUBFOLDERS.items(), desc="Folders", unit="folder"):
    path = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(path): continue
    files = [f for f in os.listdir(path) if f.endswith('.mp4')]
    for fname in tqdm(files, desc=folder, unit="video", leave=False):
        p = os.path.join(path, fname)
        tensor = process_video(p)
        if tensor is not None:
            pixel_list.append(tensor)
            label_list.append(lbl)

assert pixel_list, "No videos procesados. Verifica rutas."

# Tensores finales
dataset = TensorDataset(torch.stack(pixel_list), torch.tensor(label_list))

# División train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ----- ENTRENAMIENTO -----
optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        outputs = model(pixel_values=x).logits
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    # Validación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(pixel_values=x).logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")

# ----- INFERENCIA FINAL (OPCIONAL) -----
if USE_INFERENCE:
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=torch.stack(pixel_list).to(device)).logits.argmax(-1)
    print("Preds:", outputs.cpu().tolist())
    print("Labels:", label_list)
