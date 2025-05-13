import os
import cv2
import torch
from tqdm import tqdm
from transformers import VivitImageProcessor, VivitForVideoClassification
import numpy as np

# ----- MACROS Y CONFIGURACIÓN -----
BASE_PATH = "TFM/assets"  # Ruta base de tus carpetas
SUBFOLDERS = {
    "Violence": 1,
    "NonViolence": 0,
}
MODEL_NAME = "google/vivit-b-16x2-kinetics400"  # Modelo HuggingFace
NUM_FRAMES = 32    # Número de frames a extraer por vídeo
FRAME_STEP = 4     # Saltos entre frames
USE_INFERENCE = True  # Si quieres ejecutar inferencia al final

# Carga el procesador y el modelo
processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
model = VivitForVideoClassification.from_pretrained(MODEL_NAME)

# Función para extraer y preprocesar frames usando OpenCV
def process_video_opencv(video_path, num_frames=NUM_FRAMES, frame_step=FRAME_STEP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR al abrir vídeo: {video_path}")
        return None
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while len(frames) < num_frames and idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        idx += frame_step
    cap.release()
    if not frames:
        print(f"ADVERTENCIA: ningún frame extraído de {video_path}")
        return None
    # Rellenar con el último frame si faltan
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    frames = frames[:num_frames]
    inputs = processor(frames, return_tensors="pt")
    return inputs["pixel_values"][0]

# Listas para datos
pixel_values_list = []
labels_list = []

# Procesamiento con tqdm para mostrar progreso
for folder_name, label in tqdm(SUBFOLDERS.items(), desc="Folders", unit="folder"):
    folder_path = os.path.join(BASE_PATH, folder_name)
    if not os.path.isdir(folder_path):
        print(f"ADVERTENCIA: carpeta no encontrada: {folder_path}")
        continue
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
    for file in tqdm(files, desc=f"Procesando {folder_name}", unit="video", leave=False):
        video_path = os.path.join(folder_path, file)
        pv = process_video_opencv(video_path)
        if pv is not None:
            pixel_values_list.append(pv)
            labels_list.append(label)

# Verificación
if not pixel_values_list:
    print("ERROR: no se procesó ningún vídeo. Verifica rutas/extensiones.")
    exit(1)

# Apilar tensores
pixel_values = torch.stack(pixel_values_list)  # (N, C, T, H, W)
labels = torch.tensor(labels_list)              # (N,)
print(f"Procesados {pixel_values.shape[0]} vídeos => tensor {pixel_values.shape}")

# Inferencia opcional
if USE_INFERENCE:
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        preds = outputs.logits.argmax(-1)
    print("Predicciones:", preds.tolist())
    print("Etiquetas reales:", labels.tolist())