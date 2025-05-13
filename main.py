import os
import sys
from pathlib import Path
import cv2
import torch
from transformers import VivitImageProcessor, VivitForVideoClassification

# ----- CONFIGURACIÓN -----
# Establece aquí la ruta base donde están las carpetas 'Violence' y 'NonViolence'
BASE_PATH = Path(r"./assets")  # <--- AJUSTA esta ruta
SUBFOLDERS = {
    "Violence": 1,
    "NonViolence": 0,
}

# Comprueba que la ruta base existe
if not BASE_PATH.exists():
    print(f"ERROR: No existe la ruta base: {BASE_PATH}")
    print("Asegúrate de que la variable BASE_PATH apunte al directorio correcto.")
    sys.exit(1)

# Carga el procesador y el modelo (solo si vas a hacer inferencia)
processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

# Función para extraer y preprocesar frames de un vídeo usando OpenCV
def process_video_opencv(video_path, num_frames=32, frame_step=4):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR al abrir vídeo {video_path}")
        return None
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while len(frames) < num_frames and idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir BGR (OpenCV) a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        idx += frame_step
    cap.release()
    if not frames:
        print(f"ADVERTENCIA: No se extrajo ningún frame de {video_path}")
        return None
    # Preprocesa con VivitImageProcessor
    inputs = processor(frames, return_tensors="pt")
    return inputs["pixel_values"][0]

# Listas para datos
pixel_values_list = []
labels_list = []

# Recorre cada subcarpeta y procesa vídeos
for folder_name, label in SUBFOLDERS.items():
    folder_path = BASE_PATH / folder_name
    if not folder_path.is_dir():
        print(f"ADVERTENCIA: No existe la carpeta esperada: {folder_path}")
        continue
    for video_file in folder_path.glob("*.mp4"):
        pv = process_video_opencv(video_file)
        if pv is None:
            continue
        pixel_values_list.append(pv)
        labels_list.append(label)

if not pixel_values_list:
    print("ERROR: No se procesó ningún vídeo. Verifica tus rutas y extensiones.")
    sys.exit(1)

# Apilar tensores para entrenamiento/inferencia
pixel_values = torch.stack(pixel_values_list)  # shape: (N, C, T, H, W)
labels = torch.tensor(labels_list)             # shape: (N,)

print(f"Procesados {len(pixel_values)} vídeos (tensor {pixel_values.shape})")

# Ejemplo de inferencia (opcional)
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    preds = outputs.logits.argmax(-1)
    print("Predicciones:", preds.tolist())
    print("Etiquetas reales:", labels.tolist())