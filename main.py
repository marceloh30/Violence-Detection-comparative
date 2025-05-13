import os
import av
import torch
from transformers import VivitImageProcessor, VivitForVideoClassification

# Ruta base donde están las carpetas "Violence" y "NonViolence"
BASE_PATH = "ruta/a/tu/dataset"
SUBFOLDERS = {
    "Violence": 1,
    "NonViolence": 0
}

# Cargar el procesador y el modelo (necesario si vas a realizar inferencia)
processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

# Función para extraer frames de un vídeo
def extract_frames(video_path, num_frames=32, frame_step=4):
    container = av.open(video_path)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_step == 0 and len(frames) < num_frames:
            arr = frame.to_ndarray(format="rgb24")  # ndarray HxWx3
            frames.append(arr)
    container.close()
    return frames

# Listas para almacenar datos
pixel_values_list = []
labels_list = []

# Recorremos ambas carpetas y procesamos cada vídeo
for folder_name, label in SUBFOLDERS.items():
    folder_path = os.path.join(BASE_PATH, folder_name)
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, file_name)
        # Extraer y preprocesar frames
        frames = extract_frames(video_path)
        inputs = processor(frames, return_tensors="pt")
        # Guardar el tensor resultante y la etiqueta
        pixel_values_list.append(inputs["pixel_values"][0])  # tensor 1xCxTxH xW
        labels_list.append(label)

# Convertir listas a tensores PyTorch para entrenar/inferir
pixel_values = torch.stack(pixel_values_list)  # shape: (N, C, T, H, W)
labels = torch.tensor(labels_list)             # shape: (N,)

# Ejemplo de inferencia (opcional)
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    preds = outputs.logits.argmax(-1)
    print("Predicciones:", preds)
    print("Etiquetas reales:", labels)
