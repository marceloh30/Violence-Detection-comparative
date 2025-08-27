import logging
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VivitImageProcessor


# ----- LÓGICA DE PROCESAMIENTO DE VIDEO POR MODELO -----
# Cada una de estas funciones toma la ruta de un video y devuelve un tensor
# listo para ser consumido por su respectivo modelo.

def process_video_shared(path: str, num_frames: int, frame_step: int, resize_dim: int, crop_size: int, is_train: bool):
    """
    Función base compartida para leer y muestrear fotogramas de un video.
    Aplica redimensionamiento, recorte (aleatorio o central) y volteo horizontal.
    Devuelve una lista de fotogramas procesados como arrays de NumPy.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.warning(f"Error al abrir el vídeo: {path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logging.warning(f"Vídeo inválido o sin fotogramas: {path}")
        return None

    # Lógica de muestreo de fotogramas (común a todos los modelos)
    selectable_frames_range = total_frames - (num_frames - 1) * frame_step
    if selectable_frames_range > 0:
        start_idx = random.randint(0, selectable_frames_range - 1) if is_train else selectable_frames_range // 2
        frame_indices = [start_idx + i * frame_step for i in range(num_frames)]
    else: # Fallback para videos cortos
        available_indices = list(range(total_frames))
        frame_indices = np.linspace(0, len(available_indices) - 1, num=num_frames, dtype=int)

    processed_frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"No se pudo leer el fotograma {frame_idx} de {path}. Usando fotograma negro.")
            frame = np.zeros((resize_dim, resize_dim, 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize_dim, resize_dim))

        # Aumento de datos: volteo y recorte
        if is_train:
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1) # Volteo horizontal
            top = random.randint(0, resize_dim - crop_size)
            left = random.randint(0, resize_dim - crop_size)
        else: # Recorte central para validación
            top = (resize_dim - crop_size) // 2
            left = (resize_dim - crop_size) // 2
        
        frame = frame[top:top + crop_size, left:left + crop_size, :]
        processed_frames.append(frame)
    
    cap.release()
    return processed_frames


# ----- CLASES DE DATASET DE PYTORCH -----
# Cada clase utiliza la lógica de procesamiento anterior y la envuelve en
# un objeto Dataset de PyTorch, que puede ser utilizado por un DataLoader.

class VideoDataset(Dataset):
    """
    Clase de Dataset genérica que se adapta al modelo mediante la función de
    transformación y los argumentos que se le pasen.
    """
    def __init__(self, file_list_data, model_params, is_train, model_name):
        self.file_list_data = file_list_data
        self.is_train = is_train
        self.params = model_params
        self.model_name = model_name

        if self.model_name == 'vivit':
            self.vivit_processor = VivitImageProcessor.from_pretrained(self.params['model_name'])
            self.vivit_processor.do_resize = False
            self.vivit_processor.do_center_crop = False

    def __len__(self):
        return len(self.file_list_data)

    def __getitem__(self, idx):
        item_data = self.file_list_data[idx]
        video_path, label = item_data['path'], item_data['label']

        # 1. Procesar video para obtener lista de frames (NumPy)
        frames = process_video_shared(
            path=video_path,
            num_frames=self.params['num_frames'],
            frame_step=self.params['frame_step'],
            resize_dim=self.params['resize_dim'],
            crop_size=self.params['image_size'],
            is_train=self.is_train
        )

        if frames is None:
            logging.warning(f"Fallo al procesar vídeo {video_path}. Devolviendo dummy.")
            # Devolver un tensor dummy con la forma correcta y una etiqueta inválida
            dummy_shape = self._get_dummy_shape()
            return torch.zeros(dummy_shape), torch.tensor(-1, dtype=torch.long)
        
        # 2. Convertir a Tensor y Normalizar según el modelo
        clip_tensor = self._transform_frames(frames)
        
        return clip_tensor, torch.tensor(label, dtype=torch.long)

    def _get_dummy_shape(self):
        """Devuelve la forma del tensor esperada por cada modelo para un dummy."""
        c, t, h, w = 3, self.params['num_frames'], self.params['image_size'], self.params['image_size']
        
        if self.model_name == 'slowfast':
            t_slow = t // self.params['alpha']
            return [(c, t_slow, h, w), (c, t, h, w)] # Devuelve una tupla de formas
        else:
            return (c, t, h, w)

    def _transform_frames(self, frames):
        """Aplica la transformación y normalización final específica de cada modelo."""
        # Convertir a tensor flotante y permutar a (T, C, H, W)
        clip_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
        clip_tensor = clip_tensor.permute(0, 3, 1, 2)
        
        # Normalizar
        mean = torch.tensor(self.params['mean']).view(3, 1, 1)
        std = torch.tensor(self.params['std']).view(3, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        # Permutar a la forma final que espera el modelo (C, T, H, W)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)

        # Casos especiales
        if self.model_name == 'slowfast': # Se hace parte de lo que originalmente estaba en PackPathwayCustom
            slow_pathway = torch.index_select(
                clip_tensor, 1,
                torch.linspace(0, clip_tensor.shape[1] - 1, clip_tensor.shape[1] // self.params['alpha']).long()
            )
            return [slow_pathway, clip_tensor]
        
        if self.model_name == 'vivit':
            # ViViT espera (T, C, H, W), así que revertimos la última permutación
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)
            # El procesador de Hugging Face se encarga de la normalización final
            inputs = self.vivit_processor(images=list(clip_tensor), return_tensors="pt")
            return inputs["pixel_values"]
            
        return clip_tensor