import torch
import os


# ----- CONFIGURACION GLOBAL DEL PROYECTO -----

# --- Reproducibilidad ---
# Semilla para reproducibilidad y experimentos consistentes
RANDOM_SEED = 23
# FULL_REPRODUCIBLE en True significa algoritmos deterministas para reprod. bit a bit
# Esto puede ralentizar significativamente el entrenamiento!
FULL_REPRODUCIBLE = False # REVISAR
USE_AMP = True # REVISAR
# --- Clases Globales ---
# Mapeo unificado de etiquetas para todos los datasets y facilita consistencia
CLASSES = {"Violence": 1, "NonViolence": 0}

# --- Rutas y Directorios ---
# Uso de rutas relativas, que se basan desde la misma ruta que main.py (donde se ejecuta el proyecto)
BASE_DATA_DIR = "assets"
FILE_LIST_DIR = "dataset_file_lists"
# La salida se va a generar dinamicante en main.py para cada modelo
# Ejemplo: "results/vivit_output_seed_23/..." 
OUTPUT_DIR_BASE = "results"

# --- Configuracion del Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
# Los Data workers permiten multiprocesamiento en los DataLoaders, paralelo al proceso principal
NUM_DATA_WORKERS = 2 if os.name == 'posix' else 0

# ------------------------------------------- #

# ----- HIPERPARAMETROS GLOBALES DE ENTRENAMIENTO -----

# Epocas, Tamano de lote modificables en parametros especificos
EPOCHS = 10
BATCH_SIZE = 2 #Pequeno por grafica acotada
# Parametros de Aumento de Datos y Procesamiento
IMG_CROP_SIZE = 224
IMG_RESIZE_DIM = 256

# Mean y std utilizado en Kinetics
KINETICS_MEAN = [0.485, 0.456, 0.406]
KINETICS_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------- #

# Intentos para calculo de velocidad de inferencia en FPS:
TRIALS_FPS = 50

# ----- CONFIGURACION ESPECIFICA POR MODELO -----

# --- Parametros especificos de I3D ---
I3D_PARAMS = {
    "num_frames": 32,
    "frame_step": 4,
    "lr": 1e-5,
    "weight_decay": 2e-5,
    "image_size": IMG_CROP_SIZE,     
    "resize_dim": IMG_RESIZE_DIM, 
    "epochs": EPOCHS, #Aqui se puede variar epocas y batch size de cada modelo por separado si se desea
    "batch_size": BATCH_SIZE,
    "mean": KINETICS_MEAN,
    "std": KINETICS_STD,
}

# --- Parametros especificos de SlowFast ---
SLOWFAST_PARAMS = {
    "num_frames": 32,
    "frame_step": 4,
    "lr": 1e-5,
    "weight_decay": 2e-5,
    "image_size": IMG_CROP_SIZE,     
    "resize_dim": IMG_RESIZE_DIM, 
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "alpha": 4,
    "mean": [0.45, 0.45, 0.45],     #Valores recomendados segun ejemplos de Pytorchvideo
    "std": [0.225, 0.225, 0.225],   #Valores recomendados segun ejemplos de Pytorchvideo
}

# --- Parametros especificos de TSM ---
TSM_PARAMS = {
    "num_frames": 8, # TSM es eficiente y usa menos frames 
    "frame_step": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "image_size": IMG_CROP_SIZE,     
    "resize_dim": IMG_RESIZE_DIM,     
    "pretrained_checkpoint_path": "pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "mean": KINETICS_MEAN,
    "std": KINETICS_STD,
}

# --- Parametros especificos de ViViT ---
VIVIT_PARAMS = {
    "num_frames": 32,
    "frame_step": 4,
    "lr": 2e-5,
    "weight_decay": 1e-2,
    "image_size": IMG_CROP_SIZE,     
    "resize_dim": IMG_RESIZE_DIM, 
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "mean": KINETICS_MEAN,
    "std": KINETICS_STD,
    "model_name": "google/vivit-b-16x2-kinetics400", # Modelo base de Hugging Face
    # Habilitar Gradient Checkpointing para reducir el uso de memoria
    # a costa de un pequeño aumento en el tiempo de computacion. Esencial para ViViT.
    "use_gradient_checkpointing": True,
}

# --------------------------------------------- #