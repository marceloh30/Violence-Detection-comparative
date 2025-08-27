import logging
import torch.nn as nn
from transformers import VivitConfig, VivitForVideoClassification


# ----- VIVIT: CARGA DE MODELO -----
def load_model(num_classes: int, model_name: str, image_size: int, use_gradient_checkpointing: bool):
    """
    Carga un modelo ViViT desde Hugging Face, adapta su cabezal de clasificación
    y opcionalmente habilita gradient checkpointing para ahorrar memoria.

    Args:
        num_classes (int): El número de clases de salida para el clasificador.
        model_name (str): El identificador del modelo preentrenado en Hugging Face (ej. 'google/vivit-b-16x2-kinetics400').
        image_size (int): El tamaño (alto y ancho) de los frames de entrada para el modelo.
        use_gradient_checkpointing (bool): Si es True, habilita gradient checkpointing en el modelo base.

    Returns:
        VivitForVideoClassification: El modelo ViViT listo para el ajuste fino.
    """
    logging.info(f"Cargando modelo ViViT '{model_name}' para {num_classes} clases, image_size={image_size}.")

    # 1. Cargar la configuración del modelo preentrenado y adaptarla a nuestra tarea
    config = VivitConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        image_size=image_size,
        ignore_mismatched_sizes=True  # Permite cambiar el tamaño del cabezal de clasificación
    )

    # 2. Cargar el modelo base preentrenado para transferir sus pesos
    try:
        # Cargamos el modelo original con su cabezal de Kinetics para obtener los pesos de la base.
        base_vivit_model = VivitForVideoClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        
        # Creamos una nueva instancia del modelo con nuestra configuración (cabezal correcto para 2 clases).
        model = VivitForVideoClassification(config)
        
        # Transferimos los pesos aprendidos de la base (el "cuerpo" del Transformer) al nuevo modelo.
        # El cabezal de clasificación se quedará con una inicialización aleatoria, listo para ser entrenado.
        model.vivit.load_state_dict(base_vivit_model.vivit.state_dict())
        logging.info(f"Pesos de la base ViViT transferidos desde '{model_name}'.")

    except Exception as e:
        logging.warning(f"No se pudieron cargar los pesos preentrenados para {model_name} debido a: {e}. "
                        "Inicializando el modelo desde cero con la nueva configuración.")
        model = VivitForVideoClassification(config)

    # 3. Habilitar Gradient Checkpointing si está activado en la configuración
    if use_gradient_checkpointing:
        try:
            model.vivit.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing habilitado para ViViT.")
        except Exception as e:
            logging.warning(f"No se pudo habilitar gradient checkpointing para ViViT: {e}")
            
    return model

# Wrapper para el calculo de FLOPs con fvcore
class VivitModelWrapperForFlops(nn.Module):
    """
    Un wrapper para el modelo ViViT que asegura que la salida sea un tensor,
    facilitando el análisis de FLOPs con fvcore, que no maneja bien las salidas
    en formato de diccionario de Hugging Face.
    """
    def __init__(self, vivit_model_instance):
        super().__init__()
        self.vivit_model_instance = vivit_model_instance

    def forward(self, pixel_values):
        # El modelo ViViT devuelve un diccionario. Extraemos los 'logits' para la compatibilidad.
        return self.vivit_model_instance(pixel_values=pixel_values).logits
