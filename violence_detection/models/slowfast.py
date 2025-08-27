import torch
import torch.nn as nn
import logging

# ----- SLOWFAST: CARGA DE MODELO -----
def load_model(num_model_classes, pretrained=True):
    logging.info(f"Cargando modelo SlowFast_R50 (pretrained={pretrained}) para {num_model_classes} clases.")
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        original_head_in_features = model.blocks[6].proj.in_features
        model.blocks[6].proj = nn.Linear(original_head_in_features, num_model_classes)
    except Exception as e:
        logging.error(f"Error al cargar el modelo SlowFast desde torch.hub: {e}"); raise
    return model