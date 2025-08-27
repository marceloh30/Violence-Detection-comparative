import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50
import logging

# ----- I3D: CARGA DE MODELO -----
def load_model(num_classes: int, pretrained: bool = True):
    """
    Carga un modelo I3D ResNet-50 y adapta su cabezal de clasificación.
    """
    logging.info(f"Cargando modelo I3D_R50 (pretrained={pretrained}) para {num_classes} clases.")
    model = i3d_r50(pretrained=pretrained)
    
    # Reemplazar la capa de clasificación final
    final_projection_layer = model.blocks[-1].proj
    if isinstance(final_projection_layer, nn.Conv3d):
        model.blocks[-1].proj = nn.Conv3d(
            in_channels=final_projection_layer.in_channels,
            out_channels=num_classes,
            kernel_size=final_projection_layer.kernel_size,
            stride=final_projection_layer.stride
        )
    else: # Por si es una capa Linear
        model.blocks[-1].proj = nn.Linear(
            in_features=final_projection_layer.in_features,
            out_features=num_classes
        )
        
    return model