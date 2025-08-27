import torch
import torch.nn as nn
import torchvision.models as models
import logging
import os

# ----- TSM MÓDULO Y MODELO -----
class TemporalShift(nn.Module):
    def __init__(self, n_segment, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            logging.debug('=> Using in-place shift...')

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        # No es necesario .contiguous() aquí si la siguiente vista lo maneja
        return out.view(nt, c, h, w)

# Funcion para el desplazamiento temporal
def make_temporal_shift(net, n_segment, n_div=8, places='blockres', temporal_pool=False):
    if temporal_pool: # No se usa en la config actual, pero se mantiene por completitud
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    if places == 'blockres':
        def make_block_temporal(stage, this_n_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if isinstance(b, models.resnet.Bottleneck):
                    # Guardar conv1 original
                    conv1_original = b.conv1
                    # Crear nuevo conv1 con TemporalShift
                    # Asegurar que inplace=False si no se maneja cuidadosamente la memoria
                    b.conv1 = nn.Sequential(TemporalShift(this_n_segment, n_div, inplace=False), conv1_original)
            return nn.Sequential(*blocks)
        
        # Aplicar a cada capa de ResNet
        for i in range(1, 5): # layer1, layer2, layer3, layer4
            layer_name = f'layer{i}'
            original_layer = getattr(net, layer_name)
            shifted_layer = make_block_temporal(original_layer, n_segment_list[i-1])
            setattr(net, layer_name, shifted_layer)
    else:
        raise NotImplementedError(f"Unsupported places: {places}")

# Clase de TSM con backbone ResNet50
class TSM_ResNet50(nn.Module):
    def __init__(self, num_classes, n_segment):
        super(TSM_ResNet50, self).__init__()
        self.n_segment = n_segment
        self.num_classes = num_classes

        logging.info(f"Initializing TSM with ResNet50 backbone, num_segments={n_segment}")
        base_model = models.resnet50(weights=None) # Cargar sin pesos de ImageNet por defecto
        
        # Aplicar TSM a las capas de ResNet
        make_temporal_shift(base_model, n_segment)

        # Quitar la capa FC original de ResNet
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # Usar avgpool adaptativo
        
        num_fc_inputs = base_model.fc.in_features
        self.fc = nn.Linear(num_fc_inputs, num_classes)

    def forward(self, x): # x shape: (N, C, T, H, W)
        # Permutar a (N, T, C, H, W) y luego a (N*T, C, H, W) para ResNet2D
        n, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous() # N, T, C, H, W
        x = x.view(n * t, c, h, w) # N*T, C, H, W
        
        out = self.features(x)
        out = self.avgpool(out) # N*T, num_features, 1, 1
        out = out.view(n, t, -1) # N, T, num_features
        out = out.mean(dim=1) # N, num_features (promedio sobre la dimensión temporal)
        out = self.fc(out) # N, num_classes
        return out

# --- Función Principal para Cargar el Modelo ---
def load_model(num_model_classes, n_segment_model, checkpoint_path=None):
    # Inicializo estructura del modelo
    model = TSM_ResNet50(num_classes=num_model_classes, n_segment=n_segment_model)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint '{checkpoint_path}' no encontrado o no proporcionado. "
                        "El modelo se inicializará desde cero (o ResNet50 sin pesos).")
        return model

    logging.info(f"Cargando checkpoint desde: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Obtengo el state_dict, manejando posibles anidaciones
        if 'state_dict' in checkpoint:
            source_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            source_state_dict = checkpoint
        else:
            logging.error("El checkpoint no es un diccionario o no contiene 'state_dict'.")
            return model
            
        remapped_state_dict = {}
        model_state_keys = set(model.state_dict().keys()) # Keys esperadas por el modelo

        for src_key, src_value in source_state_dict.items():
            trg_key = src_key

            # Quito prefijo "module." si esta presente
            if trg_key.startswith('module.'):
                trg_key = trg_key[len('module.'):]

            # --- Remapping a estructura TSM_ResNet50 ---
            
            # 1. Se remapea backbone keys desde 'base_model.LAYER.BLOCK.PART' a 'features.INDEX.BLOCK.PART'
            if trg_key.startswith('base_model.'):
                temp_key = trg_key[len('base_model.'):]
                
                # Capas Top-level ResNet pasan a indices 'features':
                # conv1 -> features[0]
                # bn1 -> features[1]
                # layer1 -> features[4]
                # layer2 -> features[5]
                # layer3 -> features[6]
                # layer4 -> features[7]
                
                if temp_key.startswith('conv1.'):
                    trg_key = 'features.0.' + temp_key[len('conv1.'):]
                elif temp_key.startswith('bn1.'):
                    trg_key = 'features.1.' + temp_key[len('bn1.'):]
                elif temp_key.startswith('layer1.'):
                    trg_key = 'features.4.' + temp_key[len('layer1.'):]
                elif temp_key.startswith('layer2.'):
                    trg_key = 'features.5.' + temp_key[len('layer2.'):]
                elif temp_key.startswith('layer3.'):
                    trg_key = 'features.6.' + temp_key[len('layer3.'):]
                elif temp_key.startswith('layer4.'):
                    trg_key = 'features.7.' + temp_key[len('layer4.'):]
                else:
                    # Si no es uno de los de arribna, puede ser una key inesperada del base_model
                    # o una key que no se tiene que remapear (como la capa fc si fuese base_model.fc)
                    pass # Mantengo temp_key como es de mientras, mayor procesamiento puede aplicar

            # 2. Manejo el '.net.' de la primer conv en Bottlenecks y lo mapeo a '.1'
            # ej: 'features.4.0.conv1.net.weight' (luego del paso 1) -> 'features.4.0.conv1.1.weight'
            # Se asume que 'make_temporal_shift' pone la conv original al indice 1 de una nn.Sequential
            parts = trg_key.split('.')
            # Busco patrones al estilo de 'features.INDEX.BLOCK_IDX.conv1.net.PARAMS'
            if len(parts) > 4 and parts[-2] == 'net' and parts[-3].startswith('conv1'):
                 # Busco si modulo TemporalShift es libre de parametros, si es asi, original conv tiene el indice 1
                if not any(p.requires_grad for p in model.features[int(parts[1])][int(parts[2])].conv1[0].parameters()):
                    parts[-2] = '1' # Cambio 'net' por '1'
                    trg_key = '.'.join(parts)
                else:
                    logging.warning(f"TemporalShift en {'.'.join(parts[:-2])} podría tener parámetros. "
                                    "La reasignación de '.net' a '.1' podría ser incorrecta.")
            
            # 3. Manejo de capa de clasificacion: paso del 'new_fc' (checkpoint Kinetics) a el 'fc' con las clases que se precisan
            if trg_key.startswith('new_fc.'):
                if src_value.shape[0] != num_model_classes:
                    logging.info(f"Omitiendo capa FC del checkpoint '{src_key}' debido a la diferencia de clases "
                                 f"(checkpoint: {src_value.shape[0]}, modelo: {num_model_classes}). "
                                 "La capa FC del modelo se entrenará desde cero.")
                    continue # La salto y no la agrego a 'remapped_state_dict'
                else:
                    # Si las clases coincidieran (algo que no ocurre en este proyecto), lo mapeariamos
                    trg_key = trg_key.replace('new_fc.', 'fc.')
            
            # 4. Salto la carga de 'num_batches_tracked' para las capas de BatchNorm
            if 'num_batches_tracked' in trg_key:
                continue

            # Se agrega solamente si la target key existe en el modelo actual
            if trg_key in model_state_keys:
                remapped_state_dict[trg_key] = src_value
            elif src_key == trg_key : # Si no hubo remapping para esta key
                 if not (trg_key.startswith("fc.") and src_key.startswith("new_fc.")): # Permito que fc falte si se salta new_fc
                    logging.debug(f"Clave del checkpoint '{src_key}' no mapeada y no encontrada en el modelo.")
            # else: # Ocurrio remapping, pero la target key sigue sin estar en el modelo (seria muy raro)
                # logging.debug(f"Clave del checkpoint '{src_key}' remapeada a '{trg_key}', pero no encontrada en el modelo.")


        # Cargo el state dict remapeado (solamente pesos cargados para capas que existan y coincidan)
        msg = model.load_state_dict(remapped_state_dict, strict=False)
        logging.info(f"Mensaje de carga de state_dict (después del remapeo detallado): {msg}")

        if msg.missing_keys:
            # Missing keys esperadas: fc.weight, fc.bias si fueran saltadas intencionalmente.
            # Tambien si TemporalShift tiene parametros aprendibles y no se mapearon.
            # En este caso TemporalShift no tiene.
            is_fc_missing = all(k.startswith('fc.') for k in msg.missing_keys)
            if not (len(msg.missing_keys) <= 2 and is_fc_missing): # Permito hasta 2 fc keys faltantes
                 logging.warning(f"Claves FALTANTES (después del remapeo detallado) no esperadas o adicionales: {msg.missing_keys}")
        if msg.unexpected_keys:
            # Deberia estar vacio si hicimos el remapeo perfecto.
            logging.warning(f"Claves INESPERADAS (después del remapeo detallado): {msg.unexpected_keys}")
        
        num_loaded_layers = len(remapped_state_dict)
        total_model_layers = len(model.state_dict())
        logging.info(f"Se cargaron {num_loaded_layers} tensores de parámetros en el modelo (de {total_model_layers} capas totales en el modelo).")
        if num_loaded_layers < total_model_layers / 2 and num_loaded_layers > 0 : # Heuristica
            logging.warning("Parece que se cargó menos de la mitad de las capas del modelo. Verifica el remapeo.")
        elif num_loaded_layers == 0 and len(source_state_dict) > 0:
            logging.error("No se cargó NINGUNA capa. El remapeo de claves falló por completo.")


    except Exception as e:
        logging.error(f"Error EXCEPCIONAL cargando y remapeando el checkpoint desde {checkpoint_path}: {e}", exc_info=True)
        logging.warning("El modelo continuará con inicialización aleatoria.")
        
    return model
