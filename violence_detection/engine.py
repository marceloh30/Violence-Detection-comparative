import logging
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.amp import autocast

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, model_name, epoch_num, total_epochs):
    """
    Ejecuta una única época de entrenamiento para un modelo dado.

    Args:
        model: El modelo de PyTorch a entrenar.
        loader: El DataLoader para los datos de entrenamiento.
        criterion: La función de pérdida.
        optimizer: El optimizador.
        scaler: GradScaler para la precisión mixta.
        device: El dispositivo (CPU o CUDA).
        use_amp (bool): Si es True, utiliza precisión mixta automática.
        model_name (str): Nombre del modelo ('i3d', 'slowfast', 'tsm', 'vivit') para manejar llamadas específicas.
        epoch_num (int): Número de la época actual.
        total_epochs (int): Número total de épocas.

    Returns:
        tuple: Una tupla conteniendo (pérdida_promedio_epoca, exactitud_promedio_epoca).
    """
    model.train()
    running_loss, running_corrects, total_valid_samples = 0.0, 0, 0
    
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch_num}/{total_epochs}", leave=False)

    for inputs, labels in progress_bar:
        # Filtrar muestras que fallaron al cargar (etiqueta -1)
        valid_indices = labels != -1
        if not valid_indices.any():
            continue

        labels = labels[valid_indices].to(device, non_blocking=True)
        
        # Mover datos de entrada al dispositivo
        if isinstance(inputs, list): # Caso para SlowFast
            inputs = [tensor[valid_indices].to(device, non_blocking=True) for tensor in inputs]
        else:
            inputs = inputs[valid_indices].to(device, non_blocking=True)
        
        if (isinstance(inputs, list) and inputs[0].size(0) == 0) or (not isinstance(inputs, list) and inputs.size(0) == 0):
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            # Llamada al modelo adaptada para ViViT
            if model_name == 'vivit':
                outputs = model(pixel_values=inputs).logits
            else: # I3D, SlowFast, TSM aceptan el tensor o lista de tensores directamente
                outputs = model(inputs)
            
            # Algunos modelos como I3D pueden necesitar un aplanamiento final
            if outputs.ndim > 2:
                outputs = outputs.view(outputs.size(0), -1)

            loss = criterion(outputs, labels)
        
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        _, preds = torch.max(outputs, 1)
        current_valid_samples = labels.size(0)
        running_loss += loss.item() * current_valid_samples
        running_corrects += torch.sum(preds == labels.data)
        total_valid_samples += current_valid_samples
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(torch.sum(preds == labels.data).item() / current_valid_samples):.4f}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    epoch_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    epoch_acc = (running_corrects.double() / total_valid_samples if total_valid_samples > 0 else 0.0).item()
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, use_amp, model_name, num_classes, pos_label_value):
    """
    Evalúa el rendimiento del modelo en un conjunto de datos (validación o prueba).

    Args:
        model: El modelo de PyTorch a entrenar.
        loader: El DataLoader para los datos de entrenamiento.
        criterion: La función de pérdida.
        device: El dispositivo (CPU o CUDA).
        use_amp (bool): Si es True, utiliza precisión mixta automática.
        num_classes (int): Número de clases para el cálculo de métricas.
        pos_label_value (int): El valor de la etiqueta positiva (ej. 1 para 'Violence').

    Returns:
        tuple: Una tupla con todas las métricas (loss, acc, precision, recall, f1, confusion_matrix).
    """
    model.eval()
    running_loss, total_valid_samples = 0.0, 0
    all_preds_list, all_labels_list = [], []
    
    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            valid_indices = labels != -1
            if not valid_indices.any():
                continue

            labels_true = labels[valid_indices].to(device, non_blocking=True)
            
            if isinstance(inputs, list):
                inputs = [tensor[valid_indices].to(device, non_blocking=True) for tensor in inputs]
            else:
                inputs = inputs[valid_indices].to(device, non_blocking=True)

            if (isinstance(inputs, list) and inputs[0].size(0) == 0) or (not isinstance(inputs, list) and inputs.size(0) == 0):
                continue

            with autocast(device_type=device.type, enabled=use_amp):
                if model_name == 'vivit':
                    outputs = model(pixel_values=inputs).logits
                else:
                    outputs = model(inputs)
                
                if outputs.ndim > 2:
                    outputs = outputs.view(outputs.size(0), -1)

                loss = criterion(outputs, labels_true)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * labels_true.size(0)
            total_valid_samples += labels_true.size(0)
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels_true.cpu().numpy())
            
    if total_valid_samples == 0:
        logging.warning("No se procesaron muestras válidas durante la evaluación.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    # Calcular todas las métricas al final del bucle
    loss = running_loss / total_valid_samples
    acc = accuracy_score(all_labels_list, all_preds_list)
    
    avg_method = 'binary' if num_classes == 2 else 'macro'
    
    precision = precision_score(all_labels_list, all_preds_list, average=avg_method, pos_label=pos_label_value if avg_method=='binary' else None, zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average=avg_method, pos_label=pos_label_value if avg_method=='binary' else None, zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average=avg_method, pos_label=pos_label_value if avg_method=='binary' else None, zero_division=0)
    cm = confusion_matrix(all_labels_list, all_preds_list, labels=list(range(num_classes))).tolist()
    
    return loss, acc, precision, recall, f1, cm