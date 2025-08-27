import argparse
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importaciones desde nuestros modulos 
from . import config, utils, datasets, engine
from .models import i3d, slowfast, tsm, vivit


def setup_logging(log_dir, model_name, dataset_name):
    """Configura el logging para guardar en un archivo."""
    log_filename = f"run_{model_name}_on_{dataset_name}.log"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='w'), # Guarda en archivo
            logging.StreamHandler() # También muestra en consola
        ]
    )

def main(args):
    """Punto de entrada principal para el entrenamiento y evaluación."""
    
    # --- CONFIGURACIÓN INICIAL ---
    current_output_dir = os.path.join(
        config.OUTPUT_DIR_BASE, 
        f"{args.model_name}_outputs_seed_{config.RANDOM_SEED}",
        f"trained_on_{args.dataset_name}"
    )
    os.makedirs(current_output_dir, exist_ok=True)
    
    setup_logging(current_output_dir, args.model_name, args.dataset_name)
    utils.set_seed(config.RANDOM_SEED, config.FULL_REPRODUCIBLE)
    g = torch.Generator().manual_seed(config.RANDOM_SEED)

    # --- CARGA DE MODELO Y DATOS ---
    logging.info(f"Modelo seleccionado: {args.model_name.upper()}. Dataset de entrenamiento: {args.dataset_name.upper()}")
    
    if args.model_name == 'i3d':
        model_params = config.I3D_PARAMS
        model = i3d.load_model(num_classes=len(config.CLASSES)).to(config.DEVICE)
    elif args.model_name == 'slowfast':
        model_params = config.SLOWFAST_PARAMS
        model = slowfast.load_model(num_classes=len(config.CLASSES)).to(config.DEVICE)
    elif args.model_name == 'tsm':
        model_params = config.TSM_PARAMS
        model = tsm.load_model(num_classes=len(config.CLASSES), **model_params).to(config.DEVICE)
    elif args.model_name == 'vivit':
        model_params = config.VIVIT_PARAMS
        model = vivit.load_model(
            num_classes=len(config.CLASSES),
            model_name=model_params['model_name'],
            image_size=model_params['image_size'],
            use_gradient_checkpointing=model_params['use_gradient_checkpointing']
        ).to(config.DEVICE)
    else:
        raise ValueError(f"Modelo '{args.model_name}' no soportado.")

    # Cargar listas de archivos
    train_file_list = utils.get_dataset_file_list(args.dataset_name, "train", config.BASE_DATA_DIR, config.FILE_LIST_DIR)
    val_file_list = utils.get_dataset_file_list(args.dataset_name, "val", config.BASE_DATA_DIR, config.FILE_LIST_DIR)

    # Crear Datasets
    train_dataset = datasets.VideoDataset(train_file_list, model_params, is_train=True, model_name=args.model_name)
    val_dataset = datasets.VideoDataset(val_file_list, model_params, is_train=False, model_name=args.model_name)

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_DATA_WORKERS, pin_memory=True, generator=g, worker_init_fn=utils.seed_worker)

    # --- PREPARACIÓN PARA EL ENTRENAMIENTO ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=model_params['lr'], weight_decay=model_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=0)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    # --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
    best_val_f1 = 0.0
    metrics_json_path = os.path.join(current_output_dir, "metrics.json")
    history = {'hyperparameters': model_params, 'epochs_run': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}

    logging.info(f"Iniciando entrenamiento por {config.EPOCHS} épocas...")
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        train_loss, train_acc = engine.train_one_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config.USE_AMP, args.model_name, epoch, config.EPOCHS)
        
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = engine.evaluate(model, val_loader, criterion, config.DEVICE, config.USE_AMP, args.model_name, len(config.CLASSES), config.CLASSES["Violence"])
        
        scheduler.step()

        # Guardar historial
        history['epochs_run'].append(epoch)
        history['train_loss'].append(train_loss)
        # ... (añadir el resto de métricas al historial)
        
        logging.info(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} | Time: {time.time() - epoch_start_time:.2f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(current_output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> Nuevo mejor F1-score: {best_val_f1:.4f}. Checkpoint guardado.")
    
    utils.save_or_update_json(history, metrics_json_path)

    # --- ANÁLISIS FINAL Y CROSS-INFERENCE ---
    logging.info("Entrenamiento completado. Cargando el mejor modelo para análisis final.")
    
    # Cargar el mejor modelo guardado
    model.load_state_dict(torch.load(best_model_path))

    # ANÁLISIS DE RENDIMIENTO CON FUNCIONES SEPARADAS
    static_metrics = utils.calculate_static_metrics(model, args.model_name, model_params, config.DEVICE)
    fps_metric = utils.measure_inference_fps(model, args.model_name, model_params, config.DEVICE, config.TRIALS_FPS)

    # Combinar métricas para guardar
    performance_stats = {
        'performance_stats': {
            'parameters': int(static_metrics['parameters']),
            'gflops': static_metrics['gflops'],
            'fps': fps_metric
        }
    }

    logging.info(f"Rendimiento Final - Params: {performance_stats['performance_stats']['parameters']/1e6:.2f}M, GFLOPs: {performance_stats['performance_stats']['gflops']:.2f}, FPS: {performance_stats['performance_stats']['fps']:.2f}")
    utils.save_or_update_json(performance_stats, metrics_json_path)

    # INFERENCIA CRUZADA
    logging.info("Iniciando inferencia cruzada...")
    datasets_for_cross_inference = ["rwf2000", "rlvs", "hockey"]
    datasets_for_cross_inference.remove(args.dataset_name)

    for inference_ds_name in datasets_for_cross_inference:
        logging.info(f"--- Evaluando en: {inference_ds_name} ---")
        inf_file_list = utils.get_dataset_file_list(inference_ds_name, "all", config.BASE_DATA_DIR, config.FILE_LIST_DIR)
        inf_dataset = datasets.VideoDataset(inf_file_list, model_params, is_train=False, model_name=args.model_name)
        inf_loader = DataLoader(inf_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_DATA_WORKERS, pin_memory=True)
        
        _, inf_acc, _, _, inf_f1, _ = engine.evaluate(model, inf_loader, criterion, config.DEVICE, config.USE_AMP, args.model_name, len(config.CLASSES), config.CLASSES["Violence"])
        
        logging.info(f"  -> Resultados en {inference_ds_name}: Acc: {inf_acc:.4f}, F1: {inf_f1:.4f}")
        
        cross_inf_metrics = {f'cross_inference_on_{inference_ds_name}': {'accuracy': inf_acc, 'f1_score': inf_f1}}
        utils.save_or_update_json(cross_inf_metrics, metrics_json_path)

    logging.info("Proceso completado con éxito.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TFM: Comparativa de Modelos en Detección de Violencia.")
    parser.add_argument(
        'model_name', 
        type=str, 
        choices=['i3d', 'slowfast', 'tsm', 'vivit'],
        help='Nombre del modelo a entrenar y evaluar.'
    )
    parser.add_argument(
        'dataset_name', 
        type=str, 
        choices=['rwf2000', 'rlvs'],
        help='Dataset a usar para el entrenamiento.'
    )
    
    args = parser.parse_args()
    main(args)