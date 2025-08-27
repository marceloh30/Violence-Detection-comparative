import os
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import glob
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from .models.vivit import VivitModelWrapperForFlops
import time

# ----- CONFIGURACIÓN -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- FUNCIONES DE REPRODUCIBILIDAD -----

# --- Funcion para asignar semilla ---
def set_seed(seed_value, full_reproducibility = False):
    """
    Fija las semillas de todas las librerias relevantes para la reproducibilidad
    y configura el comportamiento determinista de CUDA si se solicita.

    Args:
        seed_value (int): El valor de la semilla a utilizar.
        full_reproducibility (bool): Si es True, configura cuDNN para un
                                     comportamiento determinista total, lo que puede
                                     afectar el rendimiento.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # Para multi-GPU
        
        if full_reproducibility:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False 
            logging.info(f"Semilla fijada a: {seed_value}. Modo de reproducibilidad total activado")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True       
            logging.info(f"Semilla fijada a: {seed_value}. Modo de alto rendimiento activado")      
    else:
        logging.info(f"Semilla fijada a: {seed_value} (CPU).")

# --- Funcion para inicializar workers del Dataloader ---
def seed_worker(worker_id):
    """
    Funcion para inicializar los workers del DataLoader y mantener la
    reproducibilidad en la carga de datos.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --------------------------------------- #

# ----- FUNCIONES DE MANEJO DE ARCHIVOS (I/O) -----

# --- Guardado de Metricas ---
def save_or_update_json(new_data: dict, json_path: str):
    """
    Lee un archivo JSON existente, actualiza su contenido con los nuevos datos
    y lo guarda de nuevo. Si el archivo no existe, lo crea. Es ideal para
    guardar el historial de entrenamiento epoca por epoca.

    Args:
        new_data (dict): Diccionario con los nuevos datos a añadir o actualizar.
        json_path (str): Ruta donde se va a guardar el archivo JSON.
    """
    # Asegura de que el directorio de salida exista
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    existing_data = {}
    # Leer el contenido actual del archivo si ya existe y no esta vacio
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"El archivo JSON {json_path} estaba corrupto. Se va a sobrescribir.")

    # Actualizar el diccionario existente con los nuevos datos.
    # El metodo .update() fusiona los diccionarios. Las claves nuevas se añaden
    # y las claves existentes en 'existing_data' se actualizan con los valores de 'new_data'.
    existing_data.update(new_data)

    # Escribir el diccionario completo de vuelta al archivo
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        logging.info(f"JSON actualizado en '{json_path}' con las claves: {list(new_data.keys())}")
    except Exception as e:
        logging.error(f"No se pudo guardar el JSON en '{json_path}': {e}")

# ----------------------------------------------- #

# ----- GENERACION DE LISTAS DE ARCHIVOS DEL DATASET -----


# --- Clases Globales (usadas por todos los datasets para consistencia) ---
# Estas son las etiquetas que los modelos esperan.
GLOBAL_CLASSES_MAP = {"Violence": 1, "NonViolence": 0} #Ver de configurar segun lo que se vea en config.py REVISAR

# --- Configuraciones Especificas por Dataset ---

# Nombre de directorios (cambiar segun lo que se tenga)
RWF_2000_SUBDIR = "RWF-2000"
RLVS_SUBDIR = "Real-Life-Violence-Dataset"
HOCKEY_FIGHTS_SUBDIR = "Hockey-Fights"
# Parametros de division de datasets
TRAIN_VAL_SPLIT_RATIO = 0.8 # porcentaje de entrenamiento, el restante es validacion
RANDOM_SEED_SPLIT = 23
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Configuracion especifica de cada dataset
DATASET_CONFIGS = {
    "rwf2000": {
        "subdir": RWF_2000_SUBDIR,
        "type": "structured_folders",
        "split_folders_map": {"train": "train", "val": "val"},
        # Mapea las carpetas "Fight"/"NonFight" de RWF-2000 a las clases globales 'Violence'/'NonViolence'
        "class_map_internal": { 
            "Fight": GLOBAL_CLASSES_MAP["Violence"],            # Ver de configurar segun lo que se vea en config.py REVISAR
            "NonFight": GLOBAL_CLASSES_MAP["NonViolence"]       # Ver de configurar segun lo que se vea en config.py REVISAR
        },
    },
    "rlvs": {
        "subdir": RLVS_SUBDIR,
        "type": "class_folders",
        # Mapea las carpetas "Violence"/"NonViolence" de RLVS a las clases globales (que ahora coinciden)
        "class_folders_map_internal": {
            "Violence": "Violence",         # Ver de configurar segun lo que se vea en config.py REVISAR
            "NonViolence": "NonViolence"    # Ver de configurar segun lo que se vea en config.py REVISAR
        },
        "global_class_map_target": GLOBAL_CLASSES_MAP,
        "train_val_ratio": TRAIN_VAL_SPLIT_RATIO,
        "random_seed": RANDOM_SEED_SPLIT,
    },
    "hockey": {
        "subdir": HOCKEY_FIGHTS_SUBDIR,
        "type": "prefix_single_folder",
        # Mapea los prefijos 'fi'/'no' de Hockey a las clases globales 'Violence'/'NonViolence'
        "prefix_map_internal": {
            "fi": "Violence",               # Ver de configurar segun lo que se vea en config.py REVISAR
            "no": "NonViolence"             # Ver de configurar segun lo que se vea en config.py REVISAR
        },
        "global_class_map_target": GLOBAL_CLASSES_MAP,
        "train_val_ratio": TRAIN_VAL_SPLIT_RATIO,
        "random_seed": RANDOM_SEED_SPLIT,
    }
}

# --- Funciones Para Generar Los JSON ---

# Funcion interna que encuentra videos en cadenas de carpetas
def _find_videos_in_structured_folders(dataset_path, split_folders_map, class_map_internal, current_split_name):
    video_files_with_labels = []
    split_folder_name = split_folders_map.get(current_split_name)
    if not split_folder_name:
        logging.error(f"StructuredFolders-like: Nombre de carpeta para split '{current_split_name}' no definido.")
        return []

    logging.debug(f"StructuredFolders-like: Buscando vídeos para split '{current_split_name}' en '{dataset_path}/{split_folder_name}'...")
    for class_name_folder, label_id in class_map_internal.items(): # class_name_folder es "Fight", "NonFight" para RWF
        class_path = os.path.join(dataset_path, split_folder_name, class_name_folder)
        if not os.path.isdir(class_path):
            logging.warning(f"StructuredFolders-like: Carpeta de clase no encontrada: {class_path}")
            continue
        
        count = 0
        for ext in VIDEO_EXTENSIONS:
            found_videos = glob.glob(os.path.join(class_path, f"*{ext}"))
            for video_path in found_videos:
                video_files_with_labels.append({"path": os.path.abspath(video_path), "label": label_id})
                count +=1
        logging.debug(f"StructuredFolders-like: Encontrados {count} vídeos en {class_path} para clase '{class_name_folder}' (etiqueta: {label_id})")
    return video_files_with_labels

# Funcion interna para encontrar los videos dentro de carpetas (cada carpeta = 1 clase)
def _find_videos_in_class_folders(dataset_path, class_folders_map_internal, global_class_map_target):
    all_video_files_with_labels = []
    logging.debug(f"ClassFolders-like: Buscando vídeos en '{dataset_path}'...")
    for specific_class_folder, global_class_name_ref in class_folders_map_internal.items(): # specific_class_folder es "Violence", "NonViolence" para RLVS
        label_id = global_class_map_target.get(global_class_name_ref) # global_class_name_ref es "Violence", "NonViolence"
        if label_id is None:
            logging.warning(f"ClassFolders-like: Clase global '{global_class_name_ref}' (de carpeta '{specific_class_folder}') no mapeada. Omitiendo.")
            continue

        class_path = os.path.join(dataset_path, specific_class_folder)
        if not os.path.isdir(class_path):
            logging.warning(f"ClassFolders-like: Carpeta de clase específica no encontrada: {class_path}")
            continue
        
        count = 0
        for ext in VIDEO_EXTENSIONS:
            found_videos = glob.glob(os.path.join(class_path, f"*{ext}"))
            for video_path in found_videos:
                all_video_files_with_labels.append({"path": os.path.abspath(video_path), "label": label_id})
                count +=1
        logging.debug(f"ClassFolders-like: Encontrados {count} vídeos en {class_path} (mapeado a '{global_class_name_ref}', etiqueta: {label_id})")
    return all_video_files_with_labels

# Funcion interna para encontrar videos segun prefijo (para formato Hockey Fights)
def _find_videos_by_prefix_in_single_folder(dataset_path, prefix_map_internal, global_class_map_target):
    all_video_files_with_labels = []
    logging.debug(f"PrefixSingleFolder-like: Buscando vídeos en '{dataset_path}'...")
    if not os.path.isdir(dataset_path):
        logging.warning(f"PrefixSingleFolder-like: Carpeta del dataset no encontrada: {dataset_path}")
        return []

    count_total = 0
    files_in_dir = os.listdir(dataset_path)
    for filename in files_in_dir:
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(VIDEO_EXTENSIONS):
            matched = False
            for prefix, global_class_name_ref in prefix_map_internal.items(): # prefix es "fi", "no" para Hockey
                if filename.lower().startswith(prefix):
                    label_id = global_class_map_target.get(global_class_name_ref) # global_class_name_ref es "Violence", "NonViolence"
                    if label_id is not None:
                        all_video_files_with_labels.append({"path": os.path.abspath(file_path), "label": label_id})
                        count_total +=1
                        matched = True
                        break
                    else:
                        logging.warning(f"PrefixSingleFolder-like: Clase global '{global_class_name_ref}' (de prefijo '{prefix}') no mapeada. Omitiendo {filename}")
    logging.debug(f"PrefixSingleFolder-like: Encontrados {count_total} vídeos con prefijos válidos en {dataset_path}")
    return all_video_files_with_labels

# Funcion interna para guardar lista a un json
def _save_file_list_to_json(file_list, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(file_list, f, indent=4)
        logging.info(f"Lista de archivos guardada en: {output_path} ({len(file_list)} vídeos)")
    except IOError as e:
        logging.error(f"Error guardando el archivo JSON {output_path}: {e}")

# Funcion interna para generar splits y guardarlos si se quiere
def _generate_and_optionally_save_splits(all_files_with_labels, dataset_name, output_dir, filename_base,
                                         train_ratio, seed, create_all_list=True, save_to_disk=True):
    if not all_files_with_labels:
        logging.warning(f"No hay archivos para dividir para {filename_base}. Devolviendo listas vacías.")
        return [], [], ([] if create_all_list else None)

    paths = [item['path'] for item in all_files_with_labels]
    labels = [item['label'] for item in all_files_with_labels]

    train_list, val_list = [], []
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = (len(unique_labels) >= 2 and all(c >= 1 for c in counts)) or \
                   (len(unique_labels) == 1 and counts[0] >= 2)

    if len(all_files_with_labels) < 2:
        logging.warning(f"Dataset {filename_base} tiene menos de 2 muestras ({len(all_files_with_labels)}). "
                        "Todo irá a 'train', 'val' estará vacío.")
        train_list = all_files_with_labels
    elif can_stratify and 0.0 < train_ratio < 1.0 :
        min_samples_per_class_for_stratify = 2 if len(unique_labels) > 1 else 1
        
        if all(c >= min_samples_per_class_for_stratify for c in counts):
            try:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    paths, labels, train_size=train_ratio, random_state=seed, stratify=labels, shuffle=True
                )
                train_list = [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)]
                val_list = [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)]
            except ValueError as e:
                logging.warning(f"División estratificada falló para {filename_base} (Error: {e}). Realizando división simple.")
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    paths, labels, train_size=train_ratio, random_state=seed, shuffle=True
                )
                train_list = [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)]
                val_list = [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)]
        else:
            logging.warning(f"No hay suficientes muestras en cada clase para división estratificada en {filename_base} "
                            f"(clases: {unique_labels}, cuentas: {counts}). Realizando división simple.")
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                paths, labels, train_size=train_ratio, random_state=seed, shuffle=True
            )
            train_list = [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)]
            val_list = [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)]
    elif train_ratio >= 1.0:
        logging.info(f"Train_ratio es {train_ratio} para {filename_base}. Todos los datos irán a 'train'.")
        train_list = all_files_with_labels
    elif train_ratio <= 0.0:
        logging.info(f"Train_ratio es {train_ratio} para {filename_base}. Todos los datos irán a 'val'.")
        val_list = all_files_with_labels
    else: 
         logging.warning(f"No se pudo realizar división estratificada para {filename_base} y train_ratio={train_ratio}. "
                    "Realizando división simple.")
         train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, train_size=train_ratio, random_state=seed, shuffle=True 
        )
         train_list = [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)]
         val_list = [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)]

    if save_to_disk:
        _save_file_list_to_json(train_list, os.path.join(output_dir, f"{filename_base}_train_files.json"))
        _save_file_list_to_json(val_list, os.path.join(output_dir, f"{filename_base}_val_files.json"))
        if create_all_list:
            _save_file_list_to_json(all_files_with_labels, os.path.join(output_dir, f"{filename_base}_all_files.json"))
            
    return train_list, val_list, (all_files_with_labels if create_all_list else None)

# Funcion para obtener el 'dataset' con la lista de archivos en cada conjunto de videos 
def get_dataset_file_list(dataset_name: str, 
                          split_name: str,
                          base_data_dir: str,
                          output_list_dir: str,
                          force_regenerate: bool = False) -> list:
    """
    Obtiene la lista de archivos para un dataset y split específico.
    Carga desde JSON si existe, o la genera y guarda.
    
    Args:
        dataset_name (str): Nombre del dataset (ej. "rwf2000").
        split_name (str): Split deseado ("train", "val", "all").
        base_data_dir (str): Ruta al directorio base que contiene las carpetas de los datasets.
        output_list_dir (str): Ruta al directorio donde se guardan/cargan los JSON con las listas de archivos.
        force_regenerate (bool): Si es True, ignora los JSON existentes y regenera las listas.

    Returns:
        list: Una lista de diccionarios: [{"path": abs_path, "label": int_label}, ...].
    """

    
    if dataset_name not in DATASET_CONFIGS:
        logging.error(f"Configuración no encontrada para el dataset: {dataset_name}")
        return []
    
    config = DATASET_CONFIGS[dataset_name]
    filename_base = dataset_name.lower().replace("-", "")
    target_json_path = os.path.join(output_list_dir, f"{filename_base}_{split_name}_files.json")

    if not force_regenerate and os.path.exists(target_json_path):
        logging.info(f"Cargando lista de archivos preexistente: {target_json_path}")
        try:
            with open(target_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error al cargar {target_json_path}: {e}. Se intentará regenerar.")
            force_regenerate = True

    logging.info(f"Generando lista de archivos para {dataset_name} (split: {split_name})...")
    dataset_abs_path = os.path.join(base_data_dir, config["subdir"])
    
    all_files_collected = []

    if config["type"] == "structured_folders": 
        if split_name in ["train", "val"]:
            generated_files = _find_videos_in_structured_folders(
                dataset_abs_path, config["split_folders_map"], config["class_map_internal"], split_name
            )
            _save_file_list_to_json(generated_files, target_json_path)
            return generated_files
        elif split_name == "all":
            # Para structured_folders, 'all' es la combinación de 'train' y 'val'
            # Asegurarse de que train y val se generen/carguen primero
            train_files = get_dataset_file_list(dataset_name, "train", base_data_dir, output_list_dir, force_regenerate)
            val_files = get_dataset_file_list(dataset_name, "val", base_data_dir, output_list_dir, force_regenerate)
            all_combined_files = train_files + val_files
            _save_file_list_to_json(all_combined_files, target_json_path)
            return all_combined_files
        else:
            logging.error(f"Split '{split_name}' no soportado para dataset tipo '{config['type']}' como {dataset_name}")
            return []

    elif config["type"] in ["class_folders", "prefix_single_folder"]: 
        all_json_path = os.path.join(output_list_dir, f"{filename_base}_all_files.json")
        if not force_regenerate and os.path.exists(all_json_path):
            logging.debug(f"Cargando lista 'all' preexistente para {dataset_name} desde {all_json_path}")
            try:
                with open(all_json_path, 'r') as f: all_files_collected = json.load(f)
            except Exception as e:
                logging.warning(f"Error cargando {all_json_path}: {e}. Se re-escaneará.")
                all_files_collected = [] 
        
        if not all_files_collected: 
            logging.debug(f"Re-escaneando archivos para {dataset_name} para generar la lista 'all'...")
            if config["type"] == "class_folders":
                all_files_collected = _find_videos_in_class_folders(
                    dataset_abs_path, config["class_folders_map_internal"], config["global_class_map_target"]
                )
            elif config["type"] == "prefix_single_folder":
                all_files_collected = _find_videos_by_prefix_in_single_folder(
                    dataset_abs_path, config["prefix_map_internal"], config["global_class_map_target"]
                )
            _save_file_list_to_json(all_files_collected, all_json_path)

        if split_name == "all":
            return all_files_collected
        elif split_name in ["train", "val"]:
            # Los splits train/val se derivan de 'all_files_collected'
            # No es necesario volver a cargar target_json_path aquí si ya se intentó al principio de la función.
            # Si llegamos aquí con force_regenerate=True o porque target_json_path no existía, debemos generar.
            logging.debug(f"Generando splits train/val para {dataset_name} a partir de la lista 'all'...")
            train_list, val_list, _ = _generate_and_optionally_save_splits(
                all_files_collected, dataset_name, output_list_dir, filename_base,
                config["train_val_ratio"], config["random_seed"], 
                create_all_list=False, # No necesitamos la lista 'all' de esta función, ya la tenemos.
                save_to_disk=True      # Asegurar que los splits train/val se guarden.
            )
            if split_name == "train":
                return train_list
            else: # split_name == "val"
                return val_list
        else:
            logging.error(f"Split '{split_name}' no soportado para dataset tipo '{config['type']}' como {dataset_name}")
            return []
    else:
        logging.error(f"Tipo de dataset desconocido: {config['type']} para {dataset_name}")
        return []
    
    
# ----- FUNCIONES DE REPRODUCIBILIDAD -----

# Calculo de metricas estaticas: GFLOPs y Parametros
def calculate_static_metrics(model, model_name, model_params, device):
    """
    Calcula las métricas estáticas de un modelo: número de parámetros y GFLOPs.

    Returns:
        dict: Un diccionario con {'parameters': count, 'gflops': count}.
    """
    logging.info("Calculando métricas estáticas (Parámetros, GFLOPs)...")
    model.eval()
    
    # Medir Parámetros
    try:
        params_count = parameter_count(model).get('', 0)
    except Exception:
        params_count = sum(p.numel() for p in model.parameters())
    
    # Medir FLOPs
    gflops = -1.0
    try:
        t, h, w = model_params['num_frames'], model_params['image_size'], model_params['image_size']
        
        if model_name == 'slowfast':
            t_slow = t // model_params['alpha']
            dummy_input = [torch.randn(1, 3, t_slow, h, w, device=device), torch.randn(1, 3, t, h, w, device=device)]
            flops_analyzer = FlopCountAnalysis(model, dummy_input)
        elif model_name == 'vivit':
            dummy_input = torch.randn(1, t, 3, h, w, device=device)
            wrapped_model = VivitModelWrapperForFlops(model).to(device)
            flops_analyzer = FlopCountAnalysis(wrapped_model, (dummy_input,))
        else: # I3D, TSM
            dummy_input = torch.randn(1, 3, t, h, w, device=device)
            flops_analyzer = FlopCountAnalysis(model, dummy_input)
        
        gflops = flops_analyzer.total() / 1e9
    except Exception as e:
        logging.error(f"No se pudieron calcular los GFLOPs para {model_name}: {e}")

    return {'parameters': params_count, 'gflops': gflops}

# Funcion para medir velocidad de inferencia en FPS de cada modelo
def measure_inference_fps(model, model_name, model_params, device, trials_fps):
    """
    Mide la velocidad de inferencia (FPS) de un modelo.

    Returns:
        float: Los fotogramas por segundo (FPS) calculados.
    """
    logging.info("Midiendo velocidad de inferencia (FPS)...")
    model.eval()
    fps = 0.0
    try:
        t, h, w = model_params['num_frames'], model_params['image_size'], model_params['image_size']
        
        if model_name == 'slowfast':
            dummy_input = [torch.randn(1, 3, t // model_params['alpha'], h, w, device=device), torch.randn(1, 3, t, h, w, device=device)]
        elif model_name == 'vivit':
            dummy_input = torch.randn(1, t, 3, h, w, device=device)
        else:
            dummy_input = torch.randn(1, 3, t, h, w, device=device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input) if model_name != 'vivit' else model(pixel_values=dummy_input)

        # Medición
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(trials_fps):
                _ = model(dummy_input) if model_name != 'vivit' else model(pixel_values=dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        fps = trials_fps / total_time if total_time > 0 else 0.0
    except Exception as e:
        logging.error(f"No se pudo calcular el FPS para {model_name}: {e}")
        
    return fps