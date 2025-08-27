import os
import sys
import logging

# --- Añadir el directorio raíz del proyecto al path ---
# Esto es crucial para que el script pueda encontrar el paquete 'violence_detection'
# Se asume que este script se ejecuta desde el directorio raíz del proyecto.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# ----------------------------------------------------

# Ahora podemos importar desde nuestro paquete
from violence_detection import utils, config

def main():
    """
    Ejecuta el proceso de generación/regeneración de las listas de archivos
    JSON para todos los datasets definidos en la configuración de utilidades.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Iniciando la generación de todas las listas de archivos de los datasets...")

    # Los datasets a procesar se leen directamente de la configuración en utils
    datasets_to_process = list(utils.DATASET_CONFIGS.keys())
    
    for dataset_name in datasets_to_process:
        print(f"\n--- Regenerando listas para el dataset: {dataset_name.upper()} ---")
        
        # Para datasets como rwf2000, que tienen splits predefinidos
        if utils.DATASET_CONFIGS[dataset_name]['type'] == 'structured_folders':
            utils.get_dataset_file_list(dataset_name, "train", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)
            utils.get_dataset_file_list(dataset_name, "val", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)
            utils.get_dataset_file_list(dataset_name, "all", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)
        
        # Para datasets que se dividen a partir de una lista 'all'
        else:
            utils.get_dataset_file_list(dataset_name, "all", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)
            utils.get_dataset_file_list(dataset_name, "train", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)
            utils.get_dataset_file_list(dataset_name, "val", config.BASE_DATA_DIR, config.FILE_LIST_DIR, force_regenerate=True)

    logging.info(f"\nProceso completado. Todas las listas de archivos deberían estar en: {os.path.abspath(config.FILE_LIST_DIR)}")


if __name__ == "__main__":
    main()