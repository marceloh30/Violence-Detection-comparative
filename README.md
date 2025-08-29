# Comparativa de Modelos en Detección de Violencia

## Resumen del Proyecto
La detección automática de violencia en video es un desafío que exige un equilibrio entre rendimiento, eficiencia y robustez. En este Trabajo de Fin de Máster se realizó un estudio comparativo de cuatro arquitecturas de aprendizaje profundo, seleccionadas por representar diferentes filosofías en el reconocimiento de acciones en video: **I3D, SlowFast, TSM y ViViT**.
Mediante un protocolo experimental estructurado, y partiendo de un preentrenamiento en Kinetics-400, se realizó un ajuste fino de cada modelo en los conjuntos de datos **RWF-2000** y **Real Life Violence Situations (RLVS)**. Adicionalmente, se evaluó la capacidad de generalización en un dataset de dominio distinto, el **Hockey Fights dataset**.
Los resultados concluyen que no existe una única arquitectura superior. La elección óptima depende de los requisitos de la aplicación, donde modelos eficientes como TSM presentan una alternativa pragmática y de gran rendimiento frente a enfoques más costosos como ViViT para su despliegue en sistemas reales.

---

## 🚀 Arquitecturas Comparadas

-   **I3D (Inflated 3D ConvNet)**: Representa una base sólida dentro de las CNNs 3D, "inflando" filtros 2D para capturar dinámicas espacio-temporales.
-   **SlowFast**: Una arquitectura de dos vías inspirada en la percepción visual humana, con un camino "lento" para el contexto espacial y uno "rápido" para movimientos de alta frecuencia.
-   **TSM (Temporal Shift Module)**: Un enfoque altamente eficiente que permite a las CNNs 2D modelar relaciones temporales con un costo computacional casi nulo, ideal para aplicaciones en tiempo real.
-   **ViViT (Video Vision Transformer)**: Representa el paradigma emergente de los Transformers, procesando "tubelets" de video para capturar relaciones globales a larga distancia.

---

## 🛠️ Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone [https://github.com/marceloh30/Violence-Detection-comparative.git](https://github.com/marceloh30/Violence-Detection-comparative.git)
cd tu_repositorio
```

### 2. Crear y Activar un Entorno Virtual
Usa el archivo `environment.yml` para recrear el entorno exacto con todas las dependencias correctas. 
Cabe destacar que se utilizó Python 3.10.18 y la versión de PyTorch 2.8.0+cu126 porque funcionaron correctamente y no generaron conflicto con otras librerías (pueden existir otras combinaciones útiles si se desea).
```bash
conda env create -f environment.yml
conda activate tfm_env
```
Esto creará y activará un entorno llamado `tfm_env` con todo lo necesario.

### 3. Descargar Pesos Pre-entrenados (Necesario para TSM)
El modelo TSM requiere un archivo de pesos pre-entrenados de Kinetics-400.
- **Descarga el archivo** desde el [repositorio oficial de TSM](https://github.com/mit-han-lab/temporal-shift-module). El archivo necesario es `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth`.
- **Crea una carpeta** llamada `pretrained_models/` en la raíz del proyecto.
- **Mueve el archivo `.pth` descargado** a la carpeta `pretrained_models/`.

### 4. Configuración de Datasets
Este proyecto no incluye los datos de video. Debes descargarlos por tu cuenta y organizarlos en la carpeta `assets/` (o la ruta que definas en `violence_detection/config.py`) con la siguiente estructura:

```
assets/
├── RWF-2000/
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
├── Real-Life-Violence-Dataset/
│   ├── Violence/
│   └── NonViolence/
└── Hockey-Fights/
    ├── fi... (videos de peleas)
    └── no... (videos sin peleas)
```

---

## ▶️ Uso del Proyecto

El flujo de trabajo se gestiona a través de dos scripts principales.

### Paso 1: Generar las Listas de Archivos (opcional)
Antes de entrenar, es necesario generar los archivos JSON que contienen las rutas y etiquetas de los videos. Ejecuta el siguiente comando desde la raíz del proyecto:
```bash
python scripts/prepare_data.py
```
Esto creará las listas de archivos `train`, `val` y `all` para cada dataset en la carpeta `dataset_file_lists/`.
Si no se ejecuta este script, al ejecutar main.py se generan los JSONs igualmente. Aun así, se puede ejecutarlo por separado para acelerar el proceso.

### Paso 2: Entrenar y Evaluar un Modelo
El script `violence_detection/main.py` es el punto de entrada para todos los experimentos. El comando para ejecutar un modelo (y un dataset en particular) es el siguiente:
```bash
python -m violence_detection.main <nombre_modelo> <nombre_dataset>
```
**Argumentos:**
-   `<nombre_modelo>`: Elige entre `i3d`, `slowfast`, `tsm`, `vivit`.
-   `<nombre_dataset>`: Elige el dataset para entrenar: `rwf2000` o `rlvs`.

**Ejemplos:**
```bash
# Entrenar y evaluar el modelo TSM en el dataset RWF-2000
python -m violence_detection.main tsm rwf2000

# Entrenar y evaluar el modelo ViViT en el dataset RLVS
python -m violence_detection.main vivit rlvs
```
El script guardará los logs, el mejor checkpoint del modelo y un archivo `metrics.json` con todos los resultados en la carpeta `results/`.

### Paso 2 Alternativo: Entrenar y Evaluar todos los Modelos (Windows)
El proyecto contiene un .bat para Windows, el cual al ejecutarlo se consulta el conjunto de datos deseado y si se desea entrenar y evaluar un solo modelo.
Una vez ejecutado y según lo ingresado, se obtendrán resultados en el json de cada modelo (en la carpeta `results/`). 
Este mismo se puede ejecutar desde el entorno virtual simplemente escribiendo el nombre del archivo: `ejecutar_modelos.bat`.

---

## 📊 Resultados Destacados

Los siguientes resultados se basan en la siguiente configuración:
-   Hardware: NVIDIA GeForce RTX 4090 24GB - Intel Core i9-13900KF - 64GB RAM.
-   Hiperparámetros en todos los modelos: 30 Épocas - Batch Size = 16

### Rendimiento de Clasificación (F1-Score en Validación)

| Modelo | Entrenado en RWF-2000 | Entrenado en RLVS |
| :--- | :---: | :---: |
| **ViViT** | **0.897** | **0.990** |
| **I3D** | 0.801 | 0.971 |
| **SlowFast** | 0.875 | 0.985 |
| **TSM** | 0.856 | 0.971 |

### Robustez: Inferencia Cruzada en Hockey Fights (F1-Score)

| Modelo | Entrenado en RWF-2000 | Entrenado en RLVS |
| :--- | :---: | :---: |
| **ViViT** | 0.507 | 0.695 |
| **I3D** | 0.685 | 0.473 |
| **SlowFast** | **0.791** | **0.758** |
| **TSM** | 0.730 | 0.684 |

### Eficiencia y Costo Computacional
| Modelo | Parámetros (M) | GFLOPs | Inferencia (FPS) |
| :--- | :---: | :---: | :---: |
| **ViViT** | 88.6 | 270.4 | 22 |
| **I3D** | 27.2 | 149.1 | 46 |
| **SlowFast** | 33.6 | 50.6 | 102 |
| **TSM** | **23.5** | **32.9** | **201** |

---

## 📄 Licencia

Este proyecto está distribuido bajo la Licencia MIT.

---
