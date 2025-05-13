import numpy as np
from datasets import load_dataset, DatasetDict
import evaluate
import torch
from transformers import AutoTokenizer, ViViTForVideoClassification

DATASET_DIR = "./assets/"

tokenizer = AutoTokenizer.from_pretrained("google/vivit-b-16x2-kinetics400")
model = ViViTForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

# Cargar el dataset (reemplazar con tu dataset)
# Se asume que el dataset tiene una columna 'video' con el path a los videos
# y una columna 'label' con "Violence" o "NonViolence"
dataset = load_dataset(DATASET_DIR)

# Preprocesar el dataset
def preprocess_function(examples):
    videos = [video for video in examples['video']] # Ajusta 'video' a tu nombre de columna
    inputs = processor(videos=videos, return_tensors="pt") # Ajusta 'video' a tu nombre de columna
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Convertir las etiquetas a valores numéricos
label2id = {"NonViolence": 0, "Violence": 1}
id2label = {0: "NonViolence", 1: "Violence"}

def convert_labels(example):
  example['labels'] = label2id[example['label']] # Ajusta 'label' a tu nombre de columna
  return example

encoded_dataset = encoded_dataset.map(convert_labels)

# Crear un conjunto de validación
if "validation" not in encoded_dataset:
  encoded_dataset = encoded_dataset["train"].train_test_split(test_size=0.2)
  encoded_dataset = DatasetDict({"train":encoded_dataset["train"], "validation": encoded_dataset["test"]})


# Entrenar o evaluar el modelo (ejemplo de evaluación)
metric = evaluate.load("accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# ... (código para entrenar o fine-tunear el modelo, utilizando compute_metrics para la evaluación)

for batch in encoded_dataset["validation"]:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits

    # ... (código para procesar los logits y calcular las métricas)
    print("Ejemplo de predicción en evaluación...")

