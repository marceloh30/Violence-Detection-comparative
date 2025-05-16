import os
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json


def main():
    # ----- MACROS Y CONFIGURACIÓN -----
    BASE_PATH = "assets"
    SUBFOLDERS = {"Violence": 1, "NonViolence": 0}
    MODEL_NAME = "google/vivit-b-16x2-kinetics400"
    NUM_FRAMES = 32                # Reducir frames para menos memoria
    FRAME_STEP = 4                 # Saltos entre frames
    BATCH_SIZE = 2                 # Reducir batch size
    EPOCHS = 3
    LR = 5e-5
    USE_MIXED_PRECISION = True     # Habilitar AMP
    USE_GRADIENT_CHECKPOINT = True # Habilitar checkpointing para memoria
    NUM_WORKERS = 4
    PIN_MEMORY = True
    NUM_LABELS = 2
    OUTPUT_DIR = "vivit_finetuned"
    INFERENCE_PATH = "assets/Pool"
    RE_TRAIN = False
    
    # ----- DISPOSITIVO GPU -----
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # ----- MODELO Y PROCESADOR -----
    processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
    config = VivitConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model_path = OUTPUT_DIR

    # Funcion para procesar cada video
    def process_video(video_path):
        video_path = os.path.normpath(video_path)
        #print(f"Procesando video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            print("No se pudo abrir el video.")
            return None
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0

        while len(frames) < NUM_FRAMES and idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                print("Ocurrio un error al leer el frame")
                idx += FRAME_STEP
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            idx += FRAME_STEP

        cap.release()
        if not frames:
            print("No se obtuvieron frames del video.")
            return None
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())
        frames = frames[:NUM_FRAMES]
        try:
            tensor = processor(frames, return_tensors="pt")
            if "pixel_values" not in tensor:
                print("'pixel_values' no está en la salida del processor")
                return None
            tensor = tensor["pixel_values"][0]
            #print(f"Tensor shape: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"Error al procesar frames con el processor: {e}")
            return None
    #Funcion para generar el dataset
    # -Toma los tensores generados de process_video y retorna un TensorDataset()
    def cargar_dataset():
        pixels, labels = [], []
        for folder, lbl in tqdm(SUBFOLDERS.items(), desc="Folders"):
            path = os.path.join(BASE_PATH, folder)
            if not os.path.isdir(path): 
                continue
            for file in tqdm(os.listdir(path), desc=folder, leave=False):
                if not file.endswith('.mp4'): 
                    continue
                t = process_video(os.path.join(path, file))
                if t is not None:
                    pixels.append(t)
                    labels.append(lbl)
        assert pixels, "No videos procesados"
        return TensorDataset(torch.stack(pixels), torch.tensor(labels))


    if (os.path.isdir(model_path) and os.listdir(model_path)) or RE_TRAIN:
        # Si ya existe modelo fine-tuned, cargarlo
        print(f"Cargando modelo fine-tuned desde '{model_path}'...")
        model = VivitForVideoClassification.from_pretrained(model_path)
        processor = VivitImageProcessor.from_pretrained(model_path)
    else:
        # Si no existe, fine-tune desde scratch
        print("No se encontró modelo fine-tuned. Comenzando entrenamiento...")
        temp = VivitForVideoClassification.from_pretrained(MODEL_NAME)
        model = VivitForVideoClassification(config)
        model.vivit.load_state_dict(temp.vivit.state_dict())
        if USE_GRADIENT_CHECKPOINT:
            model.vivit.gradient_checkpointing_enable()
        model.to(device)
        
        # Cargo dataset y genero los dataset y dataloaders
        dataset = cargar_dataset()
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # ----- ENTRENAMIENTO -----
        history = {'epoch': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        optimizer = AdamW(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if USE_MIXED_PRECISION else None

        for epoch in range(1, EPOCHS+1):
            model.train()
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION):
                    logits = model(pixel_values=x).logits
                    loss = criterion(logits, y)
                if USE_MIXED_PRECISION:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                del x, y, logits, loss
                torch.cuda.empty_cache()
            print(f"Epoch {epoch} loss: {total_loss/len(train_loader):.4f}")

            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(pixel_values=x).logits.argmax(-1).cpu().tolist()
                    ps.extend(preds) 
                    ys.extend(y.tolist())
                   
                    del x,y,preds
                    torch.cuda.empty_cache()

            acc = accuracy_score(ys, ps)
            prec = precision_score(ys, ps)
            rec = recall_score(ys, ps)
            f1 = f1_score(ys, ps)
            print(f"Epoch {epoch} - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
            history['epoch'].append(epoch)
            history['accuracy'].append(acc)
            history['precision'].append(prec)
            history['recall'].append(rec)
            history['f1'].append(f1)            

        # ----- GUARDAR MODELO FINE-TUNED -----
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, 'train_metrics.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Modelo fine-tuned y json con metricas guardado en '{OUTPUT_DIR}'")
        
    # ----- INFERENCIA EN NUEVOS VIDEOS -----

    results = []
    ys, ps = [], []
    if torch.cuda.is_available(): #Prueba cuda
        model.cuda()
    for folder, lbl in tqdm(SUBFOLDERS.items(), desc=f"Inferencia en ${INFERENCE_PATH}"):
        path = os.path.join(INFERENCE_PATH, folder)
        if not os.path.isdir(path): continue
        
        for file in tqdm(os.listdir(path), desc=folder, leave=False):
            if not file.endswith('.mp4'): continue
            vp = os.path.join(path, file)
            tensor = process_video(vp)
            if tensor is None: 
                print("Tensor es None!")
                continue
            tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                tensor = tensor.to(device)
                pred = model(pixel_values=tensor).logits.argmax(-1).item()
                ys.append(lbl)
                ps.append(pred)
            print(file,pred,lbl)
         
            del tensor
            torch.cuda.empty_cache()     
    acc = accuracy_score(ys, ps)
    prec = precision_score(ys, ps)
    rec = recall_score(ys, ps)
    f1 = f1_score(ys, ps)
    cm = confusion_matrix(ys, ps).tolist()
    print(f"Inference - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
    print("Confusion matrix:", cm)
    inf_stats = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}
    with open(os.path.join(OUTPUT_DIR, 'inference_metrics.json'), 'w') as f:
        json.dump(inf_stats, f, indent=2)                   
    print("Resultados de predicción:")
    for res in results: print(f"- Archivo: ${res["file"]}, Predicción: ${res["pred"]}, Etiqueta: ${res["etiqueta"]}")

if __name__ == '__main__':
    main()