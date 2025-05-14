import os
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig

def main():
    # ----- MACROS Y CONFIGURACIÓN -----
    BASE_PATH = "TFM/assets"
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

    # ----- DISPOSITIVO GPU -----
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ----- MODELO Y PROCESADOR -----
    NUM_LABELS = 2
    config = VivitConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
    temp_model = VivitForVideoClassification.from_pretrained(MODEL_NAME)
    model = VivitForVideoClassification(config)
    model.vivit.load_state_dict(temp_model.vivit.state_dict())
    # Gradient checkpointing reduce memoria
    if USE_GRADIENT_CHECKPOINT:
        model.vivit.gradient_checkpointing_enable()
    model.to(device)

    # ----- PREPROCESADO DE VÍDEOS -----
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        while len(frames) < NUM_FRAMES and idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            idx += FRAME_STEP
        cap.release()
        if not frames: return None
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())
        frames = frames[:NUM_FRAMES]
        return processor(frames, return_tensors="pt")["pixel_values"][0]

    def cargar_dataset(base_path = BASE_PATH, subfolders = SUBFOLDERS):
        pixels, labels = [], []
        for folder, lbl in tqdm(subfolders.items(), desc="Folders"):
            path = os.path.join(base_path, folder)
            if not os.path.isdir(path): continue
            for file in tqdm(os.listdir(path), desc=folder, leave=False):
                if not file.endswith('.mp4'): continue
                vp = os.path.join(path, file)
                tensor = process_video(vp)
                if tensor is not None:
                    pixels.append(tensor)
                    labels.append(lbl)
        assert pixels, "No videos procesados"

        return TensorDataset(torch.stack(pixels), torch.tensor(labels))
    
    dataset = cargar_dataset() #path y subfolders base ya incluidos por defecto
    
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ----- ENTRENAMIENTO -----
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

        # Validación
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(pixel_values=x).logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                del x, y, preds
        print(f"Validation Acc: {correct/total:.4f}")
        torch.cuda.empty_cache()

    # ----- INFERENCIA EN NUEVOS VIDEOS -----

    NEW_PATH = "TFM/assets/Pool"
    results = []
    for folder, lbl in tqdm(SUBFOLDERS.items(), desc=f"Inferencia en ${NEW_PATH}"):
        path = os.path.join(BASE_PATH, folder)
        if not os.path.isdir(path): continue
        
        for file in tqdm(os.listdir(path), desc=folder, leave=False):
            if not file.endswith('.mp4'): continue
            
            vp = os.path.join(path, file)
            tensor = process_video(vp)
            if tensor is not None: continue
            
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(pixel_values=tensor).logits.argmax(-1).item()
            results.append({"File":file,"pred":'Violence' if pred == 1 else 'NonViolence', "Etiqueta":f"Etiqueta: ${lbl}"}) 
            
            del tensor
            torch.cuda.empty_cache()            
      
    print("Results:", results)



if __name__ == '__main__':
    main()