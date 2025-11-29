import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
# Some constants
DATA_ROOT = "dataset"
CLASSES = ['COVID', 'Healthy', 'Non-COVID']

# Count images per class
def get_dataset_stats(data_root: str=DATA_ROOT, classes: list[str]=CLASSES) -> dict:
    stats = {}
    for cls in classes:
        img_path = os.path.join(data_root, cls, "images")
        mask_path = os.path.join(data_root, cls, "masks")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image directory not found: {img_path}")
            img_files = []
        else:
            img_files = glob.glob(os.path.join(img_path, "*.png"))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask directory not found: {mask_path}")
            mask_files = []
        else:
            mask_files = glob.glob(os.path.join(mask_path, "*.png"))
        
        stats[cls] = {
            'images': len(img_files),
            'masks': len(mask_files),
            'image_paths': img_files[:10],  # Sample paths
            'mask_paths': mask_files[:10]
        }
    return stats

def visualize_samples(n_samples: int=6, classes: list[str]=CLASSES, data_root: str=DATA_ROOT):
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(20, 10))
    
    for row, cls in enumerate(classes):
        img_files = glob.glob(os.path.join(data_root, cls, "images", "*.png"))[:n_samples]
        
        for col, img_file in enumerate(img_files):
            img = Image.open(img_file)
            axes[row, col].imshow(img, cmap='gray' if img.mode == 'L' else None)
            axes[row, col].set_title(f'{cls}\n{os.path.basename(img_file)}', 
                                    fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def visualize_image_mask_pairs(n_samples:int=3, classes:list[str]=CLASSES, data_root:str=DATA_ROOT):
    fig, axes = plt.subplots(len(classes), n_samples*3, figsize=(20, 10))
    
    for row, cls in enumerate(classes):
        img_files = glob.glob(os.path.join(data_root, cls, "images", "*.png"))[:n_samples]
        
        for col, img_file in enumerate(img_files):
            # Load image
            img = Image.open(img_file).convert('RGB')
            img_name = os.path.basename(img_file)
            
            # Load corresponding mask
            mask_file = os.path.join(data_root, cls, "masks", img_name)
            mask = Image.open(mask_file).convert('L') if os.path.exists(mask_file) else None
            
            # Resize mask to match image dimensions
            if mask:
                mask = mask.resize(img.size, Image.NEAREST)
            
            # Original image
            axes[row, col*3].imshow(img)
            axes[row, col*3].set_title(f'{cls} - Original', fontsize=10)
            axes[row, col*3].axis('off')
            
            # Mask
            if mask:
                axes[row, col*3+1].imshow(mask, cmap='gray')
                axes[row, col*3+1].set_title('Mask', fontsize=10)
                axes[row, col*3+1].axis('off')
                
                # Overlay
                img_array = np.array(img)
                mask_array = np.array(mask) > 128
                overlay = img_array.copy()
                overlay[mask_array] = [255, 0, 0]  # Red overlay
                axes[row, col*3+2].imshow(overlay)
                axes[row, col*3+2].set_title('Overlay', fontsize=10)
                axes[row, col*3+2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
# --- Model Definitions (With Pre-trained Weights and Dropout Fixes) ---

def add_dropout_to_fc(model, p=0.5, classes=CLASSES):
    """
    Replaces the final classification layer with a new one preceded by Dropout.
    This guarantees a consistent classification head for transfer learning.
    """
    # This function is critical: it sets the initial classification layer for Stage 1.
    if hasattr(model, 'fc'):
        # Standard for ResNets
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(num_features, len(classes))
        )
        return 'fc' # Return the layer name to freeze/unfreeze later
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        # Standard for VGGs
        new_classifier = list(model.classifier.children())[:-1] # Remove last layer
        last_in_features = list(model.classifier.children())[-1].in_features
        new_classifier.extend([
            nn.Dropout(p=p),
            nn.Linear(last_in_features, len(classes))
        ])
        model.classifier = nn.Sequential(*new_classifier)
        return 'classifier' # Return the layer name to freeze/unfreeze later
    return None

# FIX: Modified to directly load official models from PyTorch Hub to prevent size mismatch errors
def get_class_model(name):
    # We no longer rely on local ResNet/VGG implementations to avoid architecture mismatch.
    name_lower = name.lower()
    model = None

    print(f"Loading IMAGENET1K_V1 weights for {name}...")

    try:
        # Load the official pre-trained model directly from PyTorch Hub
        if "resnet" in name_lower:
            model = torch.hub.load('pytorch/vision:v0.10.0', name_lower, weights="IMAGENET1K_V1")
        elif "vgg" in name_lower:
            # VGG models in PyTorch Hub usually have _bn for batch normalization
            hub_name = name_lower + "_bn"
            model = torch.hub.load('pytorch/vision:v0.10.0', hub_name, weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown classification model: {name}")

    except Exception as e:
        print(f"❌ Error loading model {name} from PyTorch Hub: {e}. Model will train from scratch.")
        # Fallback in case of network error, but we expect the architecture to be correct now.
        if "resnet" in name_lower:
            from models.classification_models import ResNet
            ModelClass = ResNet.ResNet18 if name_lower == "resnet18" else ResNet.ResNet50
        elif "vgg" in name_lower:
            from models.classification_models import VGG
            ModelClass = VGG.VGG16 if name_lower == "vgg16" else VGG.VGG19
        model = ModelClass(num_classes=1000)

    if model is None:
        raise RuntimeError(f"Failed to initialize model {name}.")

    # 4. Replace the final layer and get its name
    cls_head_name = add_dropout_to_fc(model, p=0.5)
    return model, cls_head_name

def get_seg_model(name):
    from models.segmentation_models import ResnetUnet, AttentionUNet, R2U_Net, R2AttU_Net
    name_lower = name.lower()
    if name_lower == "resnetunet":
        return ResnetUnet.ResNetUnet()
    elif name_lower == "attentionunet":
        return AttentionUNet.AttentionUNet()
    elif name_lower == "r2unet":
        return R2U_Net.R2U_Net()
    elif name_lower == "r2attunet":
        return R2AttU_Net.R2AttU_Net()
    else:
        raise ValueError(f"Unknown segmentation model: {name}")
# --- END Model Definitions ---
    
# [Metrics] (Unchanged)
def acc(logits, y): return (torch.argmax(logits, 1) == y).sum().item(), y.size(0)

def iou(pred, mask, t=0.5):
    p = (pred > t).float(); inter=(p*mask).sum(); union=((p+mask)>0).float().sum()
    return (inter/(union+1e-7)).item()

# [Training] (IMPROVED Two-Stage Fine-Tuning)
def train(model, train_dl, val_dl, device, epochs, lr, name, save_dir, seg=False, cls_head_name=None):
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss() if seg else nn.CrossEntropyLoss(label_smoothing=0.1)

    # Segmentation training remains standard
    if seg:
        # FIX: Increased weight decay for stronger L2 regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        print(f"🔥 Training Segmentation model (all layers unfrozen) with LR: {lr}")
        # Use CosineAnnealing for stable convergence in segmentation
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        start_epoch = 1

    # CLASSIFICATION: Two-Stage Fine-Tuning
    else:
        FEATURE_EXTRACTION_EPOCHS = 5
        FULL_FINETUNE_EPOCHS = epochs - FEATURE_EXTRACTION_EPOCHS
        start_epoch = 1

        # STAGE 1 SETUP: Freeze all base layers, train only the classification head
        print(f"--- STAGE 1: Feature Extraction (Epochs {start_epoch}-{FEATURE_EXTRACTION_EPOCHS}) ---")
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classification head (e.g., model.fc or model.classifier)
        trainable_params = []
        if cls_head_name:
            cls_head = getattr(model, cls_head_name)
            for param in cls_head.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        # FIX: Increased weight decay for stronger L2 regularization in Stage 1
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=5e-4)
        # Use CosineAnnealing for Stage 1 for smooth LR decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FEATURE_EXTRACTION_EPOCHS)


    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    # For classification, start best score at 0.0 to save the first good model
    best_score = 0.0 if not seg else float("inf")
    patience, patience_counter = 10, 0

    start_time = time.time()

    for epoch in range(start_epoch, epochs+1):

        # STAGE 2 TRANSITION: Unfreeze all layers for fine-tuning
        if not seg and epoch == FEATURE_EXTRACTION_EPOCHS + 1:
            print(f"\n--- STAGE 2: Full Fine-Tuning (Epochs {epoch}-{epochs}) ---")

            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True

            # Re-initialize optimizer with the actual ultra-low LR for fine-tuning all layers
            # FIX: Increased weight decay for stronger L2 regularization in Stage 2
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

            # FIX: Removed 'verbose=True' due to PyTorch version compatibility error
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=3
            )
            print(f"🎯 Full fine-tuning (all layers unfrozen) with very low LR: {lr}. Using ReduceLROnPlateau scheduler.")


        # Standard training loop execution
        model.train(); running_loss = correct = total = 0
        for x, y in tqdm(train_dl, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                if seg and out.dim()==3: out = out.unsqueeze(1)

                # CrossEntropyLoss requires target to be class indices (y)
                loss = criterion(out, y)

            scaler.scale(loss).backward()

            # FIX: Gradient Clipping to prevent exploding gradients (critical for ResNet18/VGG stability)
            scaler.unscale_(optimizer) # Must unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * x.size(0)

            if not seg:
                preds = torch.argmax(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        # [VALIDATION]
        model.eval(); val_loss = val_metric = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    if seg and out.dim()==3: out = out.unsqueeze(1)
                    loss = criterion(out, y if seg else y)
                val_loss += loss.item() * x.size(0)
                if seg: val_metric += iou(torch.sigmoid(out), y)
                else:
                    preds = torch.argmax(out, 1)
                    val_metric += (preds == y).sum().item()

        val_loss /= len(val_dl.dataset)

        if seg:
            val_iou = val_metric / len(val_dl)
            score = val_loss # Use loss for segmentation
            print(f"[{name}] Ep{epoch}: TrainLoss {running_loss/len(train_dl.dataset):.3f} | ValLoss {val_loss:.3f} | IoU {val_iou:.3f}")
            # Segmentation uses loss reduction, so lower is better (inf is starting value)
            improved = val_loss < best_score
        else:
            train_acc = 100 * correct / total
            val_acc = 100 * val_metric / len(val_dl.dataset)
            score = val_acc # Use accuracy for classification
            print(f"[{name}] Ep{epoch}: TrainLoss {running_loss/len(train_dl.dataset):.3f} (Acc {train_acc:.2f}%) | ValLoss {val_loss:.3f} | ValAcc {val_acc:.2f}%")
            # Classification uses accuracy increase, so higher is better (0 is starting value)
            improved = val_acc > best_score

        # Scheduler step using the validation score
        if seg:
            # CosineAnnealing for stable segmentation training
            scheduler.step()
        else:
             if epoch <= FEATURE_EXTRACTION_EPOCHS:
                 scheduler.step() # Cosine Annealing step for Stage 1
             else:
                 # ReduceLROnPlateau step, only after Stage 2 begins
                 # We track accuracy (score) for classification
                 scheduler.step(score)


        if improved:
            best_score = score
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            # Save based on best score
            save_name = f"{name}_best_acc.pt" if not seg else f"{name}_best_loss.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}. Best score: {best_score:.2f}")
            break

    end_time = time.time()
    print(f"✅ Training for {name} finished in {(end_time - start_time) / 60:.2f} minutes.")
    return best_score
