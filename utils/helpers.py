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
import torchvision.models as models
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
def add_dropout_to_fc(model, p=0.25, classes=CLASSES):
    """Adds a Dropout layer before the final classification layer for regularization."""
    if hasattr(model, 'fc'):
        # Standard for ResNets
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(num_features, len(classes))
        )
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        # Standard for VGGs
        new_classifier = list(model.classifier.children())[:-1] # Remove last layer
        last_in_features = list(model.classifier.children())[-1].in_features
        new_classifier.extend([
            nn.Dropout(p=p),
            nn.Linear(last_in_features, len(classes))
        ])
        model.classifier = nn.Sequential(*new_classifier)
    return model

# FIX: Modified to load pre-trained state_dict manually to bypass unsupported 'weights' argument
def get_class_model(name):
    from models.classification_models import ResNet, VGG
    name_lower = name.lower()
    
    # 1. Get the model class function
    if name_lower == "resnet18":
        ModelClass = ResNet.ResNet18
    elif name_lower == "resnet50":
        ModelClass = ResNet.ResNet50
    elif name_lower == "vgg16":
        ModelClass = VGG.VGG16
    elif name_lower == "vgg19":
        ModelClass = VGG.VGG19
    else:
        raise ValueError(f"Unknown classification model: {name}")

    # 2. Instantiate the model without the 'weights' argument (FIX for the error)
    # The actual num_classes will be adjusted later by add_dropout_to_fc
    model = ModelClass(num_classes=1000) 
    
    # 3. Load pre-trained weights if available (IMAGENET1K_V1)
    try:
        if name_lower in ["resnet18", "resnet50", "vgg16", "vgg19"]:
            print(f"Loading IMAGENET1K_V1 weights for {name}...")
            # Use torchvision.models directly to avoid namespace conflicts
            try:
                # Try new API with weights parameter (torchvision >= 0.13)
                if name_lower == "resnet18":
                    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                elif name_lower == "resnet50":
                    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                elif name_lower == "vgg16":
                    pretrained_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                elif name_lower == "vgg19":
                    pretrained_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # Fallback to old API with pretrained parameter (torchvision < 0.13)
                if name_lower == "resnet18":
                    pretrained_model = models.resnet18(pretrained=True)
                elif name_lower == "resnet50":
                    pretrained_model = models.resnet50(pretrained=True)
                elif name_lower == "vgg16":
                    pretrained_model = models.vgg16(pretrained=True)
                elif name_lower == "vgg19":
                    pretrained_model = models.vgg19(pretrained=True)

            # IMPROVED: Manually transfer weights layer by layer for better compatibility
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            
            # Filter out incompatible keys (fc layer and any mismatched layers)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update model dict with pretrained weights
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Count how many layers were successfully loaded
            loaded_count = len(pretrained_dict)
            total_count = len([k for k in model_dict.keys() if 'fc' not in k])
            print(f"✅ Successfully loaded {loaded_count}/{total_count} pre-trained layers for {name}")
            
            del pretrained_model
        
    except Exception as e:
        print(f"⚠️ Could not load pre-trained weights for {name} from torchvision: {e}. Model will train from scratch.")


    # 4. Replace the final layer with one suitable for 3 classes and add Dropout
    model = add_dropout_to_fc(model, p=0.25) 
    return model

def get_seg_model(name):
    from models.segmentation_models import ResnetUnet, AttentionUNet, R2U_Net, R2AttU_Net
    # FIX: Corrected keys to match expected class names
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
    
# [Metrics] (Unchanged)
def acc(logits, y): return (torch.argmax(logits, 1) == y).sum().item(), y.size(0)

def iou(pred, mask, t=0.5): 
    p = (pred > t).float(); inter=(p*mask).sum(); union=((p+mask)>0).float().sum()
    return (inter/(union+1e-7)).item()

# [Training] (Includes Fine-tuning logic and updated patience)
def train(model, train_dl, val_dl, device, epochs, lr, name, save_dir, seg=False):
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss() if seg else nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # IMPROVED: Better fine-tuning strategy for pre-trained classification models
    # For medical images, we need to fine-tune more layers than just the head
    if not seg:
        # Check if model has ResNet or VGG structure
        has_resnet_structure = hasattr(model, 'layer1') and hasattr(model, 'layer2')
        has_vgg_structure = hasattr(model, 'features')
        
        if has_resnet_structure or has_vgg_structure:
            # Progressive fine-tuning: freeze early layers, unfreeze later layers
            # This allows the model to adapt to medical images while preserving general features
            
            # Freeze all layers first
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze later layers for ResNet (layer3, layer4, and fc)
            if has_resnet_structure:
                # Unfreeze layer3 and layer4 (deeper features)
                for param in model.layer3.parameters():
                    param.requires_grad = True
                for param in model.layer4.parameters():
                    param.requires_grad = True
                # Always unfreeze the classification head
                if hasattr(model, 'fc'):
                    for param in model.fc.parameters():
                        param.requires_grad = True
                print(f"🎯 Fine-tuning ResNet: layer3, layer4, and fc with LR: {lr}")
            
            # Unfreeze later layers for VGG (last feature blocks and classifier)
            elif has_vgg_structure:
                # Unfreeze the last two feature blocks (more task-specific features)
                if hasattr(model, 'features'):
                    # VGG features are in a Sequential, unfreeze last 2 blocks
                    features_list = list(model.features.children())
                    if len(features_list) >= 2:
                        # Unfreeze last 2 blocks (roughly last 6-8 layers)
                        for module in features_list[-8:]:
                            for param in module.parameters():
                                param.requires_grad = True
                # Always unfreeze the classifier
                if hasattr(model, 'classifier'):
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                print(f"🎯 Fine-tuning VGG: last feature blocks and classifier with LR: {lr}")
            
            # Optimizer only targets the trainable (unfrozen) parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        else:
            # Full training for models without standard structure
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            print(f"🔥 Training all layers from scratch with LR: {lr}")
    else:
        # Full training for segmentation models
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"🔥 Training segmentation model with LR: {lr}")

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    
    best_score = 0 if not seg else float("inf")
    patience, patience_counter = 10, 0 

    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train(); running_loss = correct = total = 0
        for x, y in tqdm(train_dl, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                if seg and out.dim()==3: out = out.unsqueeze(1)
                loss = criterion(out, y if seg else y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * x.size(0)
            if not seg:
                preds = torch.argmax(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        scheduler.step()

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
            print(f"[{name}] Ep{epoch}: TrainLoss {running_loss/len(train_dl.dataset):.3f} | ValLoss {val_loss:.3f} | IoU {val_iou:.3f}")
            improved = val_loss < best_score
        else:
            val_acc = 100 * val_metric / len(val_dl.dataset)
            print(f"[{name}] Ep{epoch}: TrainLoss {running_loss/len(train_dl.dataset):.3f} | ValLoss {val_loss:.3f} | ValAcc {val_acc:.2f}%")
            improved = val_acc > best_score

        if improved:
            best_score = val_loss if seg else val_acc
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}. Best score: {best_score:.2f}")
            break
    
    end_time = time.time()
    print(f"✅ Training for {name} finished in {(end_time - start_time) / 60:.2f} minutes.")
    return best_score