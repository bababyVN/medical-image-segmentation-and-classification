import os, glob, torch, warnings, time, numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# FIX: Import Pipeline from the fixed design module
from design.pipeline import Pipeline 

# [CONFIG]
DATA_ROOT = "dataset"
SAVE_ROOT = "weights"
CLS_SAVE, SEG_SAVE = [os.path.join(SAVE_ROOT, p) for p in ("classification_models", "segmentation_models")]
CLASSES = ["COVID", "Healthy", "Non-COVID"]

# --- Model Definitions (With Pre-trained Weights and Dropout Fixes) ---
def add_dropout_to_fc(model, p=0.25):
    """Adds a Dropout layer before the final classification layer for regularization."""
    if hasattr(model, 'fc'):
        # Standard for ResNets
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(num_features, len(CLASSES))
        )
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        # Standard for VGGs
        new_classifier = list(model.classifier.children())[:-1] # Remove last layer
        last_in_features = list(model.classifier.children())[-1].in_features
        new_classifier.extend([
            nn.Dropout(p=p),
            nn.Linear(last_in_features, len(CLASSES))
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
            # Use torch.hub to safely load the pre-trained weights for the model architecture
            if "resnet" in name_lower:
                pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', name_lower, weights="IMAGENET1K_V1")
            elif "vgg" in name_lower:
                 # VGG model names need to be adjusted for torch.hub (e.g., vgg16_bn)
                 hub_name = name_lower + "_bn" if "vgg" in name_lower else name_lower
                 pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', hub_name, weights="IMAGENET1K_V1")

            # Load the state dict, ignoring the final classification layer (which has 1000 output features)
            # The 'strict=False' is used because the final layer sizes will mismatch (1000 vs 3)
            model.load_state_dict(pretrained_model.state_dict(), strict=False)
            del pretrained_model
        
    except Exception as e:
        print(f"⚠️ Could not load pre-trained weights for {name} from PyTorch Hub: {e}. Model will train from scratch.")


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
# --- END Model Definitions ---


# [Datasets handle] (Unchanged)
class ClassificationDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = [(p, i) for i, cls in enumerate(CLASSES)
                             for p in glob.glob(os.path.join(root, cls, "images", "*.png"))]
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self): return len(self.samples)

class SegmentationDataset(Dataset):
    def __init__(self, root, transform):
        self.pairs = [(os.path.join(root, c, "images", n),
                       os.path.join(root, c, "masks", n))
                     for c in CLASSES for n in os.listdir(os.path.join(root, c, "images"))
                     if n.endswith(".png") and os.path.exists(os.path.join(root, c, "masks", n))]
        self.transform = transform

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self): return len(self.pairs) # FIX: Removed nested len() call
    
# [Metrics] (Unchanged)
def acc(logits, y): return (torch.argmax(logits, 1) == y).sum().item(), y.size(0)

def iou(pred, mask, t=0.5): 
    p = (pred > t).float(); inter=(p*mask).sum(); union=((p+mask)>0).float().sum()
    return (inter/(union+1e-7)).item()

# [Training] (Includes Fine-tuning logic and updated patience)
def train(model, train_dl, val_dl, device, epochs, lr, name, save_dir, seg=False):
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss() if seg else nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # FIX: Fine-tuning logic for pre-trained classification models (essential for high accuracy)
    # Check if we successfully loaded pre-trained weights by seeing if any parameter is currently frozen.
    is_pretrained = False
    for param in model.parameters():
        if not param.requires_grad:
            is_pretrained = True
            break

    if not seg and is_pretrained:
        # If pre-trained weights were loaded, we only train the newly added classification head.
        # Ensure all layers are frozen initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final layer (where we applied Dropout and Linear layer)
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        # Optimizer only targets the trainable (unfrozen) parameters
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        print(f"🎯 Fine-tuning only the classification head with LR: {lr}")
    else:
        # Full training for segmentation/scratch classification models
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"🔥 Training all layers from scratch/unfrozen with LR: {lr}")

    
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

# [MAIN]
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create save directories
    os.makedirs(CLS_SAVE, exist_ok=True)
    os.makedirs(SEG_SAVE, exist_ok=True)

    # [Preprocessing]
    pipeline = Pipeline()
    cls_transform = pipeline.get_cls_transform()
    seg_transform = pipeline.get_seg_transform()

    # [Build datasets]
    cls_ds = ClassificationDataset(DATA_ROOT, cls_transform)
    seg_ds = SegmentationDataset(DATA_ROOT, seg_transform)

    n_cls = int(0.8*len(cls_ds)); n_seg = int(0.8*len(seg_ds))

    # FIX: Check if datasets are empty before creating random_split
    if len(cls_ds) == 0:
        print("🛑 Classification dataset is empty. Cannot proceed with training.")
        exit()
    if len(seg_ds) == 0:
        # If segmentation data is missing, we can still run classification, 
        # but skip segmentation training below.
        print("⚠️ Segmentation dataset is empty. Skipping segmentation training.")
        seg_train, seg_val = [], [] # Create empty lists to allow loop to run without errors
    else:
        cls_train, cls_val = random_split(cls_ds, [n_cls, len(cls_ds)-n_cls])
        seg_train, seg_val = random_split(seg_ds, [n_seg, len(seg_ds)-n_seg])


    def make_loader(ds, bs):
        # FIX: num_workers=0 to avoid PicklingError in Colab/Kaggle environments
        # Set persistent_workers to False as num_workers=0
        if not ds: return None # Return None if dataset is empty/missing
        return DataLoader(ds, bs, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)

    cls_train_dl, cls_val_dl = make_loader(cls_train, 16), make_loader(cls_val, 16)
    seg_train_dl, seg_val_dl = make_loader(seg_train, 8), make_loader(seg_val, 8)

    # [Models to train]
    models_to_train = {
        "classification": ["ResNet18", "ResNet50", "VGG16", "VGG19"],
        "segmentation": ["ResNetUnet", "AttentionUNet", "R2Unet", "R2AttUnet"] 
    }

    results = {}

    for task, model_names in models_to_train.items():
        if task == "segmentation" and not seg_train_dl:
            print("\n--- Skipping Segmentation Training (No data found) ---")
            continue
            
        for name in model_names:
            print(f"\n--- Training {task} model: {name} ---")
            
            try:
                model = get_class_model(name) if task == "classification" else get_seg_model(name)
            except Exception as e:
                print(f"❌ Error loading model {name}: {e}. Skipping.")
                continue

            # Select correct data loaders based on task, ensuring they are not None
            if task == "classification":
                train_dl, val_dl = cls_train_dl, cls_val_dl
            else: # segmentation
                train_dl, val_dl = seg_train_dl, seg_val_dl
                
            if train_dl is None or val_dl is None:
                 print(f"🛑 Skipping training for {name}: Data loaders are not available.")
                 continue

            best = train(model,
                         train_dl,
                         val_dl,
                         device, epochs=20, lr=5e-5, name=name, 
                         save_dir=CLS_SAVE if task=="classification" else SEG_SAVE,
                         seg=(task=="segmentation"))
            results[name] = best

    # [Report]
    print("\n\n=============== Training Summary ===============")
    for k, v in sorted(results.items()):
        if "ResNet" in k or "VGG" in k:
            print(f"{k:<15}: {v:.2f}% Acc (Classification)")
        elif "Unet" in k:
            print(f"{k:<15}: {v:.4f} IoU (Segmentation)")
            
    best_cls = max([v for k, v in results.items() if "ResNet" in k or "VGG" in k], default=0)
    best_seg = min([v for k, v in results.items() if "Unet" in k], default=float('inf'))
    
    print("================================================")
    print(f"Best Classification Accuracy: {best_cls:.2f}%")
    print(f"Best Segmentation IoU Loss: {best_seg:.4f}")
    print("================================================")
