import os, glob, torch, warnings, time, numpy as np
import sys
from pathlib import Path
import cv2  # Required for Albumentations border mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# FIX: Import Pipeline from the fixed design module
from utils.pipeline import Pipeline
from utils.dataset import *
from utils.helpers import *

# [CONFIG]
DATA_ROOT = "dataset"
SAVE_ROOT = "weights"
CLS_SAVE, SEG_SAVE = [os.path.join(SAVE_ROOT, p) for p in ("classification_models", "segmentation_models")]
CLASSES = ["COVID", "Healthy", "Non-COVID"]
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")  # Directory containing train.csv, val.csv, test.csv
IMG_SIZE = 256  # Fixed input image size

# [MAIN]
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(CLS_SAVE, exist_ok=True)
    os.makedirs(SEG_SAVE, exist_ok=True)

    # --- Define Augmentation and standard Validation transforms ---
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- 1. CLASSIFICATION Transforms ---
    train_cls_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    val_cls_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # --- 2. SEGMENTATION Transforms (FIXED: Using A.Resize to ensure strict $256 \times 256$ consistency) ---
    # This replaces LongestMaxSize and PadIfNeeded to fix collation errors.

    train_seg_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE), # CRITICAL FIX: Forces strict size consistency
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], is_check_shapes=False, additional_targets={'mask': 'mask'})

    val_seg_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE), # CRITICAL FIX: Forces strict size consistency
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], is_check_shapes=False, additional_targets={'mask': 'mask'})


    # --- Data Loading and Splitting using separate transforms for train/val ---

    # Classification Datasets
    full_train_cls_ds = ClassificationDataset(DATA_ROOT, train_cls_transform)
    full_val_cls_ds = ClassificationDataset(DATA_ROOT, val_cls_transform)

    if len(full_train_cls_ds) == 0:
        print("Classification dataset is empty. Cannot proceed with training.")
        # We will continue if only segmentation data is present.

    # Classification Split
    n_cls = int(0.8*len(full_train_cls_ds));
    lengths = [n_cls, len(full_train_cls_ds) - n_cls]
    train_indices, val_indices = random_split(range(len(full_train_cls_ds)), lengths)

    cls_train = Subset(full_train_cls_ds, train_indices.indices)
    cls_val = Subset(full_val_cls_ds, val_indices.indices)

    # Segmentation Datasets
    full_train_seg_ds = SegmentationDataset(DATA_ROOT, train_seg_transform)
    full_val_seg_ds = SegmentationDataset(DATA_ROOT, val_seg_transform)

    # Segmentation Split
    if len(full_train_seg_ds) == 0:
        print("Segmentation dataset is empty. Skipping segmentation training.")
        seg_train, seg_val = [], []
    else:
        n_seg = int(0.8*len(full_train_seg_ds))
        seg_lengths = [n_seg, len(full_train_seg_ds)-n_seg]
        seg_train_indices, seg_val_indices = random_split(range(len(full_train_seg_ds)), seg_lengths)

        seg_train = Subset(full_train_seg_ds, seg_train_indices.indices)
        seg_val = Subset(full_val_seg_ds, seg_val_indices.indices)


    def make_loader(ds, bs):
        if not ds: return None
        # Increased num_workers to 4 for faster data loading
        return DataLoader(ds, bs, shuffle=True, num_workers=4, pin_memory=True)

    cls_train_dl, cls_val_dl = make_loader(cls_train, 16), make_loader(cls_val, 16)
    seg_train_dl, seg_val_dl = make_loader(seg_train, 8), make_loader(seg_val, 8)

    # Only run segmentation models as requested
    models_to_train = {
        "classification": ["ResNet50", "ResNet18", "VGG16", "VGG19"],
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
                if task == "classification":
                    # Now calling the new, fixed model loading function
                    model, cls_head_name = get_class_model(name)
                else:
                    model = get_seg_model(name)
                    cls_head_name = None # Not applicable for segmentation
            except Exception as e:
                print(f"Error loading model {name}: {e}. Skipping.")
                continue

            if task == "classification":
                train_dl, val_dl = cls_train_dl, cls_val_dl
            else:
                train_dl, val_dl = seg_train_dl, seg_val_dl

            if train_dl is None or val_dl is None:
                 print(f"Skipping training for {name}: Data loaders are not available.")
                 continue

            # Pass the classification head name only for classification tasks
            best = train(model,
                         train_dl,
                         val_dl,
                         device, epochs=20, lr=1e-6, name=name,
                         save_dir=CLS_SAVE if task=="classification" else SEG_SAVE,
                         seg=(task=="segmentation"),
                         cls_head_name=cls_head_name)
            results[name] = best

    # [Report]
    print("\n\n=============== Training Summary ===============")
    for k, v in sorted(results.items()):
        if "ResNet" in k or "VGG" in k:
            print(f"{k:<15}: {v:.2f}% Acc (Classification)")
        elif "Unet" in k:
            # Segmentation score is loss, so we report it as such, but the logic
            # for `best` might have captured IoU in a previous iteration if changed.
            # Assuming 'best' holds the loss (as per training loop)
            print(f"{k:<15}: {v:.4f} Loss (Segmentation)")

    best_cls = max([v for k, v in results.items() if "ResNet" in k or "VGG" in k], default=0)
    best_seg = min([v for k, v in results.items() if "Unet" in k], default=float('inf'))

    print("================================================")
    print(f"Best Classification Accuracy: {best_cls:.2f}%")
    print(f"Best Segmentation Loss: {best_seg:.4f}")
    print("================================================")