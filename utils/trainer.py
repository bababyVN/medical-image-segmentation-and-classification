import os, glob, torch, warnings, time, numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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

# [MAIN]
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create save directories
    os.makedirs(CLS_SAVE, exist_ok=True)
    os.makedirs(SEG_SAVE, exist_ok=True)

    # [Preprocessing]
    pipeline = Pipeline()
    # cls_transform = pipeline.get_cls_transform()
    # seg_transform = pipeline.get_seg_transform()
    cls_transform = pipeline.val_transform
    seg_transform = pipeline.val_transform

    # [CSV file paths]
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    test_csv = os.path.join(SPLITS_DIR, "test.csv")

    # Check if CSV files exist
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"🛑 CSV split files not found in {SPLITS_DIR}")
        print("Please run utils/split_dataset.py first to create train.csv, val.csv, and test.csv")
        exit()

    # [Build datasets from CSV files]
    print("Loading datasets from CSV files...")
    cls_train_ds = ClassificationDataset(DATA_ROOT, cls_transform, csv_path=train_csv, classes=CLASSES)
    cls_val_ds = ClassificationDataset(DATA_ROOT, cls_transform, csv_path=val_csv, classes=CLASSES)
    cls_test_ds = ClassificationDataset(DATA_ROOT, cls_transform, csv_path=test_csv, classes=CLASSES)
    
    seg_train_ds = SegmentationDataset(DATA_ROOT, seg_transform, csv_path=train_csv, classes=CLASSES)
    seg_val_ds = SegmentationDataset(DATA_ROOT, seg_transform, csv_path=val_csv, classes=CLASSES)
    seg_test_ds = SegmentationDataset(DATA_ROOT, seg_transform, csv_path=test_csv, classes=CLASSES)

    # Check if datasets are empty
    if len(cls_train_ds) == 0:
        print("🛑 Classification training dataset is empty. Cannot proceed with training.")
        exit()
    if len(seg_train_ds) == 0:
        print("⚠️ Segmentation training dataset is empty. Skipping segmentation training.")
        seg_train_ds, seg_val_ds, seg_test_ds = None, None, None

    print(f"\n📈 Final dataset sizes:")
    print(f"Classification - Train: {len(cls_train_ds)}, Val: {len(cls_val_ds)}, Test: {len(cls_test_ds)}")
    if seg_train_ds:
        print(f"Segmentation - Train: {len(seg_train_ds)}, Val: {len(seg_val_ds)}, Test: {len(seg_test_ds)}")

    def make_loader(ds, bs, shuffle=True):
        # FIX: num_workers=0 to avoid PicklingError in Colab/Kaggle environments
        # Set persistent_workers to False as num_workers=0
        if not ds or len(ds) == 0: return None # Return None if dataset is empty/missing
        return DataLoader(ds, bs, shuffle=shuffle, num_workers=0, pin_memory=True, persistent_workers=False)

    # Create data loaders
    cls_train_dl = make_loader(cls_train_ds, 16, shuffle=True)
    cls_val_dl = make_loader(cls_val_ds, 16, shuffle=False)
    cls_test_dl = make_loader(cls_test_ds, 16, shuffle=False)
    
    seg_train_dl = make_loader(seg_train_ds, 8, shuffle=True) if seg_train_ds else None
    seg_val_dl = make_loader(seg_val_ds, 8, shuffle=False) if seg_val_ds else None
    seg_test_dl = make_loader(seg_test_ds, 8, shuffle=False) if seg_test_ds else None

    # [Models to train]
    models_to_train = {
        "classification": ["ResNet18", "ResNet50", "VGG16", "VGG19"],
        "segmentation": ["ResNetUnet", "AttentionUNet", "R2Unet", "R2AttUnet"] 
    }

    results = {}

    for task, model_names in models_to_train.items():
        if task == "segmentation" and (not seg_train_dl or not seg_val_dl):
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
