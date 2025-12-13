import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Import from utils
from utils.pipeline import Pipeline
from utils.dataset import *
from utils.helpers import *

DATA_ROOT = "dataset"
CLASSES = ["COVID", "Healthy", "Non-COVID"]
IMG_SIZE = 224  # CLIP default size
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-6
WARMUP_EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "weights/classification_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# Text prompts for each class
# These will be used as class embeddings
text_prompts = [
    "a chest x-ray image showing COVID-19 pneumonia with ground-glass opacities",
    "a healthy normal chest x-ray image with clear lung fields",
    "a chest x-ray image showing non-COVID pneumonia infiltrates",
]


class CLIPFineTuner(nn.Module):
    """
    Wrapper for fine-tuning CLIP on classification tasks.
    """

    def __init__(self, clip_model, text_prompts, processor, device):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.device = device

        with torch.no_grad():
            text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_outputs = clip_model.get_text_features(**text_inputs)
            # Normalize text embeddings
            self.text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

    def forward(self, images):
        # Get image features
        image_features = self.clip_model.get_image_features(pixel_values=images)

        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity (logits)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.T

        return logits


def train_clip(
    model, train_dl, val_dl, device, epochs, lr, save_dir, text_prompts, classes=CLASSES
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Fine-tune only the vision encoder (freeze text encoder)
    trainable_params = []
    for name, param in model.clip_model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = 0.0
    patience, patience_counter = 10, 0

    print(f"\n{'='*80}")
    print(f"Starting CLIP Fine-Tuning")
    print(f"{'='*80}")
    print(f"Text prompts:")
    for i, prompt in enumerate(text_prompts):
        print(f"  [{classes[i]}]: {prompt}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"{'='*80}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # [TRAINING]
        model.train()
        running_loss = correct = total = 0

        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # [VALIDATION]
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total

        # Update scheduler
        scheduler.step()

        # Print epoch summary
        print(
            f"[CLIP] Ep{epoch}: TrainLoss {train_loss:.3f} (Acc {train_acc:.2f}%) | "
            f"ValLoss {val_loss:.3f} | ValAcc {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_score:
            best_score = val_acc
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "CLIP_best_acc.pt")
            torch.save(model.clip_model.state_dict(), save_path)
            print(f"Saved best model: {save_path} (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best accuracy: {best_score:.2f}%")
            break

    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_score:.2f}%")

    return best_score


def finetune(
    data_root=DATA_ROOT,
    classes=CLASSES,
    text_prompts=text_prompts,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    save_dir=SAVE_DIR,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # Load CLIP model and processor
    print("[INFO] Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_model = clip_model.to(device)

    # Create fine-tuning wrapper
    model = CLIPFineTuner(clip_model, text_prompts, processor, device)
    model = model.to(device)

    print("[INFO] Model loaded successfully!")

    # Create datasets
    print("[INFO] Loading datasets...")
    train_dataset = CLIPDataset(data_root, processor, classes, split="train")
    val_dataset = CLIPDataset(data_root, processor, classes, split="val")

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Train the model
    best_acc = train_clip(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        lr,
        save_dir,
        text_prompts,
        classes,
    )

    return best_acc


if __name__ == "__main__":
    # Run fine-tuning with default parameters
    best_acc = finetune()
    print(f"\n{'='*80}")
    print(f"CLIP Fine-Tuning Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}")
