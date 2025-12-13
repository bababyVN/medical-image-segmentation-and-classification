import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
import numpy as np
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# Import from utils
from utils.dataset import CLIPSegDataset

DATA_ROOT = "dataset"
CLASSES = ["COVID", "Healthy", "Non-COVID"]
BATCH_SIZE = 8  # CLIPSeg is memory-intensive
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5  # Lower LR for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "weights/segmentation_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# Text prompt for lung segmentation
TEXT_PROMPT = "lungs"


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + 1e-7) / (
        pred_binary.sum() + target_binary.sum() + 1e-7
    )

    return dice.item()


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = ((pred_binary + target_binary) > 0).float().sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()


class CLIPSegFineTuner(nn.Module):
    """
    Wrapper for fine-tuning CLIPSeg on segmentation tasks.
    """

    def __init__(self, clipseg_model, device):
        super(CLIPSegFineTuner, self).__init__()
        self.clipseg_model = clipseg_model
        self.device = device

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clipseg_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def train_clipseg(model, train_dl, val_dl, device, epochs, lr, save_dir, text_prompt):
    model = model.to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # Fine-tune only the decoder (freeze encoder)
    trainable_params = []
    for name, param in model.clipseg_model.named_parameters():
        if "decoder" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0.0
    patience, patience_counter = 10, 0

    print(f"\n{'='*80}")
    print(f"Starting CLIPSeg Fine-Tuning")
    print(f"{'='*80}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"{'='*80}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # [TRAINING]
        model.train()
        running_loss = running_dice = running_iou = 0
        total_samples = 0

        for batch in tqdm(
            train_dl, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False
        ):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                batch_dice = sum(
                    [calculate_dice(probs[i], labels[i]) for i in range(probs.size(0))]
                ) / probs.size(0)
                batch_iou = sum(
                    [calculate_iou(probs[i], labels[i]) for i in range(probs.size(0))]
                ) / probs.size(0)

            running_loss += loss.item() * pixel_values.size(0)
            running_dice += batch_dice * pixel_values.size(0)
            running_iou += batch_iou * pixel_values.size(0)
            total_samples += pixel_values.size(0)

        train_loss = running_loss / total_samples
        train_dice = running_dice / total_samples
        train_iou = running_iou / total_samples

        # [VALIDATION]
        model.eval()
        val_loss = val_dice_total = val_iou_total = 0
        val_samples = 0

        with torch.no_grad():
            for batch in tqdm(
                val_dl, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False
            ):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                logits = model(pixel_values, input_ids, attention_mask)
                loss = criterion(logits, labels)

                # Calculate metrics
                probs = torch.sigmoid(logits)
                batch_dice = sum(
                    [calculate_dice(probs[i], labels[i]) for i in range(probs.size(0))]
                ) / probs.size(0)
                batch_iou = sum(
                    [calculate_iou(probs[i], labels[i]) for i in range(probs.size(0))]
                ) / probs.size(0)

                val_loss += loss.item() * pixel_values.size(0)
                val_dice_total += batch_dice * pixel_values.size(0)
                val_iou_total += batch_iou * pixel_values.size(0)
                val_samples += pixel_values.size(0)

        val_loss /= val_samples
        val_dice = val_dice_total / val_samples
        val_iou = val_iou_total / val_samples

        # Update scheduler
        scheduler.step()

        # Print epoch summary
        print(
            f"[CLIPSeg] Ep{epoch}: TrainLoss {train_loss:.3f} (Dice {train_dice*100:.2f}%, IoU {train_iou*100:.2f}%) | "
            f"ValLoss {val_loss:.3f} | ValDice {val_dice*100:.2f}% | ValIoU {val_iou*100:.2f}%"
        )

        # Save best model based on Dice
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "CLIPSeg_best_loss.pt")
            torch.save(model.clipseg_model.state_dict(), save_path)
            print(f"Saved best model: {save_path} (Dice: {val_dice*100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Dice: {best_dice*100:.2f}%")
            break

    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Best Validation Dice: {best_dice*100:.2f}%")

    return best_dice


def finetune(
    data_root=DATA_ROOT,
    text_prompt=TEXT_PROMPT,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    save_dir=SAVE_DIR,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # Load CLIPSeg model and processor
    print("[INFO] Loading CLIPSeg model...")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    clipseg_model = clipseg_model.to(device)

    # Create fine-tuning wrapper
    model = CLIPSegFineTuner(clipseg_model, device)
    model = model.to(device)

    print("[INFO] Model loaded successfully!")

    # Create datasets
    print("[INFO] Loading datasets...")
    train_dataset = CLIPSegDataset(data_root, processor, text_prompt, split="train")
    val_dataset = CLIPSegDataset(data_root, processor, text_prompt, split="val")

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
    best_dice = train_clipseg(
        model, train_loader, val_loader, device, epochs, lr, save_dir, text_prompt
    )

    return best_dice


if __name__ == "__main__":
    # Run fine-tuning with default parameters
    best_dice = finetune()
    print(f"\n{'='*80}")
    print(f"CLIPSeg Fine-Tuning Complete!")
    print(f"Best Validation Dice: {best_dice*100:.2f}%")
    print(f"{'='*80}")
