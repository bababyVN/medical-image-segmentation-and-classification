import os
import sys
from pathlib import Path
import cv2
import warnings
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import ClassificationDataset, SegmentationDataset
from utils.helpers import get_class_model, get_seg_model
from models.classification_models.CLIP import CLIPClassifier, DEFAULT_TEXT_PROMPTS
from models.segmentation_models.CLIPSeg import (
    CLIPSegForSegmentation,
    DEFAULT_TEXT_PROMPT,
)

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

os.makedirs("results", exist_ok=True)
# [CONFIG]
DATA_ROOT = "dataset"
WEIGHTS_ROOT = "weights"
CLS_WEIGHTS_DIR = os.path.join(WEIGHTS_ROOT, "classification_models")
SEG_WEIGHTS_DIR = os.path.join(WEIGHTS_ROOT, "segmentation_models")
CLASSES = ["COVID", "Healthy", "Non-COVID"]
IMG_SIZE = 256


# --- METRICS FOR CLASSIFICATION ---
def calculate_classification_metrics(all_preds, all_labels):
    """
    Calculate accuracy, precision, recall, and F1 score for classification.

    Args:
        all_preds: List or array of predicted class indices
        all_labels: List or array of true class indices

    Returns:
        dict: Dictionary containing all metrics
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        all_labels, all_preds, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        all_labels, all_preds, average=None, zero_division=0
    )
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy * 100,  # Convert to percentage
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "precision_per_class": precision_per_class * 100,
        "recall_per_class": recall_per_class * 100,
        "f1_per_class": f1_per_class * 100,
        "confusion_matrix": cm,
    }


# --- METRICS FOR SEGMENTATION ---
def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
        pred: Predicted mask (after sigmoid)
        target: Ground truth mask
        threshold: Threshold for binarization

    Returns:
        float: IoU score
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = ((pred_binary + target_binary) > 0).float().sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    """
    Calculate Dice coefficient for binary segmentation.

    Args:
        pred: Predicted mask (after sigmoid)
        target: Ground truth mask
        threshold: Threshold for binarization

    Returns:
        float: Dice score
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + 1e-7) / (
        pred_binary.sum() + target_binary.sum() + 1e-7
    )

    return dice.item()


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy for segmentation.

    Args:
        pred: Predicted mask (after sigmoid)
        target: Ground truth mask
        threshold: Threshold for binarization

    Returns:
        float: Pixel accuracy
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()

    return (correct / total).item()


def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """
    Calculate all segmentation metrics.

    Args:
        pred: Predicted mask (after sigmoid)
        target: Ground truth mask
        threshold: Threshold for binarization

    Returns:
        dict: Dictionary containing all metrics
    """
    iou = calculate_iou(pred, target, threshold)
    dice = calculate_dice(pred, target, threshold)
    pixel_acc = calculate_pixel_accuracy(pred, target, threshold)

    # Calculate precision and recall for segmentation
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return {
        "iou": iou * 100,
        "dice": dice * 100,
        "pixel_accuracy": pixel_acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


# --- TEST FUNCTIONS ---
def test_classification_model(model, test_loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"Testing Classification Model: {model_name}")
    print(f"{'='*60}")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics = calculate_classification_metrics(all_preds, all_labels)

    # Print results
    print(f"\n{model_name} Test Results:")
    print(f"{'-'*60}")
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1 Score:  {metrics['f1']:.2f}%")

    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASSES):
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]:.2f}%")
        print(f"  Recall:    {metrics['recall_per_class'][i]:.2f}%")
        print(f"  F1 Score:  {metrics['f1_per_class'][i]:.2f}%")

    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'':>12}".join([f"{c:>12}" for c in CLASSES]))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"{CLASSES[i]:<12}" + "".join([f"{val:>12}" for val in row]))

    print(f"{'='*60}\n")

    return metrics


def test_segmentation_model(model, test_loader, device, model_name):
    model.eval()

    total_iou = 0
    total_dice = 0
    total_pixel_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_samples = 0

    print(f"\n{'='*60}")
    print(f"Testing Segmentation Model: {model_name}")
    print(f"{'='*60}")

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)

            # Handle dimension issues
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)

            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)

            # Calculate metrics for each sample in the batch
            for i in range(outputs.size(0)):
                metrics = calculate_segmentation_metrics(outputs[i], masks[i])
                total_iou += metrics["iou"]
                total_dice += metrics["dice"]
                total_pixel_acc += metrics["pixel_accuracy"]
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_f1 += metrics["f1"]
                num_samples += 1

    # Average metrics
    avg_metrics = {
        "iou": total_iou / num_samples,
        "dice": total_dice / num_samples,
        "pixel_accuracy": total_pixel_acc / num_samples,
        "precision": total_precision / num_samples,
        "recall": total_recall / num_samples,
        "f1": total_f1 / num_samples,
    }

    # Print results
    print(f"\n{model_name} Test Results:")
    print(f"{'-'*60}")
    print(f"IoU (Jaccard):     {avg_metrics['iou']:.2f}%")
    print(f"Dice Coefficient:  {avg_metrics['dice']:.2f}%")
    print(f"Pixel Accuracy:    {avg_metrics['pixel_accuracy']:.2f}%")
    print(f"Precision:         {avg_metrics['precision']:.2f}%")
    print(f"Recall:            {avg_metrics['recall']:.2f}%")
    print(f"F1 Score:          {avg_metrics['f1']:.2f}%")
    print(f"{'='*60}\n")

    return avg_metrics


def test_clip_classification_model(
    model, test_dataset, device, model_name, batch_size=16
):
    """
    Test CLIP classification model with its own preprocessing.

    Args:
        model: The CLIP classification model to test
        test_dataset: Test dataset (used to get image paths and labels)
        device: Device to run on (cuda/cpu)
        model_name: Name of the model for display
        batch_size: Batch size for testing

    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"Testing CLIP Classification Model: {model_name}")
    print(f"{'='*60}")

    # Process images individually due to CLIP's special preprocessing
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc=f"Testing {model_name}"):
            image, label = test_dataset[idx]

            # Convert tensor back to PIL Image for CLIP processor
            # Denormalize if needed (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            # Convert to PIL Image
            from PIL import Image

            image_pil = Image.fromarray(
                (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            # Preprocess with CLIP processor
            pixel_values = model.processor(images=image_pil, return_tensors="pt")[
                "pixel_values"
            ].to(device)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().item())
            all_labels.append(label if isinstance(label, int) else label.item())

    # Calculate metrics
    metrics = calculate_classification_metrics(all_preds, all_labels)

    # Print results
    print(f"\n{model_name} Test Results:")
    print(f"{'-'*60}")
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1 Score:  {metrics['f1']:.2f}%")

    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASSES):
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]:.2f}%")
        print(f"  Recall:    {metrics['recall_per_class'][i]:.2f}%")
        print(f"  F1 Score:  {metrics['f1_per_class'][i]:.2f}%")

    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'':>12}".join([f"{c:>12}" for c in CLASSES]))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"{CLASSES[i]:<12}" + "".join([f"{val:>12}" for val in row]))

    print(f"{'='*60}\n")

    return metrics


def test_clipseg_segmentation_model(model, test_dataset, device, model_name):
    """
    Test CLIPSeg segmentation model with its own preprocessing.

    Args:
        model: The CLIPSeg segmentation model to test
        test_dataset: Test dataset (used to get image paths and masks)
        device: Device to run on (cuda/cpu)
        model_name: Name of the model for display

    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()

    total_iou = 0
    total_dice = 0
    total_pixel_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_samples = 0

    print(f"\n{'='*60}")
    print(f"Testing CLIPSeg Segmentation Model: {model_name}")
    print(f"{'='*60}")

    # Process images individually due to CLIPSeg's special preprocessing
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc=f"Testing {model_name}"):
            image, mask = test_dataset[idx]

            # Convert tensor back to PIL Image for CLIPSeg processor
            # Denormalize if needed (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            # Convert to PIL Image
            from PIL import Image

            image_pil = Image.fromarray(
                (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            # Preprocess with CLIPSeg processor
            inputs = model.processor(
                text=[model.text_prompt],
                images=image_pil,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )

            # Apply sigmoid and resize to match mask size
            outputs = torch.sigmoid(outputs)
            if outputs.shape[-2:] != mask.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs.unsqueeze(0),
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Ensure mask has proper dimensions
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.to(device)

            # Calculate metrics
            metrics = calculate_segmentation_metrics(outputs.squeeze(), mask.squeeze())
            total_iou += metrics["iou"]
            total_dice += metrics["dice"]
            total_pixel_acc += metrics["pixel_accuracy"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_f1 += metrics["f1"]
            num_samples += 1

    # Average metrics
    avg_metrics = {
        "iou": total_iou / num_samples,
        "dice": total_dice / num_samples,
        "pixel_accuracy": total_pixel_acc / num_samples,
        "precision": total_precision / num_samples,
        "recall": total_recall / num_samples,
        "f1": total_f1 / num_samples,
    }

    # Print results
    print(f"\n{model_name} Test Results:")
    print(f"{'-'*60}")
    print(f"IoU (Jaccard):     {avg_metrics['iou']:.2f}%")
    print(f"Dice Coefficient:  {avg_metrics['dice']:.2f}%")
    print(f"Pixel Accuracy:    {avg_metrics['pixel_accuracy']:.2f}%")
    print(f"Precision:         {avg_metrics['precision']:.2f}%")
    print(f"Recall:            {avg_metrics['recall']:.2f}%")
    print(f"F1 Score:          {avg_metrics['f1']:.2f}%")
    print(f"{'='*60}\n")

    return avg_metrics


def test_all_models(device="cuda", batch_size=16):
    """
    Test all trained models (both classification and segmentation).

    Args:
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for testing

    Returns:
        dict: Dictionary containing all test results
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Test transforms (same as validation transforms)
    test_cls_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    test_seg_transform = A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        is_check_shapes=False,
        additional_targets={"mask": "mask"},
    )

    results = {}

    # --- TEST CLASSIFICATION MODELS ---
    classification_models = {
        "ResNet18": "ResNet18_best_acc.pt",
        "ResNet50": "ResNet50_best_acc.pt",
        "VGG16": "VGG16_best_acc.pt",
        "VGG19": "VGG19_best_acc.pt",
        "CLIP": "CLIP_best_acc.pt",
    }

    # Load classification test dataset
    try:
        test_cls_dataset = ClassificationDataset(
            DATA_ROOT, test_cls_transform, split="test"
        )
        test_cls_loader = DataLoader(
            test_cls_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(f"\n[INFO] Classification Test Dataset: {len(test_cls_dataset)} samples")

        for model_name, weight_file in classification_models.items():
            weight_path = os.path.join(CLS_WEIGHTS_DIR, weight_file)

            if not os.path.exists(weight_path):
                print(f"\n[WARNING] Weights not found for {model_name}: {weight_path}")
                print(f"Skipping {model_name}...")
                continue

            try:
                # Special handling for CLIP
                if model_name == "CLIP":
                    # Load CLIP model
                    model = CLIPClassifier(
                        model_name="openai/clip-vit-base-patch32",
                        num_classes=len(CLASSES),
                        text_prompts=DEFAULT_TEXT_PROMPTS,
                        device=device,
                    )

                    # Load trained weights
                    model.clip_model.load_state_dict(
                        torch.load(weight_path, map_location=device, weights_only=True)
                    )
                    model = model.to(device)

                    # Test the model with special CLIP test function
                    metrics = test_clip_classification_model(
                        model, test_cls_dataset, device, model_name
                    )
                    results[model_name] = metrics
                else:
                    # Load standard model architecture
                    model, _ = get_class_model(model_name)

                    # Load trained weights
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    model = model.to(device)

                    # Test the model
                    metrics = test_classification_model(
                        model, test_cls_loader, device, model_name
                    )
                    results[model_name] = metrics

                # Clean up
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[ERROR] Failed to test {model_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    except FileNotFoundError as e:
        print(f"\n[WARNING] Classification test dataset not found: {e}")
        print("Skipping classification model testing...")

    # --- TEST SEGMENTATION MODELS ---
    segmentation_models = {
        "ResNetUnet": "ResNetUnet_best_loss.pt",
        "AttentionUNet": "AttentionUNet_best_loss.pt",
        "R2Unet": "R2Unet_best_loss.pt",
        "R2AttUnet": "R2AttUnet_best_loss.pt",
        "CLIPSeg": "CLIPSeg_best_loss.pt",
    }

    # Load segmentation test dataset
    try:
        test_seg_dataset = SegmentationDataset(
            DATA_ROOT, test_seg_transform, split="test"
        )

        if len(test_seg_dataset) == 0:
            print(
                f"\n[WARNING] Segmentation test dataset is empty. Skipping segmentation testing."
            )
        else:
            test_seg_loader = DataLoader(
                test_seg_dataset,
                batch_size=batch_size // 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            print(
                f"\n[INFO] Segmentation Test Dataset: {len(test_seg_dataset)} samples"
            )

            for model_name, weight_file in segmentation_models.items():
                weight_path = os.path.join(SEG_WEIGHTS_DIR, weight_file)

                if not os.path.exists(weight_path):
                    print(
                        f"\n[WARNING] Weights not found for {model_name}: {weight_path}"
                    )
                    print(f"Skipping {model_name}...")
                    continue

                try:
                    # Special handling for CLIPSeg
                    if model_name == "CLIPSeg":
                        # Load CLIPSeg model
                        model = CLIPSegForSegmentation(
                            model_name="CIDAS/clipseg-rd64-refined",
                            text_prompt=DEFAULT_TEXT_PROMPT,
                            device=device,
                        )

                        # Load trained weights
                        model.clipseg_model.load_state_dict(
                            torch.load(weight_path, map_location=device)
                        )
                        model = model.to(device)

                        # Test the model with special CLIPSeg test function
                        metrics = test_clipseg_segmentation_model(
                            model, test_seg_dataset, device, model_name
                        )
                        results[model_name] = metrics
                    else:
                        # Load standard model architecture
                        model = get_seg_model(model_name)

                        # Load trained weights
                        model.load_state_dict(
                            torch.load(weight_path, map_location=device)
                        )
                        model = model.to(device)

                        # Test the model
                        metrics = test_segmentation_model(
                            model, test_seg_loader, device, model_name
                        )
                        results[model_name] = metrics

                    # Clean up
                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[ERROR] Failed to test {model_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

    except FileNotFoundError as e:
        print(f"\n[WARNING] Segmentation test dataset not found: {e}")
        print("Skipping segmentation model testing...")

    return results


def print_summary(results):
    """
    Print a summary of all test results.

    Args:
        results: Dictionary containing all test results
    """
    if not results:
        print("\n[INFO] No test results to display.")
        return

    print("\n" + "=" * 80)
    print(" " * 25 + "TEST RESULTS SUMMARY")
    print("=" * 80)

    # Classification results
    cls_models = [
        m for m in ["ResNet18", "ResNet50", "VGG16", "VGG19", "CLIP"] if m in results
    ]
    if cls_models:
        print("\nCLASSIFICATION MODELS:")
        print("-" * 80)
        print(
            f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}"
        )
        print("-" * 80)
        for model in cls_models:
            metrics = results[model]
            print(
                f"{model:<20} {metrics['accuracy']:>10.2f}% {metrics['precision']:>10.2f}% "
                f"{metrics['recall']:>10.2f}% {metrics['f1']:>10.2f}%"
            )

        # Best classification model
        best_cls_model = max(cls_models, key=lambda x: results[x]["accuracy"])
        print(
            f"\n🏆 Best Classification Model: {best_cls_model} "
            f"(Accuracy: {results[best_cls_model]['accuracy']:.2f}%)"
        )

    # Segmentation results
    seg_models = [
        m
        for m in ["ResNetUnet", "AttentionUNet", "R2Unet", "R2AttUnet", "CLIPSeg"]
        if m in results
    ]
    if seg_models:
        print("\n\nSEGMENTATION MODELS:")
        print("-" * 80)
        print(
            f"{'Model':<20} {'IoU':<10} {'Dice':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}"
        )
        print("-" * 80)
        for model in seg_models:
            metrics = results[model]
            print(
                f"{model:<20} {metrics['iou']:>8.2f}% {metrics['dice']:>8.2f}% "
                f"{metrics['precision']:>10.2f}% {metrics['recall']:>10.2f}% {metrics['f1']:>10.2f}%"
            )

        # Best segmentation model
        best_seg_model = max(seg_models, key=lambda x: results[x]["dice"])
        print(
            f"\n🏆 Best Segmentation Model: {best_seg_model} "
            f"(Dice: {results[best_seg_model]['dice']:.2f}%)"
        )

    print("=" * 80 + "\n")


def save_results_to_csv(
    results,
    cls_output_path="results/classification_test_results.csv",
    seg_output_path="results/segmentation_test_results.csv",
):
    """
    Save test results to separate CSV files for classification and segmentation.

    Args:
        results: Dictionary containing all test results
        cls_output_path: Path to save classification results CSV file
        seg_output_path: Path to save segmentation results CSV file
    """
    if not results:
        print("\n[INFO] No results to save.")
        return

    # Separate classification and segmentation results
    cls_models = [
        k
        for k in results.keys()
        if any(x in k for x in ["ResNet18", "ResNet50", "VGG", "CLIP"])
        and "Seg" not in k
    ]
    seg_models = [
        k for k in results.keys() if "Unet" in k or "UNet" in k or "CLIPSeg" in k
    ]

    # Save classification results
    if cls_models:
        cls_data = []
        for model_name in cls_models:
            metrics = results[model_name]
            row = {"Model": model_name}
            row.update(metrics)

            # Remove non-scalar values like confusion matrix and per-class metrics
            if "confusion_matrix" in row:
                del row["confusion_matrix"]
            if "precision_per_class" in row:
                del row["precision_per_class"]
            if "recall_per_class" in row:
                del row["recall_per_class"]
            if "f1_per_class" in row:
                del row["f1_per_class"]

            cls_data.append(row)

        cls_df = pd.DataFrame(cls_data)
        cls_df.to_csv(cls_output_path, index=False)
        print(f"\n[INFO] Classification results saved to: {cls_output_path}")
    else:
        print("\n[INFO] No classification results to save.")

    # Save segmentation results
    if seg_models:
        seg_data = []
        for model_name in seg_models:
            metrics = results[model_name]
            row = {"Model": model_name}
            row.update(metrics)

            seg_data.append(row)

        seg_df = pd.DataFrame(seg_data)
        seg_df.to_csv(seg_output_path, index=False)
        print(f"[INFO] Segmentation results saved to: {seg_output_path}")
    else:
        print("\n[INFO] No segmentation results to save.")


if __name__ == "__main__":
    # Run tests
    print("\n" + "=" * 80)
    print(" " * 20 + "MODEL TESTING UTILITY")
    print("=" * 80)

    # Test all models
    results = test_all_models(device="cuda", batch_size=16)

    # Print summary
    print_summary(results)

    # Save results to separate CSV files
    save_results_to_csv(
        results,
        cls_output_path="classification_test_results.csv",
        seg_output_path="segmentation_test_results.csv",
    )

    print("\n[INFO] Testing complete!")
