import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import pandas as pd
import numpy as np
from utils.pipeline import Pipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_ROOT = "dataset"
CLASSES = ["COVID", "Healthy", "Non-COVID"]

# [Datasets handle] (Fixed to handle Albumentations output and split datasets)
class ClassificationDataset(Dataset):
    def __init__(self, root, transform, split="train"):
        """
        Args:
            root: Root directory of the dataset (e.g., 'dataset')
            transform: Albumentations transform
            split: One of 'train', 'val', or 'test'
        """
        self.root = root
        self.transform = transform

        # Load split CSV file
        split_csv = os.path.join(root, "splits", f"{split}.csv")
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        df = pd.read_csv(split_csv)

        # Build samples list: (image_path, label_idx)
        self.samples = []
        for _, row in df.iterrows():
            img_id = row["id"]
            cls = row["class"]
            label_idx = CLASSES.index(cls)
            img_path = os.path.join(root, cls, "images", f"{img_id}.png")

            if os.path.exists(img_path):
                self.samples.append((img_path, label_idx))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # Ensure transform is callable (using Albumentations) and convert PIL to numpy array
        img_np = np.array(img)
        if self.transform:
            transformed = self.transform(image=img_np)
            img = transformed["image"]
        else:
            # Fallback to simple conversion if transform is None
            img = ToTensorV2()(image=img_np)["image"]
        return img, label

    def __len__(self):
        return len(self.samples)


class SegmentationDataset(Dataset):
    def __init__(self, root, transform, split="train"):
        """
        Args:
            root: Root directory of the dataset (e.g., 'dataset')
            transform: Albumentations transform
            split: One of 'train', 'val', or 'test'
        """
        self.root = root
        self.transform = transform

        # Load split CSV file
        split_csv = os.path.join(root, "splits", f"{split}.csv")
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        df = pd.read_csv(split_csv)

        # Build pairs list: (image_path, mask_path)
        self.pairs = []
        for _, row in df.iterrows():
            img_id = row["id"]
            cls = row["class"]
            img_path = os.path.join(root, cls, "images", f"{img_id}.png")
            mask_path = os.path.join(root, cls, "masks", f"{img_id}.png")

            # Only add if both image and mask exist
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        # FIX 1: Ensure mask has a channel dimension [H, W, 1] for consistent Albumentations/ToTensorV2 input
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)

        if self.transform:
            # This call requires the transform to be an A.Compose object or a compatible function
            transformed = self.transform(image=img_np, mask=mask_np)
            img = transformed["image"]
            mask = transformed["mask"]

            # FIX 2 (CRITICAL): Ensure the mask tensor is in [C, H, W] format expected by PyTorch.
            # Albumentations ToTensorV2 sometimes fails to permute 1-channel arrays correctly.
            # We check if the last dimension is 1 (the channel) and the first dimension is not 1.
            if mask.ndim == 3 and mask.shape[-1] == 1 and mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]

            # FIX 3: Ensure mask is float and normalized [0, 1] for BCEWithLogitsLoss
            # It's assumed the pixel values are 0 or 255 (binary mask)
            mask = mask.float() / 255.0
        else:
            # Fallback logic for when transform is None
            img = ToTensorV2()(image=img_np)["image"]
            mask = (
                ToTensorV2()(image=mask_np)["image"].float() / 255.0
            )  # Normalize mask

        return img, mask

    def __len__(self):
        return len(self.pairs)


class CLIPDataset(Dataset):
    """
    Dataset for CLIP classification that returns preprocessed images.
    """

    def __init__(self, root, processor, classes, split="train"):
        """
        Args:
            root: Root directory of the dataset
            processor: CLIPProcessor instance
            classes: List of class names
            split: One of 'train', 'val', or 'test'
        """
        self.root = root
        self.processor = processor
        self.classes = classes

        # Load split CSV file
        split_csv = os.path.join(root, "splits", f"{split}.csv")
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        df = pd.read_csv(split_csv)

        # Build samples list
        self.samples = []
        for _, row in df.iterrows():
            img_id = row["id"]
            cls = row["class"]
            label_idx = classes.index(cls)
            img_path = os.path.join(root, cls, "images", f"{img_id}.png")

            if os.path.exists(img_path):
                self.samples.append((img_path, label_idx))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # CLIP processor handles resizing and normalization
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim

        return pixel_values, label

    def __len__(self):
        return len(self.samples)


class CLIPSegDataset(torch.utils.data.Dataset):
    """
    Dataset for CLIPSeg fine-tuning on lung segmentation.
    """

    def __init__(self, root, processor, text_prompt, split="train"):
        self.root = root
        self.processor = processor
        self.text_prompt = text_prompt

        # Load split CSV file
        split_csv = os.path.join(root, "splits", f"{split}.csv")
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        df = pd.read_csv(split_csv)

        # Build pairs list: (image_path, mask_path)
        self.pairs = []
        for _, row in df.iterrows():
            img_id = row["id"]
            cls = row["class"]
            img_path = os.path.join(root, cls, "images", f"{img_id}.png")
            mask_path = os.path.join(root, cls, "masks", f"{img_id}.png")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Process image with text prompt
        inputs = self.processor(
            text=[self.text_prompt], images=[image], return_tensors="pt", padding=True
        )

        # Remove batch dimension
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Convert mask to tensor and resize to match CLIPSeg output size (352x352)
        # CLIPSeg outputs 352x352 by default
        mask_resized = torch.nn.functional.interpolate(
            torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0),
            size=(352, 352),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": mask_resized,
        }

    def __len__(self):
        return len(self.pairs)
