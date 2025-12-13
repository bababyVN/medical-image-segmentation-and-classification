import os
import sys
from pathlib import Path
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

DATA_ROOT = "dataset"
CLASSES = ["COVID", "Healthy", "Non-COVID"]


def get_image_ids_by_class(data_root: str = DATA_ROOT, classes: list = CLASSES):
    image_ids = []
    labels = []

    for cls in classes:
        img_path = os.path.join(data_root, cls, "images")
        if not os.path.exists(img_path):
            print(f"Warning: Image directory not found: {img_path}")
            continue

        img_files = glob.glob(os.path.join(img_path, "*.png"))

        for img_file in img_files:
            image_id = os.path.splitext(os.path.basename(img_file))[0]
            image_ids.append(image_id)
            labels.append(cls)

    return image_ids, labels


def split_dataset_stratified(
    data_root: str = DATA_ROOT,
    classes: list = CLASSES,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    print("Collecting image IDs...")
    image_ids, labels = get_image_ids_by_class(data_root, classes)

    if len(image_ids) == 0:
        raise ValueError("No images found in the dataset!")

    print(f"Total images found: {len(image_ids)}")

    image_ids = np.array(image_ids)
    labels = np.array(labels)

    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        image_ids,
        labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_state,
    )

    val_size = val_ratio / temp_ratio
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids,
        temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state,
    )

    print(f"\nSplit summary:")
    print(f"Train: {len(train_ids)} images ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"Validation: {len(val_ids)} images ({len(val_ids)/len(image_ids)*100:.1f}%)")
    print(f"Test: {len(test_ids)} images ({len(test_ids)/len(image_ids)*100:.1f}%)")

    # Print class distribution for each split
    print("\nClass distribution:")
    for split_name, split_ids, split_labels in [
        ("Train", train_ids, train_labels),
        ("Validation", val_ids, val_labels),
        ("Test", test_ids, test_labels),
    ]:
        print(f"\n{split_name}:")
        for cls in classes:
            count = np.sum(split_labels == cls)
            percentage = count / len(split_labels) * 100 if len(split_labels) > 0 else 0
            print(f"  {cls}: {count} ({percentage:.1f}%)")

    return train_ids, val_ids, test_ids, train_labels, val_labels, test_labels


def save_splits_to_csv(
    train_ids,
    val_ids,
    test_ids,
    train_labels,
    val_labels,
    test_labels,
    output_dir: str = ".",
):
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrames with both image IDs and classes
    train_df = pd.DataFrame({"id": train_ids, "class": train_labels})
    val_df = pd.DataFrame({"id": val_ids, "class": val_labels})
    test_df = pd.DataFrame({"id": test_ids, "class": test_labels})

    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nCSV files saved:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")


if __name__ == "__main__":
    # Split the dataset
    (
        train_ids,
        val_ids,
        test_ids,
        train_labels,
        val_labels,
        test_labels,
    ) = split_dataset_stratified(
        data_root=DATA_ROOT,
        classes=CLASSES,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )

    # Save to CSV files
    save_splits_to_csv(
        train_ids,
        val_ids,
        test_ids,
        train_labels,
        val_labels,
        test_labels,
        output_dir=PROJECT_ROOT / "dataset" / "splits",
    )
