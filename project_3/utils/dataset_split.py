import os
import shutil
import random
from pathlib import Path
from typing import Union

def split_and_copy_dataset(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Splits dataset in source_dir (organized by class) into train/val/test folders under target_dir.

    Parameters:
        source_dir (str or Path): Path to source directory with class folders.
        target_dir (str or Path): Path to destination directory for the split dataset.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        seed (int): Random seed for reproducibility.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    if target_dir.exists():
        raise FileExistsError(
            f"Target directory '{target_dir}' already exists. Please choose a different path or delete it first.")

    random.seed(seed)
    splits = ['train', 'val', 'test']

    for split in splits:
        (target_dir / split).mkdir(parents=True, exist_ok=True)

    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob("*.*"))
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            split_data = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }

            for split in splits:
                class_target_dir = target_dir / split / class_name
                class_target_dir.mkdir(parents=True, exist_ok=True)
                for img_path in split_data[split]:
                    shutil.copy(img_path, class_target_dir / img_path.name)

    print("Dataset successfully split and copied.")