from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_COLUMN_CANDIDATES = ("image_path", "image_id", "id_code")
LABEL_COLUMN_CANDIDATES = ("label", "diagnosis", "target")


def _infer_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    available = ", ".join(df.columns.tolist())
    raise ValueError(f"Could not find {kind} column. Available columns: {available}")


def _normalize_image_path(images_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return images_dir / candidate


def load_fundus_records(
    csv_path: str | Path,
    images_dir: str | Path,
    image_extension: str = ".png",
    binary_classification: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    records = pd.read_csv(csv_path)
    image_column = _infer_column(records, IMAGE_COLUMN_CANDIDATES, "image")
    label_column = _infer_column(records, LABEL_COLUMN_CANDIDATES, "label")

    prepared = records[[image_column, label_column]].copy()

    if image_column == "image_path":
        prepared["image_path"] = prepared[image_column].map(
            lambda value: str(_normalize_image_path(images_dir, str(value)))
        )
    else:
        prepared["image_path"] = prepared[image_column].map(
            lambda value: str(images_dir / f"{value}{image_extension}")
        )

    prepared["label"] = prepared[label_column].astype(int)
    if binary_classification:
        prepared["label"] = (prepared["label"] > 0).astype(int)

    return prepared[["image_path", "label"]]


def load_predefined_splits(
    train_csv_path: str | Path,
    train_images_dir: str | Path,
    valid_csv_path: str | Path,
    valid_images_dir: str | Path,
    test_csv_path: str | Path,
    test_images_dir: str | Path,
    image_extension: str = ".png",
    binary_classification: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_records = load_fundus_records(
        csv_path=train_csv_path,
        images_dir=train_images_dir,
        image_extension=image_extension,
        binary_classification=binary_classification,
    )
    valid_records = load_fundus_records(
        csv_path=valid_csv_path,
        images_dir=valid_images_dir,
        image_extension=image_extension,
        binary_classification=binary_classification,
    )
    test_records = load_fundus_records(
        csv_path=test_csv_path,
        images_dir=test_images_dir,
        image_extension=image_extension,
        binary_classification=binary_classification,
    )

    return train_records, valid_records, test_records


def split_records(
    records: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_holdout = val_size + test_size
    if total_holdout <= 0 or total_holdout >= 1:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    train_records, holdout_records = train_test_split(
        records,
        test_size=total_holdout,
        stratify=records["label"],
        random_state=random_state,
    )

    relative_test_size = test_size / total_holdout
    val_records, test_records = train_test_split(
        holdout_records,
        test_size=relative_test_size,
        stratify=holdout_records["label"],
        random_state=random_state,
    )

    return (
        train_records.reset_index(drop=True),
        val_records.reset_index(drop=True),
        test_records.reset_index(drop=True),
    )


def crop_dark_border(
    image: Image.Image,
    threshold: int = 12,
    padding: int = 8,
) -> Image.Image:
    grayscale = np.asarray(image.convert("L"))
    mask = grayscale > threshold

    if not mask.any():
        return image

    row_indices = np.where(mask.any(axis=1))[0]
    col_indices = np.where(mask.any(axis=0))[0]

    top = max(int(row_indices[0]) - padding, 0)
    bottom = min(int(row_indices[-1]) + padding + 1, image.height)
    left = max(int(col_indices[0]) - padding, 0)
    right = min(int(col_indices[-1]) + padding + 1, image.width)

    return image.crop((left, top, right, bottom))


def center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def apply_clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


class FundusPreprocessor:
    def __init__(
        self,
        enabled: bool = False,
        crop_dark_border_enabled: bool = False,
        crop_threshold: int = 12,
        crop_padding: int = 8,
        center_square_crop_enabled: bool = False,
        clahe_enabled: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        autocontrast_enabled: bool = False,
        autocontrast_cutoff: int = 1,
    ) -> None:
        self.enabled = enabled
        self.crop_dark_border_enabled = crop_dark_border_enabled
        self.crop_threshold = crop_threshold
        self.crop_padding = crop_padding
        self.center_square_crop_enabled = center_square_crop_enabled
        self.clahe_enabled = clahe_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.autocontrast_enabled = autocontrast_enabled
        self.autocontrast_cutoff = autocontrast_cutoff

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "FundusPreprocessor":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            crop_dark_border_enabled=bool(config.get("crop_dark_border", False)),
            crop_threshold=int(config.get("crop_threshold", 12)),
            crop_padding=int(config.get("crop_padding", 8)),
            center_square_crop_enabled=bool(config.get("center_square_crop", False)),
            clahe_enabled=bool(config.get("apply_clahe", False)),
            clahe_clip_limit=float(config.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=int(config.get("clahe_tile_grid_size", 8)),
            autocontrast_enabled=bool(config.get("apply_autocontrast", False)),
            autocontrast_cutoff=int(config.get("autocontrast_cutoff", 1)),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if not self.enabled:
            return image

        if self.crop_dark_border_enabled:
            image = crop_dark_border(
                image=image,
                threshold=self.crop_threshold,
                padding=self.crop_padding,
            )

        if self.center_square_crop_enabled:
            image = center_square_crop(image)

        if self.clahe_enabled:
            image = apply_clahe(
                image=image,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid_size,
            )

        if self.autocontrast_enabled:
            image = ImageOps.autocontrast(image, cutoff=self.autocontrast_cutoff)

        return image


class FundusImageDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        transform: Callable | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple:
        row = self.records.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"]), row["image_path"]


def default_transforms(
    image_size: int = 224,
    train: bool = False,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> transforms.Compose:
    transform_steps: list[Callable] = [FundusPreprocessor.from_config(preprocessing_cfg)]

    transform_steps.append(transforms.Resize((image_size, image_size)))

    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05,
                    hue=0.02,
                ),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def label_distribution(records: pd.DataFrame) -> dict[int, int]:
    counts = records["label"].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}
