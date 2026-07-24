from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.dataset import (  # noqa: E402
    FundusImageDataset,
    default_transforms,
    label_distribution,
    load_fundus_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate retinal dataset files.")
    parser.add_argument("--csv", required=True, help="Path to CSV annotations.")
    parser.add_argument("--images-dir", required=True, help="Directory with images.")
    parser.add_argument(
        "--image-extension",
        default=".png",
        help="Extension used when CSV stores image ids, default: .png",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size for transforms, default: 224",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Convert labels to binary: 0 = No DR, >0 = DR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_fundus_records(
        csv_path=PROJECT_ROOT / args.csv,
        images_dir=PROJECT_ROOT / args.images_dir,
        image_extension=args.image_extension,
        binary_classification=args.binary,
    )

    missing_mask = ~records["image_path"].map(lambda value: Path(value).exists())
    missing_count = int(missing_mask.sum())

    print(f"Total rows: {len(records)}")
    print(f"Class distribution: {label_distribution(records)}")
    print(f"Missing images: {missing_count}")

    if missing_count:
        preview = records.loc[missing_mask, "image_path"].head(5).tolist()
        print("First missing files:")
        for item in preview:
            print(f"  - {item}")

    valid_records = records.loc[~missing_mask].reset_index(drop=True)
    if valid_records.empty:
        raise RuntimeError("No readable images were found. Check CSV and folder paths.")

    dataset = FundusImageDataset(
        valid_records,
        transform=default_transforms(image_size=args.image_size, train=False),
    )
    image_tensor, label, image_path = dataset[0]

    print(f"First sample path: {image_path}")
    print(f"First sample tensor shape: {tuple(image_tensor.shape)}")
    print(f"First sample label: {label}")


if __name__ == "__main__":
    main()
