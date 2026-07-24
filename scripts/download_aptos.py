from __future__ import annotations

import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the APTOS 2019 dataset from Kaggle.")
    parser.add_argument(
        "--handle",
        default="mariaherrerot/aptos2019",
        help="Kaggle dataset handle.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/aptos2019",
        help="Output directory relative to the project root.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Prompt for interactive Kaggle authentication before download.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if the dataset is already cached.",
    )
    return parser.parse_args()


def find_first(directory: Path, name: str) -> Path | None:
    matches = sorted(directory.rglob(name))
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Package 'kagglehub' is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if args.login:
        kagglehub.login()

    print("Starting Kaggle download. The APTOS 2019 archive is large and may take a while.")

    try:
        download_path = kagglehub.dataset_download(
            args.handle,
            output_dir=str(output_dir),
            force_download=args.force_download,
        )
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(
            "Could not reach Kaggle. Check your internet connection, DNS, VPN/firewall, "
            "or run the script outside a restricted sandbox."
        ) from exc
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "unauthorized" in message or "forbidden" in message:
            raise SystemExit(
                "Kaggle rejected the request. Run the script with --login or configure your "
                "Kaggle API credentials first."
            ) from exc
        raise

    download_dir = Path(download_path)
    print(f"Dataset downloaded to: {download_dir}")

    csv_path = find_first(download_dir, "train.csv")
    images_dir = find_first(download_dir, "train_images")

    if csv_path:
        print(f"Found train CSV: {csv_path}")
    else:
        print("Could not find train.csv automatically.")

    if images_dir and images_dir.is_dir():
        print(f"Found image folder: {images_dir}")
    else:
        print("Could not find train_images automatically.")

    if csv_path and images_dir and images_dir.is_dir():
        print("\nNext step:")
        print(
            "python3 scripts/check_dataset.py "
            f"--csv {csv_path.relative_to(project_root)} "
            f"--images-dir {images_dir.relative_to(project_root)} "
            "--binary"
        )
    else:
        print("\nNext step:")
        print("Open the downloaded folder and locate the real paths to train.csv and train_images.")
        print("Then run scripts/check_dataset.py with those paths.")


if __name__ == "__main__":
    main()
