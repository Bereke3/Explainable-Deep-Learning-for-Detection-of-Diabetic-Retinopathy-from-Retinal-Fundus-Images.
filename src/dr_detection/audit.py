"""Read-only audit of the original APTOS splits; never changes training data."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import load_yaml_config, resolve_path

SPLITS = ("train", "valid", "test")
PAIR_COLUMNS = [
    "split_a", "id_a", "diagnosis_a", "path_a", "split_b", "id_b",
    "diagnosis_b", "path_b", "match_type", "hamming_distance",
]


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    low = cv2.dct(gray.astype(np.float32))[:8, :8].ravel()[1:]
    bits = low > np.median(low)
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


def inspect_image(row: dict, root: Path) -> dict:
    result = dict(row, status="ok", error="", file_sha256="", pixel_sha256="", phash="")
    path = resolve_path(root, row["image_path"])
    if not path.is_file():
        return dict(result, status="missing", error="Image file does not exist")
    try:
        data = path.read_bytes()
        result["file_sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            result.update(width=rgb.width, height=rgb.height)
            # Include dimensions so equal byte streams with different shapes cannot match.
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
            digest.update(rgb.tobytes())
            result["pixel_sha256"] = digest.hexdigest()
            result["phash"] = perceptual_hash(rgb)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        result.update(status="unreadable", error=str(error))
    return result


def _pair(a: dict, b: dict, kind: str, distance: int = 0) -> dict:
    return {
        "split_a": a["split"], "id_a": a["id_code"], "diagnosis_a": a["diagnosis"],
        "path_a": a["image_path"], "split_b": b["split"], "id_b": b["id_code"],
        "diagnosis_b": b["diagnosis"], "path_b": b["image_path"],
        "match_type": kind, "hamming_distance": distance,
    }


def exact_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    valid = manifest.loc[manifest["pixel_sha256"].fillna("") != ""]
    for _, group in valid.groupby("pixel_sha256", sort=True):
        for a, b in combinations(group.to_dict("records"), 2):
            kind = "identical_file" if a["file_sha256"] == b["file_sha256"] else "identical_rgb"
            pairs.append(_pair(a, b, kind))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def similar_pairs(manifest: pd.DataFrame, max_distance: int = 4) -> pd.DataFrame:
    """Candidate retrieval only: perceptual similarity is not a duplicate verdict."""
    pairs = []
    records = manifest.loc[manifest["status"] == "ok"].to_dict("records")
    for row in records:
        row["hash_int"] = int(row["phash"], 16)
    for a, b in combinations(records, 2):
        if a["split"] == b["split"] or a["pixel_sha256"] == b["pixel_sha256"]:
            continue
        distance = (a["hash_int"] ^ b["hash_int"]).bit_count()
        if distance <= max_distance:
            pairs.append(_pair(a, b, "perceptual_candidate", distance))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).sort_values(
        ["hamming_distance", "split_a", "id_a", "id_b"], ignore_index=True
    )


def run_audit(project_root: str | Path, workers: int = 2) -> dict:
    root = Path(project_root).resolve()
    config_path = root / "configs/efficientnet_b0_baseline.yaml"
    config = load_yaml_config(config_path)
    cfg = config["dataset"]
    output = root / "outputs/data_audit"
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for split in SPLITS:
        csv_path = resolve_path(root, cfg[f"{split}_csv_path"])
        frame = pd.read_csv(csv_path, dtype={"id_code": str})
        if not {"id_code", "diagnosis"}.issubset(frame.columns):
            raise ValueError(f"Expected APTOS id_code/diagnosis columns: {csv_path}")
        sources[split] = {"csv": str(csv_path.relative_to(root)), "sha256": file_sha256(csv_path)}
        for row in frame.to_dict("records"):
            image_id = str(row["id_code"])
            path = resolve_path(root, cfg[f"{split}_images_dir"]) / (image_id + cfg.get("image_extension", ".png"))
            rows.append({"split": split, "id_code": image_id, "diagnosis": row["diagnosis"],
                         "image_path": str(path.relative_to(root))})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest = pd.DataFrame(tqdm(
            pool.map(lambda row: inspect_image(row, root), rows), total=len(rows), desc="Audit images", mininterval=5
        ))
    manifest["valid_label"] = manifest["diagnosis"].isin([0, 1, 2, 3, 4])
    manifest["binary_label"] = pd.to_numeric(manifest["diagnosis"], errors="coerce").gt(0).astype(int)
    manifest.to_csv(output / "table_18_1_image_manifest.csv", index=False)
    distribution = manifest.groupby(["split", "diagnosis"]).size().unstack(fill_value=0).reindex(SPLITS)
    distribution.to_csv(output / "table_18_2_grade_distribution.csv")

    ids = []
    for image_id, group in manifest.groupby("id_code"):
        if len(group) > 1:
            ids.append({"id_code": image_id, "records": len(group),
                        "splits": ",".join(sorted(group["split"].unique())),
                        "cross_split": group["split"].nunique() > 1})
    id_matches = pd.DataFrame(ids, columns=["id_code", "records", "splits", "cross_split"])
    id_matches.to_csv(output / "table_18_3_repeated_ids.csv", index=False)
    exact = exact_pairs(manifest)
    exact.to_csv(output / "table_18_4_exact_duplicate_pairs.csv", index=False)
    similar = similar_pairs(manifest)
    similar.to_csv(output / "table_18_5_similar_cross_split_candidates.csv", index=False)
    cross = exact.loc[exact["split_a"] != exact["split_b"]]
    contaminated_test = set()
    for row in cross.to_dict("records"):
        if row["split_a"] == "test":
            contaminated_test.add(row["id_a"])
        if row["split_b"] == "test":
            contaminated_test.add(row["id_b"])
    summary = {
        "config_sha256": file_sha256(config_path), "sources": sources,
        "split_sizes": manifest.groupby("split").size().to_dict(),
        "missing_images": int((manifest["status"] == "missing").sum()),
        "unreadable_images": int((manifest["status"] == "unreadable").sum()),
        "invalid_labels": int((~manifest["valid_label"]).sum()),
        "repeated_id_groups": len(id_matches),
        "cross_split_id_groups": int(id_matches["cross_split"].sum()),
        "exact_duplicate_pairs": len(exact), "exact_cross_split_pairs": len(cross),
        "exact_duplicate_grade_conflicts": int((exact["diagnosis_a"] != exact["diagnosis_b"]).sum()),
        "exact_duplicate_binary_conflicts": int((exact["diagnosis_a"].gt(0) != exact["diagnosis_b"].gt(0)).sum()),
        "exact_cross_split_binary_conflicts": int((cross["diagnosis_a"].gt(0) != cross["diagnosis_b"].gt(0)).sum()),
        "test_images_matching_train_or_valid": sorted(contaminated_test),
        "similar_cross_split_candidates": len(similar), "phash_max_distance": 4,
        "patient_independence": "unverified: these CSVs have no patient identifiers",
        "test_protocol": "historical test results were consulted across multiple experiments",
        "near_duplicate_status": "candidates require visual review; not an exhaustive duplicate search",
    }
    summary["status"] = "issues_found" if any([
        summary["missing_images"], summary["unreadable_images"], summary["invalid_labels"],
        summary["repeated_id_groups"], summary["exact_cross_split_pairs"],
    ]) else "no_exact_overlap_found"
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
