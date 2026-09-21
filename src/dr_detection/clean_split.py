"""Deterministic binary split with exact deduplication and conservative similarity groups."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from .audit import file_sha256

SPLIT_DIR = Path("data/splits/aptos2019_clean_binary_v1")
EDGE_COLUMNS = ["id_a", "id_b", "hamming_distance"]


def similarity_groups(records: pd.DataFrame, cutoff: int = 4) -> tuple[list[str], pd.DataFrame]:
    """Group every candidate edge, not just pairs from the previous split."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    values = [int(value, 16) for value in records["phash"]]
    ids = records["id_code"].tolist()
    edges = []
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            distance = (values[a] ^ values[b]).bit_count()
            if distance <= cutoff:
                root_a, root_b = find(a), find(b)
                parent[max(root_a, root_b)] = min(root_a, root_b)
                edges.append((ids[a], ids[b], distance))
    groups = [f"similarity_{ids[find(i)]}" for i in range(len(records))]
    return groups, pd.DataFrame(edges, columns=EDGE_COLUMNS)


def create_split_frames(manifest: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = manifest.sort_values("id_code").reset_index(drop=True).copy()
    if not manifest["diagnosis"].isin(range(5)).all() or not manifest["status"].eq("ok").all():
        raise ValueError("The source manifest must contain only readable images and valid grades.")
    if manifest["id_code"].duplicated().any():
        raise ValueError("Source image identifiers must be unique.")
    manifest["label"] = manifest["diagnosis"].gt(0).astype(int)
    disagreements = manifest.groupby("pixel_sha256")["label"].nunique()
    conflict_hashes = set(disagreements.loc[disagreements > 1].index)
    quarantine = manifest.loc[manifest["pixel_sha256"].isin(conflict_hashes)].copy()
    quarantine["reason"] = "identical_pixels_with_conflicting_binary_labels"
    eligible = manifest.loc[~manifest["pixel_sha256"].isin(conflict_hashes)].copy()
    representatives = eligible.drop_duplicates("pixel_sha256", keep="first").copy()
    kept_ids = representatives.set_index("pixel_sha256")["id_code"]
    duplicate_copies = eligible.loc[eligible["id_code"] != eligible["pixel_sha256"].map(kept_ids)].copy()
    duplicate_copies["representative_id"] = duplicate_copies["pixel_sha256"].map(kept_ids)
    representatives["original_grades"] = representatives["pixel_sha256"].map(
        eligible.groupby("pixel_sha256")["diagnosis"].agg(lambda values: ",".join(map(str, sorted(set(values)))))
    )
    representatives = representatives.rename(columns={"split": "source_split"}).reset_index(drop=True)
    representatives["group_id"], edges = similarity_groups(representatives)
    # Fold choices and seed are fixed before model training, never selected by test scores.
    representatives["fold"] = -1
    # Integer group encoding preserves the deterministic lexicographic representative order.
    groups = pd.factorize(representatives["group_id"], sort=False)[0]
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(representatives, representatives["label"], groups)):
        representatives.loc[indices, "fold"] = fold
    representatives["split"] = representatives["fold"].map(
        lambda fold: "test" if fold == 0 else "valid" if fold == 1 else "train"
    )
    validate_frames(representatives, edges)
    return representatives, quarantine, duplicate_copies, edges


def validate_frames(records: pd.DataFrame, edges: pd.DataFrame) -> dict:
    if records["pixel_sha256"].duplicated().any() or records["id_code"].duplicated().any():
        raise ValueError("Exact duplicates or repeated identifiers remain in the cleaned split.")
    if records.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("A similarity group crosses split boundaries.")
    lookup = records.set_index("id_code")["split"]
    if not edges.empty:
        left, right = edges["id_a"].map(lookup), edges["id_b"].map(lookup)
        if left.isna().any() or right.isna().any() or left.ne(right).any():
            raise ValueError("A candidate similarity pair is missing or crosses split boundaries.")
    for split in ["train", "valid", "test"]:
        subset = records.loc[records["split"] == split]
        if set(subset["label"]) != {0, 1}:
            raise ValueError(f"Both binary classes must be present in {split}.")
    return {"exact_cross_split_pairs": 0, "similarity_cross_split_pairs_at_cutoff": 0,
            "group_overlap": 0, "unique_images": len(records), "similarity_groups": records["group_id"].nunique()}


def verify_clean_split(root: str | Path, verify_images: bool = False) -> dict:
    root = Path(root).resolve()
    directory = root / SPLIT_DIR
    protocol = json.loads((directory / "protocol.json").read_text())
    for name, expected in protocol["file_sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Frozen split artifact changed: {name}")
    frames = [pd.read_csv(directory / f"{split}.csv", dtype={"phash": str}).assign(split=split)
              for split in ["train", "valid", "test"]]
    combined = pd.concat(frames, ignore_index=True)
    edges = pd.read_csv(directory / "similarity_edges.csv")
    result = validate_frames(combined, edges)
    quarantine = pd.read_csv(directory / "quarantined.csv")
    if set(combined["pixel_sha256"]) & set(quarantine["pixel_sha256"]):
        raise ValueError("A quarantined conflict group is present in the split.")
    if verify_images:
        verify_file_hashes(root, combined)
    return result


def verify_file_hashes(root: Path, records: pd.DataFrame) -> None:
    paths = [root / name for name in records["image_path"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        checksums = list(tqdm(pool.map(file_sha256, paths), total=len(paths), desc="Verify image hashes", mininterval=5))
    if checksums != records["file_sha256"].tolist():
        raise ValueError("Image contents changed after the audit. Rerun the audit before splitting.")


def prepare_clean_split(root: str | Path) -> dict:
    root = Path(root).resolve()
    directory = root / SPLIT_DIR
    if directory.exists():
        raise FileExistsError(f"Preserving the frozen split: {directory}. Use verify_clean_split to check it.")
    source = root / "outputs/data_audit/table_18_1_image_manifest.csv"
    audit = json.loads((root / "outputs/data_audit/audit_summary.json").read_text())
    for item in audit["sources"].values():
        if file_sha256(root / item["csv"]) != item["sha256"]:
            raise ValueError("Source annotations changed since the audit.")
    manifest = pd.read_csv(source, dtype={"phash": str})
    verify_file_hashes(root, manifest)
    records, quarantine, copies, edges = create_split_frames(manifest)
    directory.mkdir(parents=True)
    records.to_csv(directory / "manifest.csv", index=False)
    quarantine.to_csv(directory / "quarantined.csv", index=False)
    copies.to_csv(directory / "duplicate_copies.csv", index=False)
    edges.to_csv(directory / "similarity_edges.csv", index=False)
    for split in ["train", "valid", "test"]:
        records.loc[records["split"] == split].to_csv(directory / f"{split}.csv", index=False)
    protocol = {
        "version": "aptos2019_clean_binary_v1", "seed": 42,
        "source_manifest_sha256": file_sha256(source), "source_csvs": audit["sources"],
        "source_images": len(manifest), "quarantined_images": len(quarantine),
        "quarantined_exact_groups": quarantine["pixel_sha256"].nunique(),
        "removed_duplicate_copies": len(copies), "retained_images": len(records),
        "representative_rule": "lexicographically first id_code per exact RGB group",
        "quarantine_rule": "all members of exact groups with conflicting binary labels",
        "similarity_rule": "all retained image pairs, DCT pHash distance <= 4; transitive components stay in one split",
        "similarity_interpretation": "conservative grouping only; not verified duplicates or patient identities",
        "candidate_edges": len(edges), "similarity_groups": records["group_id"].nunique(),
        "largest_group": int(records.groupby("group_id").size().max()),
        "split_rule": "StratifiedGroupKFold(10, shuffle=True, random_state=42); fold 0=test, 1=valid, 2-9=train",
        "sklearn_version": sklearn.__version__,
        "split_sizes": records.groupby("split").size().to_dict(),
        "binary_counts": {split: {str(k): int(v) for k, v in part["label"].value_counts().sort_index().items()}
                          for split, part in records.groupby("split")},
        "verification": validate_frames(records, edges),
        "limits": "Repartitioned internal data, previously explored. Not external validation; patient independence unverified; pHash misses some duplicates and can group unrelated images.",
        "file_sha256": {p.name: file_sha256(p) for p in sorted(directory.glob("*.csv"))},
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol
