from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dr_detection.clean_split import create_split_frames, similarity_groups, validate_frames
from dr_detection.audit import file_sha256
from dr_detection.clean_evaluation import evaluate_clean_model
from dr_detection.train import run_training


class CleanSplitTests(unittest.TestCase):
    def manifest(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame([
            dict(id_code=f"image_{i:03d}", diagnosis=i % 2, status="ok", split="train",
                 pixel_sha256=f"pixels_{i}", file_sha256=f"file_{i}", image_path=f"images/{i}.png",
                 phash=f"{int(rng.integers(0, 2**63)):016x}")
            for i in range(60)
        ])

    def test_deduplication_quarantine_and_order_independence(self):
        frame = self.manifest()
        duplicate = frame.iloc[0].copy()
        duplicate["id_code"] = "copy_same_label"
        conflict = frame.iloc[1].copy()
        conflict["id_code"], conflict["diagnosis"] = "copy_opposite_label", 0
        frame = pd.concat([frame, pd.DataFrame([duplicate, conflict])], ignore_index=True)
        retained, quarantine, copies, edges = create_split_frames(frame)
        self.assertEqual(len(quarantine), 2)
        self.assertEqual(len(copies), 1)
        self.assertEqual(len(retained), 59)
        self.assertFalse(set(retained.pixel_sha256) & set(quarantine.pixel_sha256))
        again = create_split_frames(frame.sample(frac=1, random_state=9))[0]
        pd.testing.assert_frame_equal(retained, again)
        self.assertEqual(validate_frames(retained, edges)["group_overlap"], 0)

    def test_similarity_connections_are_transitive(self):
        frame = pd.DataFrame({"id_code": ["a", "b", "c"], "phash": ["00", "0f", "ff"]})
        groups, edges = similarity_groups(frame, cutoff=4)
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(len(edges), 2)

    def test_cross_split_group_is_rejected(self):
        retained, _, _, edges = create_split_frames(self.manifest())
        retained.loc[1, "group_id"] = retained.loc[0, "group_id"]
        retained.loc[0, "split"] = "train"
        retained.loc[1, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "group crosses"):
            validate_frames(retained, edges)


class TrainingProtocolTests(unittest.TestCase):
    def test_validation_selection_defers_test_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ["train", "valid", "test"]:
                pd.DataFrame({"image_path": ["a", "b"], "label": [0, 1]}).to_csv(root / f"{split}.csv", index=False)
            frame = pd.DataFrame({"image_path": ["a", "b"], "label": [0, 1]})
            config = {"dataset": {f"{s}_csv_path": f"{s}.csv" for s in ["train", "valid", "test"]},
                      "training": {"num_epochs": 2, "num_classes": 2, "save_run_artifacts": True,
                                   "prevent_overwrite": True, "evaluate_test": False},
                      "paths": {"model_dir": "model", "output_dir": "output"}}
            with patch("dr_detection.train._load_records_from_config", return_value=(frame, frame, frame)), \
                 patch("dr_detection.train._create_dataloaders", return_value=("train", "valid", "test")), \
                 patch("dr_detection.train.build_model", return_value=nn.Linear(2, 2)), \
                 patch("dr_detection.train._select_device", return_value=torch.device("cpu")), \
                 patch("dr_detection.train._run_epoch", side_effect=[
                     {"f1": .8, "loss": .4}, {"f1": .9, "loss": .3},
                     {"f1": .95, "loss": .2}, {"f1": .85, "loss": .3}]) as run_epoch:
                result = run_training(config, root)
                self.assertEqual(result["best_epoch"], 1)
                self.assertFalse(result["test_evaluated"])
                self.assertIsNone(result["test_metrics"])
                self.assertEqual(run_epoch.call_count, 4)
                self.assertTrue((root / "output/training_history.json").is_file())
                with self.assertRaises(FileExistsError):
                    run_training(config, root)


class EvaluationProtocolTests(unittest.TestCase):
    def test_cached_evaluation_does_not_run_model_and_detects_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (root / "model").mkdir()
            checkpoint = root / "model/best_model.pt"
            checkpoint.write_bytes(b"frozen-checkpoint")
            split_dir = root / "data/splits/aptos2019_clean_binary_v1"
            split_dir.mkdir(parents=True)
            (split_dir / "protocol.json").write_text("{}")
            config = {"dataset": {}, "paths": {"model_dir": "model", "output_dir": "output"}}
            hashes = {}
            for split in ["train", "valid", "test"]:
                path = root / f"{split}.csv"
                path.write_text("label\n0\n1\n")
                config["dataset"][f"{split}_csv_path"] = path.name
                hashes[split] = file_sha256(path)
            (output / "config_snapshot.json").write_text(json.dumps(config))
            training = {"test_evaluated": False, "split_csv_sha256": hashes,
                        "checkpoint_sha256": file_sha256(checkpoint)}
            (output / "training_summary.json").write_text(json.dumps(training))
            artifact = output / "table_22_2_test_metrics.csv"
            artifact.write_text("metric,value\naccuracy,0.9\n")
            saved = {"checkpoint_sha256": file_sha256(checkpoint),
                     "split_protocol_sha256": file_sha256(split_dir / "protocol.json"),
                     "artifact_sha256": {artifact.name: file_sha256(artifact)}}
            (output / "evaluation_summary.json").write_text(json.dumps(saved))
            with patch("dr_detection.clean_evaluation.load_yaml_config", return_value=config), \
                 patch("dr_detection.clean_evaluation.verify_clean_split"), \
                 patch("dr_detection.clean_evaluation.BinaryPredictor") as predictor:
                self.assertEqual(evaluate_clean_model(root), saved)
                predictor.assert_not_called()
                artifact.write_text("changed")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    evaluate_clean_model(root)


if __name__ == "__main__":
    unittest.main()
