from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dr_detection.audit import exact_pairs, inspect_image, similar_pairs
from dr_detection.dataset import default_transforms
from dr_detection.inference import BinaryPredictor


class AuditTests(unittest.TestCase):
    def test_reencoded_rgb_duplicate_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (48, 48, 3), dtype=np.uint8))
            image.save(root / "a.png", compress_level=0)
            image.save(root / "b.png", compress_level=9)
            rows = [inspect_image({"image_path": name, "id_code": name, "split": split, "diagnosis": grade}, root)
                    for name, split, grade in [("a.png", "train", 0), ("b.png", "test", 1)]]
            matches = exact_pairs(pd.DataFrame(rows))
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches.iloc[0].match_type, "identical_rgb")
            self.assertNotEqual(rows[0]["file_sha256"], rows[1]["file_sha256"])
            self.assertTrue(similar_pairs(pd.DataFrame(rows)).empty)

    def test_missing_and_corrupt_images_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.png").write_bytes(b"invalid image")
            self.assertEqual(inspect_image({"image_path": "absent.png"}, root)["status"], "missing")
            self.assertEqual(inspect_image({"image_path": "bad.png"}, root)["status"], "unreadable")

    def test_phash_candidates_are_not_exact_matches(self):
        rows = [dict(split=split, id_code=split, diagnosis=0, image_path=split,
                     pixel_sha256=split, file_sha256=split, phash=hash_value, status="ok")
                for split, hash_value in [("train", "0000000000000000"), ("test", "0000000000000001")]]
        frame = pd.DataFrame(rows)
        self.assertTrue(exact_pairs(frame).empty)
        result = similar_pairs(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].hamming_distance, 1)
        self.assertEqual(result.iloc[0].match_type, "perceptual_candidate")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).mean(dim=(2, 3)))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "models").mkdir()
        (self.root / "configs").mkdir()
        self.config = {"dataset": {"binary_classification": True, "image_size": 32},
                       "training": {"model_name": "efficientnet_b0"}, "paths": {"model_dir": "models"}}
        self.config_path = self.root / "configs/efficientnet_b0_baseline.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        torch.manual_seed(42)
        torch.save({"model_name": "efficientnet_b0", "num_classes": 2, "model_state_dict": TinyModel().state_dict()},
                   self.root / "models/best_model.pt")
        Image.fromarray(np.random.default_rng(5).integers(0, 255, (36, 48, 3), dtype=np.uint8)).save(self.root / "input.png")
        self.builder = patch("dr_detection.inference.build_model", side_effect=lambda *a, **kw: TinyModel())
        self.builder.start()
        self.addCleanup(self.builder.stop)

    def test_prediction_uses_evaluation_transform_and_softmax(self):
        predictor = BinaryPredictor(self.root, device="cpu")
        with Image.open(self.root / "input.png") as image:
            expected = default_transforms(32, train=False)(image).unsqueeze(0)
        self.assertTrue(torch.equal(predictor.prepare("input.png"), expected))
        result = predictor.predict("input.png")
        self.assertAlmostEqual(result["no_dr_probability"] + result["dr_probability"], 1, places=6)
        self.assertEqual(result["predicted_label"], int(result["dr_probability"] > result["no_dr_probability"]))

    def test_gradcam_releases_hooks_and_preserves_predictions(self):
        predictor = BinaryPredictor(self.root, device="cpu")
        before = predictor.predict("input.png")
        hooks = len(predictor.target_layer._forward_hooks)
        for _ in range(2):
            rgb, overlays = predictor.explain("input.png")
            self.assertEqual(rgb.shape, (32, 32, 3))
            self.assertEqual(set(overlays), {0, 1})
            self.assertTrue(all(np.isfinite(value).all() for value in overlays.values()))
            self.assertEqual(len(predictor.target_layer._forward_hooks), hooks)
        self.assertEqual(before, predictor.predict("input.png"))

    def test_missing_input_has_clear_error(self):
        predictor = BinaryPredictor(self.root, device="cpu")
        with self.assertRaisesRegex(FileNotFoundError, "Image not found"):
            predictor.predict("not-here.png")

    def test_multiclass_config_is_rejected(self):
        self.config["dataset"]["binary_classification"] = False
        self.config_path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "binary classification"):
            BinaryPredictor(self.root, device="cpu")


if __name__ == "__main__":
    unittest.main()
