from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    efficientnet_b4,
    resnet18,
)
from tqdm import tqdm

from .config import resolve_path
from .dataset import (
    FundusImageDataset,
    default_transforms,
    load_fundus_records,
    load_predefined_splits,
    split_records,
)


class CustomFundusCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("Unsupported reduction for FocalLoss. Use 'none', 'mean', or 'sum'.")

        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        per_sample_loss = -focal_factor * target_log_probs

        if self.weight is not None:
            sample_weights = self.weight[targets]
            per_sample_loss = per_sample_loss * sample_weights

        if self.reduction == "none":
            return per_sample_loss
        if self.reduction == "sum":
            return per_sample_loss.sum()
        return per_sample_loss.mean()


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        return CustomFundusCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(
        "Unsupported model. Use one of: custom_cnn, resnet18, densenet121, efficientnet_b0, efficientnet_b4."
    )


def _create_dataloaders(
    train_records,
    val_records,
    test_records,
    image_size: int,
    batch_size: int,
    num_workers: int,
    preprocessing_cfg: dict[str, Any] | None = None,
    sampling_cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = FundusImageDataset(
        train_records,
        transform=default_transforms(
            image_size=image_size,
            train=True,
            preprocessing_cfg=preprocessing_cfg,
        ),
    )
    val_dataset = FundusImageDataset(
        val_records,
        transform=default_transforms(
            image_size=image_size,
            train=False,
            preprocessing_cfg=preprocessing_cfg,
        ),
    )
    test_dataset = FundusImageDataset(
        test_records,
        transform=default_transforms(
            image_size=image_size,
            train=False,
            preprocessing_cfg=preprocessing_cfg,
        ),
    )

    train_sampler = _build_weighted_sampler(
        train_records=train_records,
        strategy=(sampling_cfg or {}).get("strategy"),
        replacement=bool((sampling_cfg or {}).get("replacement", True)),
        num_samples=(sampling_cfg or {}).get("num_samples"),
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def _compute_metrics(
    labels: list[int],
    predictions: list[int],
    score_rows: list[list[float]] | None = None,
) -> dict[str, float]:
    metrics = {"accuracy": accuracy_score(labels, predictions)}
    num_classes = len(score_rows[0]) if score_rows else len(set(labels))

    if num_classes == 2:
        metrics.update(
            {
                "precision": precision_score(labels, predictions, zero_division=0, average="binary"),
                "recall": recall_score(labels, predictions, zero_division=0, average="binary"),
                "f1": f1_score(labels, predictions, zero_division=0, average="binary"),
            }
        )

        if score_rows is not None:
            positive_scores = [row[1] for row in score_rows]
            try:
                metrics["roc_auc"] = roc_auc_score(labels, positive_scores)
            except ValueError:
                metrics["roc_auc"] = 0.0

        return metrics

    class_labels = list(range(num_classes))
    metrics.update(
        {
            "precision_macro": precision_score(
                labels,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                labels,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                labels,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                labels,
                predictions,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            ),
            "qwk": cohen_kappa_score(labels, predictions, weights="quadratic"),
        }
    )

    if score_rows is not None:
        score_array = np.asarray(score_rows, dtype=np.float32)
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                labels,
                score_array,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                labels,
                score_array,
                labels=class_labels,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = 0.0
            metrics["roc_auc_ovr_weighted"] = 0.0

    return metrics


def _build_class_weights(
    train_records,
    num_classes: int,
    strategy: str | None,
) -> torch.Tensor | None:
    if not strategy or strategy.lower() == "none":
        return None

    if strategy.lower() != "balanced":
        raise ValueError("Unsupported class_weighting strategy. Use 'balanced' or omit it.")

    labels = train_records["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=num_classes)
    safe_counts = np.maximum(class_counts, 1)
    weights = len(labels) / (num_classes * safe_counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def _build_weighted_sampler(
    train_records,
    strategy: str | None,
    replacement: bool = True,
    num_samples: int | None = None,
    seed: int = 42,
) -> WeightedRandomSampler | None:
    if not strategy or strategy.lower() == "none":
        return None

    if strategy.lower() != "weighted":
        raise ValueError("Unsupported sampling strategy. Use 'weighted' or omit it.")

    labels = train_records["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels)
    safe_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / safe_counts.astype(np.float32)
    sample_weights = class_weights[labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=int(num_samples) if num_samples is not None else len(labels),
        replacement=replacement,
        generator=generator,
    )


def _default_monitor_metric(num_classes: int) -> str:
    return "f1" if num_classes == 2 else "f1_macro"


def _build_criterion(
    training_cfg: dict[str, Any],
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    loss_cfg = training_cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    weight_tensor = class_weights.to(device) if class_weights is not None else None

    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(weight=weight_tensor)

    if loss_name == "focal":
        gamma = float(loss_cfg.get("gamma", 2.0))
        reduction = str(loss_cfg.get("reduction", "mean")).lower()
        return FocalLoss(weight=weight_tensor, gamma=gamma, reduction=reduction)

    raise ValueError("Unsupported loss function. Use 'cross_entropy' or 'focal'.")



def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    score_rows: list[list[float]] = []

    with torch.set_grad_enabled(is_training):
        for images, batch_labels, _ in tqdm(loader, leave=False):
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(images)
            loss = criterion(logits, batch_labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = probabilities.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            score_rows.extend(probabilities.detach().cpu().tolist())

    metrics = _compute_metrics(
        labels=labels,
        predictions=predictions,
        score_rows=score_rows if score_rows else None,
    )
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_records_from_config(
    dataset_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    project_root: str | Path,
) -> tuple:
    image_extension = dataset_cfg.get("image_extension", ".png")
    binary_classification = dataset_cfg.get("binary_classification", True)

    if dataset_cfg.get("use_predefined_splits", False):
        train_records, val_records, test_records = load_predefined_splits(
            train_csv_path=resolve_path(project_root, dataset_cfg["train_csv_path"]),
            train_images_dir=resolve_path(project_root, dataset_cfg["train_images_dir"]),
            valid_csv_path=resolve_path(project_root, dataset_cfg["valid_csv_path"]),
            valid_images_dir=resolve_path(project_root, dataset_cfg["valid_images_dir"]),
            test_csv_path=resolve_path(project_root, dataset_cfg["test_csv_path"]),
            test_images_dir=resolve_path(project_root, dataset_cfg["test_images_dir"]),
            image_extension=image_extension,
            binary_classification=binary_classification,
        )
        return train_records, val_records, test_records

    records = load_fundus_records(
        csv_path=resolve_path(project_root, dataset_cfg["csv_path"]),
        images_dir=resolve_path(project_root, dataset_cfg["images_dir"]),
        image_extension=image_extension,
        binary_classification=binary_classification,
    )
    return split_records(
        records=records,
        val_size=training_cfg.get("val_size", 0.15),
        test_size=training_cfg.get("test_size", 0.15),
        random_state=training_cfg.get("seed", 42),
    )


def run_training(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    paths_cfg = config["paths"]

    _set_random_seed(training_cfg.get("seed", 42))

    model_dir = resolve_path(project_root, paths_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    train_records, val_records, test_records = _load_records_from_config(
        dataset_cfg=dataset_cfg,
        training_cfg=training_cfg,
        project_root=project_root,
    )

    train_loader, val_loader, test_loader = _create_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        image_size=dataset_cfg.get("image_size", 224),
        batch_size=training_cfg.get("batch_size", 16),
        num_workers=training_cfg.get("num_workers", 2),
        preprocessing_cfg=dataset_cfg.get("preprocessing"),
        sampling_cfg=training_cfg.get("sampling"),
        seed=training_cfg.get("seed", 42),
    )

    all_records = [train_records, val_records, test_records]
    num_classes = len({int(label) for records in all_records for label in records["label"].tolist()})
    device = _select_device()
    model = build_model(
        model_name=training_cfg.get("model_name", "resnet18"),
        num_classes=num_classes,
        pretrained=training_cfg.get("pretrained", False),
    ).to(device)

    class_weights = _build_class_weights(
        train_records=train_records,
        num_classes=num_classes,
        strategy=training_cfg.get("class_weighting"),
    )
    if class_weights is not None:
        print(f"Using class weights: {class_weights.tolist()}")
    if training_cfg.get("sampling", {}).get("strategy") == "weighted":
        print("Using weighted random sampler for the training loader.")
    if training_cfg.get("loss", {}).get("name", "cross_entropy").lower() == "focal":
        print(
            "Using focal loss "
            f"(gamma={float(training_cfg.get('loss', {}).get('gamma', 2.0)):.2f})."
        )

    criterion = _build_criterion(
        training_cfg=training_cfg,
        class_weights=class_weights,
        device=device,
    )
    optimizer = Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    best_model_path = model_dir / "best_model.pt"
    monitor_metric = training_cfg.get("monitor_metric", _default_monitor_metric(num_classes))
    best_metric_value = -1.0

    for epoch in range(1, training_cfg.get("num_epochs", 5) + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_{monitor_metric}={train_metrics[monitor_metric]:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_{monitor_metric}={val_metrics[monitor_metric]:.4f}"
        )

        if val_metrics[monitor_metric] > best_metric_value:
            best_metric_value = val_metrics[monitor_metric]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": training_cfg.get("model_name", "resnet18"),
                    "num_classes": num_classes,
                    "monitor_metric": monitor_metric,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, test_loader, criterion, device, optimizer=None)

    return {
        "best_model_path": str(best_model_path),
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": len(train_records),
            "valid": len(val_records),
            "test": len(test_records),
        },
        "monitor_metric": monitor_metric,
    }
