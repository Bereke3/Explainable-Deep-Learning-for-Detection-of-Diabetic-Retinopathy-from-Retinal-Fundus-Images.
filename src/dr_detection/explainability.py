from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


def get_default_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        return model.features[16]
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]

    raise ValueError(
        "Unsupported model for automatic Grad-CAM target layer selection. "
        "Use one of: custom_cnn, resnet18, densenet121, efficientnet_b0."
    )


def generate_gradcam_overlay(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_path: str | Path,
    image_size: int = 224,
    target_class: int | None = None,
) -> np.ndarray:
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    pil_image = Image.open(image_path).convert("RGB")
    model_device = next(model.parameters()).device
    input_tensor = preprocess(pil_image).unsqueeze(0).to(model_device)
    rgb_image = np.asarray(pil_image.resize((image_size, image_size)), dtype=np.float32) / 255.0

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
