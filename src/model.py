"""PlantDoc classifier: timm backbone with a 5-class transfer-learning head.

Weight loading order:
1. Fine-tuned checkpoint at MODEL_WEIGHTS (state_dict or {"state_dict": ...}).
2. ImageNet-pretrained backbone from timm (needs internet once) + fresh head.
3. Randomly initialised network (offline fallback) so inference always runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

CLASSES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Potato Early Blight",
    "Potato Healthy",
]

MODEL_ARCH = os.getenv("MODEL_ARCH", "efficientnet_b0")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "weights/plantdoc_effnet_b0.pth")
USE_PRETRAINED = os.getenv("USE_PRETRAINED_BACKBONE", "true").lower() == "true"

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PlantDocModel:
    def __init__(self, arch: str = MODEL_ARCH, weights: str = MODEL_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASSES
        self.weights_source = "random-init (demo)"
        self.model = self._build(arch, weights)
        if self.is_demo:
            # An untrained head emits arbitrary but often very confident logits.
            # Zero it so softmax is uniform and the trust engine reports UNCERTAIN.
            head = self.model.get_classifier()
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        self.model = self.model.to(self.device).eval()

    def _build(self, arch: str, weights: str) -> torch.nn.Module:
        n = len(self.classes)
        path = Path(weights)
        if path.is_file():
            model = timm.create_model(arch, pretrained=False, num_classes=n)
            state = torch.load(path, map_location="cpu")
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
            self.weights_source = f"fine-tuned checkpoint ({path.name})"
            return model
        if USE_PRETRAINED:
            try:
                model = timm.create_model(arch, pretrained=True, num_classes=n)
                self.weights_source = "ImageNet backbone + untrained head (demo)"
                return model
            except Exception as exc:  # offline / hub unavailable
                print(f"[model] pretrained download failed, using random init: {exc}")
        return timm.create_model(arch, pretrained=False, num_classes=n)

    @property
    def is_demo(self) -> bool:
        return "demo" in self.weights_source

    @torch.inference_mode()
    def predict(self, image, top_k: int = 3) -> list[tuple[str, float]]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        top = torch.topk(probs, k=min(top_k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]
