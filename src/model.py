"""PlantDoc classifier: timm backbone with a 6-class transfer-learning head.

Weight loading order:
0. Checkpoint at HYBRID_WEIGHTS: either a frozen-feature hybrid (train_hybrid.py,
   src/hybrid.py) or an end-to-end fine-tuned model (train_finetune.py,
   src/finetuned.py), chosen by its "type" field.
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

from src.fuzzy_engine import TRUST_THRESHOLD
from src.finetuned import FineTunedNet
from src.hybrid import HybridNet
from src.preprocess import apply_clahe

CLASSES = [
    "Potato Early Blight",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
]

MODEL_ARCH = os.getenv("MODEL_ARCH", "efficientnet_b0")
# Fine-tuned ConvNeXt-T on PlantDoc + PlantWild (DEV_LOG 2d) if present, else the frozen-feature hybrid.
HYBRID_WEIGHTS = os.getenv("HYBRID_WEIGHTS", next(
    (p for p in ("weights/plantdoc_finetuned.pth", "weights/plantdoc_hybrid.pth") if Path(p).is_file()),
    "weights/plantdoc_hybrid.pth"))
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "weights/plantdoc_effnet_b0.pth")
USE_PRETRAINED = os.getenv("USE_PRETRAINED_BACKBONE", "true").lower() == "true"

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PlantDocModel:
    def __init__(self, arch: str = MODEL_ARCH, weights: str = MODEL_WEIGHTS, checkpoint: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASSES
        self.weights_source = "random-init (demo)"
        self.use_clahe = False
        self.tta = False
        self.trust_threshold = TRUST_THRESHOLD
        self.model = self._build(arch, weights, checkpoint or HYBRID_WEIGHTS)
        if self.is_demo:
            # An untrained head emits arbitrary but often very confident logits.
            # Zero it so softmax is uniform and the trust engine reports UNCERTAIN.
            head = self.model.get_classifier()
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        self.model = self.model.to(self.device).eval()

    def _build(self, arch: str, weights: str, checkpoint: str) -> torch.nn.Module:
        n = len(self.classes)
        hybrid = Path(checkpoint)
        if hybrid.is_file():
            ckpt = torch.load(hybrid, map_location="cpu", weights_only=False)
            assert ckpt["classes"] == self.classes, "Checkpoint classes differ from CLASSES"
            self.use_clahe = bool(ckpt["clahe"])
            self.tta = True
            # The checkpoint's OOF-chosen threshold, unless TRUST_THRESHOLD is set explicitly.
            if "TRUST_THRESHOLD" not in os.environ and "trust_threshold" in ckpt:
                self.trust_threshold = float(ckpt["trust_threshold"])
            if ckpt.get("type") == "finetuned":
                self.weights_source = f"fine-tuned {ckpt['arch'].split('.')[0]}"
                model = FineTunedNet.from_checkpoint(ckpt)
            else:
                short = " + ".join(b.split(".")[0] for b in ckpt["backbones"])
                self.weights_source = f"hybrid {short}{' + CLAHE' if self.use_clahe else ''}"
                model = HybridNet.from_checkpoint(ckpt)
            model.input_transform = self.transform
            return model
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

    def transform(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalised tensor, applying CLAHE if the checkpoint was trained with it."""
        image = image.convert("RGB")
        if self.use_clahe:
            image = apply_clahe(image)
        return PREPROCESS(image)

    @torch.inference_mode()
    def diagnose(self, image, top_k: int = 3) -> tuple[list[tuple[str, float]], float | None, bool]:
        """Top-k predictions, similarity to the nearest training photo, and whether
        the photo looks out of distribution (not a supported leaf). Similarity is
        None for models without a feature bank."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        similarity, is_ood = None, False
        if isinstance(self.model, (HybridNet, FineTunedNet)):
            logits, f = self.model.forward_with_features(x)
            if self.tta:  # average with the horizontally flipped view
                logits2, f2 = self.model.forward_with_features(torch.flip(x, dims=[3]))
                logits, f = (logits + logits2) / 2, (f + f2) / 2
            if self.model.bank.numel():
                similarity = float(self.model.similarity_to_train(f)[0])
                is_ood = similarity < float(self.model.ood_threshold)
        else:
            logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        top = torch.topk(probs, k=min(top_k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)], similarity, is_ood

    def predict(self, image, top_k: int = 3) -> list[tuple[str, float]]:
        return self.diagnose(image, top_k)[0]
