"""Hybrid CNN + ViT feature-fusion classifier.

Several frozen pretrained timm backbones each produce a pooled embedding. The
embeddings are L2-normalised, concatenated, standardised and fed to a linear
head (fitted with logistic regression by train_hybrid.py). Logits are divided
by a temperature fitted on out-of-fold predictions so softmax confidence is
calibrated for the fuzzy trust engine.

For out-of-distribution detection the checkpoint also stores a bank of the
(standardised, L2-normalised) training features: a photo whose nearest
training photo is less similar than `ood_threshold` is "not a supported leaf".

Only the head and the feature bank are stored in the checkpoint (a few MB);
backbone weights come from the timm / HF Hub cache.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_backbone(name: str, pretrained: bool = True) -> nn.Module:
    kw = {"img_size": 224} if "dinov2" in name else {}
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kw).eval()


class HybridNet(nn.Module):
    def __init__(self, backbones: list[str], feat_dim: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone_names = backbones
        self.branches = nn.ModuleList(create_backbone(b, pretrained) for b in backbones)
        for i, m in enumerate(self.branches):
            cfg = timm.data.resolve_data_config({}, model=m)
            # Input arrives ImageNet-normalised; re-normalise per branch.
            scale = torch.tensor(IMAGENET_STD) / torch.tensor(cfg["std"])
            shift = (torch.tensor(IMAGENET_MEAN) - torch.tensor(cfg["mean"])) / torch.tensor(cfg["std"])
            self.register_buffer(f"scale{i}", scale.view(1, 3, 1, 1))
            self.register_buffer(f"shift{i}", shift.view(1, 3, 1, 1))
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_scale", torch.ones(feat_dim))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("bank", torch.zeros(0, feat_dim))
        self.register_buffer("ood_threshold", torch.tensor(-1.0))
        self.head = nn.Linear(feat_dim, num_classes)
        for p in self.branches.parameters():
            p.requires_grad_(False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, m in enumerate(self.branches):
            xb = x * getattr(self, f"scale{i}") + getattr(self, f"shift{i}")
            feats.append(nn.functional.normalize(m(xb), dim=1))
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calibrated logits and the standardised fused features."""
        f = (self.features(x) - self.feat_mean) / self.feat_scale
        return self.head(f) / self.temperature, f

    def similarity_to_train(self, f: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of each feature row to its nearest training photo."""
        if self.bank.numel() == 0:
            return torch.ones(f.shape[0], device=f.device)
        return (nn.functional.normalize(f, dim=1) @ self.bank.float().T).max(1).values

    def get_classifier(self) -> nn.Module:
        return self.head

    @property
    def gradcam_target(self) -> nn.Module | None:
        """Last 4-D (NCHW) feature stage of the first CNN branch, for Grad-CAM."""
        for m in self.branches:
            if getattr(m, "conv_head", None) is not None:
                return m.conv_head
            if hasattr(m, "stages") and "convnext" in type(m).__name__.lower():
                return m.stages[-1]
        return None

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "HybridNet":
        net = cls(ckpt["backbones"], ckpt["feat_mean"].numel(), len(ckpt["classes"]))
        net.feat_mean.copy_(ckpt["feat_mean"])
        net.feat_scale.copy_(ckpt["feat_scale"])
        net.temperature.fill_(float(ckpt["temperature"]))
        net.head.load_state_dict(ckpt["head"])
        if "ood_bank" in ckpt:
            net.bank = ckpt["ood_bank"].float()
            net.ood_threshold.fill_(float(ckpt["ood_threshold"]))
        return net.eval()
