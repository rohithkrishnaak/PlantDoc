"""End-to-end fine-tuned timm classifier (trained by train_finetune.py).

Same interface as HybridNet (src/hybrid.py), so PlantDocModel, the trust engine,
the out-of-distribution check and Grad-CAM work unchanged: logits are divided by
a temperature fitted on out-of-fold predictions, and the checkpoint carries an
L2-normalised bank of the training photos' pooled features.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class FineTunedNet(nn.Module):
    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        kw = {"img_size": 224} if "dinov2" in arch else {}
        self.net = timm.create_model(arch, pretrained=False, num_classes=num_classes, **kw)
        self.backbone_names = [arch]
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("bank", torch.zeros(0, self.net.num_features))
        self.register_buffer("ood_threshold", torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calibrated logits and the pooled (pre-classifier) features."""
        f = self.net.forward_head(self.net.forward_features(x), pre_logits=True)
        return self.net.get_classifier()(f) / self.temperature, f

    def similarity_to_train(self, f: torch.Tensor) -> torch.Tensor:
        if self.bank.numel() == 0:
            return torch.ones(f.shape[0], device=f.device)
        return (nn.functional.normalize(f, dim=1) @ self.bank.float().T).max(1).values

    def get_classifier(self) -> nn.Module:
        return self.net.get_classifier()

    @property
    def gradcam_target(self) -> nn.Module | None:
        if getattr(self.net, "conv_head", None) is not None:
            return self.net.conv_head
        if hasattr(self.net, "stages") and "convnext" in type(self.net).__name__.lower():
            return self.net.stages[-1]
        return None

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "FineTunedNet":
        net = cls(ckpt["arch"], len(ckpt["classes"]))
        net.net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
        net.temperature.fill_(float(ckpt["temperature"]))
        net.bank = ckpt["ood_bank"].float()
        net.ood_threshold.fill_(float(ckpt["ood_threshold"]))
        return net.eval()
