"""Grad-CAM (Gradient-weighted Class Activation Mapping) for PlantDoc models.

Produces explainable AI visual heatmaps showing which regions of the leaf
(lesions, chlorosis, veins) triggered the neural network's diagnosis.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import PREPROCESS


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None

    def _find_target_layer(self) -> torch.nn.Module:
        """Find the final convolutional layer of the backbone."""
        if getattr(self.model, "gradcam_target", None) is not None:
            return self.model.gradcam_target
        if hasattr(self.model, "conv_head") and self.model.conv_head is not None:
            return self.model.conv_head
        # Fallback search for last Conv2d layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        return [self.target_layer.register_forward_hook(forward_hook),
                self.target_layer.register_full_backward_hook(backward_hook)]

    def generate(self, image: Image.Image, target_class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Hooks live only for this call: the model is shared (cached) across app
        sessions, and hooks left behind would pile up on every request."""
        handles = self._register_hooks()
        try:
            return self._generate(image, target_class_idx)
        finally:
            for h in handles:
                h.remove()

    def _generate(self, image: Image.Image, target_class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap and overlaid visualization.

        Returns:
            (heatmap_rgb, blended_rgb) both as numpy uint8 RGB arrays of size (H, W, 3).
        """
        self.model.eval()
        orig_w, orig_h = image.size
        orig_np = np.array(image.convert("RGB"))

        # Preprocess input tensor
        transform = getattr(self.model, "input_transform", PREPROCESS)
        x = transform(image.convert("RGB")).unsqueeze(0).to(next(self.model.parameters()).device)
        x.requires_grad_(True)

        logits = self.model(x)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Gradients and activations
        grads = self.gradients.detach()  # [1, C, H, W]
        acts = self.activations.detach()  # [1, C, H, W]

        # Global average pool of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam).squeeze().cpu().numpy()  # [H, W]

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize CAM to original image size
        heatmap_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Colorize with JET colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend with original image
        blended_rgb = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)

        return heatmap_rgb, blended_rgb
