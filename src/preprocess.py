"""Shared image preprocessing: optional CLAHE contrast normalisation."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, tile: int = 8) -> Image.Image:
    """CLAHE on the L channel of LAB, so hue (lesion colour) is left untouched."""
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class CLAHE:
    """torchvision-style transform wrapper."""

    def __init__(self, clip_limit: float = 2.0, tile: int = 8):
        self.clip_limit, self.tile = clip_limit, tile

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_clahe(image, self.clip_limit, self.tile)
