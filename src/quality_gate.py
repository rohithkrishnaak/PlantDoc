"""Image Quality Gate: OpenCV features + fuzzy inference -> quality score.

Features
- Blur:       variance of the Laplacian on a resized grayscale image, plus the
              ratio of horizontal to vertical gradient energy (motion blur
              smears one direction, so the ratio drops; clean photos stay
              above ~0.75).
- Brightness: mean of the HSV V channel (0-255).
- Coverage:   area of the largest leaf regions. Green tissue defines the leaf;
              brown/yellow lesion tones count only next to green tissue, so
              bare soil (also brown) is not mistaken for leaf. Region outlines
              are filled so lesions inside the leaf count as leaf.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.fuzzy_engine import defuzzify, mu

QUALITY_PASS_THRESHOLD = float(os.getenv("QUALITY_PASS_THRESHOLD", 0.5))
_WORK_SIZE = 512  # analyse at a fixed size so thresholds are resolution independent

# Fuzzy sets on raw feature scales
BLUR_SETS = {          # Laplacian variance
    "blurry": (0, 0, 40, 100),
    "ok": (40, 100, 180, 300),
    "sharp": (180, 300, 1e9, 1e9),
}
BRIGHT_SETS = {        # HSV V mean (0-255)
    "dark": (0, 0, 50, 85),
    "good": (60, 100, 190, 225),
    "over": (195, 230, 255, 255),
}
STREAK_SETS = {        # min/max directional gradient ratio
    "streaked": (0, 0, 0.6, 0.72),
}
COVER_SETS = {         # leaf pixel fraction (0-1)
    "low": (0, 0, 0.12, 0.25),
    "ok": (0.15, 0.3, 0.45, 0.6),
    "high": (0.45, 0.6, 1.0, 1.0),
}
QUALITY_OUT = {
    "poor": (0.0, 0.0, 0.2, 0.45),
    "fair": (0.3, 0.45, 0.6, 0.75),
    "good": (0.6, 0.8, 1.0, 1.0),
}


def _to_bgr(image) -> np.ndarray:
    """Accept PIL image, RGB ndarray or BGR ndarray (assumed RGB if PIL)."""
    if hasattr(image, "convert"):  # PIL
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def compute_features(image) -> dict:
    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    scale = _WORK_SIZE / max(h, w)
    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    g = gray.astype(np.float32)
    gx = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean())
    gy = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())
    direction_ratio = min(gx, gy) / max(gx, gy, 1e-6)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[..., 2].mean())

    # Leaf mask: green tissue + brown/yellow lesions (OpenCV hue range 0-179)
    green = cv2.inRange(hsv, (25, 40, 30), (95, 255, 255))
    brown = cv2.inRange(hsv, (5, 50, 30), (25, 255, 230))
    kernel = np.ones((5, 5), np.uint8)
    # Drop isolated green specks (weeds, noise in soil) before they pull in nearby brown.
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    near_green = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = cv2.bitwise_or(green, cv2.bitwise_and(brown, near_green))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    coverage = float(np.count_nonzero(filled) / filled.size)

    green_px = int(np.count_nonzero(green))
    brown_px = int(np.count_nonzero(brown))
    brown_ratio = brown_px / max(green_px + brown_px, 1)

    return {"blur_var": blur_var, "direction_ratio": direction_ratio, "brightness": brightness,
            "coverage": min(coverage, 1.0), "brown_ratio": brown_ratio}


def fuzzy_quality(features: dict) -> tuple[float, dict]:
    b = {k: mu(features["blur_var"], v) for k, v in BLUR_SETS.items()}
    # Strong parallel veins or stripes also skew the gradient ratio, so the streak
    # signal only counts when the photo is not already clearly sharp.
    b["streaked"] = min(mu(features["direction_ratio"], STREAK_SETS["streaked"]), 1 - b["sharp"])
    l = {k: mu(features["brightness"], v) for k, v in BRIGHT_SETS.items()}
    c = {k: mu(features["coverage"], v) for k, v in COVER_SETS.items()}

    poor = max(b["blurry"], b["streaked"], l["dark"], l["over"], c["low"])
    good = min(max(b["sharp"], b["ok"]), l["good"], max(c["ok"], c["high"]))
    good = max(good, min(b["sharp"], l["good"], c["high"]))
    fair = max(min(b["ok"], l["good"]), min(l["good"], c["ok"]))

    score = defuzzify({"poor": poor, "fair": fair, "good": good}, QUALITY_OUT)
    return score, {"blur": b, "brightness": l, "coverage": c}


def _retake_tip(features: dict, m: dict) -> str:
    # Bad light also lowers sharpness and hides leaf colour, so when lighting is
    # the problem, only the lighting tip is shown: fix that first.
    if m["brightness"]["dark"] > 0.3:
        return "Too dark: move into daylight or avoid shadows on the leaf."
    if m["brightness"]["over"] > 0.3:
        return "Overexposed: avoid direct glare, shade the leaf slightly."
    tips = []
    if max(m["blur"]["blurry"], m["blur"]["streaked"]) > 0.3:
        tips.append("Image is blurry: hold the phone steady and tap to focus on the leaf.")
    if m["coverage"]["low"] > 0.3:
        tips.append("Leaf too small or not found: move closer so the leaf fills the frame.")
    return " ".join(tips) or "Image quality is borderline: retake closer, in even light, holding steady."


def check_quality(image, threshold: float = QUALITY_PASS_THRESHOLD):
    """Return (passed, score, message). Feature details are on `check_quality.last`."""
    features = compute_features(image)
    score, memberships = fuzzy_quality(features)
    passed = score >= threshold
    message = "Image quality OK." if passed else _retake_tip(features, memberships)
    check_quality.last = {"features": features, "memberships": memberships}
    return passed, score, message


check_quality.last = {}
