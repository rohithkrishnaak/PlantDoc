"""Does the quality gate reject bad photos and pass good ones?

1. Pass rate on the real (undegraded) train and test photos.
2. A synthetic degradation suite built from the test photos: motion/defocus
   blur, under-exposure, glare, and "leaf too small" (photo shrunk onto a
   soil-coloured canvas). For each, the rejection rate and which retake tip
   fired.

    python -m experiments.quality_gate_eval
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.dedup import list_images
from src.quality_gate import check_quality

OUT = Path("experiments/quality_gate_eval.json")


def blur(a):
    return cv2.GaussianBlur(a, (0, 0), sigmaX=6)


def motion_blur(a):
    k = np.zeros((25, 25), np.float32)
    k[12, :] = 1 / 25
    return cv2.filter2D(a, -1, k)


def dark(a):
    return (a.astype(np.float32) * 0.22).astype(np.uint8)


def glare(a):
    return np.clip(a.astype(np.float32) * 1.6 + 110, 0, 255).astype(np.uint8)


def too_small(a):
    h, w = a.shape[:2]
    canvas = np.full((h, w, 3), (120, 100, 80), np.uint8)  # bare-soil brown-grey
    canvas = np.clip(canvas + np.random.default_rng(0).integers(-15, 15, canvas.shape), 0, 255).astype(np.uint8)
    small = cv2.resize(a, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


DEGRADATIONS = {"defocus_blur": blur, "motion_blur": motion_blur, "dark": dark,
                "glare": glare, "leaf_too_small": too_small}
TIP_KEYS = {"blurry": "blur", "dark": "dark", "Overexposed": "glare", "small": "move closer",
            "borderline": "borderline"}


def tips_of(message: str) -> list[str]:
    return [v for k, v in TIP_KEYS.items() if k in message] or ["other"]


def main():
    report = {}
    for split in ("train", "test"):
        res = [(p, *check_quality(Image.open(p).convert("RGB"))) for p in list_images(Path("data") / split)]
        rejected = [(str(p), round(s, 3), m) for p, ok, s, m in res if not ok]
        report[f"real_{split}"] = {"n": len(res), "pass_rate": 1 - len(rejected) / len(res),
                                   "rejected": rejected}
        print(f"real {split:5s}: pass rate {1 - len(rejected) / len(res):.1%} ({len(res) - len(rejected)}/{len(res)})")

    test = list_images(Path("data/test"))
    # Only degrade photos that pass the gate when clean, so a rejection is caused by the degradation.
    clean = [p for p in test if check_quality(Image.open(p).convert("RGB"))[0]]
    for name, fn in DEGRADATIONS.items():
        tips, rejected = Counter(), 0
        for p in clean:
            ok, _, msg = check_quality(Image.fromarray(fn(np.array(Image.open(p).convert("RGB")))))
            if not ok:
                rejected += 1
                tips.update(tips_of(msg))
        report[name] = {"n": len(clean), "rejection_rate": rejected / len(clean), "tips": dict(tips)}
        print(f"{name:15s}: rejected {rejected / len(clean):6.1%} ({rejected}/{len(clean)}) tips {dict(tips)}")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
