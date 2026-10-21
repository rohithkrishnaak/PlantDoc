"""Comprehensive automated test suite for PlantDoc pipeline.

Tests:
1. Fuzzy logic primitives and Mamdani defuzzification.
2. Fuzzy Trust Engine rules and safety boundary conditions.
3. OpenCV Image Quality Gate feature extraction and retake tips.
4. Multilingual i18n dictionary completeness (EN, HI, TA).
5. Agronomic care notes coverage (ICAR/TNAU).
6. Grad-CAM visual explainability heatmap generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import unittest
import torch
from PIL import Image

from src.agri_advisory import ADVISORIES, get_advisory
from src.fuzzy_engine import (
    CONF_SETS,
    QUALITY_SETS,
    TRUST_SETS,
    defuzzify,
    evaluate_trust,
    mu,
    trapmf,
    trimf,
)
from src.gradcam import GradCAM
from src.i18n import DISEASE_TRANSLATIONS, TRANSLATIONS, t, t_disease
from src.model import CLASSES, PlantDocModel
from src.quality_gate import check_quality, compute_features, fuzzy_quality


# 1. Fuzzy Engine Tests
def test_fuzzy_trapmf():
    # Inside flat top
    assert trapmf(5, (0, 3, 7, 10)) == 1.0
    # Left rise
    assert 0.0 < trapmf(1.5, (0, 3, 7, 10)) < 1.0
    # Outside
    assert trapmf(12, (0, 3, 7, 10)) == 0.0


def test_trust_engine_safety():
    # High Quality + High Confidence -> TRUSTED
    res1 = evaluate_trust(quality=0.9, confidence=0.95)
    assert res1.status == "TRUSTED"
    assert res1.trust >= 0.65

    # Poor Quality + High Confidence -> UNCERTAIN (Critical Safety Case!)
    res2 = evaluate_trust(quality=0.3, confidence=0.95)
    assert res2.status == "UNCERTAIN"
    assert res2.trust < 0.65

    # High Quality + Low Confidence -> UNCERTAIN
    res3 = evaluate_trust(quality=0.9, confidence=0.4)
    assert res3.status == "UNCERTAIN"
    assert res3.trust < 0.65


# 2. Quality Gate Tests
def test_quality_gate_synthetic_images():
    # A. Sharp green leaf-like circle on daylight background
    img_sharp = np.ones((400, 400, 3), dtype=np.uint8) * 160
    cv2.circle(img_sharp, (200, 200), 160, (40, 180, 50), -1)
    # Add texture/leaf veins
    for i in range(50, 350, 5):
        cv2.line(img_sharp, (i, 60), (i, 340), (20, 120, 30), 2)
    passed_sharp, score_sharp, _ = check_quality(img_sharp)
    assert score_sharp > 0.50

    # B. Blurry image
    img_blurry = cv2.GaussianBlur(img_sharp, (51, 51), 0)
    passed_blurry, score_blurry, tip_blurry = check_quality(img_blurry)
    assert score_blurry < score_sharp
    assert "blurry" in tip_blurry.lower() or score_blurry < 0.5

    # C. Pitch dark image
    img_dark = np.ones((400, 400, 3), dtype=np.uint8) * 10
    passed_dark, score_dark, tip_dark = check_quality(img_dark)
    assert not passed_dark
    assert "dark" in tip_dark.lower() or "light" in tip_dark.lower()

    # D. Blank image (No leaf coverage)
    img_blank = np.ones((400, 400, 3), dtype=np.uint8) * 200
    passed_blank, score_blank, tip_blank = check_quality(img_blank)
    assert not passed_blank


# 3. i18n & Advisory Tests
def test_i18n_coverage():
    for key in ["app_title", "tab_camera", "tab_upload", "diagnosing", "uncertain_warning"]:
        for lang in ["en", "hi", "ta"]:
            translated = t(key, lang)
            assert len(translated) > 0

    for d in CLASSES:
        for lang in ["en", "hi", "ta"]:
            assert len(t_disease(d, lang)) > 0


def test_agri_advisories():
    for d in CLASSES:
        adv = get_advisory(d)
        assert adv is not None
        assert "symptoms" in adv
        assert "cultural_organic" in adv
        assert "chemical_control" in adv
        assert "source" in adv


# 4. Grad-CAM Test
def test_gradcam_execution():
    model = PlantDocModel()
    gradcam = GradCAM(model.model)
    dummy_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    heatmap, blended = gradcam.generate(dummy_img, target_class_idx=0)
    assert heatmap.shape == (224, 224, 3)
    assert blended.shape == (224, 224, 3)
    assert heatmap.dtype == np.uint8


if __name__ == "__main__":
    test_fuzzy_trapmf()
    print("[PASS] Fuzzy trapmf")
    test_trust_engine_safety()
    print("[PASS] Trust engine safety rules")
    test_quality_gate_synthetic_images()
    print("[PASS] Quality gate synthetic images")
    test_i18n_coverage()
    print("[PASS] Multilingual i18n coverage")
    test_agri_advisories()
    print("[PASS] Agronomic advisories")
    test_gradcam_execution()
    print("[PASS] Grad-CAM heatmap generation")
    print("\n[SUCCESS] ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
