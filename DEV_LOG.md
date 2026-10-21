# PlantDoc – Development Log

**NexAI 2026 Hackathon** · Track 2 (Smart Agriculture & Computer Vision) · Problem Statement 2.2: Leaf Doctor for Real Field Photos

---

## 🏗️ Architecture Pipeline

```
[Phone / Camera Upload] 
         │
         ▼
[Image Quality Gate] (OpenCV Laplacian Blur, HSV Brightness, Leaf Coverage + Fuzzy FIS)
         │
         ├─── (Score < 0.50) ──► ❌ Reject with Actionable Retake Advice ("move closer", "avoid glare")
         ▼ (Score ≥ 0.50)
[CLAHE (LAB L-channel)] ──► [Hybrid: Swin-T ⊕ DINOv2 ViT-B ⊕ ConvNeXt-T + calibrated head] ──► Top-3 Predictions
         │
         ▼
[Mamdani Fuzzy Trust Engine] (Quality Score × Model Confidence)
         │
         ├─── (Trust < 0.65) ──► ⚠️ "Not sure, please consult an agricultural expert."
         ▼ (Trust ≥ 0.65)
[Trusted Diagnosis & Explainability]
         ├── Top-3 Ranked Predictions with Confidence Bars
         ├── 🔬 Grad-CAM Heatmap (Visual Explainability / Saliency)
         ├── 🌐 Multilingual UI (English / हिंदी / தமிழ்)
         └── 🌾 Agronomic Care Notes (ICAR / TNAU extension guidelines)
```

| File | Role |
|---|---|
| `src/quality_gate.py` | Blur / brightness / leaf-coverage features + fuzzy quality score, returns `(passed, score, message)` |
| `src/fuzzy_engine.py` | Shared membership functions, Mamdani defuzzifier, Trust Engine |
| `src/model.py` | `PlantDocModel`: timm backbone, 6-class head, layered weight loading |
| `src/hybrid.py` | `HybridNet`: frozen Swin-T + DINOv2-B + ConvNeXt-T feature fusion, linear head, temperature scaling |
| `src/preprocess.py` | CLAHE contrast normalisation on the LAB L channel |
| `experiments/benchmark_backbones.py` | Frozen-backbone ± CLAHE benchmark and fusion search (5-fold CV, train split only) |
| `train_hybrid.py` | Fits the hybrid head + out-of-fold temperature → `weights/plantdoc_hybrid.pth` |
| `src/gradcam.py` | Grad-CAM saliency map (hooks the ConvNeXt branch of the hybrid, or EfficientNet `conv_head`) (Stretch Goal 1) |
| `src/i18n.py` | Multilingual localization dictionary for English, Hindi, and Tamil (Stretch Goal 2) |
| `src/agri_advisory.py` | Expert agronomic care notes cited from ICAR and TNAU (Stretch Goal 3) |
| `src/prepare_dataset.py` | PlantDoc bare git object stream extractor with Windows NTFS filename sanitization |
| `train.py` | 2-phase transfer learning fine-tuning with cosine decay and class balancing |
| `evaluate.py` | Held-out test evaluation, confusion matrix, and coverage curve plotter |
| `app.py` | Streamlit mobile-first UI with camera input, language toggle, and explainability tabs |
| `generate_pitch_deck.py` | Automated 7-slide PowerPoint presentation deck (`PlantDoc_Pitch_Deck.pptx`) |

---

## 🔬 Technical Decisions & Milestones

### 1. Dataset Extraction & Windows Compatibility
- **Dataset:** 6 classes across Tomato and Potato from official IIT PlantDoc repository (CC BY 4.0).
- **Challenge on Windows:** Internet-scraped images contained illegal NTFS characters (`?`, `&`, `=`, `%`), causing standard `git checkout` to fail.
- **Solution:** Designed `src/prepare_dataset.py` using a bare repository clone (`git clone --bare --filter=blob:none`) and stream extraction via `git cat-file -p`, validating image headers with PIL and saving with clean sequential filenames (`0001.jpg`, etc.).
- **Result:** Successfully extracted 541 valid training images and 52 held-out test images with zero corruption.

### 2. Model Training & Transfer Learning
- **Backbone:** `efficientnet_b0` pretrained on ImageNet via `timm`.
- **Classes (6):**
  1. `Potato Early Blight`
  2. `Potato Late Blight`
  3. `Tomato Bacterial Spot`
  4. `Tomato Early Blight`
  5. `Tomato Healthy`
  6. `Tomato Late Blight`
- **Training Strategy:**
  - Phase 1: 3 warmup epochs with frozen backbone (`lr=1e-3`).
  - Phase 2: 7 fine-tuning epochs with `CosineAnnealingLR` (`lr=1e-4`).
  - Class weighting in Cross-Entropy Loss to handle class size differences.
- **Outcome:** Best checkpoint saved at `weights/plantdoc_effnet_b0.pth` (16.3 MB) with **50.0% Top-1 Accuracy and 82.7% Top-3 Accuracy** on difficult, internet-sourced field photos.

- **Correction (2026-10-22):** after `train.py` was rerun, the checkpoint scored **38.5% top-1 / 80.8% top-3** on the test set (validation peaked at 40%, while training accuracy reached 70%). The 50% figure above came from an earlier run. Fine-tuning all 5 M parameters on 541 images overfits.

### 2b. CLAHE + Hybrid Feature-Fusion Model (current model)
- **GPU:** switched to `torch 2.14.1+cu130`, since the RTX 5060 (Blackwell) needs CUDA ≥ 12.8. Inference on the hybrid takes 0.06 s per image on GPU and 0.37 s on CPU.
- **Approach:** with 541 images, frozen pretrained features plus a regularised linear head beat end-to-end fine-tuning. Every backbone was benchmarked as a frozen feature extractor with logistic regression, using stratified 5-fold CV on `data/train` only. The test set was used exactly once, for the final model.
- **Single backbones (CV top-1, raw / CLAHE):**

  | Backbone | Raw | CLAHE |
  |---|---|---|
  | EfficientNet-B0 (IN-1k) | 50.1% | 50.8% |
  | EfficientNetV2-S (IN-21k) | 59.1% | 56.6% |
  | ConvNeXt-T (IN-22k) | 59.9% | 58.8% |
  | Swin-T (IN-22k) | 57.9% | 60.1% |
  | DINOv2 ViT-S/14 | 57.5% | 57.9% |
  | DINOv2 ViT-B/14 | 60.1% | 59.7% |

- **Fusion:** each backbone's embedding is L2-normalised, the embeddings are concatenated, standardised, and fed to logistic regression (class-balanced, `C=0.005`). Out of 10 two- and three-backbone combinations, **Swin-T + DINOv2-B + ConvNeXt-T** scored best. It mixes local convolutional texture (lesion spots), windowed attention, and self-supervised global shape features. Averaged over 3 CV seeds: 63.3% raw vs 62.8% CLAHE.
- **CLAHE finding:** CLAHE was applied to the L channel only, so lesion hue is preserved. The effect stayed within ±2.5 points, which is CV noise at n=541, and for the final hybrid the difference was 0.5 points. CLAHE is kept, applied the same way in training and inference, as a robustness measure for low-light and shaded field photos. It is **not** a source of accuracy gain on this dataset. The quality gate still scores the *raw* photo, so CLAHE cannot hide a dark or blurry capture.
- **Calibration:** temperature T = 1.35 was fitted on out-of-fold logits, because the fuzzy trust engine consumes softmax confidence and an overconfident head would undermine it. Prediction also averages logits over the original and horizontally flipped views.
- **Checkpoint:** `weights/plantdoc_hybrid.pth` stores only the head, scaler and temperature (small). Backbone weights come from the timm / HF Hub cache, so the first run needs internet. `PlantDocModel` loads the hybrid when this file exists and otherwise falls back to EfficientNet-B0.

### 3. Held-Out Evaluation & Coverage Curve Proof
- **Current hybrid (52 test photos):**

  | Metric | EffNet-B0 fine-tuned | **Hybrid + CLAHE** |
  |---|---|---|
  | Top-1 accuracy | 38.5% | **61.5%** |
  | Top-3 accuracy | 80.8% | **100.0%** |
  | Quality-gate pass rate | 84.6% | 84.6% |
  | Trusted rate | 32.7% | 30.8% |
  | Accuracy on TRUSTED | 47.1% | **81.3%** |

  The test set is small (±7 points at n=52), but the direction agrees with CV (+12 points over the best single backbone, +13 over frozen EfficientNet-B0).
- *Earlier EfficientNet-B0 run (superseded):*
- Evaluated on 52 held-out test photos:
  - Overall Top-1 Accuracy: **50.00%**
  - Overall Top-3 Accuracy: **76.92%**
  - Quality Gate Pass Rate: **84.62%**
  - Accuracy on TRUSTED photos: **60.00%** (+10.00% precision boost!)
  - Peak Selective Accuracy: **71.43%** at Trust Threshold $\ge 0.69$.
- **Conclusion:** The Coverage Curve mathematically proves that the Quality Gate and Mamdani Fuzzy Trust Engine filter out ambiguous and poor photos, preventing confident misdiagnoses.

### 4. Stretch Goals Implemented
1. **Grad-CAM Saliency Maps (`src/gradcam.py`):** Visual explainability hooked to `conv_head`. Shows farmers that the model inspects actual necrotic lesions rather than soil or fingers.
2. **Multilingual Localization (`src/i18n.py`):** UI strings, retake advice, and disease terminology fully translated into Hindi (हिन्दी) and Tamil (தமிழ்).
3. **Agronomic Care Notes (`src/agri_advisory.py`):** Curated cultural and chemical disease management advice cited directly from ICAR-CPRI, ICAR-IIHR, and TNAU Agritech portals.

---

## 🚫 Rejected Approaches
- **PlantVillage pre-training:** Rejected because lab-photo models learn clean backgrounds and fail on field noise (the exact trap PS 2.2 avoids).
- **End-to-end fine-tuning on 541 images:** overfit (70% train vs 40% validation). Replaced by frozen feature fusion.
- **GAN / StyleGAN augmentation:** StyleGAN needs thousands of images per class and can invent lesions, which corrupts labels. Not feasible with ~100 images per class.
- **GAN restoration (Real-ESRGAN / deblur) at inference:** hallucinated texture can add or erase lesions. Rejecting the photo and asking for a retake is safer.
- **Vision Mamba / SSMs:** `mamba-ssm` needs Linux CUDA kernels. At 224 px (196 tokens) the linear-complexity advantage is negligible, Vim-Ti has about 4× the FLOPs of EfficientNet-B0 for similar ImageNet accuracy, and a weak inductive bias needs more data than we have.
- **Per-branch CLAHE mixing:** looked +2.5 points better on one CV seed but complicates inference; across 3 seeds it was not distinguishable.
- **Hard Boolean Thresholds:** Rejected in favor of Mamdani fuzzy membership, allowing soft degradation across borderline lighting and blur.
- **Unverified Softmax Trust:** Softmax is notoriously overconfident on out-of-distribution field photos. Combining Image Quality with Softmax confidence via fuzzy logic ensures photo defects pull down the trust score.

---

## 📋 Status & Submission Checklist
- [x] Problem Statement fixed to 2.2 in all documents
- [x] Image Quality Gate implemented & unit tested
- [x] Mamdani Fuzzy Trust Engine active with exact `"Not sure..."` message
- [x] Dataset extracted (541 train, 52 test images across 6 classes)
- [x] Model trained & weights saved (`weights/plantdoc_effnet_b0.pth`)
- [x] Held-out evaluation performed & `coverage_curve.png` generated
- [x] CLAHE preprocessing + hybrid Swin-T/DINOv2-B/ConvNeXt-T model (61.5% top-1, 100% top-3, 81.3% trusted accuracy)
- [x] CUDA torch on RTX 5060; temperature-calibrated confidence
- [ ] Optional: partial fine-tuning (last stage of each branch) with the GPU
- [x] Stretch Goal 1 (Grad-CAM) completed
- [x] Stretch Goal 2 (Multilingual UI: EN, HI, TA) completed
- [x] Stretch Goal 3 (ICAR/TNAU Care Notes) completed
- [x] Automated test suite passing (`tests/test_pipeline.py`)
- [x] PowerPoint Pitch Deck generated (`PlantDoc_Pitch_Deck.pptx`)
- [x] 5-minute video presentation script drafted (`DEMO_SCRIPT.md`)
- [x] Social media posts drafted (`SOCIAL_POSTS.md`)
- [x] Git repo initialized with progressive commit history

---

## 🤖 AI Assistance & Tool Disclosure
As required by Section 3 of Hackathon Instructions:
Initial scaffolding, fuzzy rule formulation, Grad-CAM hook integration, and PowerPoint generation were assisted by AI coding tools and verified with empirical unit tests and held-out evaluations.
