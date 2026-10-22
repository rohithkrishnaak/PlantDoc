# PlantDoc – Development Log

**NexAI 2026 Hackathon** · Track 2 (Smart Agriculture & Computer Vision) · Problem Statement 2.2: Leaf Doctor for Real Field Photos

---

## 🏗️ Architecture Pipeline

```
[Phone / Camera Upload] 
         │
         ▼
[Image Quality Gate] (Laplacian + directional blur, HSV brightness, leaf coverage + Fuzzy FIS)
         │
         ├─── (Score < 0.50) ──► ❌ Reject with Actionable Retake Advice ("move closer", "avoid glare")
         ▼ (Score ≥ 0.50)
[CLAHE (LAB L-channel)] ──► [Hybrid: ConvNeXt-T ⊕ DINOv2 ViT-S + linear head] ──► Top-3 Predictions
         │
         ├─── (far from every training photo) ──► ❌ "Does not look like a supported tomato/potato leaf"
         ▼
[Mamdani Fuzzy Trust Engine] (Quality Score × Model Confidence)
         │
         ├─── (Trust < 0.61, chosen on training CV) ──► ⚠️ "Not sure, please consult an agricultural expert."
         ▼ (Trust ≥ 0.61)
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

### 2c. Data Integrity, Leakage Fix & Final Model (22 Oct morning; current model)
- **Why:** PS 2.2 warns "Internet-scraped images can repeat; check that no test image also sits in training." We hadn't checked.
- **`src/dedup.py`:** MD5 for exact copies plus a 64-bit difference hash (Hamming ≤ 4) for re-encoded or resized copies.
  - **7 of 52 test photos were also in training** (5 exact, 2 near). 5 of those carried a *different* label in train, e.g. the same photo filed as Potato Early Blight in test and Potato Late Blight in train.
  - **15 duplicate groups inside train had conflicting labels**, 31 images in total (some even potato vs tomato). That is label noise in PlantDoc itself.
  - **Action:** moved the 7 test duplicates to `data/test_removed/` and the 31 conflicting train images to `data/train_removed/`. The record is in `experiments/duplicates.json`. Result: 510 train and 45 test images, 0 cross-split duplicates.
  - The remaining 10 same-label duplicate groups in train are kept together in one fold (`StratifiedGroupKFold`), so CV never tests on a copy of a training photo.
- **Leakage check on the old model:** the 7 duplicates scored *lower* (28.6% top-1) than the clean 45 (66.7%), because their labels were contradictory. The leak did not inflate the earlier numbers, but the old figures were still on a dirty split, so all results below are on the clean one.
- **Re-benchmark (grouped CV, clean data):** removing the label noise lifted every backbone by 2–5 points.
  - Averaged over 3 CV seeds, **ConvNeXt-T + DINOv2 ViT-S + CLAHE scored 67.3%**, tied with the best 3-backbone hybrid (Swin-T + ConvNeXt-T + DINOv2-S, 67.3%) and ahead of the previous Swin-T + DINOv2-B + ConvNeXt-T (66.8%).
  - It was chosen for having **~50 M parameters instead of ~140 M**: 0.13 s per photo on CPU instead of 0.37 s, which matters on a free 2-vCPU Hugging Face Space.
  - CLAHE: +1.2 points for this pair, again within noise. It is kept for low-light robustness.
- **Trust threshold chosen without the test set:** the 0.65 threshold was set by hand before calibration. `train_hybrid.py` now sweeps thresholds on *out-of-fold* predictions plus the quality-gate scores of the raw training photos. It picks the lowest threshold whose accepted photos reach ≥ 80% accuracy: **0.61** (35.7% OOF coverage, 80.2% OOF accuracy). This value is stored in the checkpoint and used by the app.
- **"Not a supported leaf" (out-of-distribution) check:** the checkpoint stores the 510 training features. If a photo's nearest training photo has cosine similarity below **0.294** (2nd percentile of leave-group-out nearest-neighbour similarity on train), the app refuses to diagnose. In tests, non-leaf photos (astronaut, cat, coffee, rocket) are rejected, and 0 of the 45 real test leaves are rejected.
- **Quality-gate fixes from a stress test** (`experiments/quality_gate_eval.py`, degrading the 38 test photos that pass when clean):

  | Degradation | Before fixes | After fixes |
  |---|---|---|
  | Leaf too small on soil | **2.6%** rejected | **100%** |
  | Motion blur | 65.8% | 78.9% |
  | Defocus blur / dark / glare | 97% / 100% / 97% | unchanged |

  1. Bare soil matched the "brown lesion" colour range and counted as leaf. Brown now counts only within ~15 px of de-speckled green tissue, and leaf outlines are filled so lesions inside a leaf still count.
  2. Motion blur smears one direction, so a horizontal/vertical gradient-ratio signal was added. It only applies when the photo isn't already clearly sharp, because strong parallel veins also skew the ratio and a test caught that false rejection.
  3. Retake tips put lighting first: glare and darkness also lower sharpness and hide green, which used to add misleading "blurry" and "move closer" tips.
  4. The pass rate on real photos is unchanged: 89.8% train, 84.4% test.
- **App robustness:**
  - Grad-CAM registered new hooks on the shared cached model on every request (a leak); hooks are now removed after each call.
  - HEIC (iPhone) uploads are supported, unreadable files get a friendly error, and huge photos are downscaled.
  - When both a camera photo and an upload exist, the most recent one is used.
  - The "not sure" reason and the out-of-distribution message are translated into Hindi and Tamil.
- **Tests:** 12 checks pass. New ones cover: exact "not sure" text, CLAHE, duplicate hashing, small leaf on soil, image modes (L/RGBA/CMYK/P), out-of-distribution rejection, and no Grad-CAM hook leak.

### 3. Held-Out Evaluation & Coverage Curve (final, de-duplicated test set)
- **45 held-out photos, test set used once, after all choices were made on training CV:**

  | Metric | Fine-tuned EffNet-B0 (dirty split) | **ConvNeXt-T + DINOv2-S + CLAHE** | 95% bootstrap CI |
  |---|---|---|---|
  | Top-1 accuracy | 38.5% | **66.7%** | 51–80% |
  | Top-3 accuracy | 80.8% | **97.8%** | 93–100% |
  | Quality-gate pass rate | 84.6% | 84.4% | |
  | Real leaves flagged out-of-distribution | n/a | 0/45 | |
  | Answered (trusted) | 32.7% | 42.2% (19/45) | |
  | **Accuracy on answered photos** | 47.1% | **89.5%** (17/19) | 74–100% |

- Per-class F1 on all 45 photos ranges from 0.59 (Tomato Early Blight) to 0.80 (Potato Late Blight). The most common confusion is Tomato Healthy read as Bacterial Spot (3). See `evaluation_report.json`, `confusion_matrix.png` and `experiments/test_predictions.csv`.
- **Coverage curve:** answering every photo gives 66.7% correct; at the 0.61 operating point the app answers 42% and is right on 89.5%. It reduces confident wrong diagnoses but does not remove them (2 of 19 answered photos are wrong).
- *Superseded:* the first EfficientNet-B0 run reported 50% top-1, but its checkpoint was chosen on the test set (fixed in `train.py`) and the split still had duplicates. Those numbers should not be quoted.

### 4. Stretch Goals Implemented
1. **Grad-CAM Saliency Maps (`src/gradcam.py`):** Visual explainability hooked to the ConvNeXt branch of the hybrid (labelled as such in the UI). Shows farmers that the model inspects actual necrotic lesions rather than soil or fingers.
2. **Multilingual Localization (`src/i18n.py`):** UI strings, retake advice, and disease terminology fully translated into Hindi (हिन्दी) and Tamil (தமிழ்).
3. **Agronomic Care Notes (`src/agri_advisory.py`):** Curated cultural and chemical disease management advice cited directly from ICAR-CPRI, ICAR-IIHR, and TNAU Agritech portals.

---

## 🚫 Rejected Approaches
- **PlantVillage pre-training:** Rejected because lab-photo models learn clean backgrounds and fail on field noise (the exact trap PS 2.2 avoids).
- **End-to-end fine-tuning on 541 images:** overfit (70% train vs 40% validation). Replaced by frozen feature fusion.
- **GAN / StyleGAN augmentation:** StyleGAN needs thousands of images per class and can invent lesions, which corrupts labels. Not feasible with ~100 images per class.
- **GAN restoration (Real-ESRGAN / deblur) at inference:** hallucinated texture can add or erase lesions. Rejecting the photo and asking for a retake is safer.
- **Vision Mamba / SSMs:** `mamba-ssm` needs Linux CUDA kernels. At 224 px (196 tokens) the linear-complexity advantage is negligible, Vim-Ti has about 4× the FLOPs of EfficientNet-B0 for similar ImageNet accuracy, and a weak inductive bias needs more data than we have.
- **Keeping the 3-backbone Swin-T + DINOv2-B + ConvNeXt-T hybrid:** on clean data it scored no better than ConvNeXt-T + DINOv2-S (66.8% vs 67.3%) at ~3× the CPU cost.
- **Lowering the trust threshold to answer more photos:** at 0.50 the app would answer ~51% of photos at ~77% OOF accuracy. We kept the ≥ 80% target, because a wrong diagnosis means the wrong pesticide.
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
- [x] Train/test de-duplication and label-conflict removal (`src/dedup.py`): 510 train / 45 test
- [x] Final model ConvNeXt-T + DINOv2-S + CLAHE: 66.7% top-1, 97.8% top-3, 89.5% accuracy on answered photos
- [x] Trust threshold (0.61) and out-of-distribution threshold chosen on training CV only
- [x] Quality gate stress-tested and fixed (soil-as-leaf, motion blur, tip priority)
- [x] Per-class results, confusion matrix, bootstrap CIs (`evaluate.py`)
- [x] CUDA torch on RTX 5060
- [ ] Deploy to Hugging Face Spaces and test on a phone
- [ ] Native-speaker review of Hindi/Tamil text and care-note dosages
- [ ] Optional: partial fine-tuning (last stage of each branch) with the GPU
- [x] Stretch Goal 1 (Grad-CAM) completed
- [x] Stretch Goal 2 (Multilingual UI: EN, HI, TA) completed
- [x] Stretch Goal 3 (ICAR/TNAU Care Notes) completed
- [x] Automated test suite passing (`tests/test_pipeline.py`)
- [x] PowerPoint Pitch Deck generated (`PlantDoc_Pitch_Deck.pptx`)
- [x] 5-minute video presentation script drafted (`DEMO_SCRIPT.md`)
- [x] Social media posts drafted (`SOCIAL_POSTS.md`)
- [x] Git repo initialized

**Commit history note:** on the first evening, work was done locally and committed in batches (several commits within seconds at 23:09 and 23:37 on 21 Oct, and at 01:20 on 22 Oct). Commit times show when work was committed, not when each piece was written. From the morning of 22 Oct, work is committed as each task finishes.

---

## 🤖 AI Assistance & Tool Disclosure
As required by Section 3 of Hackathon Instructions:
Claude Code (Anthropic) assisted with scaffolding, fuzzy rule formulation, Grad-CAM, the backbone benchmark, de-duplication, the quality-gate stress test and documentation. Every number in this log was produced by the scripts in this repository (`evaluate.py`, `train_hybrid.py`, `experiments/*.py`, `src/dedup.py`), and the team has reviewed the code.
