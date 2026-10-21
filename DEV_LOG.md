# PlantDoc – Development Log

NexAI 2026 Hackathon · Track 2 (Smart Agriculture) · Problem 2.2

## Pipeline

```
photo ─► Quality Gate (OpenCV features ─► fuzzy FIS) ─► score < 0.5 ? reject + retake tip
                                                      └─► Classifier (timm EfficientNet-B0, 5 classes)
                                                            └─► Fuzzy Trust Engine (quality × confidence)
                                                                  └─► trust < 0.65 ? "Not sure, please consult an agricultural expert."
                                                                                  : top-3 predictions
```

| File | Role |
|---|---|
| `src/quality_gate.py` | Blur / brightness / leaf-coverage features + fuzzy quality score, returns `(passed, score, message)` |
| `src/fuzzy_engine.py` | Shared membership functions, Mamdani defuzzifier, Trust Engine |
| `src/model.py` | `PlantDocModel`: timm backbone, 5-class head, layered weight loading |
| `app.py` | Streamlit UI: camera + upload, mobile-width layout |

## Technical decisions

### Quality Gate
- **Blur:** variance of the Laplacian. The image is resized first so its longest side is 512 px. Without this, a 12 MP phone photo and a 300 px thumbnail give values on different scales, and one threshold can't work for both.
- **Brightness:** mean of the HSV V channel. It catches dark shadows and blown-out glare.
- **Leaf coverage:** HSV masks for green tissue (H 25–95) and brown/yellow lesions (H 5–25). The mask is cleaned with morphological open/close, and coverage is the area of the 5 largest external contours. Lesion tones are included on purpose: a heavily diseased brown leaf must not be rejected as "no leaf".
- **Fuzzy, not hard thresholds:** each feature gets trapezoidal sets (blurry/ok/sharp, dark/good/over, low/ok/high). The rules give a POOR / FAIR / GOOD output, defuzzified by centroid. A photo that is slightly soft but well lit and well framed degrades gracefully instead of failing on one cutoff.
- **Actionable tips** come from the membership degrees, not just the final score. For example, `coverage.low > 0.3` gives "move closer". The blur tip is hidden when the image is dark, because low light also lowers the Laplacian variance and the real fix is light.

### Fuzzy Trust Engine
- Inputs: quality score ∈ [0,1] and top-1 softmax confidence ∈ [0,1].
- Rules (OR = max, AND = min):
  - R1 Quality LOW **OR** Confidence LOW → UNCERTAIN
  - R2 Quality HIGH **AND** Confidence HIGH → HIGH
  - R3–R5 combinations involving MEDIUM → MODERATE
- Centroid defuzzification on [0,1]. If trust < **0.65**, status is `UNCERTAIN` with the exact message *"Not sure, please consult an agricultural expert."*
- Sanity checks: (Q 0.9, C 0.95) → 0.87 TRUSTED · (0.9, 0.5) → 0.32 UNCERTAIN · (0.3, 0.95) → 0.22 UNCERTAIN · (0.7, 0.75) → 0.68 TRUSTED.
- Why fuzzy rather than `conf > t`: a confident prediction on a borderline photo is exactly the field failure mode we care about. The rule base lets photo quality pull trust down even when the softmax is overconfident.
- scikit-fuzzy is used for `trapmf` / `defuzz` when it is installed. Otherwise an equivalent NumPy implementation is used, so the app never fails on that import.

### Classifier
- `timm` **EfficientNet-B0** (`MODEL_ARCH` env var switches to `mobilenetv3_large_100`). It is small (≈5 M params) and runs fast on CPU, which suits a mobile web demo.
- Weight loading, in order:
  1. A fine-tuned checkpoint at `MODEL_WEIGHTS`.
  2. The ImageNet backbone from the HF Hub with a new 5-class head.
  3. Random init when offline.
- **Demo-mode safety:** an untrained head produced 97% confidence on a synthetic image, which is a confident misdiagnosis. Whenever no fine-tuned checkpoint is loaded, the head is zero-initialised. Softmax is then uniform (0.20), so the Trust Engine always answers UNCERTAIN. The UI also shows a demo-mode banner.

## Rejected approaches
- **Hard if/else thresholds for quality:** brittle at the boundaries and no graded score to feed into trust.
- **Raw softmax threshold only:** softmax is overconfident on out-of-distribution field photos and ignores image quality.
- **scikit-fuzzy `ControlSystem` API:** heavier, rebuilds a simulation per call, and is harder to fall back from. Using the plain membership/defuzz primitives is enough for a few rules.
- **Large backbones (ViT, ResNet-50):** too slow for CPU inference on a phone-facing demo, and not needed for 5 classes.
- **Showing predictions from the untrained head in demo mode:** it would demo well but is dishonest and unsafe (wrong pesticide advice).

## Status / TODO
- [x] Quality gate, trust engine, model wrapper, Streamlit UI
- [x] Smoke-tested with synthetic good / blurry / dark / no-leaf images
- [x] App launched on `localhost:8501` (health check 200, no script exceptions)
- [ ] Fine-tune on the PlantDoc subset (5 classes, heavy augmentation) → `weights/plantdoc_effnet_b0.pth`
- [ ] Calibrate quality thresholds on real field photos
- [ ] Optional: temperature scaling to calibrate softmax confidence
- [ ] Deploy (HF Spaces / Streamlit Cloud)

## AI assistance
Scaffolding, the fuzzy rule base and the OpenCV feature code were drafted with Claude Code and then verified with smoke tests.
