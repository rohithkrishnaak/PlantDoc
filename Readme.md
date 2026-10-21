---
title: PlantDoc Leaf Doctor
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.66.1
app_file: app.py
pinned: false
---

# PlantDoc: Leaf Doctor for Real Field Photos 🌿

**NexAI Hack 2026** | **Track 2:** Smart Manufacturing & Computer Vision | **Problem Statement:** 2.2 (Agriculture)

[![Live App](https://img.shields.io/badge/Live_App-HuggingFace_Spaces-green)](https://huggingface.co/spaces/rohithkrishnaak/PlantDoc)
[![GitHub Repo](https://img.shields.io/badge/GitHub-PlantDoc-blue)](https://github.com/rohithkrishnaak/PlantDoc)
[![Presentation Deck](https://img.shields.io/badge/Pitch_Deck-PowerPoint_PPTX-orange)](./PlantDoc_Pitch_Deck.pptx)
[![Demo Script](https://img.shields.io/badge/Demo-5_Min_Script-purple)](./DEMO_SCRIPT.md)

---

## 📖 Problem Statement and Solution Overview

### The Problem
Plant diseases destroy up to 40% of Indian farmers' harvest, with early detection severely hindered by the lack of local agronomists. While standard AI models achieve ~99% accuracy on sterile lab photos (e.g., PlantVillage), they fail drastically on real smartphone photos taken in fields under variable sunlight, motion blur, and background soil clutter. Crucially, standard classifiers emit **dangerously confident false predictions**, causing farmers to purchase and spray the wrong chemical pesticides.

### Our Solution
**PlantDoc** is a mobile-first field diagnostic assistant designed with **Responsible AI and Explainability** at its core:
1. **OpenCV Image Quality Gate**: Evaluates Laplacian variance (blur), HSV V-channel mean (brightness), and dual-mask leaf canopy coverage. Rejects unusable photos with actionable retake tips (*"move closer"*, *"tap to focus"*).
2. **Transfer Learning Classifier**: EfficientNet-B0 backbone fine-tuned on real-world noisy field images from the IIT PlantDoc dataset (CC BY 4.0).
3. **Mamdani Fuzzy Trust Engine**: Combines image quality and softmax confidence through fuzzy rules. When an image is poor or the model is ambiguous, PlantDoc outputs: **`"Not sure, please consult an agricultural expert."`**
4. **Grad-CAM Explainability (Stretch Goal 1)**: Visual saliency heatmaps highlighting leaf lesions so farmers and agronomists can verify *what* drove the diagnosis.
5. **Multilingual Interface (Stretch Goal 2)**: Full UI localization in English, Hindi (हिन्दी), and Tamil (தமிழ்).
6. **ICAR / TNAU Care Notes (Stretch Goal 3)**: Expert-backed organic & chemical management advice cited from agricultural university portals.

---

## 🏗️ Architecture & Pipeline Flow

```
[Phone / Camera Upload] 
         │
         ▼
[Image Quality Gate] (OpenCV Laplacian Blur, HSV Brightness, Leaf Coverage + Fuzzy FIS)
         │
         ├─── (Score < 0.50) ──► ❌ Reject with Actionable Retake Advice ("move closer", "avoid glare")
         ▼ (Score ≥ 0.50)
[Fine-Tuned EfficientNet-B0] ──► Top-3 Predictions & Softmax Probabilities
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

---

## 📊 Evaluation & Coverage Curve Results

Evaluated strictly on held-out test field photos from the official IIT PlantDoc dataset (52 test images across 6 classes):

| Metric | Result | Description |
|---|---|---|
| **Top-1 Test Accuracy** | **50.00%** | Unfiltered baseline on difficult field photos |
| **Top-3 Test Accuracy** | **76.92%** | Correct disease within top 3 ranked candidates |
| **Quality Gate Pass Rate** | **84.62%** | Filters out extreme blur, darkness, and framing defects |
| **Accuracy on TRUSTED Photos** | **60.00%** | **+10.00% boost** when filtered by Trust Engine |
| **Peak Selective Accuracy** | **71.43%** | Achieved at Trust Threshold $\ge 0.69$ |

### Coverage Curve Analysis
![Coverage Curve](./coverage_curve.png)
As the Mamdani Trust Engine filters out low-trust and poor-quality photos, diagnostic precision on accepted images rises sharply (from 50% to over 71%), confirming that the system successfully mitigates dangerous false diagnoses.

---

## ⚙️ Setup and Execution Instructions

### Prerequisites
* Python 3.10+
* Git

### Local Installation
```bash
# 1. Clone repository
git clone https://github.com/rohithkrishnaak/PlantDoc.git
cd PlantDoc

# 2. Create virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Run the Streamlit web application
streamlit run app.py
```

### Automated Testing
Run the complete pipeline test suite (quality gate, fuzzy engine, i18n, advisories, Grad-CAM):
```bash
python tests/test_pipeline.py
```

---

## 🎯 Stretch Goals Completed

| Feature | Description | Status |
|---|---|---|
| **🔬 Grad-CAM Heatmaps** | Generates class activation maps over the leaf to highlight lesions | ✅ Completed |
| **🌐 Multilingual UI** | Toggle English, Hindi (हिन्दी), and Tamil (தமிழ்) | ✅ Completed |
| **🌾 Extension Care Notes** | ICAR/TNAU-cited organic & chemical management advice | ✅ Completed |

---

## 📊 Dataset Attribution & Licensing
* **Dataset:** [PlantDoc Dataset (GitHub)](https://github.com/pratikkayal/PlantDoc-Dataset)
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Authors:** Singh et al., Indian Institute of Technology (IIT)
* **Subset Used:** 6 target classes across Tomato and Potato (541 train, 52 held-out test images) to target high-impact Indian staple crops.

---

## 🧑‍💻 Team Members
* **Rohith Krishna A K** - [GitHub](https://github.com/rohithkrishnaak) / [LinkedIn](https://linkedin.com/in/rohithkrishnaak)

---

## 📓 Development Log & Deliverables
* **Technical Decisions & Experiments:** See [DEV_LOG.md](./DEV_LOG.md)
* **PowerPoint Pitch Deck:** [PlantDoc_Pitch_Deck.pptx](./PlantDoc_Pitch_Deck.pptx)
* **5-Minute Video Presentation Script:** [DEMO_SCRIPT.md](./DEMO_SCRIPT.md)
* **Social Media Announcements (#NEXAI2026):** [SOCIAL_POSTS.md](./SOCIAL_POSTS.md)