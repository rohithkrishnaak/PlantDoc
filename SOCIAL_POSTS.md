# NexAI Hack 2026 — Public Social Media Posts Drafts 📢

---

## 💼 LinkedIn Post Draft

**Headline:** Building PlantDoc: An Honest, Explainable Leaf Doctor for Real Indian Field Photos 🌿🇮🇳

Excited to share our solution for **#NEXAI2026** (Track 2: Smart Agriculture & Computer Vision, Problem Statement 2.2: *Leaf Doctor for Real Field Photos*)! 🚀

Plant diseases account for devastating yield losses across Indian agriculture. While standard deep learning models claim 99% accuracy on pristine lab photos, they crumble when faced with real smartphone photos taken in windy, sunny fields with soil clutter. Even worse, standard models emit dangerously confident misdiagnoses that encourage farmers to spray the wrong chemical pesticides.

To solve this, our team built **PlantDoc**:
1. 🛡️ **OpenCV Image Quality Gate**: Evaluates Laplacian blur, HSV lighting, and leaf canopy coverage via fuzzy inference, rejecting poor photos with actionable tips (*"move closer"*, *"tap to focus"*).
2. 🧠 **Transfer Learning (EfficientNet-B0)**: Fine-tuned on real field photos from the IIT PlantDoc dataset (CC BY 4.0) with field-specific augmentations.
3. ⚖️ **Mamdani Fuzzy Trust Engine**: Combines image quality with softmax confidence. If an image is borderline, PlantDoc never hallucinates—it outputs: *"Not sure, please consult an agricultural expert."*
4. 🔬 **Explainable AI (Grad-CAM)**: Visual saliency heatmaps highlighting leaf lesions so farmers know *why* the model made its prediction.
5. 🌐 **Vernacular Accessibility**: Full UI localization in English, Hindi (हिन्दी), and Tamil (தமிழ்).
6. 🌾 **ICAR & TNAU Extension Advisories**: Expert-backed cultural and chemical disease management guidelines.

🔗 **Live App:** https://huggingface.co/spaces/rohithkrishnaak/PlantDoc *(or your deployment URL)*
💻 **GitHub Repository:** https://github.com/rohithkrishnaak/PlantDoc
🎥 **Pitch Deck & Demo:** Linked in README

Huge thanks to the organizers of NexAI Hack 2026 for this meaningful problem statement!

#NEXAI2026 #AIForGood #ComputerVision #Agritech #MachineLearning #DeepLearning #PyTorch #ExplainableAI #FuzzyLogic

---

## 🐦 X (Twitter) Post Draft

Proud to present **PlantDoc** for #NEXAI2026 (Track 2: PS 2.2)! 🌿

Most crop disease AI models fail on real phone photos taken in fields. PlantDoc fixes this with:
✅ OpenCV Fuzzy Quality Gate (rejects blur/shadows with tips)
✅ EfficientNet-B0 on real field photos
✅ Mamdani Fuzzy Trust Engine (honest "Not sure" fallback)
✅ Grad-CAM heatmaps for explainability
✅ Hindi & Tamil UI + ICAR/TNAU care notes

💻 Repo: https://github.com/rohithkrishnaak/PlantDoc
🌐 Demo: https://huggingface.co/spaces/rohithkrishnaak/PlantDoc

#NEXAI2026 #Agritech #AI #ComputerVision #PyTorch
