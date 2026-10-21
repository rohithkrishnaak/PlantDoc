# PlantDoc: Leaf Doctor for Real Field Photos 🌿

**NexAI 2026 Hackathon** | **Track 2:** Smart Agriculture | **Problem Statement:** 2.1

[![Deployment URL](https://img.shields.io/badge/Live_App-Click_Here-success)](#) *(Update with your deployed link or Hugging Face Space)*
[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red)](#) *(Update with your video link)*

## 📖 Problem Statement and Solution Overview
Plant diseases cost farmers a massive share of their yield. While many AI models can classify diseases from clean lab photos (like PlantVillage), they fail in the real world. 

**Our Solution:** We built a mobile-friendly field diagnostic tool. Unlike standard classifiers, our app includes an **Image Quality Gate** that rejects blurry, dark, or poorly framed photos with actionable retake advice (e.g., "move closer"). If the image passes, it runs through a lightweight, transfer-learned model. To prevent confident misdiagnoses that could lead to the wrong pesticide use, the model outputs "Not sure, please consult an expert" if the confidence falls below our strict threshold.

## 🏗️ Architecture and Technology Stack
*   **Computer Vision (Quality Gate):** OpenCV (Laplacian variance for blur detection, HSV/Grayscale thresholds for brightness).
*   **Deep Learning Model:** PyTorch + `timm` (Transfer learning via pre-trained EfficientNet-B0 / MobileNetV3).
*   **Frontend UI:** Streamlit / Gradio (optimized for mobile web).
*   **Deployment:** Hugging Face Spaces / Streamlit Community Cloud.

## ⚙️ Setup and Execution Instructions

### Prerequisites
*   Python 3.10+
*   Git

### Local Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rohithkrishnaak/PlantDoc.git
   cd PlantDoc
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Copy the example environment file and update if needed.
   ```bash
   cp .env.example .env
   ```
5. **Run the App:**
   ```bash
   streamlit run app.py  # Or `python app.py` if using Gradio
   ```

## 📊 Dataset Attribution
*   **Dataset:** [PlantDoc Dataset (GitHub)](https://github.com/pratikkayal/PlantDoc-Dataset)
*   **Licence:** CC BY 4.0
*   **Note:** We restricted training to a subset of 3-5 specific plant species to optimize for hackathon time constraints and combat extreme class imbalance, using heavy augmentation.

## 🧑‍💻 Team Members
*   **Rohith Krishna A K** - [GitHub](https://github.com/rohithkrishnaak) / [LinkedIn](#)
*   *(Add teammate 2)*
*   *(Add teammate 3)*

## 📓 Development Log
Check out our [DEV_LOG.md](./DEV_LOG.md) to see our engineering decisions, rejected approaches, and how we utilized AI assistance during the 24 hours.