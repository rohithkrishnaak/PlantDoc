"""PlantDoc - Leaf Doctor for Real Field Photos (Streamlit UI).

NexAI Hack 2026 · Track 2 (Smart Agriculture & Computer Vision) · Problem Statement 2.2
Features:
- Image Quality Gate (Blur, Brightness, Coverage) + Actionable Retake Tips
- Fine-Tuned EfficientNet-B0 Classifier on Real Field Photos
- Mamdani Fuzzy Trust Engine (Safe expert fallback on low trust)
- Grad-CAM Visual Explainability Heatmap (Stretch Goal 1)
- Multilingual Interface: English, Hindi, Tamil (Stretch Goal 2)
- ICAR / TNAU Cited Agronomic Care Notes (Stretch Goal 3)
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # iPhone photos are HEIC by default
    from pillow_heif import register_heif_opener

    register_heif_opener()
    UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]
except ImportError:
    UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

MAX_SIDE = 2048  # phone photos are downscaled; the pipeline works at <= 512 px anyway

from src.agri_advisory import get_advisory
from src.fuzzy_engine import UNCERTAIN_MESSAGE, evaluate_trust
from src.gradcam import GradCAM
from src.i18n import LANGUAGES, t, t_disease
from src.model import PlantDocModel
from src.quality_gate import check_quality

st.set_page_config(page_title="PlantDoc", page_icon="🌿", layout="centered")

# Mobile-friendly CSS styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 680px; }
    button { min-height: 2.8rem; font-weight: 600; }
    .stAlert { border-radius: 8px; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> PlantDocModel:
    return PlantDocModel()


# Language Selector in Sidebar / Header
col_title, col_lang = st.columns([0.7, 0.3])
with col_lang:
    selected_lang_name = st.selectbox(
        "🌐 Language",
        options=list(LANGUAGES.values()),
        index=0,
        label_visibility="collapsed",
    )
    lang = [k for k, v in LANGUAGES.items() if v == selected_lang_name][0]

with col_title:
    st.title(t("app_title", lang))

st.caption(t("app_caption", lang))

model = load_model()
if model.is_demo:
    st.info(t("demo_mode_banner", lang, source=model.weights_source))

def _mark_latest(which: str):
    st.session_state["latest_input"] = which


tab_cam, tab_up = st.tabs([t("tab_camera", lang), t("tab_upload", lang)])
with tab_cam:
    cam = st.camera_input(t("camera_label", lang), key="cam", on_change=_mark_latest, args=("cam",))
with tab_up:
    up = st.file_uploader(t("upload_label", lang), type=UPLOAD_TYPES, key="up",
                          on_change=_mark_latest, args=("up",))

# Use whichever input the user touched most recently when both hold a photo.
if cam is not None and up is not None:
    source = up if st.session_state.get("latest_input") == "up" else cam
else:
    source = cam or up
if source is None:
    st.info(t("tip_initial", lang))
    st.stop()

# 1. Display Image (respecting phone EXIF rotation)
try:
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("Could not read this file as an image. Please upload a JPG or PNG photo.")
    st.stop()
image.thumbnail((MAX_SIDE, MAX_SIDE))
st.image(image, width="stretch", caption="Captured Leaf")

# 2. Image Quality Gate
passed, q_score, q_msg = check_quality(image)
details = check_quality.last["features"]
memberships = check_quality.last["memberships"]

st.progress(q_score, text=f"{t('quality_score_text', lang)}: {q_score:.2f} / 1.00")

if not passed:
    # Select localized tip(s); same priority as quality_gate._retake_tip (lighting first)
    if memberships["brightness"]["dark"] > 0.3:
        localized_tip = t("tip_dark", lang)
    elif memberships["brightness"]["over"] > 0.3:
        localized_tip = t("tip_overexposed", lang)
    else:
        tips = []
        if max(memberships["blur"]["blurry"], memberships["blur"]["streaked"]) > 0.3:
            tips.append(t("tip_blurry", lang))
        if memberships["coverage"]["low"] > 0.3:
            tips.append(t("tip_low_coverage", lang))
        localized_tip = " ".join(tips) or t("tip_borderline", lang)

    st.error(f"{t('quality_rejected', lang)} {localized_tip}")
    with st.expander("🔍 Quality diagnostics"):
        st.json(details)
    st.stop()

# 3. Model Inference & Trust Engine
with st.spinner(t("diagnosing", lang)):
    preds, similarity, is_ood = model.diagnose(image, top_k=3)

# 3b. Out-of-distribution check: not a supported tomato/potato leaf
if is_ood:
    st.error(t("ood_rejected", lang))
    with st.expander("🔍 Why?"):
        st.write(f"Similarity to the closest training photo: `{similarity:.3f}` "
                 f"(minimum `{float(model.model.ood_threshold):.3f}`)")
    st.stop()

top1_name, top1_conf = preds[0]
trust = evaluate_trust(q_score, top1_conf, threshold=model.trust_threshold)

# 4. Results Presentation
if trust.status == "UNCERTAIN":
    st.warning(t("uncertain_warning", lang))
    st.write(t("uncertain_reason", lang, trust=trust.trust, threshold=model.trust_threshold))
else:
    translated_disease = t_disease(top1_name, lang)
    st.success(f"{t('diagnosis_title', lang)}: **{translated_disease}**")

    # Top-3 Probabilities
    st.subheader(t("top3_title", lang))
    for name, p in preds:
        lbl = t_disease(name, lang)
        st.progress(p, text=f"{lbl}: {p:.1%}")

    # Stretch Goal 1: Grad-CAM Explainable AI Heatmap
    with st.expander(t("gradcam_title", lang), expanded=True):
        st.caption(t("gradcam_caption", lang))
        try:
            gradcam = GradCAM(model.model)
            heatmap_rgb, blended_rgb = gradcam.generate(image)
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.image(image, caption="Original Leaf", width="stretch")
            with col_cam:
                st.image(blended_rgb, caption=("Grad-CAM (ConvNeXt branch of the hybrid)"
                                               if model.model.__class__.__name__ == "HybridNet"
                                               else "Grad-CAM Saliency"), width="stretch")
        except Exception as e:
            st.info(f"Grad-CAM visualization unavailable: {e}")

    # Stretch Goal 3: Agronomic Care Notes (ICAR/TNAU)
    advisory = get_advisory(top1_name)
    if advisory:
        with st.expander(t("care_notes_title", lang), expanded=True):
            st.markdown(f"**Pathogen:** *{advisory.get('pathogen', 'N/A')}*")
            st.markdown(f"**Symptoms:** {advisory['symptoms'].get(lang, advisory['symptoms']['en'])}")
            st.markdown(f"**🌿 Organic & Cultural Control:** {advisory['cultural_organic'].get(lang, advisory['cultural_organic']['en'])}")
            st.markdown(f"**🧪 Chemical Management:** {advisory['chemical_control'].get(lang, advisory['chemical_control']['en'])}")
            st.caption(f"Source: {advisory.get('source', 'Agricultural Extension Service')}")

# Diagnostics Expander
with st.expander(t("why_result", lang)):
    st.write(f"**{t('trust_score_title', lang)}:** `{trust.trust:.2f}` (Threshold: {model.trust_threshold:.2f})")
    st.write(f"**Top-1 Softmax Confidence:** `{top1_conf:.2f}` | **Image Quality Score:** `{q_score:.2f}`")
    st.json({"features": details, "fuzzy_memberships": trust.memberships})
