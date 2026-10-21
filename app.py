"""PlantDoc - Leaf Doctor for Real Field Photos (Streamlit UI)."""
from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image, ImageOps

from src.fuzzy_engine import UNCERTAIN_MESSAGE, evaluate_trust
from src.model import PlantDocModel
from src.quality_gate import check_quality

st.set_page_config(page_title="PlantDoc", page_icon="🌿", layout="centered")
st.markdown(
    "<style>.block-container{padding-top:1.5rem;max-width:640px}"
    "button{min-height:3rem}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> PlantDocModel:
    return PlantDocModel()


st.title("🌿 PlantDoc")
st.caption("Leaf Doctor for real field photos")

model = load_model()
if model.is_demo:
    st.info(f"Demo mode: weights = {model.weights_source}. Predictions are not meaningful "
            "until a fine-tuned checkpoint is placed at MODEL_WEIGHTS.")

tab_cam, tab_up = st.tabs(["📷 Camera", "📁 Upload"])
with tab_cam:
    cam = st.camera_input("Take a photo of a single leaf")
with tab_up:
    up = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"])

source = cam or up
if source is None:
    st.write("Tip: fill the frame with one leaf, in daylight, avoiding harsh shadows.")
    st.stop()

# 1. Display image (respect EXIF orientation from phones)
image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
st.image(image, width="stretch")

# 2. Quality gate
passed, q_score, q_msg = check_quality(image)
details = check_quality.last["features"]
st.progress(q_score, text=f"Image quality: {q_score:.2f}")

if not passed:
    st.error(f"Photo rejected. {q_msg}")
    with st.expander("Quality details"):
        st.json(details)
    st.stop()

# 3. Model + trust engine
with st.spinner("Diagnosing..."):
    preds = model.predict(image, top_k=3)
trust = evaluate_trust(q_score, preds[0][1])

# 4 / 5. Result
if trust.status == "UNCERTAIN":
    st.warning(UNCERTAIN_MESSAGE)
else:
    st.success(f"Diagnosis: **{preds[0][0]}**")
    st.subheader("Top-3 predictions")
    for name, p in preds:
        st.progress(p, text=f"{name}: {p:.1%}")

with st.expander("Why this result?"):
    st.write(f"Trust score: **{trust.trust:.2f}** (threshold 0.65)")
    st.write(f"Top-1 confidence: {preds[0][1]:.2f} | Quality: {q_score:.2f}")
    st.json({"features": details, "fuzzy": trust.memberships})
