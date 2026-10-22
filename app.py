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

import base64
import hashlib
import html
import io
from pathlib import Path

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
HERO_SVG = Path(__file__).parent / "assets" / "hero_leaf.svg"

from src.agri_advisory import get_advisory
from src.fuzzy_engine import evaluate_trust
from src.gradcam import GradCAM
from src.i18n import LANGUAGES, t, t_disease
from src.model import PlantDocModel
from src.quality_gate import check_quality

st.set_page_config(page_title="PlantDoc · Leaf Doctor", page_icon="🌿", layout="centered")

# Minimal, nature-inspired theme: sage background, soft cards, deep leaf-green accents.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    :root {
        --leaf: #4f7a3a; --leaf-dark: #2f4a23; --sage: #e6eedb; --mist: #f4f7ee;
        --card: #ffffff; --ink: #1f2d1a; --muted: #6b7a63;
        --warn-bg: #fdf3e1; --warn: #9a6512; --bad-bg: #fbeae6; --bad: #a43d2b;
    }
    html, body, [class*="css"], .stMarkdown, button, input, select {
        font-family: 'Nunito', system-ui, sans-serif;
    }
    .stApp { background: var(--mist); }
    .block-container { padding: 1rem 1rem 3rem; max-width: 560px; }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }

    .hero { border-radius: 28px; overflow: hidden; position: relative; margin-bottom: 0.9rem;
            box-shadow: 0 8px 24px rgba(47, 74, 35, 0.10); }
    .hero img { display: block; width: 100%; height: auto; }
    .hero-text { position: absolute; left: 22px; top: 18px; }
    .hero-brand { font-size: 1.7rem; font-weight: 800; color: var(--leaf-dark); line-height: 1.1; }
    .hero-brand span { color: var(--leaf); }
    .hero-tag { font-size: 0.92rem; color: var(--leaf-dark); max-width: 46%; margin-top: 4px; opacity: 0.85; }
    .hero-chip { display: inline-block; margin-top: 8px; padding: 3px 10px; border-radius: 999px;
                 background: rgba(255,255,255,0.75); color: var(--leaf-dark); font-size: 0.75rem; font-weight: 700; }

    .steps { display: flex; gap: 6px; margin: 0.4rem 0 1rem; }
    .step { flex: 1; text-align: center; padding: 7px 4px; border-radius: 999px; font-size: 0.78rem;
            font-weight: 700; background: var(--sage); color: var(--muted); }
    .step.done { background: #cfe0bb; color: var(--leaf-dark); }
    .step.active { background: var(--leaf); color: #fff; }

    .card { background: var(--card); border-radius: 22px; padding: 16px 18px; margin: 0 0 0.9rem;
            box-shadow: 0 2px 10px rgba(47, 74, 35, 0.06); color: var(--ink); }
    .card h4 { margin: 0 0 8px; font-size: 1rem; color: var(--ink); }
    .card ul { margin: 0; padding-left: 1.1rem; color: var(--muted); }
    .card li { margin: 3px 0; }
    .card.warn { background: var(--warn-bg); }
    .card.warn h4 { color: var(--warn); }
    .card.bad { background: var(--bad-bg); }
    .card.bad h4 { color: var(--bad); }
    .card .muted { color: var(--muted); font-size: 0.88rem; }

    .checks { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
    .pill { padding: 5px 11px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .pill.ok { background: #dcebcb; color: var(--leaf-dark); }
    .pill.no { background: var(--bad-bg); color: var(--bad); }

    .dx-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; }
    .dx-name { font-size: 1.45rem; font-weight: 800; color: var(--leaf-dark); margin: 2px 0 10px; }
    .bar-row { margin: 9px 0; }
    .bar-top { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--ink); }
    .bar-top b { font-weight: 800; }
    .bar { height: 9px; background: var(--sage); border-radius: 999px; overflow: hidden; margin-top: 4px; }
    .bar > div { height: 100%; background: var(--leaf); border-radius: 999px; }
    .bar.dim > div { background: #a9c98b; }

    .foot { text-align: center; color: var(--muted); font-size: 0.78rem; margin-top: 1.4rem; }

    [data-testid="stTabs"] button { border-radius: 999px; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px; min-height: 2.9rem; font-weight: 700; width: 100%;
        background: var(--leaf); color: #fff; border: none;
    }
    .stButton > button:hover { background: var(--leaf-dark); color: #fff; }
    [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
        border-radius: 20px; background: var(--card);
    }
    [data-testid="stExpander"] { background: var(--card); border-radius: 18px; border: none; }
    [data-testid="stImage"] img { border-radius: 18px; max-height: 340px; object-fit: cover; }
    @media (max-width: 420px) {
        .hero-brand { font-size: 1.4rem; }
        .hero-tag { font-size: 0.8rem; max-width: 50%; }
        .dx-name { font-size: 1.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> PlantDocModel:
    return PlantDocModel()


@st.cache_resource(show_spinner=False)
def hero_data_uri() -> str:
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(HERO_SVG.read_bytes()).decode()
    except OSError:
        return ""


def card(body: str, kind: str = "") -> None:
    st.markdown(f'<div class="card {kind}">{body}</div>', unsafe_allow_html=True)


def steps(active: int) -> None:
    """Render the 3-step tracker into the slot reserved above the photo input."""
    labels = [t("step_photo", lang), t("step_check", lang), t("step_result", lang)]
    cells = "".join(
        f'<div class="step {"done" if i < active else "active" if i == active else ""}">{i + 1}. {html.escape(l)}</div>'
        for i, l in enumerate(labels)
    )
    step_slot.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)


def bars(preds, dim: bool = False) -> str:
    rows = []
    for name, p in preds:
        rows.append(
            f'<div class="bar-row"><div class="bar-top"><span>{html.escape(t_disease(name, lang))}</span>'
            f'<b>{p:.0%}</b></div><div class="bar{" dim" if dim else ""}"><div style="width:{p * 100:.1f}%"></div></div></div>'
        )
    return "".join(rows)


def reset_inputs() -> None:
    st.session_state["input_gen"] = st.session_state.get("input_gen", 0) + 1
    st.session_state.pop("latest_input", None)


def _mark_latest(which: str) -> None:
    st.session_state["latest_input"] = which


# ---- Language + hero -------------------------------------------------------
lang_names = list(LANGUAGES.values())
_, col_lang = st.columns([0.55, 0.45])
with col_lang:
    selected_lang_name = st.selectbox("🌐 Language", options=lang_names, index=0, label_visibility="collapsed")
lang = next(k for k, v in LANGUAGES.items() if v == selected_lang_name)

hero_img = hero_data_uri()
st.markdown(
    f"""
    <div class="hero">
      {f'<img src="{hero_img}" alt="">' if hero_img else '<div style="height:200px;background:#dfeacf"></div>'}
      <div class="hero-text">
        <div class="hero-brand">Plant<span>Doc</span></div>
        <div class="hero-tag">{html.escape(t("hero_tagline", lang))}</div>
        <div class="hero-chip">🌱 {html.escape(t("hero_crops", lang))}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
if model.is_demo:
    st.info(t("demo_mode_banner", lang, source=model.weights_source))

# ---- 1. Photo input --------------------------------------------------------
gen = st.session_state.get("input_gen", 0)
step_slot = st.empty()
tab_up, tab_cam = st.tabs([t("tab_upload", lang), t("tab_camera", lang)])
with tab_up:
    up = st.file_uploader(t("upload_label", lang), type=UPLOAD_TYPES, key=f"up{gen}",
                          on_change=_mark_latest, args=("up",))
with tab_cam:
    cam = st.camera_input(t("camera_label", lang), key=f"cam{gen}", on_change=_mark_latest, args=("cam",))

# Use whichever input the user touched most recently when both hold a photo.
if cam is not None and up is not None:
    source = up if st.session_state.get("latest_input") == "up" else cam
else:
    source = cam or up

if source is None:
    steps(0)
    card(
        f"<h4>📸 {html.escape(t('photo_tips_title', lang))}</h4><ul>"
        + "".join(f"<li>{html.escape(t(k, lang))}</li>" for k in ("photo_tip_1", "photo_tip_2", "photo_tip_3"))
        + "</ul>"
    )
    st.markdown(f'<div class="foot">{html.escape(t("footer", lang))}</div>', unsafe_allow_html=True)
    st.stop()

raw = source.getvalue()
try:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
except (UnidentifiedImageError, OSError):
    steps(0)
    card(f"<h4>{html.escape(t('image_unreadable', lang))}</h4>", "bad")
    st.stop()
image.thumbnail((MAX_SIDE, MAX_SIDE))
st.image(image, width="stretch", caption=t("your_photo", lang))


# ---- 2. Quality gate + inference (cached per photo so language changes are instant)
@st.cache_data(show_spinner=False, max_entries=8)
def analyse(digest: str, _image: Image.Image) -> dict:
    passed, q_score, _ = check_quality(_image)
    out = {"passed": passed, "q_score": q_score, **check_quality.last}
    if passed:
        preds, similarity, is_ood = model.diagnose(_image, top_k=3)
        out.update(preds=preds, similarity=similarity, is_ood=is_ood)
    return out


@st.cache_data(show_spinner=False, max_entries=8)
def heatmap(digest: str, _image: Image.Image):
    return GradCAM(model.model).generate(_image)[1]


digest = hashlib.sha1(raw).hexdigest()
with st.spinner(t("diagnosing", lang)):
    res = analyse(digest, image)

details, m = res["features"], res["memberships"]
light_ok = m["brightness"]["dark"] <= 0.3 and m["brightness"]["over"] <= 0.3
sharp_ok = max(m["blur"]["blurry"], m["blur"]["streaked"]) <= 0.3
frame_ok = m["coverage"]["low"] <= 0.3
pills = "".join(
    f'<span class="pill {"ok" if ok else "no"}">{"✓" if ok else "✕"} {html.escape(t(k, lang))}</span>'
    for k, ok in (("check_sharp", sharp_ok), ("check_light", light_ok), ("check_frame", frame_ok))
)
q_line = f'<div class="muted" style="margin-top:8px">{html.escape(t("quality_score_text", lang))}: {res["q_score"]:.2f} / 1.00</div>'

if not res["passed"]:
    # Same priority as quality_gate._retake_tip: fix lighting first.
    if m["brightness"]["dark"] > 0.3:
        tips = [t("tip_dark", lang)]
    elif m["brightness"]["over"] > 0.3:
        tips = [t("tip_overexposed", lang)]
    else:
        tips = ([t("tip_blurry", lang)] if not sharp_ok else []) + ([t("tip_low_coverage", lang)] if not frame_ok else [])
        tips = tips or [t("tip_borderline", lang)]
    steps(1)
    card(
        f"<h4>🔁 {html.escape(t('retake_title', lang))}</h4><div class='checks'>{pills}</div>"
        f"<ul style='margin-top:10px'>{''.join(f'<li>{html.escape(x)}</li>' for x in tips)}</ul>{q_line}",
        "bad",
    )
    st.button(t("new_photo", lang), on_click=reset_inputs)
    with st.expander("🔍 " + t("why_result", lang)):
        st.json(details)
    st.stop()

card(f"<h4>✅ {html.escape(t('quality_passed', lang))}</h4><div class='checks'>{pills}</div>{q_line}")

# ---- 3. Diagnosis ----------------------------------------------------------
preds = res["preds"]
if res["is_ood"]:
    steps(2)
    card(f"<h4>🍃 {html.escape(t('ood_title', lang))}</h4><div>{html.escape(t('ood_rejected', lang))}</div>", "bad")
    st.button(t("new_photo", lang), on_click=reset_inputs)
    with st.expander("🔍 " + t("why_result", lang)):
        st.write(f"Similarity to the closest training photo: `{res['similarity']:.3f}` "
                 f"(minimum `{float(model.model.ood_threshold):.3f}`)")
    st.stop()

top1_name, top1_conf = preds[0]
trust = evaluate_trust(res["q_score"], top1_conf, threshold=model.trust_threshold)
steps(3)

if trust.status == "UNCERTAIN":
    card(
        f"<h4>{html.escape(t('uncertain_warning', lang))}</h4>"
        f"<div>{html.escape(t('uncertain_reason', lang, trust=trust.trust, threshold=model.trust_threshold))}</div>"
        f"<div style='margin-top:8px'><b>📞 {html.escape(t('expert_help', lang))}</b></div>",
        "warn",
    )
    card(f"<div class='dx-label'>{html.escape(t('possible_matches', lang))}</div>{bars(preds, dim=True)}")
else:
    note = f"<div class='muted'>{html.escape(t('healthy_note', lang))}</div>" if "Healthy" in top1_name else ""
    card(
        f"<div class='dx-label'>{html.escape(t('diagnosis_title', lang))}</div>"
        f"<div class='dx-name'>{html.escape(t_disease(top1_name, lang))}</div>{note}"
        f"<div class='dx-label' style='margin-top:12px'>{html.escape(t('top3_title', lang))}</div>{bars(preds)}"
    )

    # Stretch Goal 1: Grad-CAM heatmap
    with st.expander(t("gradcam_title", lang)):
        st.caption(t("gradcam_caption", lang))
        try:
            blended = heatmap(digest, image)
            c1, c2 = st.columns(2)
            c1.image(image, caption=t("original", lang), width="stretch")
            c2.image(blended, caption=t("heatmap", lang), width="stretch")
        except Exception:
            st.info(t("gradcam_unavailable", lang))

    # Stretch Goal 3: care notes (ICAR / TNAU)
    advisory = get_advisory(top1_name)
    if advisory:
        def loc(field: str) -> str:
            return html.escape(advisory[field].get(lang, advisory[field]["en"]))

        with st.expander(t("care_notes_title", lang), expanded=True):
            st.markdown(
                f"**{t('pathogen', lang)}:** *{advisory.get('pathogen', 'N/A')}*\n\n"
                f"**{t('symptoms', lang)}:** {loc('symptoms')}\n\n"
                f"**🌿 {t('organic_control', lang)}:** {loc('cultural_organic')}\n\n"
                f"**🧪 {t('chemical_control', lang)}:** {loc('chemical_control')}"
            )
            st.caption(f"{t('source', lang)}: {advisory.get('source', 'Agricultural Extension Service')}")

st.button(t("new_photo", lang), on_click=reset_inputs)

with st.expander("🔍 " + t("why_result", lang)):
    st.write(f"**{t('trust_score_title', lang)}:** `{trust.trust:.2f}` (threshold {model.trust_threshold:.2f})")
    st.write(f"**Top-1 confidence:** `{top1_conf:.2f}` · **Quality:** `{res['q_score']:.2f}`")
    st.json({"features": details, "fuzzy_memberships": trust.memberships})

st.markdown(f'<div class="foot">{html.escape(t("footer", lang))}</div>', unsafe_allow_html=True)
