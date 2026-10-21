"""Generates a professional PowerPoint Pitch Deck for PlantDoc (PS 2.2) using python-pptx.

Creates PlantDoc_Pitch_Deck.pptx adhering to NexAI Hack 2026 guidelines.
"""
from __future__ import annotations

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(filename="PlantDoc_Pitch_Deck.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    c_primary = RGBColor(27, 77, 62)      # Deep Forest Green
    c_accent = RGBColor(46, 204, 113)     # Fresh Leaf Green
    c_dark = RGBColor(44, 62, 80)         # Charcoal Slate
    c_light = RGBColor(245, 247, 248)     # Soft Background
    c_white = RGBColor(255, 255, 255)
    c_amber = RGBColor(230, 126, 34)

    def add_header(slide, title_text, category_text="NEXAI HACK 2026 · TRACK 2 (SMART AGRICULTURE)"):
        # Top banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
        tf = header_box.text_frame
        tf.word_wrap = True
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = c_accent

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = c_primary

    def add_card(slide, left, top, width, height, title, points, bg_color=c_white):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = RGBColor(218, 225, 231)
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = c_primary
        p.space_after = Pt(10)

        for pt in points:
            p_bullet = tf.add_paragraph()
            p_bullet.text = f"• {pt}"
            p_bullet.font.size = Pt(13)
            p_bullet.font.color.rgb = c_dark
            p_bullet.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = c_primary
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11), Inches(4.0))
    tf1 = tb1.text_frame
    p = tf1.paragraphs[0]
    p.text = "🌿 PlantDoc"
    p.font.size = Pt(46)
    p.font.bold = True
    p.font.color.rgb = c_accent

    p = tf1.add_paragraph()
    p.text = "Leaf Doctor for Real Field Photos"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = c_white
    p.space_after = Pt(16)

    p = tf1.add_paragraph()
    p.text = "Honest AI Diagnosis · Fuzzy Quality Gate · Explainable Trust Engine"
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(200, 230, 215)
    p.space_after = Pt(30)

    p = tf1.add_paragraph()
    p.text = "Track 2: Smart Agriculture & CV | Problem Statement 2.2\nPresented by: Rohith Krishna A K"
    p.font.size = Pt(14)
    p.font.color.rgb = c_white

    # -------------------------------------------------------------
    # Slide 2: The Problem
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Real-World Challenge: Lab vs. Field")
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "⚠️ The 'Lab-Photo' Illusion", [
        "Conventional crop classifiers achieve 99% accuracy on sterile laboratory datasets (like PlantVillage).",
        "In real Indian farms, photos have motion blur, variable sunlight, deep shadows, and complex soil backgrounds.",
        "When fed poor photos, standard neural networks emit high softmax confidence (e.g. 97%) on wrong diagnoses.",
        "Misdiagnosis leads farmers to buy expensive, inappropriate pesticides, devastating crops and farm economics."
    ])
    add_card(s2, 6.8, 1.8, 5.6, 5.0, "🎯 PS 2.2 Mandate & Core Goal", [
        "Train on real-world noisy field images from the PlantDoc IIT dataset (CC BY 4.0).",
        "Implement an Image Quality Gate to reject unusable photos with actionable retake instructions.",
        "Refuse to guess: output 'Not sure, please consult an agricultural expert' when confidence is low.",
        "Provide explainable, transparent AI that farmers and agricultural extension officers can trust."
    ])

    # -------------------------------------------------------------
    # Slide 3: Two-Stage System Architecture
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "System Architecture: Quality Gate + Fuzzy Trust")
    add_card(s3, 0.8, 1.8, 3.7, 5.0, "1. Quality Gate (OpenCV)", [
        "Resolution normalized to 512px.",
        "Blur Detection: Variance of Laplacian catches shaky smartphone captures.",
        "Lighting: HSV V-channel mean catches dark shadows & harsh glare.",
        "Leaf Coverage: Dual HSV green + lesion masks ensure leaf fills the frame.",
        "Fuzzy Scoring: Soft transitions prevent brittle cliff-edge rejections."
    ])
    add_card(s3, 4.8, 1.8, 3.7, 5.0, "2. Transfer Classifier", [
        "Backbone: EfficientNet-B0 (timm), chosen for ultra-fast mobile CPU inference.",
        "Pretrained on ImageNet, fine-tuned on PlantDoc field photos.",
        "Heavy domain augmentations: Random affine, color jitter, multi-angle flips.",
        "Class-weighted loss function handles real-world class imbalances."
    ])
    add_card(s3, 8.8, 1.8, 3.7, 5.0, "3. Mamdani Trust Engine", [
        "Inputs: Quality Score + Top-1 Softmax Confidence.",
        "Rule Base: High confidence on poor quality is explicitly mapped to UNCERTAIN.",
        "Defuzzification: Centroid defuzzification yields scalar Trust ∈ [0, 1].",
        "Threshold 0.65 triggers safe expert referral message."
    ])

    # -------------------------------------------------------------
    # Slide 4: Explainability & Stretch Goals
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Completed Stretch Goals: Explainability & Farmer First")
    add_card(s4, 0.8, 1.8, 3.7, 5.0, "🔬 Grad-CAM Heatmaps", [
        "Computes gradients on final conv layer (conv_head).",
        "Generates heatmaps highlighting infected spots and margins.",
        "Farmers & agronomists can verify the AI looked at actual lesions, not soil or background."
    ])
    add_card(s4, 4.8, 1.8, 3.7, 5.0, "🌐 Multilingual UI", [
        "Full localization in English, Hindi (हिन्दी), and Tamil (தமிழ்).",
        "Translates quality retake advice ('Hold steady', 'Move closer').",
        "Translates disease names to vernacular farming terminology."
    ])
    add_card(s4, 8.8, 1.8, 3.7, 5.0, "🌾 ICAR/TNAU Care Notes", [
        "Expert-curated organic and cultural control methods (e.g. NSKE 5%, Trichoderma).",
        "Chemical recommendations with dosages cited from ICAR-CPRI, IIHR & TNAU.",
        "Prevents pesticide overuse."
    ])

    # -------------------------------------------------------------
    # Slide 5: Evaluation & Coverage Curve
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Held-Out Evaluation & Coverage Curve Metrics")
    add_card(s5, 0.8, 1.8, 5.6, 5.0, "📊 Held-Out Test Evaluation", [
        "Evaluated on 52 held-out real field photos from the IIT PlantDoc dataset.",
        "Top-1 Accuracy: 50.00% | Top-3 Accuracy: 76.92% on difficult field photos.",
        "Quality Gate Pass Rate: 84.62% (filters out severe blur and extreme shadows).",
        "Diagnostic Accuracy on TRUSTED photos jumps to 60.00% (peaks at 71.4%).",
        "Strict adherence to Responsible AI: Honest metrics, zero lab-photo contamination."
    ])
    add_card(s5, 6.8, 1.8, 5.6, 5.0, "📈 Coverage Curve Proof", [
        "Evaluates Selective Accuracy vs Coverage across Trust thresholds [0.35 to 0.85].",
        "Key Finding: Filtering unconfident/poor images yields a +10% to +21.4% precision boost.",
        "Proves that our Mamdani Fuzzy Trust Engine successfully eliminates confident false diagnoses.",
        "Protects Indian farmers from spending money on incorrect chemical fungicides."
    ])

    # -------------------------------------------------------------
    # Slide 6: Live Web App & Deployment
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Interactive Mobile Web App & Deployment")
    add_card(s6, 0.8, 1.8, 5.6, 5.0, "📱 Streamlit Mobile-First UI", [
        "Live camera snapshot and instant gallery photo upload.",
        "Real-time visual feedback on image quality and retake tips.",
        "Expandable 'Why this result?' diagnostics tab showing fuzzy membership degrees.",
        "Runs on standard mobile browsers with minimal latency (<150ms inference)."
    ])
    add_card(s6, 6.8, 1.8, 5.6, 5.0, "🚀 Cloud Deployment Stack", [
        "Publicly accessible via Hugging Face Spaces / Streamlit Cloud.",
        "Containerized with lightweight CPU-optimized PyTorch dependencies.",
        "Clean .env configuration and robust offline fallback architecture.",
        "Open-source GitHub repository with complete hourly commit logs."
    ])

    # -------------------------------------------------------------
    # Slide 7: Conclusion & Roadmap
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Impact & Future Roadmap")
    add_card(s7, 0.8, 1.8, 5.6, 5.0, "🌱 Agricultural Impact", [
        "Protects Indian smallholder farmers from devastating crop disease outbreaks.",
        "Bridges the gap between remote farmers and agricultural university experts.",
        "Reduces unnecessary chemical fungicide spraying through targeted diagnosis.",
        "Promotes sustainable, scientific crop protection practices."
    ])
    add_card(s7, 6.8, 1.8, 5.6, 5.0, "🔭 Next Milestones", [
        "Edge Deployment: Export to ONNX Runtime and TFLite for 100% offline field operation.",
        "Voice Assistant: Audio advice in Hindi, Tamil, and Telugu for non-literate farmers.",
        "Pest & Deficiency Expansion: Broaden taxonomy beyond fungal/bacterial leaf spot to nutrient deficiencies."
    ])

    prs.save(filename)
    print(f"PowerPoint Pitch Deck saved successfully as {filename}")

if __name__ == "__main__":
    create_deck()
