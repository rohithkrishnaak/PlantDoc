"""Multilingual Internationalization (i18n) Module for PlantDoc.

Supports English (en), Hindi (hi), and Tamil (ta) for:
- Interface headers and controls
- Quality gate rejection warnings and actionable retake tips
- Disease classification labels
- Trust score explanations and extension advisories
"""
from __future__ import annotations

LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
}

TRANSLATIONS = {
    "app_title": {
        "en": "🌿 PlantDoc",
        "hi": "🌿 प्लांटडॉक (PlantDoc)",
        "ta": "🌿 பிளான்ட்டாக் (PlantDoc)",
    },
    "app_caption": {
        "en": "Leaf Doctor for real field photos with Explainable Trust & Quality Gate",
        "hi": "वास्तविक खेत की तस्वीरों के लिए पत्ता डॉक्टर - गुणवत्ता जांच और भरोसेमंद निदान",
        "ta": "உண்மையான கள புகைப்படங்களுக்கான இலை மருத்துவர் - நம்பகமான பயிர் நோய் கண்டறிதல்",
    },
    "tab_camera": {
        "en": "📷 Camera",
        "hi": "📷 कैमरा",
        "ta": "📷 கேமரா",
    },
    "tab_upload": {
        "en": "📁 Upload",
        "hi": "📁 फ़ोटो अपलोड करें",
        "ta": "📁 பதிவேற்றம்",
    },
    "camera_label": {
        "en": "Take a photo of a single leaf",
        "hi": "पौधे की एक पत्ती की तस्वीर लें",
        "ta": "ஒரு இலையின் புகைப்படத்தை எடுக்கவும்",
    },
    "upload_label": {
        "en": "Upload a leaf photo",
        "hi": "पत्ती की तस्वीर अपलोड करें",
        "ta": "இலை புகைப்படத்தை பதிவேற்றவும்",
    },
    "tip_initial": {
        "en": "Tip: Fill the frame with one leaf, in natural daylight, avoiding harsh shadows.",
        "hi": "सलाह: प्राकृतिक दिन के उजाले में पत्ती को फ्रेम में भरें और छाया से बचें।",
        "ta": "குறிப்பு: ஒரு இலையை மட்டும் சட்டகத்தில் நிரப்பவும், நிழல்களைத் தவிர்க்கவும்.",
    },
    "quality_score_text": {
        "en": "Image Quality Score",
        "hi": "तस्वीर की गुणवत्ता स्कोर",
        "ta": "படத்தின் தர மதிப்பீடு",
    },
    "quality_rejected": {
        "en": "Photo rejected by Quality Gate.",
        "hi": "गुणवत्ता जांच द्वारा तस्वीर अस्वीकृत की गई।",
        "ta": "படத்தின் தரம் போதுமானதாக இல்லை.",
    },
    "tip_blurry": {
        "en": "Image is blurry: hold the phone steady and tap to focus on the leaf.",
        "hi": "तस्वीर धुंधली है: फोन को स्थिर रखें और पत्ती पर फोकस करने के लिए स्क्रीन टैप करें।",
        "ta": "படம் மங்கலாக உள்ளது: போனை அசையாமல் பிடித்து இலையின் மீது ஃபோகஸ் செய்யவும்.",
    },
    "tip_dark": {
        "en": "Too dark: move into daylight or avoid shadows on the leaf.",
        "hi": "बहुत अंधेरा है: दिन की रोशनी में जाएं या पत्ती पर छाया से बचें।",
        "ta": "மிகவும் இருட்டாக உள்ளது: வெளிச்சத்திற்குச் செல்லவும் அல்லது நிழலைத் தவிர்க்கவும்.",
    },
    "tip_overexposed": {
        "en": "Overexposed: avoid direct glare, shade the leaf slightly.",
        "hi": "अधिक चमक: सीधे तेज धूप से बचें, पत्ती पर हल्की छाया करें।",
        "ta": "அதிக வெளிச்சம்: நேரடி சூரிய ஒளியைத் தவிர்க்கவும், லேசான நிழல் கொடுக்கவும்.",
    },
    "tip_low_coverage": {
        "en": "Leaf too small or not found: move closer so the leaf fills the frame.",
        "hi": "पत्ती बहुत छोटी है या नहीं दिखी: फोन को पत्ती के करीब ले जाएं ताकि वह फ्रेम भर ले।",
        "ta": "இலை மிகச் சிறியதாக உள்ளது: இலை திரையை நிரப்பும் வகையில் நெருக்கமாக எடுக்கவும்.",
    },
    "ood_rejected": {
        "en": "This does not look like a tomato or potato leaf this app was trained on. "
              "Please photograph a single tomato or potato leaf.",
        "hi": "यह उस टमाटर या आलू की पत्ती जैसी नहीं दिखती जिस पर यह ऐप प्रशिक्षित है। "
              "कृपया टमाटर या आलू की एक पत्ती की तस्वीर लें।",
        "ta": "இது இந்த செயலி பயிற்சி பெற்ற தக்காளி அல்லது உருளைக்கிழங்கு இலை போல் தெரியவில்லை. "
              "தயவுசெய்து ஒரு தக்காளி அல்லது உருளைக்கிழங்கு இலையை புகைப்படம் எடுக்கவும்.",
    },
    "uncertain_reason": {
        "en": "Trust score {trust:.2f} is below the threshold {threshold:.2f}: the photo quality was "
              "borderline or the disease signs were ambiguous. A confident wrong diagnosis can lead "
              "to the wrong pesticide.",
        "hi": "भरोसा स्कोर {trust:.2f} सीमा {threshold:.2f} से कम है: तस्वीर की गुणवत्ता सीमांत थी या "
              "रोग के लक्षण स्पष्ट नहीं थे। गलत निदान से गलत कीटनाशक का उपयोग हो सकता है।",
        "ta": "நம்பக மதிப்பெண் {trust:.2f} வரம்பு {threshold:.2f}-க்கு கீழ் உள்ளது: புகைப்படத் தரம் "
              "எல்லைக்குட்பட்டது அல்லது நோய் அறிகுறிகள் தெளிவாக இல்லை. தவறான கண்டறிதல் தவறான "
              "பூச்சிக்கொல்லிக்கு வழிவகுக்கும்.",
    },
    "tip_borderline": {
        "en": "Quality is borderline: retake closer, in even light, holding steady.",
        "hi": "गुणवत्ता सीमांत है: फोन स्थिर रखकर अच्छी रोशनी में दोबारा तस्वीर लें।",
        "ta": "தர எல்லைக்குட்பட்டது: சமமான வெளிச்சத்தில் போனை அசைக்காமல் மீண்டும் எடுக்கவும்.",
    },
    "diagnosing": {
        "en": "Diagnosing leaf health...",
        "hi": "पत्ती के स्वास्थ्य की जांच की जा रही है...",
        "ta": "இலை ஆரோக்கியத்தை ஆராய்கிறது...",
    },
    "uncertain_warning": {
        "en": "⚠️ Not sure, please consult an agricultural expert.",
        "hi": "⚠️ निश्चित नहीं है, कृपया नजदीकी कृषि विशेषज्ञ या कृषि विज्ञान केंद्र से सलाह लें।",
        "ta": "⚠️ உறுதி செய்ய முடியவில்லை, தயவுசெய்து வேளாண்மை நிபுணரை அணுகவும்.",
    },
    "diagnosis_title": {
        "en": "Diagnosis",
        "hi": "निदान",
        "ta": "நோய் கண்டறிதல்",
    },
    "top3_title": {
        "en": "Top-3 Disease Probabilities",
        "hi": "शीर्ष-3 रोग संभावनाएँ",
        "ta": "முதல் 3 நோய் சாத்தியக்கூறுகள்",
    },
    "trust_score_title": {
        "en": "Trust Score",
        "hi": "भरोसा स्कोर (Trust Score)",
        "ta": "நம்பகத்தன்மை மதிப்பீடு",
    },
    "why_result": {
        "en": "Why this result? (Trust & Quality Diagnostics)",
        "hi": "यह परिणाम क्यों मिला? (भरोसा और गुणवत्ता विश्लेषण)",
        "ta": "இந்த முடிவு ஏன்? (நம்பகத்தன்மை & தர ஆய்வுகள்)",
    },
    "gradcam_title": {
        "en": "🔬 Grad-CAM Saliency Map (Explainable AI)",
        "hi": "🔬 ग्रैड-कैम हीटमैप (एआई व्याख्या - पत्ती पर लक्षण क्षेत्र)",
        "ta": "🔬 கிராட்-கேம் வெப்ப வரைபடம் (நோய் அறிகுறி பகுதிகள்)",
    },
    "gradcam_caption": {
        "en": "Red/Yellow zones highlight visual leaf regions that drove the model's diagnosis.",
        "hi": "लाल/पीले क्षेत्र उन स्थानों को दर्शाते हैं जिनके आधार पर मॉडल ने रोग पहचाना।",
        "ta": "சிவப்பு/மஞ்சள் பகுதிகள் மாதிரி நோய் கண்டறிந்த முக்கிய இலை பகுதிகளைக் காட்டுகின்றன.",
    },
    "care_notes_title": {
        "en": "🌾 Agronomic Care & Management Guide (ICAR / TNAU Extension)",
        "hi": "🌾 कृषि प्रबंधन एवं उपचार सलाह (आईसीएआर / टीएनएयू)",
        "ta": "🌾 வேளாண் மேலாண்மை மற்றும் சிகிச்சை குறிப்புகள் (ICAR / TNAU)",
    },
    "demo_mode_banner": {
        "en": "Demo mode: weights = {source}. Train on PlantDoc to enable full field diagnosis.",
        "hi": "डेमो मोड: वजन = {source}। पूर्ण क्षेत्र निदान सक्षम करने के लिए प्लांटडॉक पर प्रशिक्षित करें।",
        "ta": "டெமோ பயன்முறை: {source}. முழுமையான கள ஆய்வுக்கு மாதிரி பயிற்சி தேவை.",
    },
}

# Strings for the redesigned UI (hero, step flow, quality checklist, result cards).
TRANSLATIONS.update({
    "hero_tagline": {
        "en": "Snap a leaf. Get an honest diagnosis.",
        "hi": "पत्ती की फोटो लें, ईमानदार निदान पाएं।",
        "ta": "இலையை படம் எடுங்கள், நேர்மையான கண்டறிதலைப் பெறுங்கள்.",
    },
    "hero_crops": {
        "en": "Tomato · Potato",
        "hi": "टमाटर · आलू",
        "ta": "தக்காளி · உருளைக்கிழங்கு",
    },
    "step_photo": {"en": "Photo", "hi": "फोटो", "ta": "புகைப்படம்"},
    "step_check": {"en": "Quality check", "hi": "गुणवत्ता जांच", "ta": "தர சோதனை"},
    "step_result": {"en": "Diagnosis", "hi": "निदान", "ta": "கண்டறிதல்"},
    "photo_tips_title": {
        "en": "How to take a good photo",
        "hi": "अच्छी फोटो कैसे लें",
        "ta": "நல்ல புகைப்படம் எடுப்பது எப்படி",
    },
    "photo_tip_1": {
        "en": "One leaf, filling most of the frame",
        "hi": "एक ही पत्ती, फ्रेम का अधिकांश भाग भरे",
        "ta": "ஒரே இலை, சட்டகத்தின் பெரும்பகுதியை நிரப்ப வேண்டும்",
    },
    "photo_tip_2": {
        "en": "Soft daylight, no harsh shadow or glare",
        "hi": "हल्की धूप, कड़ी छाया या चमक नहीं",
        "ta": "மென்மையான பகல் ஒளி, கடும் நிழல் அல்லது பளபளப்பு இல்லை",
    },
    "photo_tip_3": {
        "en": "Hold steady and tap the leaf to focus",
        "hi": "फोन स्थिर रखें और फोकस के लिए पत्ती पर टैप करें",
        "ta": "போனை அசையாமல் பிடித்து இலையைத் தொட்டு ஃபோகஸ் செய்யவும்",
    },
    "check_sharp": {"en": "Sharp", "hi": "स्पष्ट", "ta": "தெளிவு"},
    "check_light": {"en": "Lighting", "hi": "रोशनी", "ta": "வெளிச்சம்"},
    "check_frame": {"en": "Leaf fills frame", "hi": "पत्ती फ्रेम भरे", "ta": "இலை சட்டகத்தை நிரப்புகிறது"},
    "quality_passed": {
        "en": "Photo looks good. Analysing the leaf.",
        "hi": "फोटो ठीक है। पत्ती की जांच हो रही है।",
        "ta": "புகைப்படம் நன்றாக உள்ளது. இலையை ஆராய்கிறது.",
    },
    "retake_title": {
        "en": "Please retake the photo",
        "hi": "कृपया दोबारा फोटो लें",
        "ta": "தயவுசெய்து மீண்டும் புகைப்படம் எடுக்கவும்",
    },
    "your_photo": {"en": "Your photo", "hi": "आपकी फोटो", "ta": "உங்கள் புகைப்படம்"},
    "image_unreadable": {
        "en": "Could not read this file as an image. Please upload a JPG or PNG photo.",
        "hi": "यह फ़ाइल तस्वीर के रूप में नहीं खुली। कृपया JPG या PNG फोटो अपलोड करें।",
        "ta": "இந்தக் கோப்பை படமாகப் படிக்க முடியவில்லை. JPG அல்லது PNG புகைப்படத்தைப் பதிவேற்றவும்.",
    },
    "confidence": {"en": "confidence", "hi": "विश्वास", "ta": "நம்பிக்கை"},
    "possible_matches": {
        "en": "Closest matches (not a diagnosis)",
        "hi": "निकटतम मिलान (यह निदान नहीं है)",
        "ta": "நெருக்கமான பொருத்தங்கள் (இது கண்டறிதல் அல்ல)",
    },
    "expert_help": {
        "en": "Call the Kisan Call Centre (1800-180-1551, free) or visit your nearest Krishi Vigyan Kendra.",
        "hi": "किसान कॉल सेंटर (1800-180-1551, निःशुल्क) पर कॉल करें या नजदीकी कृषि विज्ञान केंद्र जाएं।",
        "ta": "கிசான் அழைப்பு மையத்தை (1800-180-1551, இலவசம்) அழைக்கவும் அல்லது அருகிலுள்ள வேளாண் அறிவியல் மையத்தை அணுகவும்.",
    },
    "healthy_note": {
        "en": "No disease signs found. Keep monitoring the lower leaves every few days.",
        "hi": "कोई रोग लक्षण नहीं मिले। हर कुछ दिनों में निचली पत्तियों की जांच करते रहें।",
        "ta": "நோய் அறிகுறிகள் இல்லை. சில நாட்களுக்கு ஒருமுறை கீழ் இலைகளைக் கண்காணிக்கவும்.",
    },
    "ood_title": {
        "en": "Not a supported leaf",
        "hi": "समर्थित पत्ती नहीं",
        "ta": "ஆதரிக்கப்படும் இலை அல்ல",
    },
    "pathogen": {"en": "Pathogen", "hi": "रोगजनक", "ta": "நோய்க்காரணி"},
    "symptoms": {"en": "Symptoms", "hi": "लक्षण", "ta": "அறிகுறிகள்"},
    "organic_control": {"en": "Organic & cultural control", "hi": "जैविक एवं सांस्कृतिक नियंत्रण", "ta": "இயற்கை மற்றும் சாகுபடி கட்டுப்பாடு"},
    "chemical_control": {"en": "Chemical management", "hi": "रासायनिक प्रबंधन", "ta": "இரசாயன மேலாண்மை"},
    "source": {"en": "Source", "hi": "स्रोत", "ta": "ஆதாரம்"},
    "original": {"en": "Original", "hi": "मूल", "ta": "அசல்"},
    "heatmap": {"en": "Heatmap", "hi": "हीटमैप", "ta": "வெப்ப வரைபடம்"},
    "gradcam_unavailable": {
        "en": "Heatmap unavailable for this photo.",
        "hi": "इस फोटो के लिए हीटमैप उपलब्ध नहीं है।",
        "ta": "இந்தப் புகைப்படத்திற்கு வெப்ப வரைபடம் இல்லை.",
    },
    "new_photo": {"en": "Check another leaf", "hi": "दूसरी पत्ती जांचें", "ta": "மற்றொரு இலையைச் சோதிக்கவும்"},
    "footer": {
        "en": "A decision aid, not a replacement for an expert. Always confirm before spraying.",
        "hi": "यह सहायक उपकरण है, विशेषज्ञ का विकल्प नहीं। छिड़काव से पहले पुष्टि करें।",
        "ta": "இது ஒரு உதவிக் கருவி, நிபுணருக்கு மாற்று அல்ல. தெளிப்பதற்கு முன் உறுதிப்படுத்தவும்.",
    },
})

DISEASE_TRANSLATIONS = {
    "Tomato Early Blight": {
        "en": "Tomato Early Blight",
        "hi": "टमाटर - अगेती झुलसा (Early Blight)",
        "ta": "தக்காளி - ஆரம்ப கருகல் நோய் (Early Blight)",
    },
    "Tomato Bacterial Spot": {
        "en": "Tomato Bacterial Spot",
        "hi": "टमाटर - जीवाणु पत्ता धब्बा (Bacterial Spot)",
        "ta": "தக்காளி - பாக்டீரியா இலைப்புள்ளி நோய் (Bacterial Spot)",
    },
    "Tomato Late Blight": {
        "en": "Tomato Late Blight",
        "hi": "टमाटर - पछेती झुलसा (Late Blight)",
        "ta": "தக்காளி - தாமத கருகல் நோய் (Late Blight)",
    },
    "Tomato Healthy": {
        "en": "Tomato Healthy Leaf",
        "hi": "टमाटर - स्वस्थ पत्ता (Healthy)",
        "ta": "தக்காளி - ஆரோக்கியமான இலை (Healthy)",
    },
    "Potato Early Blight": {
        "en": "Potato Early Blight",
        "hi": "आलू - अगेती झुलसा (Early Blight)",
        "ta": "உருளைக்கிழங்கு - ஆரம்ப கருகல் நோய் (Early Blight)",
    },
    "Potato Late Blight": {
        "en": "Potato Late Blight",
        "hi": "आलू - पछेती झुलसा (Late Blight)",
        "ta": "உருளைக்கிழங்கு - தாமத கருகல் நோய் (Late Blight)",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Fetch localized string with fallback to English."""
    val = TRANSLATIONS.get(key, {}).get(lang) or TRANSLATIONS.get(key, {}).get("en", key)
    if kwargs:
        val = val.format(**kwargs)
    return val


def t_disease(name: str, lang: str = "en") -> str:
    """Translate disease name with fallback to English."""
    entry = DISEASE_TRANSLATIONS.get(name)
    if not entry:
        return name
    return entry.get(lang, entry.get("en", name))
