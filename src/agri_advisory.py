"""Agronomic extension care notes and management guides for crop diseases.

Cited from ICAR (Indian Council of Agricultural Research) & TNAU (Tamil Nadu Agricultural University)
Agritech Portals to provide actionable, expert-backed field guidance.
"""
from __future__ import annotations

ADVISORIES = {
    "Tomato Early Blight": {
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": {
            "en": "Concentric 'target-board' ring lesions with yellow halo, initially on older lower leaves.",
            "hi": "पुरानी निचली पत्तियों पर पीले घेरे के साथ संकेंद्रित 'टारगेट-बोर्ड' छल्लेदार धब्बे।",
            "ta": "கீழ் இலைகளில் மஞ்சள் வளையத்துடன் கூடிய வட்ட வடிவ கருகல் புள்ளிகள்.",
        },
        "cultural_organic": {
            "en": "Remove infected lower leaves to restrict spore spread. Mulch soil to prevent rain splash. Spray 5% Neem Seed Kernel Extract (NSKE) or Trichoderma viride (5g/L).",
            "hi": "रोगग्रस्त निचली पत्तियों को हटाकर नष्ट करें। मिट्टी पर पुआल बिछाएं। 5% नीम गिरी अर्क (NSKE) या ट्राइकोडर्मा विरिडी (5 ग्राम/लीटर) का छिड़काव करें।",
            "ta": "பாதிக்கப்பட்ட கீழ் இலைகளை அகற்றி அழிக்கவும். 5% வேப்பங்கொட்டை சாறு அல்லது டிரைக்கோடெர்மா விரிடி தெளிக்கவும்.",
        },
        "chemical_control": {
            "en": "Foliar spray of Mancozeb 75% WP @ 2g/L or Chlorothalonil 75% WP @ 2g/L at 10–14 day intervals (TNAU Agritech).",
            "hi": "मैंकोजेब 75% WP (2 ग्राम/लीटर) या क्लोरोथैलोनिल (2 ग्राम/लीटर) का 10-14 दिनों के अंतराल पर छिड़काव करें।",
            "ta": "மேன்கோசெப் (Mancozeb) 2 கிராம்/லிட்டர் அல்லது குளோரோதலோனில் தெளிக்கவும் (TNAU வழிகாட்டல்).",
        },
        "source": "ICAR-IIHR & TNAU Agritech Portal",
    },
    "Tomato Bacterial Spot": {
        "pathogen": "Xanthomonas perforans / vesicatoria (Bacteria)",
        "symptoms": {
            "en": "Small, angular water-soaked dark brown spots that may coalesce; yellow halo around spots.",
            "hi": "पत्तियों पर छोटे, कोणीय जल-सिक्त गहरे भूरे/काले धब्बे जो सूखकर छेद बना देते हैं।",
            "ta": "இலைகளில் சிறிய, நீர் ஊறிய பழுப்பு நிற புள்ளிகள் மற்றும் மஞ்சள் வளையம்.",
        },
        "cultural_organic": {
            "en": "Avoid overhead sprinkler irrigation. Use disease-free certified seeds. Spray Pseudomonas fluorescens @ 5g/L as preventive biocontrol.",
            "hi": "फव्वारा सिंचाई से बचें ताकि पत्तियां गीली न रहें। स्यूडोमोनास फ्लोरेसेंस (5 ग्राम/लीटर) का जैविक छिड़काव करें।",
            "ta": "தெளிப்பு பாசனத்தைத் தவிர்க்கவும். சூடோமோனாஸ் ஃப்ளோரசன்ஸ் 5 கிராம்/லிட்டர் தெளிக்கவும்.",
        },
        "chemical_control": {
            "en": "Copper Oxychloride 50% WP (2.5 g/L) mixed with Streptocycline (1 g per 10 L water) sprayed at first symptom onset (ICAR).",
            "hi": "कॉपर ऑक्सीक्लोराइड (2.5 ग्राम/लीटर) के साथ स्ट्रेप्टोसाइक्लिन (1 ग्राम प्रति 10 लीटर पानी) मिलाकर छिड़कें।",
            "ta": "காப்பர் ஆக்சிகுளோரைடு (2.5 கிராம்/லிட்டர்) + ஸ்ட்ரெப்டோமைசின் கலந்து தெளிக்கவும்.",
        },
        "source": "ICAR-Indian Institute of Vegetable Research (IIVR)",
    },
    "Tomato Late Blight": {
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": {
            "en": "Rapidly expanding water-soaked greasy lesions, white cottony mildew under leaf margins in humid weather.",
            "hi": "पत्तियों पर तेजी से फैलने वाले पानी से भीगे भूरे-काले धब्बे; नम मौसम में पत्ती के नीचे सफेद फफूंद।",
            "ta": "இலைகளில் வேகமாகப் பரவும் நீர் கசிந்த கருப்பு புள்ளிகள், கீழ் பகுதியில் வெள்ளை பூஞ்சை படலம்.",
        },
        "cultural_organic": {
            "en": "Ensure proper plant spacing for canopy aeration. Destroy severely infected plants. Avoid watering late in the evening.",
            "hi": "हवा के संचार के लिए पौधों के बीच उचित दूरी रखें। गंभीर रूप से रोगग्रस्त पौधों को जला दें।",
            "ta": "செடிகளுக்கு இடையே போதிய இடைவெளி விட்டு காற்றோட்டத்தை அதிகரிக்கவும்.",
        },
        "chemical_control": {
            "en": "Immediate spray of Metalaxyl + Mancozeb (Ridomil MZ @ 2g/L) or Cymoxanil + Mancozeb @ 2g/L upon early detection.",
            "hi": "लक्षण दिखते ही रिडोमिल एमजेड (मेटालैक्सिल + मैंकोजेब 2 ग्राम/लीटर) का छिड़काव करें।",
            "ta": "ரிடோமில் எம்.இசட் (Ridomil MZ @ 2 கிராம்/லிட்டர்) உடனடி தெளிப்பு அவசியம்.",
        },
        "source": "ICAR-CPRI / TNAU Plant Pathology Advisory",
    },
    "Tomato Healthy": {
        "pathogen": "None (Healthy Tissue)",
        "symptoms": {
            "en": "Vibrant green leaf tissue with intact venation and no chlorosis or necrosis.",
            "hi": "पत्ती का रंग चमकीला हरा, शिराएं स्पष्ट और किसी भी प्रकार के धब्बे रहित।",
            "ta": "ஆரோக்கியமான பச்சையம் மற்றும் கறைகள் அற்ற இலை பரப்பு.",
        },
        "cultural_organic": {
            "en": "Maintain regular drip irrigation and balanced N-P-K fertigation. Scout field weekly for pest or disease vectors (whiteflies/aphids).",
            "hi": "नियमित ड्रिप सिंचाई और संतुलित पोषण बनाए रखें। सफेद मक्खी या कीटों के लिए साप्ताहिक निरीक्षण करें।",
            "ta": "வழக்கமான சொட்டு நீர் பாசனம் மற்றும் ஊட்டச்சத்து மேலாண்மையை தொடரவும்.",
        },
        "chemical_control": {
            "en": "No chemical treatment needed. Preventive organic spray of Neem oil (3ml/L) recommended.",
            "hi": "किसी रासायनिक कीटनाशक की आवश्यकता नहीं है। सुरक्षात्मक नीम तेल (3 मिली/लीटर) का उपयोग करें।",
            "ta": "இரசாயன மருந்துகள் தேவையில்லை. முன்னெச்சரிக்கையாக வேப்பெண்ணெய் (3 மிலி/லிட்டர்) தெளிக்கலாம்.",
        },
        "source": "Good Agricultural Practices (GAP) Guide",
    },
    "Potato Early Blight": {
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": {
            "en": "Dark brown circular spots with characteristic concentric rings; leaf tissue becomes dry and brittle.",
            "hi": "पत्तियों पर गोलाकार गहरे भूरे छल्लेदार धब्बे; पुरानी पत्तियां सूखकर मुड़ने लगती हैं।",
            "ta": "வட்ட வடிவ அடர் பழுப்பு புள்ளிகள், இலைகள் காய்ந்து உடையும் தன்மை பெறும்.",
        },
        "cultural_organic": {
            "en": "Use certified seed tubers. Crop rotation with non-solanaceous crops for 2 years. Maintain field hygiene.",
            "hi": "प्रमाणित रोगमुक्त आलू कंदों का उपयोग करें। गैर-सोलानेसी फसलों के साथ फसल चक्र अपनाएं।",
            "ta": "சான்றளிக்கப்பட்ட விதை கிழங்குகளை பயன்படுத்தவும். பயிர் சுழற்சி முறை பின்பற்றவும்.",
        },
        "chemical_control": {
            "en": "Foliar application of Mancozeb 75% WP (2 to 2.5 kg/ha) or Propineb 70% WP @ 2g/L at symptom onset (ICAR-CPRI).",
            "hi": "लक्षण दिखते ही मैंकोजेब 75% WP (2 ग्राम/लीटर) या प्रोपिनेब (2 ग्राम/लीटर) का छिड़काव करें।",
            "ta": "மேன்கோசெப் 75% WP (2 கிராம்/லிட்டர்) தெளிக்கவும் (ICAR-CPRI வழிகாட்டல்).",
        },
        "source": "ICAR-Central Potato Research Institute (CPRI), Shimla",
    },
    "Potato Late Blight": {
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": {
            "en": "Water-soaked irregular blackish-brown spots on leaf tips and margins; devastating rapid blighting of foliage in cool, humid weather.",
            "hi": "पत्ती के किनारों पर काले-भूरे गीले धब्बे; ठंडे और नम मौसम में फसल तेजी से झुलस जाती है।",
            "ta": "இலை விளிம்புகளில் நீர் ஊறிய கருப்பு புள்ளிகள்; குளிர்ந்த காற்றில் வேகமாக பரவி செடியை அழிக்கும்.",
        },
        "cultural_organic": {
            "en": "Follow CPRI Indo-Blightcast advisories. Proper earthing up (mounding) to protect underground tubers from spores. Kill haulms prior to harvest.",
            "hi": "आलू की मेड़ों को अच्छी तरह चढ़ाएं ताकि कंद तक फफूंद न पहुंचे। खुदाई से 10 दिन पहले बेलें काट दें।",
            "ta": "கிழங்குகளை மூடி மண் அணைக்கவும். அறுவடைக்கு முன் செடிகளை வெட்டி அகற்றவும்.",
        },
        "chemical_control": {
            "en": "Preventive: Mancozeb 75 WP @ 2.5 g/L. Curative: Fenamidone + Mancozeb (Sectin @ 3 g/L) or Cymoxanil + Mancozeb @ 2.5 g/L (ICAR-CPRI).",
            "hi": "रोकथाम के लिए मैंकोजेब (2.5 ग्राम/लीटर)। बीमारी आने पर साइमोक्सानिल + मैंकोजेब (2.5 ग्राम/लीटर) छिड़कें।",
            "ta": "பாதுகாப்பிற்கு மேன்கோசெப்; நோய் தீவிரமடைந்தால் சைமோக்சானில் + மேன்கோசெப் தெளிக்கவும்.",
        },
        "source": "ICAR-Central Potato Research Institute (CPRI), Shimla",
    },
}


def get_advisory(disease_name: str) -> dict | None:
    return ADVISORIES.get(disease_name)
