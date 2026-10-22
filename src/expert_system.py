"""Synthetic agronomist: a rule-based expert system that turns a diagnosis plus field
conditions into an action plan (urgency, treatment track, doses scaled to the farm,
spray calendar, prevention, safety, escalation).

Inputs are the classifier's disease, the farmer's profile and optional field conditions.
Facts are derived first (severity, weather risk, urgency), then rules fire against the
facts and append to the plan. Every rule that fires is reported, so the advice is
explainable. Dosages come from the ICAR / TNAU notes in agri_advisory.py; this is
decision support, not a substitute for a local agronomist or the product label.
"""
from __future__ import annotations

from src.agri_advisory import get_advisory

KISAN_CALL_CENTRE = "1800-180-1551"
KVK_FINDER = "https://kvk.icar.gov.in"
ACRE_PER_HA = 0.4047


def L(en: str, hi: str, ta: str) -> dict:
    return {"en": en, "hi": hi, "ta": ta}


def tx(d: dict, lang: str, **kw) -> str:
    s = d.get(lang) or d["en"]
    return s.format(**kw) if kw else s


# ---- disease knowledge -------------------------------------------------------
META = {
    "Tomato Early Blight":   {"crop": "tomato", "kind": "fungal",   "base": 1, "interval": (10, 14)},
    "Tomato Bacterial Spot": {"crop": "tomato", "kind": "bacterial", "base": 2, "interval": (7, 10)},
    "Tomato Late Blight":    {"crop": "tomato", "kind": "oomycete", "base": 3, "interval": (5, 7)},
    "Tomato Healthy":        {"crop": "tomato", "kind": "healthy",  "base": 0, "interval": (14, 14)},
    "Potato Early Blight":   {"crop": "potato", "kind": "fungal",   "base": 1, "interval": (10, 14)},
    "Potato Late Blight":    {"crop": "potato", "kind": "oomycete", "base": 3, "interval": (5, 7)},
}

# (track, product, dose, unit, basis)  basis: per_l = dose per litre of spray, per_ha = kg/ha
NSKE = ("organic", L("Neem seed kernel extract (NSKE) 5%", "नीम गिरी अर्क (NSKE) 5%", "வேப்பங்கொட்டை சாறு (NSKE) 5%"), 50, "ml", "per_l")
TRICHO = ("organic", L("Trichoderma viride", "ट्राइकोडर्मा विरिडी", "டிரைக்கோடெர்மா விரிடி"), 5, "g", "per_l")
PSEUDO = ("organic", L("Pseudomonas fluorescens", "स्यूडोमोनास फ्लोरेसेंस", "சூடோமோனாஸ் ஃப்ளோரசென்ஸ்"), 5, "g", "per_l")
NEEM_OIL = ("organic", L("Neem oil", "नीम का तेल", "வேப்ப எண்ணெய்"), 3, "ml", "per_l")
TREATMENTS = {
    "Tomato Early Blight": [NSKE, TRICHO,
        ("chemical", L("Mancozeb 75% WP", "मैंकोजेब 75% WP", "மேன்கோசெப் 75% WP"), 2, "g", "per_l"),
        ("chemical", L("Chlorothalonil 75% WP", "क्लोरोथैलोनिल 75% WP", "குளோரோதலோனில் 75% WP"), 2, "g", "per_l")],
    "Tomato Bacterial Spot": [PSEUDO,
        ("chemical", L("Copper oxychloride 50% WP", "कॉपर ऑक्सीक्लोराइड 50% WP", "காப்பர் ஆக்சிகுளோரைடு 50% WP"), 2.5, "g", "per_l"),
        ("chemical", L("Streptocycline (with copper)", "स्ट्रेप्टोसाइक्लिन (कॉपर के साथ)", "ஸ்ட்ரெப்டோசைக்ளின் (காப்பருடன்)"), 0.1, "g", "per_l")],
    "Tomato Late Blight": [
        ("chemical", L("Metalaxyl + Mancozeb (Ridomil MZ)", "मेटालैक्सिल + मैंकोजेब (रिडोमिल MZ)", "மெட்டாலாக்சில் + மேன்கோசெப் (ரிடோமில் MZ)"), 2, "g", "per_l"),
        ("chemical", L("Cymoxanil + Mancozeb", "साइमोक्सानिल + मैंकोजेब", "சைமோக்சானில் + மேன்கோசெப்"), 2, "g", "per_l")],
    "Tomato Healthy": [NEEM_OIL],
    "Potato Early Blight": [
        ("chemical", L("Mancozeb 75% WP", "मैंकोजेब 75% WP", "மேன்கோசெப் 75% WP"), 2.5, "kg", "per_ha"),
        ("chemical", L("Propineb 70% WP", "प्रोपीनेब 70% WP", "புரோபினெப் 70% WP"), 2, "g", "per_l")],
    "Potato Late Blight": [
        ("chemical", L("Mancozeb 75% WP (preventive)", "मैंकोजेब 75% WP (रोकथाम)", "மேன்கோசெப் 75% WP (தடுப்பு)"), 2.5, "g", "per_l"),
        ("chemical", L("Fenamidone + Mancozeb (curative)", "फेनामिडोन + मैंकोजेब (उपचार)", "ஃபெனாமிடோன் + மேன்கோசெப் (சிகிச்சை)"), 3, "g", "per_l"),
        ("chemical", L("Cymoxanil + Mancozeb (curative)", "साइमोक्सानिल + मैंकोजेब (उपचार)", "சைமோக்சானில் + மேன்கோசெப் (சிகிச்சை)"), 2.5, "g", "per_l")],
}

SPRAY_L_PER_ACRE = {"seedling": 100, "vegetative": 200, "flowering": 300, "fruiting": 400, "harvest": 400}

LABELS = {
    "severity": {"none": L("None", "कोई नहीं", "இல்லை"), "mild": L("Mild", "हल्का", "லேசான"),
                 "moderate": L("Moderate", "मध्यम", "மிதமான"), "severe": L("Severe", "गंभीर", "கடுமையான")},
    "risk": {"low": L("Low", "कम", "குறைவு"), "moderate": L("Moderate", "मध्यम", "மிதமான"),
             "high": L("High", "अधिक", "அதிகம்")},
    "stage": {"seedling": L("Seedling", "पौध", "நாற்று"), "vegetative": L("Vegetative", "वानस्पतिक", "வளர்ச்சி நிலை"),
              "flowering": L("Flowering", "फूल अवस्था", "பூக்கும் நிலை"), "fruiting": L("Fruit / tuber set", "फल / कंद बनना", "காய் / கிழங்கு பிடிப்பு"),
              "harvest": L("Near harvest", "कटाई के करीब", "அறுவடை நெருக்கம்")},
    "urgency": {
        0: L("Monitor", "निगरानी", "கண்காணிப்பு"), 1: L("Treat within 7 days", "7 दिनों में उपचार", "7 நாட்களில் சிகிச்சை"),
        2: L("Treat within 3 days", "3 दिनों में उपचार", "3 நாட்களில் சிகிச்சை"), 3: L("Treat within 48 hours", "48 घंटे में उपचार", "48 மணி நேரத்தில் சிகிச்சை"),
        4: L("Act today", "आज ही कार्रवाई करें", "இன்றே நடவடிக்கை")},
    "track": {
        "preventive": L("Preventive care", "रोकथाम", "தடுப்பு பராமரிப்பு"),
        "organic": L("Organic-first", "जैविक प्राथमिकता", "இயற்கை முறை முன்னுரிமை"),
        "integrated": L("Integrated (cultural + chemical)", "समेकित (सांस्कृतिक + रासायनिक)", "ஒருங்கிணைந்த (சாகுபடி + இரசாயனம்)"),
        "hold": L("Hold treatment", "उपचार रोकें", "சிகிச்சையை நிறுத்தவும்")},
}

T = {
    "scout": (L("Scout the whole field", "पूरे खेत का निरीक्षण करें", "முழு வயலையும் ஆய்வு செய்யவும்"),
              L("Walk a zig-zag path and check about 20 plants. Mark infected plants and photograph new symptoms to track spread.",
                "ज़िग-ज़ैग चलकर लगभग 20 पौधे जाँचें। रोगी पौधों को चिह्नित करें और नए लक्षणों की फ़ोटो लें।",
                "ஜிக்-ஜாக் முறையில் சுமார் 20 செடிகளைச் சோதிக்கவும். பாதிக்கப்பட்ட செடிகளைக் குறித்து, புதிய அறிகுறிகளைப் படம் எடுக்கவும்.")),
    "remove": (L("Remove and destroy infected leaves", "रोगी पत्तियाँ हटाकर नष्ट करें", "பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்"),
               L("Pick off infected leaves, bag them and burn or bury them away from the field. Do not compost.",
                 "रोगी पत्तियाँ तोड़कर थैले में रखें और खेत से दूर जलाएँ या दबाएँ। खाद न बनाएँ।",
                 "பாதிக்கப்பட்ட இலைகளைப் பறித்து, பையில் போட்டு வயலிலிருந்து தூரத்தில் எரிக்கவும் அல்லது புதைக்கவும். உரமாக்க வேண்டாம்.")),
    "uproot": (L("Uproot severely infected plants", "गंभीर रोगी पौधे उखाड़ें", "கடுமையாக பாதித்த செடிகளை பிடுங்கவும்"),
               L("Plants with most leaves blighted keep releasing spores. Uproot, bag and bury or burn them to protect neighbours.",
                 "जिन पौधों की अधिकतर पत्तियाँ झुलस चुकी हैं वे बीजाणु छोड़ते रहते हैं। उन्हें उखाड़कर दबाएँ या जलाएँ।",
                 "பெரும்பாலான இலைகள் கருகிய செடிகள் தொடர்ந்து வித்துகளை பரப்பும். அவற்றைப் பிடுங்கி புதைக்கவும் அல்லது எரிக்கவும்.")),
    "spray_chem": (L("Start the spray programme", "छिड़काव कार्यक्रम शुरू करें", "தெளிப்பு திட்டத்தைத் தொடங்கவும்"),
                   L("Apply the first spray in the cool morning or evening, wetting both sides of the leaves. Follow the calendar below.",
                     "पहला छिड़काव ठंडी सुबह या शाम को करें, पत्तियों के दोनों ओर। नीचे दिए कैलेंडर का पालन करें।",
                     "முதல் தெளிப்பை குளிர்ந்த காலை அல்லது மாலையில், இலைகளின் இருபுறமும் படுமாறு செய்யவும். கீழே உள்ள அட்டவணையைப் பின்பற்றவும்.")),
    "spray_bio": (L("Start the bio-control sprays", "जैव-नियंत्रण छिड़काव शुरू करें", "உயிரி கட்டுப்பாட்டு தெளிப்பைத் தொடங்கவும்"),
                  L("Bio-agents work best before heavy infection. Spray weekly in the evening and cover both leaf surfaces.",
                    "जैव-कारक भारी संक्रमण से पहले सबसे अच्छा काम करते हैं। हर हफ़्ते शाम को दोनों सतहों पर छिड़कें।",
                    "உயிரி காரணிகள் அதிக தொற்றுக்கு முன் சிறப்பாக செயல்படும். வாரந்தோறும் மாலையில் இருபுறமும் தெளிக்கவும்.")),
    "rain": (L("Spray before the rain, not after", "बारिश से पहले छिड़काव करें", "மழைக்கு முன் தெளிக்கவும்"),
             L("Wet, humid weather speeds up spread. Spray 1-2 days before forecast rain so it dries on the leaf; repeat if washed off within 6 hours.",
               "नम मौसम में रोग तेज़ी से फैलता है। बारिश के 1-2 दिन पहले छिड़कें; 6 घंटे में धुल जाए तो दोहराएँ।",
               "ஈரப்பதமான வானிலையில் நோய் வேகமாகப் பரவும். மழைக்கு 1-2 நாள் முன் தெளிக்கவும்; 6 மணி நேரத்தில் கழுவப்பட்டால் மீண்டும் தெளிக்கவும்.")),
    "water": (L("Water at the base, early in the day", "जड़ में सिंचाई करें, सुबह जल्दी", "அடிப்பகுதியில், காலையிலேயே நீர் பாய்ச்சவும்"),
              L("Wet leaves spread disease. Use drip or furrow irrigation, avoid sprinklers and avoid evening watering.",
                "गीली पत्तियाँ रोग फैलाती हैं। ड्रिप या नाली सिंचाई करें; फव्वारा और शाम की सिंचाई से बचें।",
                "ஈரமான இலைகள் நோயைப் பரப்பும். சொட்டு அல்லது வாய்க்கால் பாசனம் செய்யவும்; தெளிப்பான் மற்றும் மாலை நீர்ப்பாசனத்தைத் தவிர்க்கவும்.")),
    "sanitize": (L("Disinfect tools; stay out when leaves are wet", "औज़ार साफ़ करें; गीले में खेत में न जाएँ", "கருவிகளைக் கிருமி நீக்கவும்; இலை ஈரமாக இருக்கும்போது வயலில் வேலை செய்ய வேண்டாம்"),
                 L("Bacteria travel on hands, tools and wet leaves. Clean secateurs and avoid field work while foliage is wet.",
                   "जीवाणु हाथ, औज़ार और गीली पत्तियों से फैलते हैं। कैंची साफ़ करें और गीली पत्तियों में काम न करें।",
                   "பாக்டீரியா கை, கருவி, ஈர இலைகள் மூலம் பரவும். கத்திரியைச் சுத்தம் செய்து, இலை ஈரமாக இருக்கும்போது வேலை செய்வதைத் தவிர்க்கவும்.")),
    "earthing": (L("Earth up the ridges", "मेड़ों पर मिट्टी चढ़ाएँ", "மேடுகளில் மண் அணைக்கவும்"),
                 L("Mound soil over tubers so spores washing down cannot reach them, and keep ridges well formed.",
                   "कंदों पर मिट्टी चढ़ाएँ ताकि बीजाणु उन तक न पहुँचें; मेड़ें मज़बूत रखें।",
                   "கிழங்குகள் மீது மண் அணைத்து, வித்துகள் அவற்றை அடையாமல் தடுக்கவும்; மேடுகளை நன்கு அமைக்கவும்.")),
    "rotate": (L("Rotate products between sprays", "छिड़काव में दवाएँ बदलते रहें", "தெளிப்புகளுக்கு இடையே மருந்துகளை மாற்றவும்"),
               L("Alternate between the listed products to slow resistance. Never exceed the label dose.",
                 "प्रतिरोध धीमा करने के लिए सूची की दवाएँ बदल-बदलकर प्रयोग करें। लेबल से अधिक मात्रा न डालें।",
                 "எதிர்ப்புத்திறனைத் தாமதப்படுத்த பட்டியலிலுள்ள மருந்துகளை மாற்றி மாற்றிப் பயன்படுத்தவும். லேபிள் அளவை மீறாதீர்கள்.")),
    "phi": (L("Respect the waiting period before harvest", "कटाई से पहले प्रतीक्षा अवधि का पालन करें", "அறுவடைக்கு முன் காத்திருப்பு காலத்தைப் பின்பற்றவும்"),
            L("You are close to harvest. Prefer bio-control, or the product with the shortest pre-harvest interval on its label, and count the waiting days before picking.",
              "आप कटाई के करीब हैं। जैव-नियंत्रण चुनें या लेबल पर सबसे कम प्रतीक्षा अवधि वाली दवा लें और तुड़ाई से पहले दिन गिनें।",
              "அறுவடை நெருங்கிவிட்டது. உயிரி கட்டுப்பாடு அல்லது லேபிளில் குறைந்த காத்திருப்பு காலமுள்ள மருந்தைத் தேர்ந்தெடுத்து, பறிப்பதற்கு முன் நாட்களைக் கணக்கிடவும்.")),
    "block": (L("Treat block by block", "खंड-दर-खंड उपचार करें", "பகுதி பகுதியாக சிகிச்சை செய்யவும்"),
              L("On larger plots, treat the affected block plus a 2-3 row buffer first, and keep scouting the rest weekly.",
                "बड़े खेत में पहले प्रभावित खंड और 2-3 पंक्ति का बफ़र उपचारित करें; बाकी की साप्ताहिक निगरानी करें।",
                "பெரிய நிலத்தில் முதலில் பாதித்த பகுதியையும் 2-3 வரிசை இடைவெளியையும் சிகிச்சை செய்து, மற்றவற்றை வாரந்தோறும் கண்காணிக்கவும்.")),
    "followup": (L("Re-photograph in 5 days", "5 दिन में दोबारा फ़ोटो लें", "5 நாட்களில் மீண்டும் படம் எடுக்கவும்"),
                 L("Photograph the same plants again and compare in History. If lesions keep spreading after two sprays, call an expert.",
                   "उन्हीं पौधों की फ़ोटो फिर लें और इतिहास में तुलना करें। दो छिड़काव के बाद भी धब्बे बढ़ें तो विशेषज्ञ से बात करें।",
                   "அதே செடிகளை மீண்டும் படம் எடுத்து வரலாற்றில் ஒப்பிடவும். இரண்டு தெளிப்புக்குப் பின்னும் புள்ளிகள் பரவினால் நிபுணரை அழைக்கவும்.")),
    "cultural": (L("Cultural and organic practices (ICAR / TNAU)", "सांस्कृतिक व जैविक उपाय (ICAR / TNAU)", "சாகுபடி மற்றும் இயற்கை நடைமுறைகள் (ICAR / TNAU)"), None),
    "hold": (L("Do not spray yet", "अभी छिड़काव न करें", "இப்போது தெளிக்க வேண்டாம்"),
             L("A confident-sounding but wrong spray wastes money and can harm the crop. Wait for expert confirmation of the disease.",
               "गलत दवा का छिड़काव पैसे की बर्बादी है और फ़सल को नुकसान पहुँचा सकता है। रोग की विशेषज्ञ पुष्टि का इंतज़ार करें।",
               "தவறான மருந்து தெளிப்பது பணத்தை வீணாக்கி பயிரை பாதிக்கலாம். நோயை நிபுணர் உறுதிப்படுத்தும் வரை காத்திருக்கவும்.")),
    "retake": (L("Re-photograph in good light", "अच्छी रोशनी में दोबारा फ़ोटो लें", "நல்ல வெளிச்சத்தில் மீண்டும் படம் எடுக்கவும்"),
               L("Take 2-3 close photos of different affected leaves in daylight, one leaf filling the frame, and show them to an expert.",
                 "दिन के उजाले में प्रभावित अलग-अलग पत्तियों की 2-3 पास की फ़ोटो लें, फ्रेम में एक पत्ती हो, और विशेषज्ञ को दिखाएँ।",
                 "பகல் வெளிச்சத்தில் பாதித்த வெவ்வேறு இலைகளின் 2-3 நெருக்கமான படங்களை, சட்டகத்தில் ஒரு இலை நிறைய, எடுத்து நிபுணரிடம் காட்டவும்.")),
    "isolate": (L("Limit spread while you wait", "प्रतीक्षा के दौरान फैलाव रोकें", "காத்திருக்கும் நேரத்தில் பரவலைத் தடுக்கவும்"),
                L("Remove the worst leaves, clean tools, avoid working in wet foliage and keep irrigation off the leaves.",
                  "सबसे ख़राब पत्तियाँ हटाएँ, औज़ार साफ़ रखें, गीली पत्तियों में काम न करें और पत्तियों पर सिंचाई न करें।",
                  "மிகவும் பாதித்த இலைகளை அகற்றி, கருவிகளைச் சுத்தம் செய்து, ஈர இலைகளில் வேலை செய்வதையும் இலைமீது நீர்ப்பாசனத்தையும் தவிர்க்கவும்.")),
    "weekly": (L("Scout weekly", "हर हफ़्ते निरीक्षण करें", "வாரந்தோறும் ஆய்வு செய்யவும்"),
               L("Check lower leaves and the undersides for early spots, and look for whiteflies and aphids that carry viruses.",
                 "निचली पत्तियों और उनके नीचे शुरुआती धब्बे देखें; सफ़ेद मक्खी और माहू पर नज़र रखें।",
                 "கீழ் இலைகள் மற்றும் அடிப்பகுதியில் ஆரம்ப புள்ளிகளைப் பார்க்கவும்; வெள்ளை ஈ, அசுவினி உள்ளதா எனவும் கவனிக்கவும்.")),
}

PREVENT = {
    "fungal": [L("Rotate with non-solanaceous crops for 2-3 years.", "2-3 साल तक गैर-सोलेनेसी फ़सलों के साथ फ़सल चक्र अपनाएँ।", "2-3 ஆண்டுகள் சோலனேசி அல்லாத பயிர்களுடன் சுழற்சி செய்யவும்."),
               L("Stake and prune lower leaves so air moves through the canopy; mulch to stop soil splash.", "पौधों को सहारा दें, निचली पत्तियाँ छाँटें और मल्च बिछाएँ ताकि हवा चले और मिट्टी के छींटे न लगें।", "செடிகளுக்கு ஆதாரம் கொடுத்து, கீழ் இலைகளை நீக்கி, மண் தெறிப்பைத் தடுக்க தழைக்கூளம் இடவும்."),
               L("Use certified seed and avoid excess nitrogen, which makes foliage soft and susceptible.", "प्रमाणित बीज लें और अधिक नाइट्रोजन से बचें, इससे पत्तियाँ कोमल होकर रोग-ग्रस्त होती हैं।", "சான்றிதழ் பெற்ற விதைகளைப் பயன்படுத்தி, அதிக நைட்ரஜனைத் தவிர்க்கவும்.")],
    "oomycete": [L("Plant with wide spacing; avoid dense canopy and waterlogging.", "पर्याप्त दूरी पर रोपें; घनी छत्र और जलभराव से बचें।", "போதிய இடைவெளியில் நடவும்; அடர்த்தியான இலைப்பரப்பு, நீர்தேக்கத்தைத் தவிர்க்கவும்."),
                 L("Watch humid, cool, cloudy spells; start protective sprays before the disease arrives (potato: follow ICAR-CPRI Indo-Blightcast).", "नम, ठंडे, बादल वाले मौसम पर नज़र रखें; रोग आने से पहले सुरक्षात्मक छिड़काव करें (आलू: ICAR-CPRI इंडो-ब्लाइटकास्ट)।", "ஈரமான, குளிர்ந்த, மேகமூட்டமான நாட்களைக் கவனித்து, நோய் வருவதற்கு முன் பாதுகாப்பு தெளிப்பு செய்யவும் (உருளை: ICAR-CPRI இண்டோ-ப்ளைட்காஸ்ட்)."),
                 L("Destroy cull piles and volunteer plants, and use certified disease-free seed.", "फेंके गए कंद/पौधे और स्वतः उगे पौधे नष्ट करें; प्रमाणित रोगमुक्त बीज लें।", "கழிவுக் குவியல்கள், தானாக முளைத்த செடிகளை அழித்து, சான்றிதழ் பெற்ற நோயற்ற விதையைப் பயன்படுத்தவும்.")],
    "bacterial": [L("Use certified or treated seed and healthy transplants.", "प्रमाणित या उपचारित बीज और स्वस्थ पौध लें।", "சான்றிதழ் பெற்ற அல்லது நேர்த்தி செய்த விதை, ஆரோக்கிய நாற்றுகளைப் பயன்படுத்தவும்."),
                  L("Avoid overhead irrigation and work in the field when leaves are dry.", "ऊपर से सिंचाई न करें और पत्तियाँ सूखी हों तभी खेत में काम करें।", "மேலிருந்து நீர்ப்பாசனம் தவிர்த்து, இலைகள் உலர்ந்திருக்கும்போது மட்டும் வயலில் வேலை செய்யவும்."),
                  L("Rotate away from tomato/pepper for 2 years and control weeds that host the bacteria.", "2 साल तक टमाटर/मिर्च न लगाएँ और जीवाणु के मेज़बान खरपतवार हटाएँ।", "2 ஆண்டுகள் தக்காளி/மிளகாய் பயிரிடாமல், பாக்டீரியாவை ஆதரிக்கும் களைகளை அகற்றவும்.")],
    "healthy": [L("Keep drip irrigation and balanced N-P-K nutrition.", "ड्रिप सिंचाई और संतुलित N-P-K पोषण रखें।", "சொட்டு நீர்ப்பாசனம், சமச்சீர் N-P-K ஊட்டச்சத்து வழங்கவும்."),
                L("Space plants for airflow and remove weeds that harbour pests.", "हवा के लिए पौधों में दूरी रखें और कीटों को आश्रय देने वाले खरपतवार हटाएँ।", "காற்றோட்டத்திற்கு இடைவெளி விட்டு, பூச்சிகளுக்கு இடமளிக்கும் களைகளை அகற்றவும்."),
                L("Photograph a few leaves every week; early detection is the cheapest treatment.", "हर हफ़्ते कुछ पत्तियों की फ़ोटो लें; जल्दी पहचान सबसे सस्ता उपचार है।", "வாரந்தோறும் சில இலைகளைப் படம் எடுக்கவும்; முன்கூட்டியே கண்டறிதலே மலிவான சிகிச்சை.")],
}

SAFETY = [
    L("Read the product label: follow dose, re-entry interval and pre-harvest waiting period. Use only products registered with CIB&RC.", "लेबल पढ़ें: मात्रा, प्रवेश अंतराल और कटाई-पूर्व प्रतीक्षा अवधि का पालन करें। केवल CIB&RC पंजीकृत दवाएँ प्रयोग करें।", "லேபிளைப் படித்து அளவு, மீண்டும் நுழையும் இடைவெளி, அறுவடைக்கு முந்தைய காத்திருப்பு காலத்தைப் பின்பற்றவும். CIB&RC பதிவு பெற்ற மருந்துகளை மட்டும் பயன்படுத்தவும்."),
    L("Wear gloves, mask and full sleeves; wash hands and clothes after spraying.", "दस्ताने, मास्क और पूरी बाँह के कपड़े पहनें; छिड़काव के बाद हाथ-कपड़े धोएँ।", "கையுறை, முகமூடி, முழுக்கை ஆடை அணிந்து, தெளித்த பின் கை, உடைகளைக் கழுவவும்."),
    L("Spray in calm, cool hours, away from water sources, and in the evening to protect pollinating bees.", "शांत, ठंडे समय में, जल स्रोतों से दूर और मधुमक्खियों की सुरक्षा के लिए शाम को छिड़कें।", "காற்றற்ற, குளிர்ந்த நேரத்தில், நீர்நிலைகளிலிருந்து தள்ளி, தேனீக்களைப் பாதுகாக்க மாலையில் தெளிக்கவும்."),
    L("Do not mix products unless the label allows it; keep chemicals in original containers away from children.", "लेबल अनुमति न दे तो दवाएँ न मिलाएँ; रसायन मूल डिब्बे में बच्चों से दूर रखें।", "லேபிள் அனுமதிக்காவிட்டால் மருந்துகளைக் கலக்க வேண்டாம்; இரசாயனங்களை குழந்தைகளிடமிருந்து விலக்கி அசல் கொள்கலனில் வைக்கவும்."),
]

DISCLAIMER = L(
    "Decision support based on ICAR / TNAU guidance and your inputs. It does not replace a local agronomist or the product label. Doses are approximate for the spray volume shown.",
    "यह ICAR / TNAU मार्गदर्शन और आपके इनपुट पर आधारित निर्णय-सहायता है। यह स्थानीय कृषि विशेषज्ञ या दवा के लेबल का विकल्प नहीं है। मात्रा दिए गए छिड़काव आयतन के लिए अनुमानित है।",
    "இது ICAR / TNAU வழிகாட்டுதல் மற்றும் உங்கள் உள்ளீடுகளை அடிப்படையாகக் கொண்ட முடிவு ஆதரவு. உள்ளூர் வேளாண் நிபுணர் அல்லது மருந்து லேபிளுக்கு மாற்றல்ல. அளவுகள் காட்டப்பட்ட தெளிப்பு அளவுக்கு தோராயமானவை.")


# ---- facts -------------------------------------------------------------------
def _num(v, lo, hi, default):
    try:
        return max(lo, min(float(v), hi))
    except (TypeError, ValueError):
        return default


def clean_ctx(ctx: dict | None, profile: dict | None) -> dict:
    ctx, profile = ctx or {}, profile or {}
    pick = lambda v, ok, d: v if v in ok else d
    return {
        "severity": pick(ctx.get("severity"), ("auto", "mild", "moderate", "severe"), "auto"),
        "humidity": pick(ctx.get("humidity"), ("low", "medium", "high"), "medium"),
        "rain": bool(ctx.get("rain", False)),
        "stage": pick(ctx.get("stage"), tuple(SPRAY_L_PER_ACRE), "vegetative"),
        "acres": _num(ctx.get("acres", profile.get("farm_acres", 1.0)), 0.1, 5000.0, 1.0),
        "organic_only": bool(ctx.get("organic_only", profile.get("organic_only", False))),
    }


def _estimate_severity(brown_ratio: float | None) -> tuple[str, str]:
    if brown_ratio is None:
        return "moderate", "assumed"
    return ("mild" if brown_ratio < 0.08 else "moderate" if brown_ratio < 0.25 else "severe"), "photo"


def _fmt_qty(qty: float, unit: str) -> str:
    if unit == "g" and qty >= 1000:
        return f"{qty / 1000:.2f} kg"
    if unit == "ml" and qty >= 1000:
        return f"{qty / 1000:.2f} L"
    return f"{qty:.0f} {unit}" if qty >= 10 else f"{qty:.1f} {unit}"


def _treatment_rows(disease: str, track: str, acres: float, volume: float, lang: str) -> list[dict]:
    rows = []
    for kind, name, dose, unit, basis in TREATMENTS.get(disease, []):
        if track == "organic" and kind != "organic":
            continue
        if track == "preventive" and kind != "organic":
            continue
        if basis == "per_ha":
            dose_txt, total = f"{dose:g} kg/ha", f"{dose * acres * ACRE_PER_HA:.2f} kg"
        else:
            total = _fmt_qty(dose * volume * acres, unit)
            dose_txt = f"{dose:g} {unit} / L"
        rows.append({"kind": kind, "name": tx(name, lang), "dose": dose_txt, "total": total})
    return rows


def build_plan(disease: str | None, lang: str = "en", *, confidence: float | None = None,
               trust: float | None = None, brown_ratio: float | None = None, uncertain: bool = False,
               candidates: list[str] | None = None, ctx: dict | None = None,
               profile: dict | None = None) -> dict:
    lang = lang if lang in ("en", "hi", "ta") else "en"
    c = clean_ctx(ctx, profile)
    fired: list[dict] = []        # explainability trail

    def why(rule: str, en: str, hi: str, ta: str):
        fired.append({"rule": rule, "why": tx(L(en, hi, ta), lang)})

    actions: list[dict] = []

    def act(key: str, priority: str, detail_override: str | None = None):
        title, detail = T[key]
        actions.append({"id": key, "priority": priority, "title": tx(title, lang),
                        "detail": detail_override or (tx(detail, lang) if detail else "")})

    # --- Not enough certainty: safe interim plan -----------------------------------
    if uncertain or disease not in META:
        act("hold", "now"); act("retake", "now"); act("isolate", "now"); act("scout", "week")
        why("R10", "The model is not confident enough, so spraying is held back.",
            "मॉडल पर्याप्त भरोसेमंद नहीं है, इसलिए छिड़काव रोका गया है।", "மாதிரி போதுமான நம்பிக்கை தரவில்லை, எனவே தெளிப்பு நிறுத்தப்பட்டுள்ளது.")
        return {
            "kind": "uncertain", "disease": None, "candidates": candidates or [],
            "urgency": {"level": 1, "label": tx(LABELS["urgency"][1], lang)},
            "track": {"id": "hold", "label": tx(LABELS["track"]["hold"], lang)},
            "headline": tx(L("Diagnosis not confirmed: get an expert to look before you spray",
                             "निदान की पुष्टि नहीं: छिड़काव से पहले विशेषज्ञ को दिखाएँ",
                             "கண்டறிதல் உறுதியாகவில்லை: தெளிப்பதற்கு முன் நிபுணரிடம் காட்டவும்"), lang),
            "facts": [], "actions": actions, "treatment": [], "spray_volume": None, "schedule": [],
            "prevention": [tx(x, lang) for x in PREVENT["fungal"][:2]], "safety": [],
            "escalate": _escalation("required", lang), "reasoning": fired, "rules": [f["rule"] for f in fired],
            "disclaimer": tx(DISCLAIMER, lang), "ctx": c,
        }

    m = META[disease]
    kind, crop = m["kind"], m["crop"]
    adv = get_advisory(disease) or {}

    # --- Derived facts ---------------------------------------------------------
    if kind == "healthy":
        severity, sev_src = "none", "photo"
    elif c["severity"] != "auto":
        severity, sev_src = c["severity"], "you"
    else:
        severity, sev_src = _estimate_severity(brown_ratio)

    score, reasons = 0, []
    if c["humidity"] == "high":
        score += 1; reasons.append(L("high humidity", "अधिक नमी", "அதிக ஈரப்பதம்"))
    if c["rain"]:
        score += 1; reasons.append(L("recent / forecast rain", "हाल की / संभावित बारिश", "சமீபத்திய / எதிர்பார்க்கும் மழை"))
    if kind == "oomycete" and c["humidity"] != "low":
        score += 1; reasons.append(L("late blight thrives in humid weather", "पछेती झुलसा नम मौसम में पनपता है", "கருகல் நோய் ஈரமான வானிலையில் வளரும்"))
    risk = "high" if score >= 2 else "moderate" if score == 1 else "low"

    urgency = 0 if kind == "healthy" else min(4, m["base"] + {"mild": 0, "moderate": 1, "severe": 2}[severity] + (1 if risk == "high" else 0))
    volume = SPRAY_L_PER_ACRE[c["stage"]]
    near_harvest = c["stage"] == "harvest"
    big_farm = c["acres"] >= 5

    # --- Treatment track ---------------------------------------------------------
    escalate = None
    if kind == "healthy":
        track = "preventive"
        why("R1", "Healthy leaf: preventive care only.", "स्वस्थ पत्ती: केवल रोकथाम।", "ஆரோக்கியமான இலை: தடுப்பு பராமரிப்பு மட்டும்.")
    elif c["organic_only"]:
        track = "organic"
        why("R6", "You chose organic-only, so chemicals are left out.", "आपने केवल जैविक चुना है, इसलिए रसायन शामिल नहीं।", "நீங்கள் இயற்கை முறையை மட்டும் தேர்ந்தெடுத்ததால் இரசாயனங்கள் சேர்க்கப்படவில்லை.")
        if severity == "severe" or kind == "oomycete":
            escalate = "recommended"
            why("R6b", "Organic methods alone may not stop this disease at this stage.", "इस अवस्था में केवल जैविक उपाय इस रोग को रोक न पाएँ।", "இந்த நிலையில் இயற்கை முறைகள் மட்டும் இந்நோயை நிறுத்தாமல் போகலாம்.")
    elif severity == "mild" and risk != "high" and kind != "oomycete":
        track = "organic"
        why("R3", "Mild infection in calm weather: start organic-first.", "हल्का संक्रमण, सामान्य मौसम: जैविक से शुरू करें।", "லேசான தொற்று, சாதாரண வானிலை: இயற்கை முறையில் தொடங்கவும்.")
    else:
        track = "integrated"
        why("R4", "Severity, weather or disease type calls for cultural control plus a protective spray.", "गंभीरता, मौसम या रोग के प्रकार के कारण सांस्कृतिक नियंत्रण के साथ सुरक्षात्मक छिड़काव ज़रूरी है।", "தீவிரம், வானிலை அல்லது நோய் வகை காரணமாக சாகுபடி கட்டுப்பாட்டுடன் பாதுகாப்பு தெளிப்பும் தேவை.")
    if near_harvest and track == "integrated":
        why("R9", "Close to harvest: waiting periods matter.", "कटाई के करीब: प्रतीक्षा अवधि महत्वपूर्ण है।", "அறுவடை நெருங்கியுள்ளது: காத்திருப்பு காலம் முக்கியம்.")
    if severity == "severe" and kind != "fungal":
        escalate = "required" if kind == "oomycete" else (escalate or "recommended")
    if urgency >= 4 and not escalate:
        escalate = "recommended"
    if kind == "oomycete":
        why("R8", "Late blight can wipe out a field in days, so urgency is raised.", "पछेती झुलसा कुछ दिनों में खेत बर्बाद कर सकता है, इसलिए तात्कालिकता बढ़ाई गई।", "கருகல் நோய் சில நாட்களில் வயலை அழிக்கும், எனவே அவசரம் உயர்த்தப்பட்டுள்ளது.")

    # --- Actions -------------------------------------------------------------------
    if kind == "healthy":
        act("weekly", "ongoing")
        if adv:
            act("cultural", "ongoing", tx(adv["cultural_organic"], lang))
        if c["humidity"] == "high" or c["rain"]:
            act("water", "week")
    else:
        act("scout", "now")
        act("uproot" if severity == "severe" and kind == "oomycete" else "remove", "now")
        if kind == "bacterial":
            act("sanitize", "now")
        if track == "integrated":
            act("spray_chem", "now")
            if len([t for t in TREATMENTS[disease] if t[0] == "chemical"]) > 1:
                act("rotate", "week")
                why("R11", "More than one product is listed, so rotate them to avoid resistance.", "एक से अधिक दवाएँ सूचीबद्ध हैं, प्रतिरोध से बचने के लिए बदलकर प्रयोग करें।", "ஒன்றுக்கு மேற்பட்ட மருந்துகள் உள்ளதால் எதிர்ப்பைத் தவிர்க்க மாற்றிப் பயன்படுத்தவும்.")
        elif any(t[0] == "organic" for t in TREATMENTS[disease]):
            act("spray_bio", "now")
        if risk != "low" or c["rain"]:
            act("rain", "now")
            why("R4w", "Weather risk is {}.".format(tx(LABELS["risk"][risk], lang)), "मौसम का जोखिम: {}।".format(tx(LABELS["risk"][risk], lang)), "வானிலை அபாயம்: {}.".format(tx(LABELS["risk"][risk], lang)))
        act("water", "week")
        if crop == "potato" and kind == "oomycete":
            act("earthing", "week")
        if near_harvest and track == "integrated":
            act("phi", "now")
        if big_farm:
            act("block", "week")
            why("R13", "Farm is {:g} acres, so treat and scout block by block.".format(c["acres"]), "खेत {:g} एकड़ का है, इसलिए खंड-दर-खंड काम करें।".format(c["acres"]), "பண்ணை {:g} ஏக்கர், எனவே பகுதி பகுதியாக செய்யவும்.".format(c["acres"]))
        if adv:
            act("cultural", "ongoing", tx(adv["cultural_organic"], lang))
        act("followup", "ongoing")

    # --- Calendar ------------------------------------------------------------------
    lo, hi = m["interval"]
    interval = lo if (risk == "high" or urgency >= 3) else hi
    if track == "organic":
        interval = 7
    schedule: list[dict] = []
    if kind == "healthy":
        schedule = [{"day": 0, "task": tx(L("Weekly scouting and photo check", "साप्ताहिक निरीक्षण और फ़ोटो जाँच", "வாராந்திர ஆய்வு, படச் சோதனை"), lang)},
                    {"day": 14, "task": tx(L("Optional neem oil spray if the season is humid", "नमी का मौसम हो तो वैकल्पिक नीम तेल छिड़काव", "ஈரமான பருவமாக இருந்தால் விருப்ப வேப்ப எண்ணெய் தெளிப்பு"), lang)}]
    else:
        first = L("Remove infected leaves, then first spray", "रोगी पत्तियाँ हटाएँ, फिर पहला छिड़काव", "பாதித்த இலைகளை அகற்றி, முதல் தெளிப்பு")
        second = L("Second spray (switch product if two are listed)", "दूसरा छिड़काव (दो दवाएँ हों तो बदलें)", "இரண்டாம் தெளிப்பு (இரண்டு மருந்துகள் இருந்தால் மாற்றவும்)")
        third = L("Third spray, then re-photograph and compare", "तीसरा छिड़काव, फिर दोबारा फ़ोटो लेकर तुलना करें", "மூன்றாம் தெளிப்பு, பின்னர் மீண்டும் படம் எடுத்து ஒப்பிடவும்")
        schedule = [{"day": 0, "task": tx(first, lang)}, {"day": interval, "task": tx(second, lang)},
                    {"day": 2 * interval, "task": tx(third, lang)}]

    plan = {
        "kind": "healthy" if kind == "healthy" else "disease",
        "disease": disease, "crop": crop, "pathogen": adv.get("pathogen"), "source": adv.get("source"),
        "urgency": {"level": urgency, "label": tx(LABELS["urgency"][urgency], lang)},
        "track": {"id": track, "label": tx(LABELS["track"][track], lang)},
        "headline": _headline(kind, urgency, lang),
        "facts": [
            {"key": "severity", "value": tx(LABELS["severity"][severity], lang), "source": sev_src,
             "level": severity},
            {"key": "risk", "value": tx(LABELS["risk"][risk], lang), "level": risk,
             "detail": ", ".join(tx(r, lang) for r in reasons)},
            {"key": "stage", "value": tx(LABELS["stage"][c["stage"]], lang)},
            {"key": "farm", "value": f"{c['acres']:g} ac"},
        ],
        "actions": actions,
        "treatment": _treatment_rows(disease, track, c["acres"], volume, lang),
        "spray_volume": {"per_acre": volume, "total": round(volume * c["acres"])},
        "schedule": schedule,
        "prevention": [tx(x, lang) for x in PREVENT[kind]],
        "safety": [tx(x, lang) for x in SAFETY] if track == "integrated" else [],
        "escalate": _escalation(escalate, lang) if escalate else None,
        "reasoning": fired, "rules": [f["rule"] for f in fired],
        "disclaimer": tx(DISCLAIMER, lang), "ctx": c,
    }
    return plan


def _headline(kind: str, urgency: int, lang: str) -> str:
    if kind == "healthy":
        return tx(L("Leaf looks healthy: keep it that way", "पत्ती स्वस्थ दिखती है: ऐसे ही बनाए रखें", "இலை ஆரோக்கியமாக உள்ளது: இப்படியே பராமரிக்கவும்"), lang)
    h = {0: L("Low risk: monitor closely", "कम जोखिम: बारीकी से निगरानी करें", "குறைந்த அபாயம்: உன்னிப்பாகக் கண்காணிக்கவும்"),
         1: L("Start treatment this week", "इसी हफ़्ते उपचार शुरू करें", "இந்த வாரமே சிகிச்சையைத் தொடங்கவும்"),
         2: L("Spray soon, before it spreads further", "जल्दी छिड़काव करें, इससे पहले कि रोग और फैले", "நோய் மேலும் பரவும் முன் விரைவில் தெளிக்கவும்"),
         3: L("Treat within 48 hours to stop the spread", "फैलाव रोकने के लिए 48 घंटे में उपचार करें", "பரவலை நிறுத்த 48 மணி நேரத்தில் சிகிச்சை செய்யவும்"),
         4: L("Act today: this disease spreads fast", "आज ही कार्रवाई करें: यह रोग तेज़ी से फैलता है", "இன்றே நடவடிக்கை: இந்நோய் வேகமாகப் பரவும்")}
    return tx(h[urgency], lang)


def _escalation(level: str, lang: str) -> dict:
    msg = {
        "required": L("Please have a local expert confirm the disease before you buy or spray anything.",
                      "कुछ भी ख़रीदने या छिड़कने से पहले स्थानीय विशेषज्ञ से रोग की पुष्टि कराएँ।",
                      "எதையும் வாங்கும் அல்லது தெளிக்கும் முன் உள்ளூர் நிபுணரிடம் நோயை உறுதிப்படுத்தவும்."),
        "recommended": L("Have a Krishi Vigyan Kendra (KVK) officer confirm and tune this plan for your field.",
                         "कृषि विज्ञान केंद्र (KVK) के अधिकारी से इस योजना की पुष्टि और अपने खेत के अनुसार सुधार कराएँ।",
                         "கிருஷி விக்யான் கேந்திரா (KVK) அலுவலரிடம் இத்திட்டத்தை உறுதிப்படுத்தி உங்கள் வயலுக்கு ஏற்ப மாற்றவும்."),
    }[level]
    return {"level": level, "message": tx(msg, lang), "phone": KISAN_CALL_CENTRE, "kvk_url": KVK_FINDER,
            "phone_label": tx(L("Kisan Call Centre (toll-free)", "किसान कॉल सेंटर (निःशुल्क)", "கிசான் கால் சென்டர் (கட்டணமில்லா)"), lang),
            "kvk_label": tx(L("Find your nearest KVK", "नज़दीकी KVK खोजें", "அருகிலுள்ள KVK-ஐத் தேடவும்"), lang)}
