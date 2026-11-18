import logging
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


logger = logging.getLogger(__name__)

APP_DIR = Path(settings.BASE_DIR) / "detector"
MODEL_PATH = APP_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"

LABELS = {
    0: "Fake News",
    1: "Real News",
}

HINDI_STOP_WORDS = {
    "और", "एक", "है", "में", "को", "का", "से", "पर", "की", "के", "यह", "वह",
    "था", "थी", "थे", "हैं", "हूं", "हो", "मैं", "तुम", "आप", "ये", "वे",
    "इस", "उस", "कि", "तो", "भी", "ही", "तक", "साथ", "बिना",
}

GUJARATI_STOP_WORDS = {
    "અને", "એક", "છે", "માં", "ને", "નો", "થી", "પર", "ની", "ના", "આ", "તે",
    "હતો", "હતી", "હતા", "હું", "તમે", "કે", "તો", "પણ", "જ", "સુધી", "સાથે",
}

ALL_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(HINDI_STOP_WORDS).union(GUJARATI_STOP_WORDS)

FAKE_STYLE_CUES = {
    "high_importance": [
        "mainstream media is hiding", "mainstream media won't report",
        "share this", "forward this", "before it gets deleted",
        "forward to everyone", "share with everyone",
        "never actually acted", "body double did all", "body double",
        "सबको फॉरवर्ड", "सभी को फॉरवर्ड", "हटाने से पहले",
        "બધાને ફોરવર્ડ", "ડિલીટ થાય તે પહેલા", "શેર કરો",
    ],
    "medium_importance": [
        "breaking", "shocking", "urgent", "bombshell", "alert",
        "exposed", "revealed", "whistleblower", "cover-up", "secret",
        "gag order", "suppressed", "censored", "viral claim",
        "तोड़फोड़", "सनसनीखेज", "खुलासा", "चौंकाने वाला", "गुप्त",
        "તોડફોડ", "ખુલાસો", "સનસનાટીભર્યું", "ચોંકાવનારું", "ગુપ્ત", "છુપાવી",
    ],
    "low_importance": [
        "medical alert", "secretly", "leaked", "warning",
        "चेतावनी", "अलर्ट", "તાત્કાલિક", "ચેતવણી", "એલર્ટ",
    ],
}

REAL_STYLE_CUES = [
    "सरकार", "केंद्र सरकार", "राज्य सरकार", "मंत्रालय", "कृषि मंत्री", "प्रधानमंत्री",
    "योजना", "आर्थिक सहायता", "लाभ मिलेगा", "लागू होगी", "उद्घाटन", "लोकार्पण",
    "સરકાર", "કેન્દ્ર સરકાર", "રાજ્ય સરકાર", "મંત્રાલય", "મંત્રી", "પ્રધાનમંત્રી",
    "યોજના", "આર્થિક સહાય", "લાભ મળશે", "લાગુ થશે", "લોકાર્પણ", "શરૂ થયું",
    "data center", "digital hub", "gift city", "launched", "inaugurated",
    "official", "minister", "ministry", "scheme", "will benefit",
]


class ModelNotReadyError(RuntimeError):
    pass


def _fake_style_score(text):
    lowered = (text or "").lower()
    score = 0

    for cue in FAKE_STYLE_CUES["high_importance"]:
        if cue.lower() in lowered:
            score += 2

    for cue in FAKE_STYLE_CUES["medium_importance"]:
        if cue.lower() in lowered:
            score += 1

    for cue in FAKE_STYLE_CUES["low_importance"]:
        if cue.lower() in lowered:
            score += 0.5

    return score


def _real_style_score(text):
    lowered = (text or "").lower()
    return sum(1 for cue in REAL_STYLE_CUES if cue.lower() in lowered)


def clean_text(text):
    """Apply preprocessing that supports English, Hindi, and Gujarati."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = "".join(
        ch if (ch.isspace() or unicodedata.category(ch).startswith(("L", "N"))) else " "
        for ch in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered_words = [word for word in words if len(word) > 1 and word not in ALL_STOP_WORDS]
    return " ".join(filtered_words)


@lru_cache(maxsize=1)
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise ModelNotReadyError("ML model is not trained yet. Run: python detector/train_model.py")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Fake news ML artifacts loaded from disk")
    return model, vectorizer


def _sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _get_probabilities(model, vector_input):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(vector_input)[0]

    if hasattr(model, "decision_function"):
        score = model.decision_function(vector_input)
        score = float(np.ravel(score)[0])
        real_probability = _sigmoid(score)
        return np.array([1 - real_probability, real_probability])

    prediction = int(model.predict(vector_input)[0])
    probabilities = np.zeros(len(LABELS), dtype="float32")
    probabilities[prediction] = 1.0
    return probabilities


def predict_news(news_title="", news_content=""):
    """Predict whether a news item is real, fake, or uncertain."""
    try:
        combined_text = f"{news_title or ''} {news_content or ''}".strip()

        if len(combined_text) < 20:
            return "Uncertain", 0.0

        cleaned_text = clean_text(combined_text)
        if not cleaned_text or len(cleaned_text) < 10:
            return "Uncertain", 0.0

        model, vectorizer = load_artifacts()
        vector_input = vectorizer.transform([cleaned_text])

        prediction = int(model.predict(vector_input)[0])
        probabilities = _get_probabilities(model, vector_input)

        fake_prob = round(float(probabilities[0]) * 100, 2)
        real_prob = round(float(probabilities[1]) * 100, 2)
        confidence = max(fake_prob, real_prob)

        fake_style_score = _fake_style_score(combined_text)
        real_style_score = _real_style_score(combined_text)

        logger.info(
            "Prediction: model=%s, fake_prob=%s%%, real_prob=%s%%, fake_style=%s, real_style=%s",
            prediction,
            fake_prob,
            real_prob,
            fake_style_score,
            real_style_score,
        )

        if fake_style_score >= 2.5 and real_prob < 90:
            return "Fake News", max(confidence, 70.0)

        if fake_style_score >= 1 and confidence < 68:
            return "Fake News", max(confidence, 55.0)

        if real_style_score >= 3 and fake_style_score == 0 and fake_prob < 68:
            return "Real News", max(real_prob, 68.0)

        if real_style_score >= 2 and fake_style_score == 0 and prediction == 0 and confidence < 58:
            return "Real News", max(real_prob, 62.0)

        if prediction == 0 and confidence >= 58.0:
            return "Fake News", confidence

        if prediction == 1 and confidence >= 58.0 and fake_style_score < 1:
            return "Real News", confidence

        if real_prob >= fake_prob + 15 and fake_style_score < 1:
            return "Real News", real_prob

        if fake_prob >= real_prob + 15 and real_style_score == 0:
            return "Fake News", fake_prob

        if confidence < 58:
            return "Uncertain", confidence

        result = LABELS.get(prediction, "Prediction Failed")
        return result, confidence

    except ModelNotReadyError as exc:
        logger.warning("%s", exc)
        return "Model Not Trained", 0.0
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return "Prediction Failed", 0.0


def predict_news_payload(news_title="", news_content=""):
    label, confidence = predict_news(news_title, news_content)
    return {
        "prediction": label,
        "confidence": confidence,
        "is_success": label not in {"Model Not Trained", "Prediction Failed", "Uncertain"},
    }
