import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "detector"
DATASET_DIR = APP_DIR / "dataset"
MERGED_DATASET = DATASET_DIR / "merged" / "news_dataset.csv"
LEGACY_FAKE_DATASET = DATASET_DIR / "Fake.csv"
LEGACY_TRUE_DATASET = DATASET_DIR / "True.csv"
HI_GU_DATASET = DATASET_DIR / "hi_gudata.csv"

MODEL_PATH = APP_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"
REPORT_PATH = APP_DIR / "model_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_WORD_FEATURES = 150000
MAX_CHAR_FEATURES = 180000
MIN_DF = 2
MAX_DF = 0.95

LABEL_FAKE = 0
LABEL_REAL = 1
REQUIRED_COLUMNS = ["title", "text", "label", "category", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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


def repair_mojibake(value):
    """Repair Hindi/Gujarati text that was saved as UTF-8 read through Latin-1."""
    if not isinstance(value, str):
        return value
    if "à" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def clean_text(text):
    """Normalize text before TF-IDF so training and prediction use one vocabulary."""
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
    return " ".join(word for word in text.split() if len(word) > 1 and word not in ALL_STOP_WORDS)


def normalize_label(value):
    """Accept human labels and numeric labels in dataset CSVs."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"real", "true", "1", "real news"}:
            return LABEL_REAL
        if label in {"fake", "false", "0", "fake news"}:
            return LABEL_FAKE
    try:
        return int(value)
    except (ValueError, TypeError):
        return LABEL_FAKE


def load_hi_gu_dataset():
    """Load Hindi+Gujarati examples and repair encoding when needed."""
    if not HI_GU_DATASET.exists():
        logger.info("hi_gudata.csv not found at %s", HI_GU_DATASET)
        return None

    data = pd.read_csv(HI_GU_DATASET, encoding="utf-8").fillna("")
    for column in ("title", "text", "category", "source", "language"):
        if column in data.columns:
            data[column] = data[column].apply(repair_mojibake)

    if "title" not in data.columns or "text" not in data.columns:
        logger.warning("hi_gudata.csv missing title or text columns")
        return None

    if "label" not in data.columns:
        logger.warning("hi_gudata.csv missing label column")
        return None

    if "category" not in data.columns:
        data["category"] = "hindi_gujarati"

    if "source" not in data.columns:
        if "language" in data.columns:
            data["source"] = data["language"].apply(lambda value: value if value in ["hi", "gu"] else "hi")
        else:
            data["source"] = "hi"

    data["label"] = data["label"].apply(normalize_label)
    data = data[data["label"].isin([LABEL_FAKE, LABEL_REAL])]

    logger.info(
        "Loaded hi_gudata.csv: %s rows (Fake: %s, Real: %s)",
        len(data),
        (data["label"] == LABEL_FAKE).sum(),
        (data["label"] == LABEL_REAL).sum(),
    )
    return data[REQUIRED_COLUMNS]


def load_merged_dataset():
    """Load the main multi-domain dataset."""
    if not MERGED_DATASET.exists():
        logger.info("Merged dataset not found at %s", MERGED_DATASET)
        return None

    data = pd.read_csv(MERGED_DATASET, encoding="utf-8").fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        logger.warning("Merged dataset missing columns: %s", missing)
        return None

    data["label"] = data["label"].apply(normalize_label)
    data = data[data["label"].isin([LABEL_FAKE, LABEL_REAL])]

    logger.info(
        "Loaded merged dataset: %s rows (Fake: %s, Real: %s)",
        len(data),
        (data["label"] == LABEL_FAKE).sum(),
        (data["label"] == LABEL_REAL).sum(),
    )
    return data[REQUIRED_COLUMNS]


def load_legacy_dataset():
    """Fallback for Kaggle-style Fake.csv and True.csv files."""
    if not LEGACY_FAKE_DATASET.exists() or not LEGACY_TRUE_DATASET.exists():
        return None

    fake_df = pd.read_csv(LEGACY_FAKE_DATASET, encoding="utf-8").fillna("")
    true_df = pd.read_csv(LEGACY_TRUE_DATASET, encoding="utf-8").fillna("")

    title_col = None
    text_col = None
    for column in fake_df.columns:
        lowered = column.lower()
        if "title" in lowered or "headline" in lowered:
            title_col = column
        if "text" in lowered or "content" in lowered or "article" in lowered:
            text_col = column

    if not text_col:
        logger.warning("Legacy dataset missing text column")
        return None

    fake_data = pd.DataFrame(
        {
            "title": fake_df[title_col] if title_col else "",
            "text": fake_df[text_col],
            "label": LABEL_FAKE,
            "category": "legacy",
            "source": "legacy_fake_csv",
        }
    )
    true_data = pd.DataFrame(
        {
            "title": true_df[title_col] if title_col else "",
            "text": true_df[text_col],
            "label": LABEL_REAL,
            "category": "legacy",
            "source": "legacy_true_csv",
        }
    )

    combined = pd.concat([fake_data, true_data], ignore_index=True)
    logger.info("Loaded legacy dataset: %s rows (Fake: %s, Real: %s)", len(combined), len(fake_data), len(true_data))
    return combined


def load_dataset():
    """Load all available datasets into the training format."""
    all_data = []

    merged_data = load_merged_dataset()
    if merged_data is not None:
        all_data.append(merged_data)

    hi_gu_data = load_hi_gu_dataset()
    if hi_gu_data is not None:
        all_data.append(hi_gu_data)

    if not all_data:
        legacy_data = load_legacy_dataset()
        if legacy_data is not None:
            all_data.append(legacy_data)

    if not all_data:
        raise FileNotFoundError(
            "No training dataset found. Add detector/dataset/merged/news_dataset.csv, "
            "detector/dataset/hi_gudata.csv, or detector/dataset/Fake.csv and True.csv."
        )

    data = pd.concat(all_data, ignore_index=True)
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)

    data = data[data["content"].str.len() > 20]
    data = data.drop_duplicates(subset=["content"])

    valid_categories = data.groupby("category")["label"].nunique().loc[lambda counts: counts == 2].index
    skipped_categories = sorted(set(data["category"]) - set(valid_categories))
    if skipped_categories:
        logger.warning("Skipping one-sided categories: %s", skipped_categories)
        data = data[data["category"].isin(valid_categories)]

    logger.info("Final dataset rows: %s", len(data))
    logger.info("Fake rows: %s", (data["label"] == LABEL_FAKE).sum())
    logger.info("Real rows: %s", (data["label"] == LABEL_REAL).sum())
    logger.info("Categories: %s", data["category"].value_counts().to_dict())
    logger.info("Sources: %s", data["source"].value_counts().to_dict())
    return data


def build_model():
    """SGD with log-loss works well on sparse multilingual TF-IDF features."""
    return SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        early_stopping=False,
        n_jobs=-1,
    )


def evaluate(model, x_test_vec, y_test):
    predictions = model.predict(x_test_vec)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Fake News", "Real News"],
            zero_division=0,
        ),
    }


def main():
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60 + "\n")

    data = load_dataset()
    if len(data) < 100:
        logger.warning("Only %s rows available. Training may not be effective.", len(data))

    if data["label"].nunique() != 2:
        raise ValueError("Training requires both fake and real samples.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    word_vectorizer = TfidfVectorizer(
        max_features=MAX_WORD_FEATURES,
        ngram_range=(1, 2),
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer = FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ],
        n_jobs=-1,
    )

    logger.info("Fitting TF-IDF vectorizer")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = build_model()
    logger.info("Training SGDClassifier(log_loss)")
    model.fit(x_train_vec, y_train)

    metrics = evaluate(model, x_test_vec, y_test)
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)

    report = {
        "model": "SGDClassifier(log_loss) with word+char TF-IDF",
        "label_mapping": {"0": "Fake News", "1": "Real News"},
        "dataset_size": int(len(data)),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "test_size": TEST_SIZE,
        "max_word_features": MAX_WORD_FEATURES,
        "max_char_features": MAX_CHAR_FEATURES,
        "category_distribution": data["category"].value_counts().to_dict(),
        "source_distribution": data["source"].value_counts().to_dict(),
        "metrics": metrics,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Vectorizer saved: {VECTORIZER_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training failed")
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)
