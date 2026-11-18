import argparse
import logging
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "detector" / "dataset"
CATEGORIES = [
    "politics",
    "sports",
    "technology",
    "health",
    "finance",
    "entertainment",
]

RAW_FAKE_NAMES = {"fake.csv", "false.csv"}
RAW_TRUE_NAMES = {"true.csv", "real.csv"}
OUTPUT_COLUMNS = ["title", "text", "label", "category", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pick_text_columns(data):
    """
    common fake news
    Preferred columns are title and text. If text is missing, common content columns are used as fallback.
    """
    lower_columns = {column.lower(): column for column in data.columns}
    title_column = lower_columns.get("title") or lower_columns.get("headline")
    text_column = (
        lower_columns.get("text")
        or lower_columns.get("content")
        or lower_columns.get("article")
        or lower_columns.get("body")
        or lower_columns.get("claim")
    )

    if not title_column and not text_column:
        raise ValueError(f"Could not find title/text columns. Found: {list(data.columns)}")

    title = data[title_column].fillna("").astype(str) if title_column else ""
    text = data[text_column].fillna("").astype(str) if text_column else title
    return title, text


def convert_file(input_path, category, label):
    data = pd.read_csv(input_path).fillna("")
    title, text = pick_text_columns(data)

    output = pd.DataFrame(
        {
            "title": title,
            "text": text,
            "label": label,
            "category": category,
            "source": input_path.stem,
        },
        columns=OUTPUT_COLUMNS,
    )

    output["title"] = output["title"].astype(str).str.strip()
    output["text"] = output["text"].astype(str).str.strip()
    output = output[output["title"].str.len() > 0]
    output = output[output["text"].str.len() > 40]
    output = output.drop_duplicates(subset=["title", "text"])
    return output


def prepare_category(category):
    category_dir = DATASET_DIR / category
    if not category_dir.exists():
        logger.warning("Missing category folder: %s", category_dir)
        return

    frames = []
    for path in category_dir.glob("*.csv"):
        name = path.name.lower()
        if name.startswith("prepared_") or name.startswith("newsapi_"):
            continue

        if name in RAW_FAKE_NAMES:
            logger.info("Converting fake file: %s", path)
            frames.append(convert_file(path, category, label=0))
        elif name in RAW_TRUE_NAMES:
            logger.info("Converting true file: %s", path)
            frames.append(convert_file(path, category, label=1))

    if not frames:
        logger.info("No raw Fake.csv/True.csv files found for %s", category)
        return

    output = pd.concat(frames, ignore_index=True)
    output = output.drop_duplicates(subset=["title", "text"])
    output_path = category_dir / f"prepared_{category}.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %s rows to %s", len(output), output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert category Fake.csv/True.csv files to title,text,label,category,source format."
    )
    parser.add_argument("--category", choices=CATEGORIES + ["all"], default="all")
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else [args.category]
    for category in categories:
        prepare_category(category)


if __name__ == "__main__":
    main()
