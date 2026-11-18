import argparse
import logging
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "detector" / "dataset"
MERGED_DIR = DATASET_DIR / "merged"
MERGED_FILE = MERGED_DIR / "news_dataset.csv"
HI_GU_FILE = DATASET_DIR / "hi_gudata.csv"  
REQUIRED_COLUMNS = ["title", "text", "label", "category", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def csv_files():
    for path in DATASET_DIR.glob("*/*.csv"):
        if path.parent.name != "merged":
            if path.name.lower() in {"fake.csv", "true.csv", "false.csv", "real.csv"}:
                continue
            yield path


def load_hi_gu_data():
    """Load hi_gudata.csv if it exists and is properly formatted."""
    if not HI_GU_FILE.exists():
        return None
    
    try:
        data = pd.read_csv(HI_GU_FILE, encoding='utf-8').fillna("")
        for col in REQUIRED_COLUMNS:
            if col not in data.columns:
                if col == "source":
                    data[col] = "hi_gu"
                elif col == "category":
                    data[col] = "general"
                else:
                    data[col] = ""
        
        if "label" in data.columns:
            def clean_label(val):
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    if val_lower in ['fake', 'false', '0']:
                        return 0
                    elif val_lower in ['real', 'true', '1']:
                        return 1
                try:
                    return int(float(val))
                except:
                    return 0
            data["label"] = data["label"].apply(clean_label)
        
        logger.info("Loaded hi_gudata.csv: %s rows", len(data))
        return data[REQUIRED_COLUMNS]
    except Exception as e:
        logger.warning("Could not load hi_gudata.csv: %s", str(e))
        return None


def load_category_file(path):
    data = pd.read_csv(path).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return data[REQUIRED_COLUMNS]


def merge(output_path=MERGED_FILE):
    frames = []
    
    
    for path in csv_files():
        logger.info("Loading %s", path)
        frames.append(load_category_file(path))
    
   
    hi_gu_data = load_hi_gu_data()
    if hi_gu_data is not None:
        logger.info("Adding hi_gudata.csv to merge")
        frames.append(hi_gu_data)
    
    if not frames:
        raise FileNotFoundError(
            "No category CSV files found. Add files under detector/dataset/<category>/ or hi_gudata.csv"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["title"] = merged["title"].astype(str).str.strip()
    merged["text"] = merged["text"].astype(str).str.strip()
    merged["category"] = merged["category"].astype(str).str.lower().str.strip()
    merged["source"] = merged["source"].astype(str).str.strip()
    merged = merged[merged["title"].str.len() > 0]
    merged = merged[merged["text"].str.len() > 40]
    merged = merged.drop_duplicates(subset=["title", "text", "category"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info("Saved %s rows to %s", len(merged), output_path)
    logger.info("Labels: %s", merged["label"].value_counts().to_dict())
    logger.info("Categories: %s", merged["category"].value_counts().to_dict())
    logger.info("Sources: %s", merged["source"].value_counts().to_dict())
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge category datasets into one training CSV.")
    parser.add_argument("--output", default=str(MERGED_FILE))
    args = parser.parse_args()
    merge(args.output)


if __name__ == "__main__":
    main()