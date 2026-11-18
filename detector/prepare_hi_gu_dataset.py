import logging
import csv
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "detector"
DATASET_DIR = APP_DIR / "dataset"
HI_GU_FILE = DATASET_DIR / "hi_gudata.csv"
REQUIRED_COLUMNS = ["title", "text", "label", "category", "source"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def read_csv_safe(filepath, encoding='utf-8'):
    """
    Safely read CSV with error handling for inconsistent columns.
    Tries multiple strategies to parse problematic CSV files.
    """
    
    try:
        data = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
        logger.info("Successfully read CSV with pandas (strategy 1)")
        return data
    except Exception as e:
        logger.warning("Strategy 1 failed: %s", str(e))
    
    try:
        data = pd.read_csv(
            filepath, 
            encoding=encoding, 
            on_bad_lines='skip',
            quoting=csv.QUOTE_ALL,
            escapechar='\\'
        )
        logger.info("Successfully read CSV with quoting (strategy 2)")
        return data
    except Exception as e:
        logger.warning("Strategy 2 failed: %s", str(e))
    
    
    try:
        rows = []
        with open(filepath, 'r', encoding=encoding) as f:
            
            header_line = f.readline().strip()
            headers = header_line.split(',')
            logger.info("Found headers: %s", headers)
            
            
            line_num = 2
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                
                try:
                    import io
                    csv_reader = csv.reader(io.StringIO(line), quotechar='"', doublequote=True)
                    row = next(csv_reader)
                    
                    if len(row) < len(headers):
                        row.extend([''] * (len(headers) - len(row)))
                    elif len(row) > len(headers):
                        extra = row[len(headers)-1:]
                        row = row[:len(headers)-1] + [' '.join(extra)]
                    
                    rows.append(row)
                except Exception:
                    parts = line.split(',', len(headers)-1)
                    if len(parts) < len(headers):
                        parts.extend([''] * (len(headers) - len(parts)))
                    rows.append(parts)
                
                line_num += 1
                if line_num > 100000:  # Increased limit for larger files
                    logger.warning("Reached row limit, stopping early")
                    break
        
        data = pd.DataFrame(rows, columns=headers[:len(headers)])
        logger.info("Successfully read CSV with manual parsing (strategy 3) - %s rows", len(data))
        return data
        
    except Exception as e:
        logger.error("All parsing strategies failed: %s", str(e))
        return None


def validate_hi_gu_file():
    """Check if hi_gudata.csv has required columns and format."""
    if not HI_GU_FILE.exists():
        logger.error("hi_gudata.csv not found at %s", HI_GU_FILE)
        logger.info("Please place your hi_gudata.csv file in: %s", DATASET_DIR)
        return False
    
    logger.info("Validating %s...", HI_GU_FILE)
    
    # Try to read the file safely
    data = read_csv_safe(HI_GU_FILE)
    
    if data is None:
        logger.error("Could not read hi_gudata.csv")
        return False
    
    data = data.fillna("")
    
    # Check if we have the required columns (case insensitive)
    data_columns_lower = {col.lower(): col for col in data.columns}
    
    column_mapping = {}
    for required in REQUIRED_COLUMNS:
        if required.lower() in data_columns_lower:
            column_mapping[required] = data_columns_lower[required.lower()]
    
    missing = [col for col in REQUIRED_COLUMNS if col not in column_mapping]
    
    if missing:
        logger.error("hi_gudata.csv missing columns: %s", missing)
        logger.info("Available columns: %s", list(data.columns))
        logger.info("Required columns: %s", REQUIRED_COLUMNS)
        logger.info("Run --fix to attempt repair")
        return False
    
    # Rename columns to standard names
    for required, actual in column_mapping.items():
        if required != actual:
            data = data.rename(columns={actual: required})
    
    # Clean and validate label column
    try:
        # Convert labels to numeric
        def clean_label(val):
            if pd.isna(val):
                return 0
            val_str = str(val).lower().strip()
            if val_str in ['fake', 'false', '0', 'fake news']:
                return 0
            elif val_str in ['real', 'true', '1', 'real news']:
                return 1
            try:
                return int(float(val_str))
            except:
                logger.warning("Could not convert label: %s", val_str[:50])
                return 0
        
        data["label"] = data["label"].apply(clean_label)
        
    except Exception as e:
        logger.error("Error processing labels: %s", str(e))
        return False
    
    # Validate data quality
    fake_count = (data["label"] == 0).sum()
    real_count = (data["label"] == 1).sum()
    total_rows = len(data)
    
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info("File: %s", HI_GU_FILE)
    logger.info("Total rows: %s", total_rows)
    logger.info("Fake news (0): %s", fake_count)
    logger.info("Real news (1): %s", real_count)
    
    if fake_count == 0 or real_count == 0:
        logger.warning("Dataset only has one label type! Training will fail.")
        logger.warning("Add samples of the missing label type.")
    else:
        logger.info("Both label types present")
    
    # Check category distribution
    if "category" in data.columns:
        logger.info("\nCategory distribution:")
        for cat, count in data["category"].value_counts().items():
            logger.info("  %s: %s", cat, count)
    
   
    if "source" in data.columns:
        logger.info("\nSource distribution:")
        for src, count in data["source"].value_counts().items():
            logger.info("  %s: %s", src, count)
    
    # Check for empty fields
    empty_titles = (data["title"].str.len() == 0).sum()
    empty_texts = (data["text"].str.len() == 0).sum()
    
    if empty_titles > 0:
        logger.warning("%s rows have empty titles", empty_titles)
    if empty_texts > 0:
        logger.warning("%s rows have empty text", empty_texts)
    
  
    logger.info("\nSample rows (first 3):")
    for i, row in data.head(3).iterrows():
        title_preview = str(row.get('title', ''))[:60]
        text_preview = str(row.get('text', ''))[:60]
        logger.info("  %d. Label=%s | %s...", i+1, row.get('label', '?'), title_preview)
        logger.info("     Text: %s...", text_preview)
    
    duplicates = data.duplicated(subset=["title", "text"]).sum()
    if duplicates > 0:
        logger.warning(" %s duplicate rows found", duplicates)
    
    logger.info("=" * 60)
    
    # Provide recommendations
    if total_rows < 100:
        logger.warning(" Dataset is small (%s rows). Model may not perform well.", total_rows)
        logger.info("Recommendation: Add more data (at least 500 rows for decent performance)")
    
    if fake_count < 50 or real_count < 50:
        logger.warning(" Imbalanced dataset. Add more samples of the minority class.")
    
    logger.info("\n Validation complete!")
    logger.info("Run 'python detector/train_model.py' to train the model.")
    
    return True


def fix_hi_gu_format():
    """Fix common formatting issues in hi_gudata.csv without generating new data."""
    if not HI_GU_FILE.exists():
        logger.error("File not found at %s", HI_GU_FILE)
        logger.info("Please place your hi_gudata.csv file in: %s", DATASET_DIR)
        return False
    
    logger.info("Fixing %s...", HI_GU_FILE)
    
    # Backup original file
    backup_path = HI_GU_FILE.with_suffix('.csv.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(HI_GU_FILE, backup_path)
        logger.info("Backed up original file to %s", backup_path)
    
    
    data = read_csv_safe(HI_GU_FILE)
    
    if data is None:
        logger.error("Could not read the file. Cannot fix automatically.")
        logger.info("Please check your CSV file manually for formatting issues.")
        return False
    
    original_count = len(data)
    logger.info("Read %s rows", original_count)
    
    # Clean text fields - handle missing columns gracefully
    if "title" in data.columns:
        data["title"] = data["title"].astype(str).str.strip()
    else:
        logger.warning("No 'title' column found, creating empty column")
        data["title"] = ""
    
    if "text" in data.columns:
        data["text"] = data["text"].astype(str).str.strip()
    else:
        logger.error("No 'text' column found! Cannot fix.")
        return False
    
    # Convert labels to 0/1
    def convert_label(label):
        try:
            if pd.isna(label):
                return 0
            label_str = str(label).lower().strip()
            if label_str in ['fake', 'false', '0', 'fake news']:
                return 0
            elif label_str in ['real', 'true', '1', 'real news']:
                return 1
            return int(float(label_str)) if label_str else 0
        except:
            return 0
    
    if "label" in data.columns:
        data["label"] = data["label"].apply(convert_label)
    else:
        logger.error("No 'label' column found! Cannot fix.")
        return False
    
    # Ensure source is set
    if "source" not in data.columns:
        if "language" in data.columns:
            data["source"] = data["language"].apply(lambda x: str(x) if str(x) in ["hi", "gu"] else "hi")
            logger.info("Created 'source' column from 'language' column")
        else:
            data["source"] = "hi"
            logger.info("Created 'source' column with default value 'hi'")
    
    # Ensure category is set
    if "category" not in data.columns:
        data["category"] = "general"
        logger.info("Created 'category' column with default value 'general'")
    
    # Remove duplicate rows
    before_dedup = len(data)
    data = data.drop_duplicates(subset=["title", "text"])
    after_dedup = len(data)
    if before_dedup > after_dedup:
        logger.info("Removed %s duplicate rows", before_dedup - after_dedup)
    
    # Remove rows with empty text
    before_filter = len(data)
    data = data[data["text"].str.len() > 20]
    after_filter = len(data)
    if before_filter > after_filter:
        logger.info("Removed %s rows with text length < 20 characters", before_filter - after_filter)
    
    # Ensure we have all required columns in correct order
    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            if col in ["source", "category"]:
                data[col] = "general" if col == "category" else "hi"
            else:
                data[col] = ""
    
    data[REQUIRED_COLUMNS].to_csv(HI_GU_FILE, index=False, encoding='utf-8', quoting=1)
    
    logger.info("=" * 60)
    logger.info("FIX COMPLETE")
    logger.info("=" * 60)
    logger.info("Original rows: %s", original_count)
    logger.info("After cleaning: %s rows", len(data))
    logger.info("Fake news (0): %s", (data["label"] == 0).sum())
    logger.info("Real news (1): %s", (data["label"] == 1).sum())
    logger.info("=" * 60)
    logger.info(" Fixed file saved to: %s", HI_GU_FILE)
    logger.info("Original backup saved to: %s", backup_path)
    
    return True


def show_stats():
    """Display detailed statistics about the dataset."""
    if not HI_GU_FILE.exists():
        logger.error("hi_gudata.csv not found")
        return
    
    data = read_csv_safe(HI_GU_FILE)
    if data is None:
        return
    
    data = data.fillna("")
    
    # Ensure label column exists
    if "label" in data.columns:
        data["label"] = data["label"].apply(lambda x: 0 if str(x).lower() in ['fake', 'false', '0'] else 1)
    else:
        logger.error("No label column found")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info("Total rows: %s", len(data))
    logger.info("Fake news: %s (%.1f%%)", (data["label"] == 0).sum(), 
                (data["label"] == 0).sum() / len(data) * 100)
    logger.info("Real news: %s (%.1f%%)", (data["label"] == 1).sum(),
                (data["label"] == 1).sum() / len(data) * 100)
    
    if "source" in data.columns:
        logger.info("\nLanguage distribution:")
        for src, count in data["source"].value_counts().items():
            logger.info("  %s: %s (%.1f%%)", src, count, count / len(data) * 100)
    
    if "category" in data.columns:
        logger.info("\nCategory distribution:")
        for cat, count in data["category"].value_counts().head(10).items():
            logger.info("  %s: %s", cat, count)
    

    data["text_len"] = data["text"].astype(str).str.len()
    logger.info("\nText length statistics:")
    logger.info("  Minimum: %s chars", data["text_len"].min())
    logger.info("  Maximum: %s chars", data["text_len"].max())
    logger.info("  Average: %.0f chars", data["text_len"].mean())
    
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and prepare hi_gudata.csv for training")
    parser.add_argument("--fix", action="store_true", help="Fix formatting issues in existing file")
    parser.add_argument("--validate", action="store_true", help="Validate existing file")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    args = parser.parse_args()
    
    if args.fix:
        success = fix_hi_gu_format()
        sys.exit(0 if success else 1)
    elif args.stats:
        show_stats()
    else:
        # Default to validate
        success = validate_hi_gu_file()
        sys.exit(0 if success else 1)