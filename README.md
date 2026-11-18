# Fake News Detection System

A Django web application that uses machine learning to classify submitted news as likely fake, likely real, or uncertain.

## Screenshot

<!-- Add a screenshot or GIF here, for example: ![Application screenshot](docs/screenshot.png) -->

## Features

- Submit a news headline and article text for analysis.
- Receive a Fake News, Real News, or Uncertain prediction with a confidence score.
- Register, sign in, and review personal prediction history in a dashboard.
- Retrain the text classifier and save the resulting model artifacts.

## Technology stack

- Python 3.11
- Django
- scikit-learn (word and character TF-IDF features, `SGDClassifier`)
- pandas, NumPy, SciPy, joblib
- SQLite

## How it works

1. The application combines the submitted title and article text and normalizes the text.
2. A fitted word and character TF-IDF vectorizer transforms the text into model features.
3. An `SGDClassifier` estimates whether the example resembles fake or real news in its training data.
4. The application applies confidence thresholds and a small set of text cues to select a label.
5. Low confidence is returned as **Uncertain** rather than forcing a classification.

The output is a model estimate based on training data and text patterns. It does not verify claims against authoritative sources.

## Model details

- Classifier: `SGDClassifier(loss="log_loss")`
- Features: word-level and character-level TF-IDF
- Training script: `detector/train_model.py`
- Dataset: place prepared data at `detector/dataset/merged/news_dataset.csv`, or use the supported Hindi/Gujarati or legacy dataset paths described in the training script.
- Dataset size: [fill in after your final training run]
- Accuracy: [fill in from `detector/model_report.json`]
- Weighted F1 score: [fill in from `detector/model_report.json`]

Metrics are meaningful only for the stated dataset and evaluation split. Review for data leakage, class imbalance, and performance on new sources before making performance claims.

## Installation and setup

### Windows PowerShell

```powershell
git clone https://github.com/<my-username>/fake-news-detection-system.git
cd fake-news-detection-system
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set a fresh `DJANGO_SECRET_KEY`. Keep `DJANGO_DEBUG=True` only for local development. The project reads these values directly from the process environment; for local use, load `.env` before starting Django. For example, in PowerShell:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
python manage.py migrate
python manage.py runserver
```

### Linux / macOS

```bash
git clone https://github.com/<my-username>/fake-news-detection-system.git
cd fake-news-detection-system
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set a fresh `DJANGO_SECRET_KEY` in `.env`, then export the variables before running Django. One simple option is `set -a; source .env; set +a` in Bash.

```bash
python manage.py migrate
python manage.py runserver
```

The settings use environment variables and do not parse `.env` automatically. Load those values in your shell as shown above, or install/configure a dotenv loader if you prefer automatic loading.

## Usage

1. Start the development server using the commands above.
2. Open `http://127.0.0.1:8000/` in a browser.
3. Create an account or open the news checker.
4. Enter a headline and article text, then review the prediction and confidence.
5. Sign in to review prediction history on your dashboard.

To retrain the model, make sure a supported dataset is available and run:

```bash
python detector/train_model.py
```

This writes `detector/fake_news_model.pkl`, `detector/vectorizer.pkl`, and `detector/model_report.json`.

## Project structure

```text
.
├── detector/
│   ├── dataset/                 # Training data (CSV files are ignored by Git)
│   ├── templates/detector/      # Django HTML templates
│   ├── static/                  # CSS, JavaScript, and images
│   ├── train_model.py           # Model training and evaluation
│   ├── utils.py                 # Text cleaning and prediction
│   ├── views.py                 # Web views and authentication flows
│   ├── fake_news_model.pkl      # Trained classifier artifact
│   └── vectorizer.pkl           # Fitted TF-IDF feature transformer
├── newsdetect/                  # Django project settings and URL config
├── manage.py
├── requirements.txt
└── README.md
```

## Disclaimer

Predictions are an aid to fact-checking, not a verdict. The system can be wrong, and a confidence score is not proof that a claim is true or false. Verify important claims with reliable sources.

## Author

Hari — Summer Research 2026

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
