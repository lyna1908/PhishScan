# PhishScan — Phishing Email Detector

A machine-learning-powered web application that detects phishing emails using handcrafted features extracted from email metadata and body content.

---

## Project Structure

```
phishing/
├── app.py                   # Flask web server (main entry point)
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment config (Render / gunicorn)
├── .gitignore
├── README.md
│
├── extension/               # Browser extension files
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
│
├── backend/                 # ML training scripts
│   ├── prepare_data.py
│   ├── features.py
│   ├── train.py
│   └── feature_importance.py
│   └── model/
│       └── best_model.pkl   # ML Model
│
├── frontend/
│   └── templates/
│       └── index.html       # Web UI
│
├── results/                 # Evaluation plots
└── data/                    # Datasets (ignored)
```

---

## Running the App Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Flask server
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Reproducing the Model (Optional)

Run the pipeline scripts **from inside the `backend/` directory**:

```bash
cd backend

# 1. Clean & merge raw datasets → data/emails.csv
python prepare_data.py

# 2. Extract features → data/features.csv
python features.py

# 3. Train models → model/best_model.pkl + results/*.png
python train.py

# 4. (Optional) Plot feature importance → results/feature_importance.png
python feature_importance.py
```

---

## Features Used for Detection

| Feature | Description |
|---|---|
| `url_count` | Number of URLs in the body |
| `has_ip_url` | URL with raw IP address |
| `has_short_url` | URL shortener detected |
| `urgent_keyword` | Count of urgency words |
| `is_free_email` | Sender uses Gmail / Yahoo / etc. |
| `subject_urgent` | Urgency words in the subject |
| `body_length` | Total character count |
| `has_html` | HTML tags present in body |
| `html_text_ratio` | Ratio of HTML markup to text |
| `urls` | Boolean — at least one URL |

---

## Models Compared

- Logistic Regression
- **Random Forest** ← selected (best F1)
- SVM

---

## Datasets

- **CEAS 2008** — spam/ham email corpus
- **Nazario Phishing** — curated phishing collection
- **SpamAssassin** — public spam corpus

> Data files are excluded from this repository due to size. Place them in the `data/` directory before running the pipeline.
