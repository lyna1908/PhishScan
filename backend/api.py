from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
import os

# ── Flask App ────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load Model ───────────────────────────────────
model_path = os.path.join(
    os.path.dirname(__file__),
    'model',
    'best_model.pkl'
)
print("Loading model from:", model_path)

model = joblib.load(model_path)

print("Model loaded successfully!")

# ── Feature Lists ────────────────────────────────
shorteners = [
    'bit.ly',
    'tinyurl',
    'goo.gl',
    't.co',
    'ow.ly',
    'short.io'
]

urgent_words = [
    'verify',
    'suspended',
    'immediately',
    'click now',
    'confirm',
    'limited',
    'urgent',
    'account',
    'winner',
    'password',
    'update',
    'login',
    'secure',
    'bank'
]

free_providers = [
    'gmail.com',
    'yahoo.com',
    'hotmail.com',
    'outlook.com',
    'aol.com',
    'mail.com'
]

# ── Feature Extraction ───────────────────────────
def extract_features(sender, subject, body):

    urls_list = re.findall(r'http[s]?://\S+', body)

    url_count = len(urls_list)

    urls = int(url_count > 0)

    has_ip_url = int(
        bool(
            re.search(
                r'http[s]?://\d+\.\d+\.\d+\.\d+',
                body
            )
        )
    )

    has_short_url = int(
        any(s in body for s in shorteners)
    )

    urgent_keyword = sum(
        w in body.lower() for w in urgent_words
    )

    subject_urgent = int(
        any(w in subject.lower() for w in urgent_words)
    )

    domain_match = re.search(
        r'@([\w\.-]+)',
        sender
    )

    domain = (
        domain_match.group(1).lower()
        if domain_match
        else ''
    )

    is_free_email = int(
        domain in free_providers
    )

    has_html = int(
        bool(re.search(r'<[a-zA-Z]+', body))
    )

    body_text = re.sub(
        r'<[^>]+>',
        '',
        body
    )

    body_length = len(body_text)

    if body_length > 0:
        html_text_ratio = len(body) / body_length
    else:
        html_text_ratio = 0

    data = {
        'url_count': url_count,
        'has_ip_url': has_ip_url,
        'has_short_url': has_short_url,
        'urgent_keyword': urgent_keyword,
        'is_free_email': is_free_email,
        'subject_urgent': subject_urgent,
        'body_length': body_length,
        'has_html': has_html,
        'urls': urls,
        'html_text_ratio': html_text_ratio
    }

    df = pd.DataFrame([data])

    # IMPORTANT:
    # Match exact training feature order
    df = df[model.feature_names_in_]

    return df

# ── Home Route ───────────────────────────────────
@app.route('/')
def home():

    return "PhishScan API is running!"

# ── Analyze Route ────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze_email():

    try:

        data = request.json

        sender = data.get('sender', '')
        subject = data.get('subject', '')
        body = data.get('body', '')

        features = extract_features(
            sender,
            subject,
            body
        )

        proba = model.predict_proba(features)[0][1]

        pred = int(
            model.predict(features)[0]
        )

        explanations = []

        f = features.iloc[0]

        if f['url_count'] > 0:
            explanations.append(
                f"{int(f['url_count'])} URL(s) detected"
            )

        if f['has_ip_url']:
            explanations.append(
                "IP-based URL detected"
            )

        if f['has_short_url']:
            explanations.append(
                "Shortened URL detected"
            )

        if f['urgent_keyword'] > 0:
            explanations.append(
                "Urgent keywords found"
            )

        if f['is_free_email']:
            explanations.append(
                "Free email provider used"
            )

        if f['subject_urgent']:
            explanations.append(
                "Urgent subject line"
            )

        if f['has_html']:
            explanations.append(
                "HTML content detected"
            )

        response = {
            'prediction': (
                'phishing'
                if pred == 1
                else 'legitimate'
            ),
            'probability': round(
                proba * 100,
                2
            ),
            'details': explanations
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500

# ── Run Server ───────────────────────────────────
if __name__ == '__main__':

    print("Starting Flask API...")
    print("http://127.0.0.1:5000")

port = int(os.environ.get("PORT", 5000))

app.run(
    host="0.0.0.0",
    port=port
)