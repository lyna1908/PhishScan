from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
import os

app = Flask(__name__)
CORS(app)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')
model = joblib.load(model_path)

# Feature lists
shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'short.io']
urgent_words = ['verify', 'suspended', 'immediately', 'click now',
                'confirm', 'limited', 'urgent', 'account', 'winner',
                'password', 'update', 'login', 'secure', 'bank']
free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com',
                  'outlook.com', 'aol.com', 'mail.com']


def extract_features(sender, subject, body):

    urls_list = re.findall(r'http[s]?://\S+', body)
    url_count = len(urls_list)
    urls = int(url_count > 0)

    has_ip_url = int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', body)))
    has_short_url = int(any(s in body for s in shorteners))

    urgent_keyword = sum(w in body.lower() for w in urgent_words)
    subject_urgent = int(any(w in subject.lower() for w in urgent_words))

    domain_match = re.search(r'@([\w\.-]+)', sender)
    domain = domain_match.group(1).lower() if domain_match else ''
    is_free_email = int(domain in free_providers)

    has_html = int(bool(re.search(r'<[a-zA-Z]+', body)))

    body_text = re.sub(r'<[^>]+>', '', body)
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
    df = df[model.feature_names_in_]

    return df

    app.run(debug=True, port=5000)