from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
import email

app = Flask(__name__, template_folder='../frontend/templates')
model = joblib.load('model/best_model.pkl')

shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'short.io']
urgent_words = ['verify', 'suspended', 'immediately', 'click now',
                'confirm', 'limited', 'urgent', 'account', 'winner',
                'password', 'update', 'login', 'secure', 'bank']
free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com',
                  'outlook.com', 'aol.com', 'mail.com']

def extract_features(sender, subject, body):
    url_count       = len(re.findall(r'http[s]?://\S+', body))
    has_ip_url      = int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', body)))
    has_short_url   = int(any(s in body for s in shorteners))
    urgent_keyword  = sum(w in body.lower() for w in urgent_words)
    domain_match    = re.search(r'@([\w\.-]+)', sender)
    domain          = domain_match.group(1).lower() if domain_match else ''
    is_free_email   = int(domain in free_providers)
    subject_urgent  = int(any(w in subject.lower() for w in urgent_words))
    body_length     = len(body)
    has_html        = int(bool(re.search(r'<[a-zA-Z]+', body)))
    html_len        = len(re.findall(r'<[^>]+>', body))
    html_text_ratio = round(html_len / len(body), 4) if len(body) > 0 else 0
    urls            = int(url_count > 0)

    return pd.DataFrame([[
        url_count, has_ip_url, has_short_url, urgent_keyword,
        is_free_email, subject_urgent, body_length,
        has_html, html_text_ratio, urls
    ]], columns=[
        'url_count', 'has_ip_url', 'has_short_url', 'urgent_keyword',
        'is_free_email', 'subject_urgent', 'body_length',
        'has_html', 'html_text_ratio', 'urls'
    ])

def parse_eml(raw_bytes):
    msg     = email.message_from_bytes(raw_bytes)
    sender  = msg.get('From', '')
    subject = msg.get('Subject', '')
    body    = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return sender, subject, body

def build_triggers(f):
    triggers = []
    if f['url_count'] > 0:
        triggers.append(f"[!] {int(f['url_count'])} URL(s) detected in body")
    if f['has_ip_url']:
        triggers.append("[!] IP-based URL detected — suspicious")
    if f['has_short_url']:
        triggers.append("[!] URL shortener detected")
    if f['urgent_keyword'] > 0:
        triggers.append(f"[!] {int(f['urgent_keyword'])} urgent keyword(s) found")
    if f['is_free_email']:
        triggers.append("[!] Sent from free email provider")
    if f['subject_urgent']:
        triggers.append("[!] Urgent keywords in subject line")
    if f['has_html']:
        triggers.append("[!] HTML content embedded in body")
    if f['html_text_ratio'] > 0.1:
        triggers.append(f"[!] High HTML ratio — {f['html_text_ratio']}")
    if not triggers:
        triggers.append("[ok] No major threats detected")
    return triggers

def build_response(features, proba, pred):
    f = features.iloc[0]
    return jsonify({
        'probability': round(proba * 100, 1),
        'prediction':  int(pred),
        'triggers':    build_triggers(f),
        'features': {
            'url_count':       int(f['url_count']),
            'has_ip_url':      int(f['has_ip_url']),
            'has_short_url':   int(f['has_short_url']),
            'urgent_keyword':  int(f['urgent_keyword']),
            'is_free_email':   int(f['is_free_email']),
            'subject_urgent':  int(f['subject_urgent']),
            'has_html':        int(f['has_html']),
            'html_text_ratio': float(f['html_text_ratio'])
        }
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data     = request.json
    sender   = data.get('sender', '')
    subject  = data.get('subject', '')
    body     = data.get('body', '')
    features = extract_features(sender, subject, body)
    proba    = model.predict_proba(features)[0][1]
    pred     = model.predict(features)[0]
    return build_response(features, proba, pred)

@app.route('/analyze-eml', methods=['POST'])
def analyze_eml():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    raw              = request.files['file'].read()
    sender, subject, body = parse_eml(raw)
    features         = extract_features(sender, subject, body)
    proba            = model.predict_proba(features)[0][1]
    pred             = model.predict(features)[0]
    response         = build_response(features, proba, pred)
    response_data    = response.get_json()
    response_data['sender']  = sender
    response_data['subject'] = subject
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)