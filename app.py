from flask import Flask, render_template, request, jsonify, send_file
import joblib, re, email, urllib.parse
import io, os, zipfile
from datetime import datetime

app = Flask(__name__, template_folder='frontend/templates')

# ML model kept as secondary confidence signal
try:
    _ml_model    = joblib.load('model/best_model.pkl')
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# BeautifulSoup (optional — for link-text mismatch & form detection)
try:
    from bs4 import BeautifulSoup
    BS4 = True
except ImportError:
    BS4 = False

# python-whois (optional — for domain age)
try:
    import whois as _whois
    WHOIS = True
except ImportError:
    WHOIS = False

# ── Constants ──────────────────────────────────────────────────────────────────
SHORTENERS   = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','short.io','is.gd','buff.ly']
URGENT_WORDS = ['verify','suspended','immediately','click now','confirm',
                'limited','urgent','winner','password','bank','credit card','ssn']
FREE_EMAIL   = ['gmail.com','yahoo.com','hotmail.com','outlook.com','aol.com','mail.com']
BRAND_DOMAINS = {
    'paypal':'paypal.com', 'apple':'apple.com', 'amazon':'amazon.com',
    'microsoft':'microsoft.com', 'google':'google.com', 'netflix':'netflix.com',
    'dhl':'dhl.com', 'fedex':'fedex.com', 'instagram':'instagram.com',
    'facebook':'facebook.com', 'twitter':'twitter.com', 'linkedin':'linkedin.com',
}

# Max score per feature
# DATA-DRIVEN (Gradient Boosting ensemble, 46,527 samples):
#   body_length=0.625, url_count=0.117, html_text_ratio=0.115, is_free_email=0.061
#   Low-importance (<0.05): has_ip_url, has_short_url, urgent_keyword, subject_urgent, has_html
# Runtime features (not in dataset) retain literature-based weights.
MAX_SCORES = {
    # -- Dataset-validated features (data-driven weights) --
    'url_count':          6,   # importance=0.117 (was 12)
    'ip_based_url':       4,   # importance=0.016 (was 15) — low in dataset
    'url_shortener':      4,   # importance=0.004 (was 12) — low in dataset
    'urgent_keywords':    4,   # importance=0.023 (was 15) — low in dataset
    'domain_trust':       5,   # importance=0.061 (was 10)
    'subject_urgent':     4,   # importance=0.018 (was  7) — low in dataset
    'html_content':       4,   # importance=0.002 (was  4) — unchanged
    'html_ratio':         6,   # importance=0.115 (was  8)
    'body_length':       15,   # importance=0.625 (was  8) — DOMINANT FEATURE
    # -- Runtime features (literature-based, not in dataset) --
    'brand_impersonation':14,
    'link_text_mismatch': 15,
    'form_presence':      13,
    'subdomain_depth':     8,
    'domain_age':         12,
    'ssl_validity':        6,
}
MAX_TOTAL   = sum(MAX_SCORES.values())   # 120
# Heuristic-only fallback thresholds (used when ML is unavailable)
RISK_LOW_PCT    = 30.0
RISK_MEDIUM_PCT = 59.5

# From results/optimized_weights.json
ML_OPTIMAL_THRESHOLD = 36.0

# ── Helpers ────────────────────────────────────────────────────────────────────
def _domain(sender):
    m = re.search(r'@([\w\.-]+)', sender)
    return m.group(1).lower() if m else ''

def _urls(body):
    return re.findall(r'https?://[^\s<>"\']+', body)

def _url_protocol_flags(urls):
    has_http = any(u.lower().startswith('http://') for u in urls)
    has_https = any(u.lower().startswith('https://') for u in urls)
    return has_http, has_https

# ── 15 Feature scorers ─────────────────────────────────────────────────────────
def s_url_count(urls):
    n = len(urls)
    if n == 0: return 0
    if n == 1: return 1
    if n <= 2: return 3
    if n <= 4: return 5
    return 6   # max=6

def s_ip_url(body):
    return 4 if re.search(r'https?://\d+\.\d+\.\d+\.\d+', body) else 0   # max=4

def s_shortener(body):
    return 4 if any(s in body for s in SHORTENERS) else 0   # max=4

def s_urgent_kw(body, subject):
    text = (body + ' ' + subject).lower()
    hits = sum(1 for w in URGENT_WORDS if w in text)
    if hits == 0: return 0
    if hits == 1: return 1
    if hits == 2: return 2
    if hits == 3: return 3
    return 4   # max=4

def s_domain_trust(domain):
    if not domain:
        return 5
    score = 0
    base = (domain.split('.')[0] if domain else '').lower()

    if domain in FREE_EMAIL:
        score += 2
    for brand, official in BRAND_DOMAINS.items():
        if brand in domain and domain != official:
            score += 3
            break
    if re.match(r'^[\d\-]+$', base):
        score += 2
    elif len(base) >= 10:
        vowels = sum(c in 'aeiou' for c in base)
        if len(base) > 0 and (vowels / len(base)) < 0.25:
            score += 1
    return min(score, MAX_SCORES['domain_trust'])

def s_subject_urgent(subject):
    return 4 if any(w in subject.lower() for w in URGENT_WORDS) else 0   # max=4

def s_html_content(body):
    return 4 if re.search(r'<[a-zA-Z]', body) else 0   # max=4 (unchanged)

def s_html_ratio(body):
    if not body: return 0
    tags = re.findall(r'<[^>]+>', body)
    ratio = len(tags) / max(len(body), 1)
    if ratio >= 0.18: return 6
    if ratio >= 0.10: return 4
    if ratio >= 0.05: return 2
    return 0   # max=6

def s_body_length(body):
    # Body length is useful, but should not dominate by itself.
    n = len(body or '')
    if n == 0:     return 12
    if n < 40:     return 10
    if n < 120:    return 6
    if n <= 1200:  return 0
    if n <= 3000:  return 3
    return 5

def s_brand_impersonation(body, subject, domain):
    text = (body + ' ' + subject).lower()
    urls = _urls(body)
    url_hosts = []
    for u in urls:
        try:
            host = urllib.parse.urlparse(u).hostname
            url_hosts.append(host or '')
        except Exception:
            pass

    for brand, official in BRAND_DOMAINS.items():
        if brand not in text:
            continue

        sender_is_official = (domain == official)
        urls_are_official = all((official in h) for h in url_hosts) if url_hosts else False
        if sender_is_official and urls_are_official:
            continue
        return 14
    return 0

def s_link_mismatch(body):
    if not BS4:
        return 0
    try:
        for a in BeautifulSoup(body, 'html.parser').find_all('a', href=True):
            href = a.get('href', '').strip()
            text = a.get_text(' ', strip=True).lower()
            hd = re.search(r'https?://([^/\s]+)', href)
            td = re.search(r'https?://([^/\s]+)', text)

            if hd and td and hd.group(1).lower() != td.group(1).lower():
                return 15

            if hd:
                href_host = hd.group(1).lower()
                for brand, official in BRAND_DOMAINS.items():
                    if brand in text and official not in href_host:
                        return 15
    except Exception:
        pass
    return 0

def s_form_presence(body):
    if BS4:
        try:
            soup = BeautifulSoup(body, 'html.parser')
            forms = soup.find_all('form')
            if not forms:
                return 0

            for f in forms:
                action = (f.get('action', '') or '').lower()
                inputs = ' '.join((i.get('type', '') or '').lower() for i in f.find_all('input'))
                looks_credential = any(k in inputs for k in ['password', 'email', 'tel'])
                if re.search(r'https?://', action) and looks_credential:
                    return 13
            return 4
        except Exception:
            pass
    return 4 if re.search(r'<form', body or '', re.IGNORECASE) else 0

def s_subdomain_depth(urls):
    max_d = 0
    for url in urls:
        try:
            host  = urllib.parse.urlparse(url).hostname or ''
            depth = max(0, len(host.split('.')) - 2)
            max_d = max(max_d, depth)
        except Exception:
            pass
    if max_d >= 4: return 8
    if max_d == 3: return 4
    return 0

def s_domain_age(domain):
    # Domain age is only meaningful for real registered domains.
    # Free-email providers are not sender-owned domains in phishing context.
    if domain in FREE_EMAIL:
        return 0
    if not WHOIS or not domain: return 0
    try:
        info    = _whois.whois(domain)
        created = info.creation_date
        if isinstance(created, list): created = created[0]
        if created:
            age = (datetime.now() - created).days
            if age < 30:  return 12
            if age < 180: return 6
    except Exception:
        pass
    return 0

def s_ssl_validity(urls):
    """
    Returns SSL validity as a risk score and protocol state.
    Rules:
      - Any HTTP URL => SSL validity risk = max (insecure link)
      - HTTPS-only URLs => SSL validity risk = 0
      - No URL => SSL validity risk = 0
    """
    has_http, has_https = _url_protocol_flags(urls)
    if has_http:
        return 6, 'INSECURE_HTTP'
    if has_https:
        return 0, 'HTTPS_ONLY'
    return 0, 'NO_URL'

def _apply_consistency_rules(scores, domain, urls, ssl_state):
    """
    Enforce constraints between mutually related features.
    """
    notes = []

    # Rule 1: HTTP vs SSL validity must be mutually consistent.
    if ssl_state == 'INSECURE_HTTP' and scores['ssl_validity'] == 0:
        scores['ssl_validity'] = MAX_SCORES['ssl_validity']
        notes.append('Adjusted ssl_validity: HTTP URL detected, forced insecure SSL score.')
    if ssl_state == 'HTTPS_ONLY' and scores['ssl_validity'] > 0:
        scores['ssl_validity'] = 0
        notes.append('Adjusted ssl_validity: HTTPS-only URLs cannot be marked insecure.')

    # Rule 2: Domain age is not applicable for free-email providers.
    if domain in FREE_EMAIL and scores['domain_age'] > 0:
        scores['domain_age'] = 0
        notes.append('Adjusted domain_age: free-email provider domains are excluded from age risk.')

    # Rule 3: No URLs -> subdomain depth and SSL checks must be zero.
    if not urls:
        if scores['subdomain_depth'] > 0:
            scores['subdomain_depth'] = 0
            notes.append('Adjusted subdomain_depth: no URLs present.')
        if scores['ssl_validity'] > 0:
            scores['ssl_validity'] = 0
            notes.append('Adjusted ssl_validity: no URLs present.')

    return scores, notes

def _cap_scores(scores):
    """
    Safety clamp: no feature score may exceed its declared max or go negative.
    """
    capped = {}
    notes = []
    for k, v in scores.items():
        max_v = MAX_SCORES.get(k, v)
        new_v = max(0, min(v, max_v))
        capped[k] = new_v
        if new_v != v:
            notes.append(f'Adjusted {k}: capped from {v} to {new_v}.')
    return capped, notes

def _combine_verdict(heuristic_pct, ml_proba):
    """
    Final decision strategy based on user thresholds:
      - If 0 heuristic features triggered -> Always LEGITIMATE (Safe Override)
      - > 59.5%: PHISHING DETECTED
      - 30% to 59.5%: SUSPICIOUS
      - < 30%: LEGITIMATE
    """
    # Safe Override: If no heuristic indicators, force Legitimate
    if heuristic_pct == 0:
        return 'LEGITIMATE', 0, 0.0

    if ml_proba is None:
        risk = heuristic_pct
    else:
        risk = round((0.60 * ml_proba) + (0.40 * heuristic_pct), 1)

    if risk >= RISK_MEDIUM_PCT:
        return 'PHISHING DETECTED', 1, risk
    if risk >= RISK_LOW_PCT:
        return 'SUSPICIOUS', 0, risk
    return 'LEGITIMATE', 0, risk

# ── Core analysis ──────────────────────────────────────────────────────────────
def analyze_email(sender, subject, body):
    sender  = str(sender) if sender is not None else ''
    subject = str(subject) if subject is not None else ''
    body    = str(body) if body is not None else ''

    domain = _domain(sender)
    urls   = _urls(body)
    ssl_score, ssl_state = s_ssl_validity(urls)

    scores = {
        'url_count':          s_url_count(urls),
        'ip_based_url':       s_ip_url(body),
        'url_shortener':      s_shortener(body),
        'urgent_keywords':    s_urgent_kw(body, subject),
        'domain_trust':       s_domain_trust(domain),
        'subject_urgent':     s_subject_urgent(subject),
        'html_content':       s_html_content(body),
        'html_ratio':         s_html_ratio(body),
        'body_length':        s_body_length(body),
        'brand_impersonation':s_brand_impersonation(body, subject, domain),
        'link_text_mismatch': s_link_mismatch(body),
        'form_presence':      s_form_presence(body),
        'subdomain_depth':    s_subdomain_depth(urls),
        'domain_age':         s_domain_age(domain),
        'ssl_validity':       ssl_score,
    }
    scores, consistency_notes = _apply_consistency_rules(scores, domain, urls, ssl_state)
    scores, cap_notes = _cap_scores(scores)

    risk_score = sum(scores.values())
    risk_pct   = round(risk_score / MAX_TOTAL * 100, 1)

    # ML model — secondary signal (shown in UI only)
    ml_proba = None
    if ML_AVAILABLE:
        try:
            raw = _ml_model.predict_proba([[
                len(urls),
                1 if s_ip_url(body) else 0,
                1 if s_shortener(body) else 0,
                sum(w in body.lower() for w in URGENT_WORDS),
                1 if domain in FREE_EMAIL else 0,
                1 if s_subject_urgent(subject) else 0,
                len(body),
                1 if s_html_content(body) else 0,
                round(len(re.findall(r'<[^>]+>', body)) / len(body), 4) if body else 0,
                1 if urls else 0,
            ]])[0][1]
            ml_proba = round(raw * 100, 1)
        except Exception:
            ml_proba = None

    verdict, pred, combined_risk = _combine_verdict(risk_pct, ml_proba)

    triggers = _build_triggers(scores, len(urls), ssl_state)

    return {
        'risk_score':   risk_score,
        'max_score':    MAX_TOTAL,
        'risk_pct':     risk_pct,
        'combined_risk': combined_risk,
        'verdict':      verdict,
        'prediction':   pred,
        'ml_probability': ml_proba,
        'ssl_state':     ssl_state,
        'consistency_notes': consistency_notes + cap_notes,
        'triggers':     triggers,
        'scores':       scores,
        'max_scores':   MAX_SCORES,
    }

def _build_triggers(scores, url_count, ssl_state):
    t = []
    if scores['url_count']          > 0:  t.append(f"[!] {url_count} URL(s) detected in body")
    if scores['ip_based_url']       > 0:  t.append("[!] IP-based URL detected")
    if scores['url_shortener']      > 0:  t.append("[!] URL shortener detected")
    if scores['urgent_keywords']    > 0:  t.append(f"[!] Urgent keywords found (score: {scores['urgent_keywords']})")
    if scores['domain_trust']       >= 5: t.append(f"[!] Suspicious sender domain (trust score: {scores['domain_trust']})")
    if scores['subject_urgent']     > 0:  t.append("[!] Urgency language in subject line")
    if scores['html_content']       > 0:  t.append("[!] HTML content embedded in body")
    if scores['html_ratio']         > 0:  t.append("[!] High HTML-to-text ratio")
    if scores['body_length']        >= 6: t.append("[!] Abnormal email body length pattern")
    if scores['brand_impersonation']> 0:  t.append("[!] Brand impersonation detected")
    if scores['link_text_mismatch'] > 0:  t.append("[!] Link text mismatches actual URL")
    if scores['form_presence']      > 0:  t.append("[!] HTML form collecting data detected")
    if scores['subdomain_depth']    > 0:  t.append("[!] Deep subdomain structure in URL")
    if scores['domain_age']         > 0:  t.append(f"[!] Newly registered domain (score: {scores['domain_age']})")
    if ssl_state == 'INSECURE_HTTP' and scores['ssl_validity'] > 0:
        t.append("[!] Insecure HTTP link detected (no SSL)")
    elif ssl_state == 'HTTPS_ONLY':
        t.append("[ok] URLs use HTTPS")
    if not t:
        t.append("[ok] No threat indicators detected")
    return t

# ── EML parser ─────────────────────────────────────────────────────────────────
def parse_eml(raw_bytes):
    msg     = email.message_from_bytes(raw_bytes)
    sender  = msg.get('From', '')
    subject = msg.get('Subject', '')
    body    = ''

    def _decode_part(part):
        payload = part.get_payload(decode=True)
        if payload is None:
            raw = part.get_payload()
            return raw if isinstance(raw, str) else ''
        charset = part.get_content_charset() or 'utf-8'
        try:
            return payload.decode(charset, errors='ignore')
        except Exception:
            return payload.decode('utf-8', errors='ignore')

    if msg.is_multipart():
        text_parts = []
        html_parts = []
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            ctype = part.get_content_type()
            if ctype == 'text/plain':
                text_parts.append(_decode_part(part))
            elif ctype == 'text/html':
                html_parts.append(_decode_part(part))
        body = '\n'.join(text_parts).strip() or '\n'.join(html_parts).strip()
    else:
        body = _decode_part(msg)
    return sender, subject, body

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data    = request.json or {}
    result  = analyze_email(
        data.get('sender', ''),
        data.get('subject', ''),
        data.get('body', '')
    )
    return jsonify(result)

@app.route('/analyze-eml', methods=['POST'])
def analyze_eml():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    uploaded_file = request.files['file']
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({'error': 'No file selected'}), 400
    if not uploaded_file.filename.lower().endswith('.eml'):
        return jsonify({'error': 'Invalid file type. Please upload a .eml file.'}), 400

    raw = uploaded_file.read()
    if not raw:
        return jsonify({'error': 'Uploaded file is empty'}), 400

    try:
        sender, subject, body = parse_eml(raw)
    except Exception:
        return jsonify({'error': 'Unable to parse .eml file'}), 400

    if not any([sender.strip(), subject.strip(), body.strip()]):
        return jsonify({'error': 'Could not extract email content from file'}), 400

    result = analyze_email(sender, subject, body)
    result['sender']  = sender
    result['subject'] = subject
    result['body']    = body
    return jsonify(result)

@app.route('/download-extension', methods=['GET'])
def download_extension():
    extension_dir = os.path.join(app.root_path, 'extension')
    if not os.path.isdir(extension_dir):
        return jsonify({'error': 'Extension folder not found'}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(extension_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, extension_dir)
                zf.write(file_path, arcname)
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='phishscan-extension.zip'
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
