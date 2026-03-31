import streamlit as st
import joblib
import pandas as pd
import re

# ── Load model ────────────────────────────────────────────────────
model = joblib.load('best_model.pkl')

# ── Feature extraction (same as features.py) ──────────────────────
shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'short.io']
urgent_words = ['verify', 'suspended', 'immediately', 'click now',
                'confirm', 'limited', 'urgent', 'account', 'winner',
                'password', 'update', 'login', 'secure', 'bank']
free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com',
                  'outlook.com', 'aol.com', 'mail.com']

def extract_features(sender, subject, body):
    url_count      = len(re.findall(r'http[s]?://\S+', body))
    has_ip_url     = int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', body)))
    has_short_url  = int(any(s in body for s in shorteners))
    urgent_keyword = sum(w in body.lower() for w in urgent_words)
    domain_match   = re.search(r'@([\w\.-]+)', sender)
    domain         = domain_match.group(1).lower() if domain_match else ''
    is_free_email  = int(domain in free_providers)
    subject_urgent = int(any(w in subject.lower() for w in urgent_words))
    body_length    = len(body)
    has_html       = int(bool(re.search(r'<[a-zA-Z]+', body)))
    urls           = int(url_count > 0)

    return pd.DataFrame([[
        url_count, has_ip_url, has_short_url, urgent_keyword,
        is_free_email, subject_urgent, body_length, has_html, urls
    ]], columns=[
        'url_count', 'has_ip_url', 'has_short_url', 'urgent_keyword',
        'is_free_email', 'subject_urgent', 'body_length', 'has_html', 'urls'
    ])

# ── UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")

st.title("🛡️ Phishing Email Detector")
st.markdown("Paste the email details below to check if it's phishing or legitimate.")

sender  = st.text_input("📧 Sender Email", placeholder="example@domain.com")
subject = st.text_input("📝 Subject", placeholder="Your account has been suspended...")
body    = st.text_area("📨 Email Body", height=200, placeholder="Paste email body here...")

if st.button("🔍 Analyze"):
    if not body:
        st.warning("Please paste the email body at minimum.")
    else:
        features = extract_features(
            sender or '',
            subject or '',
            body
        )
        proba = model.predict_proba(features)[0][1]
        pred  = model.predict(features)[0]

        st.divider()

        # ── Result ────────────────────────────────────────────────
        if pred == 1:
            st.error(f"🚨 PHISHING DETECTED — {proba*100:.1f}% probability")
        else:
            st.success(f"✅ LEGITIMATE — {proba*100:.1f}% phishing probability")

        # ── Feature breakdown ─────────────────────────────────────
        st.subheader("🔎 Why?")
        f = features.iloc[0]
        if f['url_count'] > 0:
            st.write(f"• 🔗 {int(f['url_count'])} URL(s) found in body")
        if f['has_ip_url']:
            st.write("• ⚠️ IP-based URL detected")
        if f['has_short_url']:
            st.write("• ⚠️ URL shortener detected")
        if f['urgent_keyword'] > 0:
            st.write(f"• 🚨 {int(f['urgent_keyword'])} urgent keyword(s) found")
        if f['is_free_email']:
            st.write("• 📮 Sent from a free email provider")
        if f['subject_urgent']:
            st.write("• ⚠️ Urgent keywords in subject line")
        if f['has_html']:
            st.write("• 🖥️ HTML content detected in body")
        if f['body_length'] < 100:
            st.write("• 📏 Very short email body")
