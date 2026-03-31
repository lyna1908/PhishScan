import pandas as pd
import re
from bs4 import BeautifulSoup

df = pd.read_csv('../data/emails_parsed.csv')

shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'short.io']
urgent_words = ['verify', 'suspended', 'immediately', 'click now',
                'confirm', 'limited', 'urgent', 'account', 'winner',
                'password', 'update', 'login', 'secure', 'bank']
free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com',
                  'outlook.com', 'aol.com', 'mail.com']

def count_urls(text):
    if pd.isna(text): return 0
    return len(re.findall(r'http[s]?://\S+', str(text)))

def has_ip_url(text):
    if pd.isna(text): return 0
    return int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', str(text))))

def has_short_url(text):
    if pd.isna(text): return 0
    return int(any(s in str(text) for s in shorteners))

def count_urgent(text):
    if pd.isna(text): return 0
    return sum(w in str(text).lower() for w in urgent_words)

def get_sender_domain(sender):
    if pd.isna(sender): return ''
    match = re.search(r'@([\w\.-]+)', str(sender))
    return match.group(1).lower() if match else ''

def is_free_email(domain):
    return int(domain in free_providers)

def subject_urgent(subject):
    if pd.isna(subject): return 0
    return int(any(w in str(subject).lower() for w in urgent_words))

def body_length(text):
    if pd.isna(text): return 0
    return len(str(text))

def has_html(text):
    if pd.isna(text): return 0
    return int(bool(re.search(r'<[a-zA-Z]+', str(text))))

def html_text_ratio(text):
    if pd.isna(text): return 0
    html_len = len(re.findall(r'<[^>]+>', str(text)))
    text_len = len(str(text))
    if text_len == 0: return 0
    return round(html_len / text_len, 4)

df['url_count']       = df['body'].apply(count_urls)
df['has_ip_url']      = df['body'].apply(has_ip_url)
df['has_short_url']   = df['body'].apply(has_short_url)
df['urgent_keyword']  = df['body'].apply(count_urgent)
df['sender_domain']   = df['sender'].apply(get_sender_domain)
df['is_free_email']   = df['sender_domain'].apply(is_free_email)
df['subject_urgent']  = df['subject'].apply(subject_urgent)
df['body_length']     = df['body'].apply(body_length)
df['has_html']        = df['body'].apply(has_html)
df['html_text_ratio'] = df['body'].apply(html_text_ratio)

features = df[[
    'url_count', 'has_ip_url', 'has_short_url',
    'urgent_keyword', 'is_free_email', 'subject_urgent',
    'body_length', 'has_html', 'html_text_ratio', 'urls', 'label'
]]

print("Feature matrix shape:", features.shape)
print("\nSample:\n", features.head())
features.to_csv('../data/features.csv', index=False)
print("\n✅ features.csv saved!")