const byId = (id) => document.getElementById(id);

const openBtn = byId('openAnalysis');
const analyzeCurrentBtn = byId('analyzeCurrentEmail');
const statusEl = byId('status');
const resultBox = byId('resultBox');
const resultTitle = byId('resultTitle');
const resultSub = byId('resultSub');

let latestEmail = { sender: '', subject: '', body: '' };
let savedBaseUrl = 'http://localhost:5000';

chrome.storage.sync.get(['phishscanBaseUrl'], (result) => {
  savedBaseUrl = (result.phishscanBaseUrl || 'http://localhost:5000').trim();
});

function getBaseUrl() {
  return (savedBaseUrl || 'http://localhost:5000').trim().replace(/\/$/, '');
}

function openAnalysisPage(sender, subject, body) {
  const params = new URLSearchParams({ sender: sender || '', subject: subject || '', body: body || '', auto: '1' });
  const url = `${getBaseUrl()}/?${params.toString()}`;
  chrome.tabs.create({ url });
}

function showVerdict(data) {
  const verdict = (data.verdict || '').toUpperCase();
  let cls = 'warn';
  let title = `Suspicious (${data.risk_score}/${data.max_score})`;

  if (verdict.includes('PHISHING')) {
    cls = 'phish';
    title = `Phishing Detected (${data.risk_score}/${data.max_score})`;
  } else if (verdict.includes('LEGITIMATE')) {
    cls = 'safe';
    title = `Looks Safe (${data.risk_score}/${data.max_score})`;
  }

  resultBox.style.display = 'block';
  resultTitle.className = `result-title ${cls}`;
  resultTitle.textContent = title;
  resultSub.textContent = `Risk: ${data.risk_pct}% | Verdict: ${data.verdict}`;
}

async function analyzeOnServer(sender, subject, body) {
  const endpoint = `${getBaseUrl()}/analyze`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sender, subject, body })
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || 'Analysis request failed.');
  }
  return data;
}

openBtn.addEventListener('click', () => {
  openAnalysisPage(latestEmail.sender, latestEmail.subject, latestEmail.body);
});

analyzeCurrentBtn.addEventListener('click', () => {
  statusEl.textContent = 'Reading opened Gmail email...';
  resultBox.style.display = 'none';
  openBtn.style.display = 'none';

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs && tabs[0];
    if (!activeTab || !activeTab.id) {
      statusEl.textContent = 'No active tab found.';
      return;
    }

    if (!activeTab.url || !activeTab.url.startsWith('https://mail.google.com/')) {
      statusEl.textContent = 'Open a Gmail email first, then click again.';
      return;
    }

    chrome.tabs.sendMessage(activeTab.id, { type: 'GET_EMAIL_CONTENT' }, async (response) => {
      if (chrome.runtime.lastError) {
        statusEl.textContent = 'Cannot read this page yet. Refresh Gmail tab and retry.';
        return;
      }

      if (!response || !response.ok) {
        statusEl.textContent = (response && response.error) || 'Email content not detected.';
        return;
      }

      const sender = response.sender || '';
      const subject = response.subject || '';
      const body = response.body || '';

      latestEmail = { sender, subject, body };

      statusEl.textContent = 'Email detected. Running quick verification...';

      try {
        const result = await analyzeOnServer(sender, subject, body);
        showVerdict(result);
        openBtn.style.display = 'block';
        statusEl.textContent = 'Verification completed.';
      } catch (err) {
        statusEl.textContent = err.message || 'Could not verify email.';
      }
    });
  });
});
