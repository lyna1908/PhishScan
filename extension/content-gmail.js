function pickText(selectors) {
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent && el.textContent.trim()) {
      return el.textContent.trim();
    }
  }
  return '';
}

function extractGmailEmail() {
  const subject = pickText(['h2.hP', 'h2[data-legacy-thread-id]']);

  let sender = '';
  const senderEmailEl = document.querySelector('span[email]');
  if (senderEmailEl && senderEmailEl.getAttribute('email')) {
    sender = senderEmailEl.getAttribute('email').trim();
  }
  if (!sender) {
    sender = pickText(['span[email]', '.gD']);
  }

  const body = pickText(['div.a3s.aiL', 'div.a3s']);

  return { sender, subject, body };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== 'GET_EMAIL_CONTENT') {
    return;
  }

  const data = extractGmailEmail();
  const ok = Boolean((data.body || '').trim());

  if (!ok) {
    sendResponse({ ok: false, error: 'No opened email content detected. Open an email first.' });
    return;
  }

  sendResponse({ ok: true, ...data });
});
