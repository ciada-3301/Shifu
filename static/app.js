/* ═══════════════════════════════════════════
   Shifu — app.js
   Handles: conversations, SSE chat, file uploads,
            markdown rendering, code copy, auto-resize
   ═══════════════════════════════════════════ */

// ── Markdown + Highlight.js setup ──────────

marked.setOptions({ breaks: true, gfm: true });

const renderer = new marked.Renderer();

// Override code blocks to inject copy button + language label
renderer.code = (code, language) => {
  const lang = language || 'text';
  let highlighted;
  try {
    highlighted = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  } catch {
    highlighted = hljs.highlightAuto(code).value;
  }
  const id = `cb-${Math.random().toString(36).slice(2, 8)}`;
  return `
<div class="code-block-wrap">
  <div class="code-header">
    <span class="code-lang">${lang}</span>
    <button class="copy-btn" data-target="${id}" onclick="copyCode(this)">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>
      Copy
    </button>
  </div>
  <pre><code id="${id}" class="hljs language-${lang}">${highlighted}</code></pre>
</div>`;
};

marked.use({ renderer });


// ── State ───────────────────────────────────

const state = {
  conversations: [],          // [{id, title, messages:[]}]
  activeConvId: null,
  pendingFiles: [],           // [{file_id, original_name, path, file_type, previewUrl?}]
  isLoading: false,
};


// ── DOM refs ────────────────────────────────

const $ = id => document.getElementById(id);
const messagesContainer = $('messagesContainer');
const welcomeScreen     = $('welcomeScreen');
const messageInput      = $('messageInput');
const sendBtn           = $('sendBtn');
const attachBtn         = $('attachBtn');
const fileInput         = $('fileInput');
const filePreviewBar    = $('filePreviewBar');
const convoList         = $('convoList');
const chatTitle         = $('chatTitle');
const newChatBtn        = $('newChatBtn');
const clearBtn          = $('clearBtn');
const sidebarToggle     = $('sidebarToggle');
const mobileMenuBtn     = $('mobileMenuBtn');
const sidebar           = $('sidebar');


// ── Conversation management ─────────────────

function createConversation() {
  const conv = {
    id: crypto.randomUUID(),
    title: 'New conversation',
    messages: [],
  };
  state.conversations.unshift(conv);
  state.activeConvId = conv.id;
  return conv;
}

function getActiveConv() {
  return state.conversations.find(c => c.id === state.activeConvId);
}

function setConvTitle(id, title) {
  const conv = state.conversations.find(c => c.id === id);
  if (conv && conv.title === 'New conversation') {
    conv.title = title.slice(0, 46) + (title.length > 46 ? '…' : '');
  }
  renderConvoList();
}

function switchConversation(id) {
  state.activeConvId = id;
  renderConvoList();
  renderMessages();
  chatTitle.textContent = getActiveConv()?.title || 'Conversation';
  closeMobileSidebar();
}


// ── Sidebar rendering ───────────────────────

function renderConvoList() {
  convoList.innerHTML = '';
  if (!state.conversations.length) return;

  state.conversations.forEach(conv => {
    const item = document.createElement('div');
    item.className = `convo-item ${conv.id === state.activeConvId ? 'active' : ''}`;
    item.innerHTML = `
      <span class="convo-icon">💬</span>
      <span class="convo-label">${escHtml(conv.title)}</span>
    `;
    item.addEventListener('click', () => switchConversation(conv.id));
    convoList.appendChild(item);
  });
}


// ── Message rendering ───────────────────────

function renderMessages() {
  const conv = getActiveConv();
  messagesContainer.innerHTML = '';

  if (!conv || !conv.messages.length) {
    messagesContainer.appendChild(buildWelcomeScreen());
    return;
  }

  conv.messages.forEach(msg => appendMessageDOM(msg));
  scrollToBottom();
}

function buildWelcomeScreen() {
  const div = document.createElement('div');
  div.id = 'welcomeScreen';
  div.className = 'welcome-screen';
  div.innerHTML = `
    <div class="welcome-icon">🥷</div>
    <h1 class="welcome-title">Shifu is ready</h1>
    <p class="welcome-sub">Your personal AI assistant, powered by CrewAI agents.<br/>Ask anything — search the web, read files, browse directories.</p>
    <div class="welcome-chips">
      <button class="chip" data-prompt="Search the web for the latest AI news today">🔍 Latest AI news</button>
      <button class="chip" data-prompt="List the files in my current directory">📁 Browse directory</button>
      <button class="chip" data-prompt="Write a Python function to read a PDF and extract text">🐍 Python snippet</button>
      <button class="chip" data-prompt="Explain the difference between RAG and fine-tuning">🧠 RAG vs fine-tuning</button>
    </div>
  `;
  div.querySelectorAll('.chip').forEach(chip =>
    chip.addEventListener('click', () => handleChip(chip.dataset.prompt))
  );
  return div;
}

function appendMessageDOM(msg, animate = false) {
  const row = document.createElement('div');
  row.className = `message-row ${msg.role}`;
  if (animate) row.style.animationDuration = '0.3s';

  const avatarChar = msg.role === 'user' ? '👤' : '🥷';
  const avatarClass = msg.role === 'user' ? 'user-avatar' : 'ai-avatar';

  let filesHtml = '';
  if (msg.files && msg.files.length) {
    filesHtml = `<div class="attached-files">` +
      msg.files.map(f => `
        <div class="file-chip">
          <span class="file-chip-icon">${fileIcon(f.file_type)}</span>
          <span>${escHtml(f.original_name)}</span>
        </div>
      `).join('') +
    `</div>`;
  }

  const contentHtml = msg.role === 'assistant'
    ? `<div class="md-content">${marked.parse(msg.content || '')}</div>`
    : `<p>${escHtml(msg.content || '')}</p>`;

  row.innerHTML = `
    <div class="avatar ${avatarClass}">${avatarChar}</div>
    <div class="bubble">
      ${contentHtml}
      ${filesHtml}
    </div>
  `;

  messagesContainer.appendChild(row);
  if (animate) scrollToBottom();
  return row;
}

function appendThinkingBubble() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = 'thinkingRow';
  row.innerHTML = `
    <div class="avatar ai-avatar">🥷</div>
    <div class="bubble thinking-bubble">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
  return row;
}

function removeThinkingBubble() {
  document.getElementById('thinkingRow')?.remove();
}


// ── Send message ────────────────────────────

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text && !state.pendingFiles.length) return;
  if (state.isLoading) return;

  // Ensure active conversation
  if (!state.activeConvId) createConversation();

  const conv = getActiveConv();

  // Remove welcome screen
  const ws = messagesContainer.querySelector('.welcome-screen');
  if (ws) ws.remove();

  // Build user message
  const userMsg = {
    role: 'user',
    content: text,
    files: state.pendingFiles.map(f => ({
      file_id: f.file_id,
      original_name: f.original_name,
      path: f.path,
      file_type: f.file_type
    }))
  };
  conv.messages.push(userMsg);
  appendMessageDOM(userMsg, true);

  if (conv.title === 'New conversation' && text) setConvTitle(conv.id, text);
  chatTitle.textContent = conv.title;
  renderConvoList();

  // Clear input
  const filesForRequest = [...userMsg.files];
  messageInput.value = '';
  autoResizeTextarea();
  clearPendingFiles();
  setSendState(true);

  // Show thinking
  appendThinkingBubble();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conv.id,
        message: text,
        files: filesForRequest,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.type === 'ping') continue;

          removeThinkingBubble();

          if (evt.type === 'done') {
            const aiMsg = { role: 'assistant', content: evt.content };
            conv.messages.push(aiMsg);
            appendMessageDOM(aiMsg, true);
          } else if (evt.type === 'error') {
            const errMsg = { role: 'assistant', content: `⚠️ ${evt.content}` };
            conv.messages.push(errMsg);
            appendMessageDOM(errMsg, true);
          }
        } catch { /* ignore parse errors on partial lines */ }
      }
    }
  } catch (err) {
    removeThinkingBubble();
    const errMsg = { role: 'assistant', content: `**Connection error:** ${err.message}` };
    conv.messages.push(errMsg);
    appendMessageDOM(errMsg, true);
  }

  setSendState(false);
}


// ── File uploads ────────────────────────────

attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  const files = Array.from(fileInput.files);
  fileInput.value = '';
  for (const file of files) await uploadFile(file);
});

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();

    const entry = {
      file_id: data.file_id,
      original_name: data.original_name,
      path: data.path,
      file_type: data.file_type,
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
    };
    state.pendingFiles.push(entry);
    renderFilePreviewBar();
    updateSendBtn();
  } catch (err) {
    console.error('Upload error:', err);
  }
}

function clearPendingFiles() {
  state.pendingFiles.forEach(f => {
    if (f.previewUrl) URL.revokeObjectURL(f.previewUrl);
  });
  state.pendingFiles = [];
  renderFilePreviewBar();
}

function removeFile(idx) {
  const f = state.pendingFiles[idx];
  if (f?.previewUrl) URL.revokeObjectURL(f.previewUrl);
  state.pendingFiles.splice(idx, 1);
  renderFilePreviewBar();
  updateSendBtn();
}

function renderFilePreviewBar() {
  filePreviewBar.innerHTML = '';
  state.pendingFiles.forEach((f, i) => {
    const item = document.createElement('div');
    item.className = 'preview-item';

    const thumb = f.previewUrl
      ? `<img class="preview-thumb" src="${f.previewUrl}" alt="" />`
      : `<span class="preview-icon">${fileIcon(f.file_type)}</span>`;

    item.innerHTML = `
      ${thumb}
      <span class="preview-name">${escHtml(f.original_name)}</span>
      <button class="preview-remove" title="Remove" onclick="removeFile(${i})">×</button>
    `;
    filePreviewBar.appendChild(item);
  });
}


// ── UI helpers ──────────────────────────────

function setSendState(loading) {
  state.isLoading = loading;
  sendBtn.disabled = loading || (!messageInput.value.trim() && !state.pendingFiles.length);
  if (loading) {
    sendBtn.classList.add('loading');
  } else {
    sendBtn.classList.remove('loading');
  }
}

function updateSendBtn() {
  sendBtn.disabled = state.isLoading ||
    (!messageInput.value.trim() && !state.pendingFiles.length);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });
}

function autoResizeTextarea() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 180) + 'px';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fileIcon(type) {
  if (type === 'pdf')   return '📄';
  if (type === 'image') return '🖼️';
  return '📎';
}

function handleChip(prompt) {
  messageInput.value = prompt;
  autoResizeTextarea();
  updateSendBtn();
  sendMessage();
}

// Global: copy code blocks
window.copyCode = function(btn) {
  const targetId = btn.dataset.target;
  const codeEl = document.getElementById(targetId);
  if (!codeEl) return;
  navigator.clipboard.writeText(codeEl.textContent).then(() => {
    btn.textContent = '✓ Copied';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
      btn.classList.remove('copied');
    }, 2000);
  });
};


// ── Sidebar toggle ───────────────────────────

function closeMobileSidebar() {
  sidebar.classList.remove('mobile-open');
}

sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});

mobileMenuBtn.addEventListener('click', () => {
  sidebar.classList.toggle('mobile-open');
});

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
  if (window.innerWidth <= 680 &&
      sidebar.classList.contains('mobile-open') &&
      !sidebar.contains(e.target) &&
      e.target !== mobileMenuBtn) {
    closeMobileSidebar();
  }
});


// ── Keyboard & input events ──────────────────

messageInput.addEventListener('input', () => {
  autoResizeTextarea();
  updateSendBtn();
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

newChatBtn.addEventListener('click', () => {
  createConversation();
  renderMessages();
  renderConvoList();
  chatTitle.textContent = 'New conversation';
  clearPendingFiles();
  closeMobileSidebar();
});

clearBtn.addEventListener('click', () => {
  const conv = getActiveConv();
  if (!conv || !conv.messages.length) return;
  if (!confirm('Clear this conversation?')) return;
  conv.messages = [];
  conv.title = 'New conversation';
  renderMessages();
  renderConvoList();
  chatTitle.textContent = 'New conversation';
});

// Drag-and-drop on the whole page
document.addEventListener('dragover', e => { e.preventDefault(); });
document.addEventListener('drop', async e => {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files);
  for (const f of files) await uploadFile(f);
});

// Chips on the initial welcome screen
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => handleChip(chip.dataset.prompt));
});


// ── Boot ─────────────────────────────────────

(function init() {
  createConversation();
  renderConvoList();
  renderMessages();
  messageInput.focus();
})();