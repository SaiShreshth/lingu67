"""
HTML Template for the web interface.

Extracted and adapted from original chatbot_ui.py.
"""

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lingu 67</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

<style>
:root {
    --bg-main: #0e1117;
    --bg-panel: #161b22;
    --bg-user: #2563eb;
    --bg-bot: #1f2937;
    --bg-footer: #0b0f14;
    --text-main: #e5e7eb;
    --text-muted: #9ca3af;
    --border: rgba(255,255,255,0.08);
    --radius: 14px;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif;
}

html, body {
    height: 100%;
    background: var(--bg-main);
    color: var(--text-main);
    overflow: hidden;
}

body {
    display: flex;
    flex-direction: column;
}

.chat-container {
    flex: 1;
    max-width: 860px;
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    border-right: 1px solid var(--border);
    overflow: hidden;
}

header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header .title { font-weight: 600; }
header .status { font-size: 0.8rem; color: var(--text-muted); }

.messages-area {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.message {
    max-width: 85%;
    padding: 14px 18px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 0.95rem;
    word-wrap: break-word;
}

.message.bot {
    align-self: flex-start;
    background: var(--bg-bot);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
}

.message.user {
    align-self: flex-end;
    background: var(--bg-user);
    color: white;
    border-bottom-right-radius: 4px;
}

.message p { margin-bottom: 10px; }
.message p:last-child { margin-bottom: 0; }
.message pre { 
    background: #0d1117; 
    padding: 10px; 
    border-radius: 6px; 
    overflow-x: auto; 
    margin: 10px 0;
}
.message code { font-family: monospace; }

.input-area {
    padding: 14px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex;
    gap: 10px;
}

.input-wrapper {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
}

.input-wrapper input {
    width: 100%;
    padding: 12px 14px;
    padding-right: 45px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #020617;
    color: var(--text-main);
    outline: none;
}

.send-btn {
    position: absolute;
    right: 6px;
    background: transparent;
    border: none;
    padding: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
}
.send-btn:hover { background: rgba(255,255,255,0.05); }

.send-icon {
    width: 20px;
    height: 20px;
    fill: var(--bg-user);
    transition: transform 0.2s;
}
.send-btn:hover .send-icon { transform: scale(1.1); }

.footer-drawer {
    width: 100%;
    background: var(--bg-footer);
    border-top: 1px solid var(--border);
    transition: height 0.3s ease;
    overflow: hidden;
    position: relative;
    z-index: 10;
}

.footer-drawer.collapsed { height: 46px; }
.footer-drawer.expanded { height: 260px; }

.drawer-header {
    height: 46px;
    padding: 0 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
    background: var(--bg-footer);
}

.drawer-header span { font-size: 0.85rem; color: var(--text-muted); }

.drawer-content {
    padding: 18px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 14px;
    overflow-y: auto;
    height: calc(260px - 46px);
}

.file-item {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background 0.2s;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 5px;
}

.file-item:hover { background: rgba(255,255,255,0.12); }

.file-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    width: 100%;
    font-weight: 500;
}

.delete-btn-red {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
    margin-top: 6px;
    transition: all 0.2s;
}

.delete-btn-red:hover { background: #ef4444; color: white; }

.upload-zone {
    border: 1px dashed var(--border);
    background: transparent;
    color: var(--text-muted);
}
.upload-zone:hover { border-color: var(--bg-user); color: var(--text-main); }
.upload-zone label {
    cursor: pointer;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.file-item.loading { pointer-events: none; opacity: 0.6; }

.file-item .loader {
    display: none;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 24px;
    height: 24px;
    border: 3px solid rgba(255,255,255,0.2);
    border-top-color: var(--bg-user);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

.file-item.loading .loader { display: block; }
.file-item.loading .file-name,
.file-item.loading .delete-btn-red { opacity: 0.3; }

.upload-zone.uploading label { opacity: 0.5; }
.upload-zone .loader { display: none; }
.upload-zone.uploading .loader { display: block; }

@keyframes spin { to { transform: translate(-50%, -50%) rotate(360deg); } }

@media (max-width: 640px) {
    .message { max-width: 90%; }
    .chat-container { border: none; }
}
</style>
</head>

<body>

<div class="chat-container">
    <header>
        <div class="title">Lingu 67 <span style="font-size:0.7rem;color:var(--text-muted)">(v2.0)</span></div>
        <div class="status">Online</div>
    </header>

    <div class="messages-area" id="messages">
        <div class="message bot">
            <div class="message-content">Hello! I'm ready when you are.</div>
        </div>
    </div>

    <div class="input-area">
        <div class="input-wrapper">
            <input id="msg" placeholder="Type a message…" autocomplete="off">
            <button onclick="send()" class="send-btn">
                <svg viewBox="0 0 24 24" class="send-icon">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
            </button>
        </div>
    </div>
</div>

<!-- FOOTER DRAWER -->
<div class="footer-drawer collapsed" id="drawer">
    <div class="drawer-header" onclick="toggleDrawer()">
        <span>📂 Project Files</span>
        <span>▲</span>
    </div>
    <div class="drawer-content" id="fileList"></div>
</div>

<script>
const messages = document.getElementById("messages");
const input = document.getElementById("msg");
const drawer = document.getElementById("drawer");
const fileList = document.getElementById("fileList");
let history = [];

marked.setOptions({breaks:true});

input.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        send();
    }
});

function toggleDrawer() {
    drawer.classList.toggle("collapsed");
    drawer.classList.toggle("expanded");
}

function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = "message " + who;
    div.innerHTML = `<div class="message-content">${marked.parse(text)}</div>`;
    messages.appendChild(div);
    div.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
    messages.scrollTop = messages.scrollHeight;
}

async function send() {
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    addMessage(text, "user");
    history.push({role: "user", content: text});

    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = '<div class="message-content"></div>';
    messages.appendChild(div);
    const contentDiv = div.querySelector(".message-content");
    messages.scrollTop = messages.scrollHeight;

    let fullResponse = "";
    
    try {
        const response = await fetch("/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, history })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            fullResponse += chunk;
            contentDiv.innerHTML = marked.parse(fullResponse);
            div.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
            messages.scrollTop = messages.scrollHeight;
        }
        
        history.push({ role: "assistant", content: fullResponse });
    } catch (e) {
        contentDiv.innerHTML = "Error: " + e;
    }
}

async function loadFiles() {
    const r = await fetch("/files");
    const j = await r.json();
    
    fileList.innerHTML = `
    <div class="file-item upload-zone" id="uploadZone">
        <label>+ Upload <input type="file" id="fileUploadInput" style="display:none"></label>
        <div class="loader"></div>
    </div>`;
    
    j.files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.id = 'file-' + btoa(f.name).replace(/[^a-zA-Z0-9]/g, '');
        
        const nameDiv = document.createElement('div');
        nameDiv.className = 'file-name';
        nameDiv.title = f.name;
        nameDiv.textContent = f.name;
        
        const btn = document.createElement('button');
        btn.className = 'delete-btn-red';
        btn.textContent = 'Delete';
        btn.dataset.filename = f.name;
        
        const loader = document.createElement('div');
        loader.className = 'loader';
        
        item.appendChild(nameDiv);
        item.appendChild(btn);
        item.appendChild(loader);
        fileList.appendChild(item);
    });
    
    document.getElementById('fileUploadInput').addEventListener('change', function() {
        uploadFile(this);
    });
}

fileList.addEventListener('click', async function(e) {
    if (e.target.classList.contains('delete-btn-red')) {
        e.stopPropagation();
        const name = e.target.dataset.filename;
        const fileCard = e.target.closest('.file-item');
        
        if (!confirm("Delete '" + name + "'?")) return;
        
        fileCard.classList.add('loading');
        
        try {
            const res = await fetch('/files/' + encodeURIComponent(name), { method: 'DELETE' });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Server error');
            }
            loadFiles();
        } catch (err) {
            fileCard.classList.remove('loading');
            alert('Error: ' + err.message);
        }
    }
});

async function uploadFile(input) {
    const f = input.files[0];
    if (!f) return;
    
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.classList.add('uploading');
    
    const d = new FormData();
    d.append("file", f);
    
    try {
        await fetch("/files/upload", { method: "POST", body: d });
        loadFiles();
    } catch (err) {
        uploadZone.classList.remove('uploading');
        alert('Upload failed: ' + err.message);
    }
}

loadFiles();
</script>

</body>
</html>
'''
