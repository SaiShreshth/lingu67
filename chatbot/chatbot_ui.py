"""
Memory Assistant - Flask Chatbot Web UI (FINAL - MOBILE FIXED)
"""

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from flask_cors import CORS
import os, json, logging, hashlib, uuid, threading, sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from server.local_client import LocalLLMClient
from config import (
    MODEL_SERVER_URL as LLM_API_URL,
    QDRANT_PATH,
    PROFILE_PATH,
    CHAT_LOG_PATH,
    FILE_METADATA_PATH
)

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

# ---------------- BOT ----------------
class MemoryAssistantBot:
    def __init__(self):
        self.llm = LocalLLMClient(LLM_API_URL)
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        self.collection = "episodic_memory"
        self._init_collection()
        self.files = self._load(FILE_METADATA_PATH)
        self.profile = self._load(PROFILE_PATH)
        
        # Rebuild file metadata if empty but data exists
        if not self.files:
            self._rebuild_file_metadata()

    def _init_collection(self):
        try:
            self.qdrant.get_collection(self.collection)
        except:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

    def _load(self, path):
        if os.path.exists(path):
            try:
                return json.load(open(path))
            except:
                pass
        return {}
    
    def _rebuild_file_metadata(self):
        """Rebuild file metadata by scanning Qdrant for unique source values"""
        logging.info("Rebuilding file metadata from Qdrant...")
        try:
            # Scroll through all points to find unique sources
            sources = set()
            offset = None
            while True:
                results, offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["source", "type"],
                )
                for point in results:
                    if point.payload and point.payload.get("type") == "file":
                        source = point.payload.get("source")
                        if source:
                            sources.add(source)
                if offset is None:
                    break
            
            # Build file metadata
            for source in sources:
                self.files[source] = {"point_ids": [], "chunk_count": 0, "recovered": True}
            
            if self.files:
                json.dump(self.files, open(FILE_METADATA_PATH, "w"), indent=2)
                logging.info(f"Recovered {len(self.files)} files from Qdrant: {list(self.files.keys())}")
        except Exception as e:
            logging.error(f"Failed to rebuild file metadata: {e}")

    def _embed(self, text):
        return self.llm.embed(text)

    def chat(self, msg, history):
        # --- 1. Profile Memory (Permanent User Facts) ---
        profile_text = ""
        if self.profile:
            profile_text = f"User Profile:\n{json.dumps(self.profile, indent=2)}"
        
        # --- 2. Long-Term Memory (Qdrant Vector Search) ---
        long_term_memories = []
        try:
            vec = self._embed(msg)
            res = self.qdrant.query_points(
                self.collection, vec, limit=5, with_payload=True
            )
            long_term_memories = [p.payload["text"] for p in res.points if p.payload]
        except Exception as e:
            logging.warning(f"Memory search error: {e}")
        
        # --- 3. File-Specific Search (if query mentions a file) ---
        file_memories = []
        mentioned_files = self._detect_file_mention(msg)
        if mentioned_files:
            for filename in mentioned_files:
                file_chunks = self._search_by_filename(filename)
                file_memories.extend(file_chunks)
        
        # --- 4. Recent Conversation Log (Short-Term Context) ---
        recent_log = self._get_recent_log(5)
        
        # --- 5. Build Enhanced System Prompt ---
        system = f"""You are Lingu 67, an AI assistant with persistent memory.

=== USER PROFILE (Permanent Facts) ===
{profile_text or "No profile data yet."}

=== LONG-TERM MEMORY (Retrieved from vector DB) ===
{chr(10).join(long_term_memories) if long_term_memories else "No relevant memories found."}

=== FILE CONTENT ===
{chr(10).join(file_memories) if file_memories else "No specific file content retrieved."}

=== UPLOADED FILES ===
{", ".join(self.files.keys()) if self.files else "None"}

=== RECENT CONVERSATION LOG ===
{recent_log or "No recent history."}

=== INSTRUCTIONS ===
1. Use the memory sections above to provide context-aware responses.
2. Remember user preferences and facts from their profile.
3. When asked about files, reference the FILE CONTENT section.
4. Be helpful, accurate, and personalized based on the user's history.
"""

        messages = [{"role": "system", "content": system}]
        messages += history[-10:]  # Last 10 messages as short-term memory
        messages.append({"role": "user", "content": msg})

        # Yield chunks and accumulate for storage
        full_response = ""
        for chunk in self.llm.chat(messages, max_tokens=8000, temperature=0.7, stream=True):
            full_response += chunk
            yield chunk
        
        # Background tasks: store memory and update profile
        threading.Thread(target=self._store, args=(msg, full_response), daemon=True).start()
        threading.Thread(target=self._update_profile, args=(msg, full_response), daemon=True).start()

    def _detect_file_mention(self, query):
        """Detect if user mentions any uploaded filenames using smart matching."""
        mentioned = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Common file reference keywords that apply to any uploaded file
        file_keywords = {"file", "document", "pdf", "book", "textbook", "paper", "uploaded", "chapter"}
        
        for filename in self.files.keys():
            # Extract searchable parts from filename
            base_name = os.path.splitext(filename)[0].lower()
            
            # Remove common separators and get words
            clean_name = base_name.replace("-", " ").replace("_", " ").replace(".", " ")
            file_words = set(clean_name.split())
            
            # Method 1: Exact filename match
            if filename.lower() in query_lower or base_name in query_lower:
                mentioned.append(filename)
                continue
            
            # Method 2: Word overlap (at least one significant word matches)
            # Filter out common words like 'the', 'a', 'of', numbers
            significant_words = {w for w in file_words if len(w) > 3 and not w.isdigit()}
            if significant_words & query_words:
                mentioned.append(filename)
                continue
            
            # Method 3: If user uses generic file terms and there's only one file
            if len(self.files) == 1 and (query_words & file_keywords):
                mentioned.append(filename)
                continue
            
            # Method 4: Fuzzy substring match (any 4+ char word from filename in query)
            for word in significant_words:
                if len(word) >= 4 and word in query_lower:
                    mentioned.append(filename)
                    break
        
        return list(set(mentioned))  # Remove duplicates
    
    def _search_by_filename(self, filename, limit=8):
        """Retrieve chunks from a specific file using Qdrant filtering."""
        try:
            vec = self._embed(f"file {filename}")
            res = self.qdrant.query_points(
                collection_name=self.collection,
                query=vec,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            return [p.payload.get("text", "") for p in res.points if p.payload]
        except Exception as e:
            logging.warning(f"File search error: {e}")
            return []
    
    def _get_recent_log(self, n_last=5):
        """Get recent interactions from chat log file."""
        if not os.path.exists(CHAT_LOG_PATH):
            return ""
        try:
            with open(CHAT_LOG_PATH, 'rb') as f:
                try:
                    f.seek(-2048, os.SEEK_END)
                except OSError:
                    pass
                last_chunk = f.read().decode('utf-8', errors='ignore')
            interactions = [i.strip() for i in last_chunk.split('-'*20) if i.strip()]
            return "\n---\n".join(interactions[-n_last:])
        except:
            return ""

    def _store(self, u, a):
        """Store conversation in vector DB and chat log."""
        blob = f"User: {u}\nAI: {a}"
        uid = str(uuid.UUID(hashlib.md5(blob.encode()).hexdigest()))
        try:
            self.qdrant.upsert(
                self.collection,
                [models.PointStruct(
                    id=uid,
                    vector=self._embed(blob),
                    payload={"text": blob, "type": "conversation", "timestamp": datetime.now().isoformat()}
                )]
            )
            # Also append to chat log file
            with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}]\nUser: {u}\nAI: {a}\n{'-'*20}\n")
        except Exception as e:
            logging.error(f"Storage error: {e}")
    
    def _update_profile(self, user_msg, ai_response):
        """Extract and update user profile facts from conversation."""
        try:
            # Simple pattern-based extraction for common facts
            extract_prompt = f"""Extract any NEW personal facts about the user from this exchange.
User said: "{user_msg}"
AI replied: "{ai_response}"

Return a JSON object with any new facts (e.g., name, job, preferences).
Return {{}} if no new facts.
JSON:"""
            
            result = self.llm.complete(extract_prompt, max_tokens=100, temperature=0)
            
            # Parse the result
            import re
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                new_facts = json.loads(match.group(0))
                if new_facts:
                    # Merge with existing profile
                    self.profile.update(new_facts)
                    with open(PROFILE_PATH, 'w') as f:
                        json.dump(self.profile, f, indent=2)
                    logging.info(f"Profile updated: {new_facts}")
        except Exception as e:
            logging.debug(f"Profile update skipped: {e}")

    def get_files(self):
        self.files = self._load(FILE_METADATA_PATH)
        return [{"name": k} for k in self.files]

    def ingest(self, path, name):
        """Ingest file with parallel batched embedding"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        text = open(path, "r", errors="ignore").read()
        
        # Use larger chunks to reduce total count
        chunk_size = 1024
        overlap = 512
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        
        total = len(chunks)
        print(f"\n[UPLOAD] Starting: {name}")
        print(f"[UPLOAD] File size: {len(text):,} chars → {total:,} chunks")
        start_time = time.time()
        
        # Split into batches
        BATCH_SIZE = 500  # Larger batches
        batches = [chunks[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
        
        def process_batch(batch_idx, batch_chunks):
            """Process a single batch - runs in parallel"""
            try:
                vectors = self.llm.embed_batch(batch_chunks)
                return batch_idx, batch_chunks, vectors, None
            except Exception as e:
                return batch_idx, batch_chunks, None, str(e)
        
        # Process batches in parallel (4 workers)
        all_results = [None] * len(batches)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_batch, i, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx, batch_chunks, vectors, error = future.result()
                completed += 1
                
                if error:
                    logging.warning(f"Batch {batch_idx} failed: {error}")
                    # Fallback to sequential
                    vectors = [self._embed(c) for c in batch_chunks]
                
                all_results[batch_idx] = (batch_chunks, vectors)
                
                progress = completed * 100 // len(batches)
                print(f"[UPLOAD] Embedding: {completed}/{len(batches)} batches ({progress}%)", end='\r', flush=True)
        
        print(f"\n[UPLOAD] Building index...")
        
        # Build all points
        all_ids = []
        all_points = []
        chunk_idx = 0
        
        for batch_chunks, vectors in all_results:
            for c, vec in zip(batch_chunks, vectors):
                pid = str(uuid.uuid4())
                all_ids.append(pid)
                all_points.append(models.PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "text": f"[{name} #{chunk_idx}]\n{c}",
                        "source": name,
                        "type": "file",
                        "timestamp": datetime.now().isoformat()
                    }
                ))
                chunk_idx += 1
        
        # Batch upsert to Qdrant
        print(f"[UPLOAD] Saving to database...")
        UPSERT_BATCH = 1000
        for i in range(0, len(all_points), UPSERT_BATCH):
            self.qdrant.upsert(self.collection, all_points[i:i+UPSERT_BATCH])
        
        self.files[name] = {"point_ids": all_ids, "chunk_count": total}
        json.dump(self.files, open(FILE_METADATA_PATH, "w"), indent=2)
        
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        print(f"[UPLOAD] ✓ Complete: {name} ({total:,} chunks in {elapsed:.1f}s, {rate:.0f}/s)")
        return {"success": True}

    def delete(self, name):
        ids = self.files[name]["point_ids"]
        self.qdrant.delete(self.collection, models.PointIdsList(points=ids))
        del self.files[name]
        json.dump(self.files, open(FILE_METADATA_PATH, "w"), indent=2)
        return {"success": True}

bot = None
def get_bot():
    global bot
    if not bot:
        bot = MemoryAssistantBot()
    return bot

# ---------------- UI ----------------
HTML = """<!DOCTYPE html>
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

/* ===== APP LAYOUT ===== */
body {
    display: flex;
    flex-direction: column;
}

/* ===== CHAT CONTAINER ===== */
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
    overflow: hidden; /* Ensure messages scroll inside */
}

/* ===== HEADER ===== */
header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header .title {
    font-weight: 600;
}

header .status {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* ===== MESSAGES ===== */
.messages-area {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
}

/* Message bubbles */
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

/* ===== INPUT ===== */
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
    padding-right: 45px; /* Space for button */
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
.send-btn:hover .send-icon {
    transform: scale(1.1);
}

/* ===== FOOTER DRAWER ===== */
.footer-drawer {
    width: 100%;
    background: var(--bg-footer);
    border-top: 1px solid var(--border);
    transition: height 0.3s ease;
    overflow: hidden;
    position: relative;
    z-index: 10;
}

/* Heights */
.footer-drawer.collapsed {
    height: 46px;
}

.footer-drawer.expanded {
    height: 260px;
}

/* Header */
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

.drawer-header span {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Content */
.drawer-content {
    padding: 18px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 14px;
    overflow-y: auto;
    height: calc(260px - 46px);
}

/* File item */
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

.file-item:hover {
    background: rgba(255,255,255,0.12);
}

.file-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    width: 100%;
    font-weight: 500;
}

.delete-btn-red {
    background: rgba(239, 68, 68, 0.15); /* #ef4444 with opacity */
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
    margin-top: 6px;
    transition: all 0.2s;
}

.delete-btn-red:hover {
    background: #ef4444;
    color: white;
}

.upload-zone {
    border: 1px dashed var(--border);
    background: transparent;
    color: var(--text-muted);
}
.upload-zone:hover {
    border-color: var(--bg-user);
    color: var(--text-main);
}
.upload-zone label {
    cursor: pointer;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Loading overlay for file operations */
.file-item.loading {
    pointer-events: none;
    opacity: 0.6;
}

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

.file-item.loading .loader {
    display: block;
}

.file-item.loading .file-name,
.file-item.loading .delete-btn-red {
    opacity: 0.3;
}

.upload-zone.uploading label {
    opacity: 0.5;
}

.upload-zone .loader {
    display: none;
}

.upload-zone.uploading .loader {
    display: block;
}

@keyframes spin {
    to { transform: translate(-50%, -50%) rotate(360deg); }
}

/* ===== MOBILE ===== */
@media (max-width: 640px) {
    .message { max-width: 90%; }
    .chat-container { border: none; }
}
</style>
</head>

<body>

<div class="chat-container">
    <header>
        <div class="title">Lingu 67</div>
        <div class="status">Online</div>
    </header>

    <div class="messages-area" id="messages">
        <div class="message bot">
            <div class="message-content">Hello! I’m ready when you are.</div>
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
    <div class="drawer-content" id="fileList">
        <!-- Files will be loaded here -->
    </div>
</div>

<script>
const messages = document.getElementById("messages");
const input = document.getElementById("msg"); // Global input variable
const drawer = document.getElementById("drawer");
const fileList = document.getElementById("fileList");
let history = [];

marked.setOptions({breaks:true});

// Enable Enter key to send
input.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent default newline if any
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

    // Assistant placeholder
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = '<div class="message-content"></div>';
    messages.appendChild(div);
    const contentDiv = div.querySelector(".message-content");
    messages.scrollTop = messages.scrollHeight;

    let fullResponse = "";
    
    try {
        const response = await fetch("/api/chat", {
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
    const r = await fetch("/api/files");
    const j = await r.json();
    
    // Clear and rebuild file list
    fileList.innerHTML = `
    <div class="file-item upload-zone" id="uploadZone">
        <label>+ Upload <input type="file" id="fileUploadInput" style="display:none"></label>
        <div class="loader"></div>
    </div>`;
    
    j.files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.id = 'file-' + btoa(f.name).replace(/[^a-zA-Z0-9]/g, ''); // Unique ID
        
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
    
    // Re-attach upload listener
    document.getElementById('fileUploadInput').addEventListener('change', function() {
        uploadFile(this);
    });
}

// Event delegation for delete buttons
fileList.addEventListener('click', async function(e) {
    if (e.target.classList.contains('delete-btn-red')) {
        e.stopPropagation();
        const name = e.target.dataset.filename;
        const fileCard = e.target.closest('.file-item');
        
        if (!confirm("Are you sure you want to delete '" + name + "'?")) return;
        
        // Show loading state
        fileCard.classList.add('loading');
        
        try {
            const res = await fetch('/api/files/' + encodeURIComponent(name), { method: 'DELETE' });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Server error');
            }
            loadFiles();
        } catch (err) {
            fileCard.classList.remove('loading');
            alert('Error deleting file: ' + err.message);
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
        await fetch("/api/upload", { method: "POST", body: d });
        loadFiles();
    } catch (err) {
        uploadZone.classList.remove('uploading');
        alert('Upload failed: ' + err.message);
    }
}

loadFiles();

// Allow Enter key
input.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        send();
    }
});
</script>

</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/chat", methods=["POST"])
def chat():
    d=request.json
    return Response(
        stream_with_context(get_bot().chat(d["message"], d["history"])),
        mimetype="text/plain"
    )

@app.route("/api/files")
def files(): return jsonify({"files":get_bot().get_files()})

@app.route("/api/upload", methods=["POST"])
def upload():
    f=request.files["file"]
    p=f"/tmp/{f.filename}"
    f.save(p)
    r=get_bot().ingest(p,f.filename)
    os.remove(p)
    return jsonify(r)

@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    res = get_bot().delete(filename)
    return jsonify(res)

if __name__=="__main__":
    app.run("0.0.0.0",7860,debug=False)
