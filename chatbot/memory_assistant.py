import os
import sys
import json
import logging
import hashlib
import re
import ast
import uuid
import threading
import copy
from datetime import datetime
from typing import List, Dict, Any, Set

# Third-party imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from local_client import LocalLLMClient

# --- CONFIGURATION ---
LLM_API_URL = "http://localhost:8000"  # Local FastAPI server
QDRANT_PATH = "./qdrant_local"  # Local file-based Qdrant storage (no Docker needed)
PROFILE_PATH = "./user_profile.json"
CHAT_LOG_PATH = "./conversation_log.txt"
FILE_METADATA_PATH = "./file_metadata.json"

CTX_SIZE = 4096 
HISTORY_LIMIT = 15

# Context overflow management
MAX_TOKENS_PER_MESSAGE = 512
CONTEXT_SAFETY_MARGIN = 500
OVERFLOW_STORAGE_PATH = "./context_overflow.json"

# --- LOGGING ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("MemorySystem")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

class SemanticRouter:
    def __init__(self, api_url: str):
        self.llm_client = LocalLLMClient(api_url)
        logger.info(f"✓ Using embedding API at {api_url}")

    def encode(self, text: str) -> List[float]:
        return self.llm_client.embed(text)

class Brain:
    def __init__(self, api_url: str, ctx_size: int):
        self.ctx_size = ctx_size
        self.llm_client = LocalLLMClient(api_url)
        self.overflow_storage = []
        self._load_overflow_context()
        logger.info(f"✓ Connected to LLM API at {api_url}")
        self.lock = threading.Lock()
    
    def _load_overflow_context(self):
        """Load previously overflowed context from disk"""
        try:
            if os.path.exists(OVERFLOW_STORAGE_PATH):
                with open(OVERFLOW_STORAGE_PATH, 'r') as f:
                    self.overflow_storage = json.load(f)
                logger.info(f"Loaded {len(self.overflow_storage)} overflow messages")
        except Exception as e:
            logger.warning(f"Could not load overflow context: {e}")
            self.overflow_storage = []

    def _save_overflow_context(self):
        """Save overflowed context to disk"""
        try:
            with open(OVERFLOW_STORAGE_PATH, 'w') as f:
                json.dump(self.overflow_storage, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save overflow context: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation: 1 token ≈ 4 characters"""
        return len(text) // 4

    def _manage_context_window(self, messages: List[Dict]) -> List[Dict]:
        """Manage context window, moving overflow to storage"""
        total_tokens = sum(self._estimate_tokens(m.get('content', '')) for m in messages)
        available_tokens = self.ctx_size - CONTEXT_SAFETY_MARGIN - 512  # Reserve for response
        
        if total_tokens <= available_tokens:
            return messages
        
        logger.info(f"Context overflow: {total_tokens} tokens > {available_tokens} available")
        
        # Separate system and other messages
        system_msgs = [m for m in messages if m.get('role') == 'system']
        other_msgs = [m for m in messages if m.get('role') != 'system']
        
        # First, truncate system messages if they're too long
        system_tokens = sum(self._estimate_tokens(m.get('content', '')) for m in system_msgs)
        max_system_tokens = available_tokens // 2  # Allow system to use at most 50% of context
        
        if system_tokens > max_system_tokens:
            logger.info(f"System message too long ({system_tokens} tokens), truncating to {max_system_tokens}")
            # Truncate each system message proportionally
            for msg in system_msgs:
                content = msg.get('content', '')
                msg_tokens = self._estimate_tokens(content)
                if msg_tokens > max_system_tokens:
                    # Keep roughly the right number of characters
                    target_chars = max_system_tokens * 4
                    msg['content'] = content[:target_chars] + "\n...[truncated for context limit]"
            system_tokens = max_system_tokens
        
        remaining_tokens = available_tokens - system_tokens
        
        # Now fit as many conversation messages as possible
        kept_messages = []
        overflow_messages = []
        current_tokens = 0
        
        for msg in reversed(other_msgs):
            msg_tokens = self._estimate_tokens(msg.get('content', ''))
            if current_tokens + msg_tokens <= remaining_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                overflow_messages.insert(0, msg)
        
        # Store overflow messages
        if overflow_messages:
            self.overflow_storage.extend(overflow_messages)
            self.overflow_storage = self.overflow_storage[-50:]  # Keep only last 50
            self._save_overflow_context()
            logger.info(f"Moved {len(overflow_messages)} messages to overflow storage")
        
        return system_msgs + kept_messages 

    def extract_json_from_text(self, text: str) -> Dict:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(0))
                except: return ast.literal_eval(match.group(0))
            return {}
        except: return {}

    def extract_keywords(self, text_chunk: str) -> str:
        prompt = (
            f"Text: \"{text_chunk[:500]}\"\n" 
            f"Task: Extract 3-5 key entities or terms from the text above to use as a search query.\n"
            f"Output ONLY the keywords separated by spaces."
        )
        with self.lock:
            output = self.llm_client.complete(prompt, max_tokens=32, temperature=0.0)
        return output.strip()

    def detect_file_intent(self, query: str) -> bool:
        if any(w in query.lower() for w in ["upload", "ingest", "attach file", "read this file"]):
            return True

        sys_prompt = "You are an Intent Classifier."
        user_prompt = (
            f"User Input: '{query}'\n"
            f"Task: Does this input indicate the user is ABOUT to upload or provide a file?\n"
            f"Reply YES only if they say 'let me upload', 'here is the file', 'ingest this'.\n"
            f"Reply NO if they are asking a question about an existing file.\n"
            f"Answer (YES/NO):"
        )
        
        with self.lock:
            output = self.llm_client.chat(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=5,
                temperature=0.0
            )
        decision = output.strip().upper()
        return "YES" in decision

    def check_sufficiency(self, history: List[Dict], query: str) -> bool:
        if not history:
            if "?" in query or any(w in query.lower() for w in ["what", "analyze", "check", "read", "file", "csv"]):
                return False 
            return True
            
        if any(w in query.lower() for w in ["file", "csv", "document", "data", "analyze", "read", "summary"]):
            return False

        prompt = (
            f"User Input: '{query}'\n"
            f"Task: Do we need to search the Database to answer this?\n"
            f"Answer (YES/NO):"
        )
        with self.lock:
            output = self.llm_client.complete(prompt, max_tokens=3, temperature=0.0)
        decision = output.strip().upper()
        
        if "YES" in decision: return False 
        return True 

    def consolidate_profile(self, current_profile_text: str, history: List[Dict], user_input: str, ai_response: str) -> Dict[str, str]:
        recent_context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        sys_prompt = "You are a Profile Manager. Update the User's Permanent Profile JSON. Return JSON only."
        user_prompt = (
            f"--- CURRENT PROFILE ---\n{current_profile_text}\n"
            f"--- INPUT ---\nUser: {user_input}\n"
            f"Task: Extract NEW user facts (Name, Job, Preferences). Return {{}} if none."
        )
        
        with self.lock:
            output = self.llm_client.chat(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=128,
                temperature=0.0
            )
        return self.extract_json_from_text(output)

    def _summarize_overflow(self) -> str:
        """Create a brief summary of overflow context"""
        if not self.overflow_storage:
            return ""
        
        recent_overflow = self.overflow_storage[-10:]
        overflow_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')[:100]}" for m in recent_overflow])
        
        prompt = f"Briefly summarize this conversation history in 2-3 sentences:\n{overflow_text}"
        try:
            summary = self.llm_client.complete(prompt, max_tokens=128, temperature=0.3)
            return summary.strip()
        except:
            return ""

    def _recursive_process(self, system_prompt: str, context_parts: List[str], query: str) -> str:
        """Process large context recursively by summarizing chunks"""
        if len(context_parts) <= 1:
            full_context = context_parts[0] if context_parts else ""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {full_context}\n\nQuestion: {query}"}
            ]
            response = self.llm_client.chat(messages, max_tokens=512, temperature=0.7)
            return response
        
        # Recursive case: summarize chunks first
        summaries = []
        for i, part in enumerate(context_parts):
            logger.info(f"Processing context chunk {i+1}/{len(context_parts)}")
            summary_prompt = f"Summarize the key information from this context:\n{part[:1500]}"
            messages = [
                {"role": "system", "content": "You are a summarization assistant."},
                {"role": "user", "content": summary_prompt}
            ]
            summary = self.llm_client.chat(messages, max_tokens=256, temperature=0.3)
            summaries.append(summary)
        
        # Combine summaries and answer
        combined_summary = "\n\n".join(summaries)
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarized Context: {combined_summary}\n\nQuestion: {query}"}
        ]
        response = self.llm_client.chat(final_messages, max_tokens=512, temperature=0.7)
        return response

    def generate_response(self, system_context: str, history: List[Dict], query: str, use_overflow: bool = False) -> str:
        messages = [{"role": "system", "content": system_context}]
        
        if use_overflow and self.overflow_storage:
            overflow_summary = self._summarize_overflow()
            if overflow_summary:
                messages.append({"role": "system", "content": f"Previous context summary: {overflow_summary}"})
        
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        # Manage context window BEFORE sending
        managed_messages = self._manage_context_window(messages)
        
        # Double-check total size
        total_tokens = sum(self._estimate_tokens(m.get('content', '')) for m in managed_messages)
        logger.info(f"Sending {len(managed_messages)} messages (~{total_tokens} tokens) to LLM")
        
        response = self.llm_client.chat(managed_messages, max_tokens=512, temperature=0.7)
        return response
    
    def generate_response_with_large_context(self, system_context: str, context_chunks: List[str], query: str) -> str:
        """Generate response when context is too large - uses recursive processing"""
        return self._recursive_process(system_context, context_chunks, query)

class MemoryManager:
    def __init__(self, qdrant_path: str, profile_path: str, log_path: str, router: SemanticRouter, brain: Brain):
        self.router = router
        self.brain = brain
        self.profile_path = profile_path
        self.log_path = log_path
        self.file_metadata_path = FILE_METADATA_PATH  # NEW
        
        try:
            # Use local file-based storage instead of Docker server
            self.client = QdrantClient(path=qdrant_path)
            logger.info(f"✓ Connected to local Qdrant at {qdrant_path}")
        except Exception as e:
            logger.critical(f"❌ Database Connection Failed: {e}")
            sys.exit(1)

        self.collection_name = "episodic_memory"
        self._init_collection()
        
        if not os.path.exists(self.profile_path):
            with open(self.profile_path, 'w') as f: json.dump({}, f)
        
        # NEW: Load file metadata on startup
        self.uploaded_files = self._load_file_metadata()

    def _init_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except:
            logger.info(f"Creating Qdrant collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384, 
                    distance=models.Distance.COSINE
                )
            )
    
    # NEW: File metadata persistence
    def _load_file_metadata(self) -> Dict[str, Dict]:
        """Load file metadata from disk."""
        if not os.path.exists(self.file_metadata_path):
            return {}
        try:
            with open(self.file_metadata_path, 'r') as f:
                return json.load(f)
        except:
            logger.warning("Failed to load file metadata, starting fresh")
            return {}
    
    def _save_file_metadata(self):
        """Save file metadata to disk."""
        try:
            with open(self.file_metadata_path, 'w') as f:
                json.dump(self.uploaded_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")

    # ENHANCED: Track uploaded files with their metadata
    def ingest_file(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"\033[91m[System: File not found at {file_path}]\033[0m")
            return

        try:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f: text = f.read()
            
            chunk_size = 512
            overlap = 50
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
            
            print(f"\033[90m[System: Processing {len(chunks)} chunks from {filename}...]\033[0m")
            
            # NEW: Track point IDs for this file
            point_ids = []
            points = []
            for i, chunk in enumerate(chunks):
                augmented_chunk = f"[File: {filename} Part {i}] \n{chunk}"
                vector = self.router.encode(augmented_chunk)
                doc_id = str(uuid.uuid4())
                point_ids.append(doc_id)  # NEW: Track IDs
                
                points.append(models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload={"text": augmented_chunk, "source": filename, "type": "file", "timestamp": datetime.now().isoformat()}
                ))
            
            self.client.upsert(collection_name=self.collection_name, points=points)
            
            # NEW: Save metadata
            self.uploaded_files[filename] = {
                "point_ids": point_ids,
                "chunk_count": len(chunks),
                "upload_date": datetime.now().isoformat(),
                "file_path": file_path
            }
            self._save_file_metadata()
            
            print(f"\033[92m[System: Successfully ingested {filename} into Long-Term Memory]\033[0m")
            
        except Exception as e:
            logger.error(f"Ingestion Error: {e}")
    
    # NEW: Search for specific file content using Qdrant filters
    def search_by_filename(self, filename: str, n_results: int = 10) -> List[str]:
        """Retrieve all chunks from a specific file using Qdrant filtering."""
        try:
            # Create a dummy vector for filtering
            dummy_query = f"file {filename}"
            vector = self.router.encode(dummy_query)
            
            # Search with filter for specific filename
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=n_results,
                with_payload=True
            )
            
            # Handle response - it returns QueryResponse object
            if hasattr(search_result, 'points'):
                memories = [hit.payload.get('text', '') for hit in search_result.points if hit.payload]
            else:
                memories = []
            
            return memories
        except Exception as e:
            logger.error(f"File-specific search error: {e}")
            logger.debug(f"Error details:", exc_info=True)
            return []
    
    # NEW: Get list of uploaded files
    def get_uploaded_files_list(self) -> str:
        """Return a formatted list of uploaded files."""
        if not self.uploaded_files:
            return "No files uploaded yet."
        
        file_list = ["Uploaded Files:"]
        for filename, metadata in self.uploaded_files.items():
            upload_date = metadata.get('upload_date', 'Unknown')
            chunk_count = metadata.get('chunk_count', 0)
            file_list.append(f"  - {filename} ({chunk_count} chunks, uploaded: {upload_date[:10]})")
        
        return "\n".join(file_list)
    
    # NEW: Detect if query mentions specific filename
    def detect_filename_in_query(self, query: str) -> List[str]:
        """Detect if user mentions any uploaded filenames in their query."""
        mentioned_files = []
        query_lower = query.lower()
        for filename in self.uploaded_files.keys():
            # Check for exact filename or basename without extension
            base_name = os.path.splitext(filename)[0].lower()
            if filename.lower() in query_lower or base_name in query_lower:
                mentioned_files.append(filename)
        return mentioned_files

    def get_profile(self) -> str:
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
            if not data: return "User Profile: Empty."
            return f"User Facts & Reminders:\n{json.dumps(data, indent=2)}"
        except: return ""

    def get_raw_profile_json(self) -> str:
        try:
            with open(self.profile_path, 'r') as f: return f.read()
        except: return "{}"

    def update_profile(self, new_facts: Dict[str, str]):
        data = {}
        try:
            with open(self.profile_path, 'r') as f: data = json.load(f)
        except: pass
        
        updated = False
        for k, v in new_facts.items():
            if k not in data or data[k] != v:
                data[k] = v
                updated = True
        
        if updated:
            with open(self.profile_path, 'w') as f: json.dump(data, f, indent=2)
            print(f"\n\033[95m[Background Update: Learned new fact -> {new_facts}]\033[0m")
            print("\033[94mYou:\033[0m ", end="", flush=True)

    def search_layer(self, query_text: str, n_results: int = 5) -> List[str]:
        try:
            vector = self.router.encode(query_text)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=n_results,
                with_payload=True
            )
            
            # Handle response
            if hasattr(search_result, 'points'):
                memories = [hit.payload.get('text', '') for hit in search_result.points if hit.payload]
            else:
                memories = []
            
            return memories
        except Exception as e:
            logger.error(f"Layer Search Error: {e}")
            return []

    # ENHANCED: Improved episodic search with file-specific routing
    def search_episodic(self, user_query: str) -> str:
        unique_memories: Set[str] = set()
        
        # NEW: Check if user mentions specific files
        mentioned_files = self.detect_filename_in_query(user_query)
        if mentioned_files:
            print(f"\033[90m  -> Detected file mention: {', '.join(mentioned_files)}\033[0m")
            for filename in mentioned_files:
                file_memories = self.search_by_filename(filename, n_results=8)
                unique_memories.update(file_memories)
                print(f"\033[90m  -> Retrieved {len(file_memories)} chunks from {filename}\033[0m")
        
        print("\033[90m  -> Layer 1: Searching User Input...\033[0m")
        layer1_results = self.search_layer(user_query, n_results=5)
        unique_memories.update(layer1_results)
        
        if layer1_results:
            l1_context = " ".join(layer1_results)
            keywords_l2 = self.brain.extract_keywords(l1_context)
            print(f"\033[90m  -> Layer 2: Searching Keywords '{keywords_l2}'...\033[0m")
            layer2_results = self.search_layer(keywords_l2, n_results=5)
            unique_memories.update(layer2_results)
            
            if layer2_results and ("analyze" in user_query.lower() or "file" in user_query.lower()):
                l2_context = " ".join(layer2_results)
                keywords_l3 = self.brain.extract_keywords(l2_context)
                print(f"\033[90m  -> Layer 3: Searching Keywords '{keywords_l3}'...\033[0m")
                layer3_results = self.search_layer(keywords_l3, n_results=5)
                unique_memories.update(layer3_results)

        print(f"\033[93m[Deep Search: Found {len(unique_memories)} unique memories]\033[0m")
        if not unique_memories: return "No relevant memories found."
        return "Deep Search Memories:\n" + "\n".join([f"- {m}" for m in unique_memories])

    def recall_recent(self, n_last: int = 5) -> str:
        """OPTIMIZED: Only reads the last few KB of the file, preserving RAM."""
        if not os.path.exists(self.log_path): return ""
        try:
            # Read only last 2000 bytes (~30 lines) instead of whole file
            with open(self.log_path, 'rb') as f:
                try:
                    f.seek(-2048, os.SEEK_END)
                except OSError:
                    pass # File is smaller than 2KB
                last_chunk = f.read().decode('utf-8', errors='ignore')
            
            interactions = [i.strip() for i in last_chunk.split('-'*20) if i.strip()]
            return "Recent Conversation History:\n" + "\n---\n".join(interactions[-n_last:])
        except: return ""

    def save_interaction(self, user_input: str, ai_response: str):
        text_blob = f"User: {user_input} | AI: {ai_response}"
        vector = self.router.encode(text_blob)
        doc_uuid = str(uuid.UUID(hashlib.md5(text_blob.encode()).hexdigest()))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=doc_uuid, vector=vector, payload={"text": text_blob, "type": "conversation", "timestamp": datetime.now().isoformat()})]
        )
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\nUser: {user_input}\nAI: {ai_response}\n{'-'*20}\n")

def background_profile_update(brain, memory, profile_text, history_copy, user_input, response):
    try:
        new_facts = brain.consolidate_profile(profile_text, history_copy, user_input, response)
        if new_facts:
            memory.update_profile(new_facts)
    except Exception as e:
        logger.error(f"Background Update Failed: {e}")

def main():
    print(f"=== Booting Up Memory System v3.8 (Local API Mode + Context Overflow Management) ===")
    try:
        router = SemanticRouter(LLM_API_URL)
        brain = Brain(LLM_API_URL, CTX_SIZE)
        memory = MemoryManager(QDRANT_PATH, PROFILE_PATH, CHAT_LOG_PATH, router, brain)
    except Exception as e:
        logger.critical(f"Startup Failed: {e}")
        return

    short_term_history = [] 
    print("\n>>> Model Ready. Type 'exit' to quit, 'files' to list uploaded files.\n")

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")
            if user_input.lower() in ['exit', 'quit']:
                print("Saving and shutting down...")
                break
            
            # NEW: List files command
            if user_input.lower() == 'files':
                print(f"\033[96m{memory.get_uploaded_files_list()}\033[0m")
                continue

            # --- FEATURE: File Ingestion Check ---
            is_upload = brain.detect_file_intent(user_input)
            if is_upload:
                confirm = input("\033[93m[System] Did you mean to upload a file? (y/n): \033[0m")
                if confirm.lower() == 'y':
                    path = input("\033[93m[System] Enter absolute file path: \033[0m").strip('"') 
                    memory.ingest_file(path)
                    short_term_history.append({"role": "system", "content": f"User uploaded file: {os.path.basename(path)}"})
                    continue 
            
            # --- Normal Flow ---
            is_sufficient = brain.check_sufficiency(short_term_history, user_input)
            
            long_term_context = ""
            if not is_sufficient:
                print("\033[90m[Thinking: Searching Long-Term...]\033[0m")
                vector_memories = memory.search_episodic(user_input)
                recent_memories = memory.recall_recent(n_last=5)
                long_term_context = f"{vector_memories}\n\n{recent_memories}"
            
            persistent_profile = memory.get_profile()
            raw_profile_json = memory.get_raw_profile_json()
            
            # ENHANCED: Include uploaded files list in system prompt
            uploaded_files_context = memory.get_uploaded_files_list()

            system_prompt = (
                f"You are a helpful assistant.\n"
                f"--- LONG TERM MEMORY ---\n{long_term_context}\n"
                f"--- AVAILABLE FILES ---\n{uploaded_files_context}\n"
                f"--- INSTRUCTIONS ---\n"
                f"1. Use the memory above to answer questions.\n"
                f"2. If the answer is in a [File: ...] block, use that data.\n"
                f"3. When user asks about files, reference the AVAILABLE FILES list.\n"
                f"4. ALWAYS adhere to the User Profile below for facts about the user.\n"
                f"{persistent_profile}"
            )

            response = brain.generate_response(system_prompt, short_term_history, user_input)
            print(f"\033[92mAI:\033[0m {response}")

            short_term_history.append({"role": "user", "content": user_input})
            short_term_history.append({"role": "assistant", "content": response})
            if len(short_term_history) > HISTORY_LIMIT:
                short_term_history = short_term_history[-HISTORY_LIMIT:]

            memory.save_interaction(user_input, response)

            history_snapshot = copy.deepcopy(short_term_history)
            t = threading.Thread(
                target=background_profile_update,
                args=(brain, memory, raw_profile_json, history_snapshot, user_input, response),
                daemon=True
            )
            t.start()

        except KeyboardInterrupt:
            print("\nShutdown.")
            break
        except Exception as e:
            logger.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()
