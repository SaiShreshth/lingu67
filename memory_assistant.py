import os
import sys
import json
import logging
import hashlib
import re
import ast
from datetime import datetime
from typing import List, Dict, Any

# Third-party imports
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- CONFIGURATION ---
# Ensure you have downloaded this exact model file
MODEL_PATH = "qwen2.5-3b-instruct-q4_k_m.gguf" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./memory_db"
PROFILE_PATH = "./user_profile.json"
CHAT_LOG_PATH = "./conversation_log.txt"

# Hardware Settings (Optimized for RTX 3050 4GB)
CTX_SIZE = 4096       # Context Window
GPU_LAYERS = 35       # Offload everything to GPU
HISTORY_LIMIT = 15    # Keep last 15 messages in RAM

# --- LOGGING SETUP ---
# Console gets clean info, File gets raw debug data
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler('system_debug.log', mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("MemorySystem")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class SemanticRouter:
    """CPU-based Router for converting text to vectors."""
    def __init__(self, model_name: str):
        logger.info(f"Loading Router ({model_name}) on CPU...")
        self.model = SentenceTransformer(model_name, device='cpu')

    def encode(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class MemoryManager:
    """Handles Episodic (Vector DB), Semantic (JSON), and Logging (Text)."""
    def __init__(self, db_path: str, profile_path: str, log_path: str, router: SemanticRouter):
        self.router = router
        self.profile_path = profile_path
        self.log_path = log_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="episodic_memory")

    def get_profile(self) -> str:
        """Reads Semantic Memory (Facts)."""
        if not os.path.exists(self.profile_path):
            return "User Profile: No known facts yet."
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
            return f"User Facts & Reminders:\n{json.dumps(data, indent=2)}"
        except:
            return ""

    def update_profile(self, new_facts: Dict[str, str]):
        """Updates Semantic Memory with new facts."""
        data = {}
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
            except: pass
        
        # Merge new facts
        data.update(new_facts)
        
        with open(self.profile_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Log to console so user knows it happened
        print(f"\033[90m[Memory: Learned new fact: {new_facts}]\033[0m")

    def search_episodic(self, query: str, n_results: int = 3) -> str:
        """Searches Long-Term Memory (Vector DB)."""
        try:
            vector = self.router.encode(query)
            results = self.collection.query(query_embeddings=[vector], n_results=n_results)
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            # Deduplicate results
            unique_memories = list(set(results['documents'][0]))
            return "Retrieved Long-Term Memories:\n" + "\n".join([f"- {m}" for m in unique_memories])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ""

    def save_interaction(self, user_input: str, ai_response: str):
        """Saves to Episodic DB and Raw Log file."""
        text_blob = f"User: {user_input} | AI: {ai_response}"
        vector = self.router.encode(text_blob)
        doc_id = hashlib.md5(text_blob.encode()).hexdigest()
        
        # 1. Vector DB
        self.collection.add(documents=[text_blob], embeddings=[vector], ids=[doc_id])
        
        # 2. Text Log
        with open(self.log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\nUser: {user_input}\nAI: {ai_response}\n{'-'*20}\n")

class Brain:
    """The Intelligence (runs on GPU)."""
    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int):
        if not os.path.exists(model_path):
            logger.critical(f"Model not found at {model_path}! Download the GGUF file.")
            sys.exit(1)
        
        logger.info("Loading Brain on RTX GPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            verbose=False # Reduce C++ spam
        )

    def extract_json_from_text(self, text: str) -> Dict:
        """
        Robustly extracts JSON object by finding the first valid { } block.
        Handles cases where LLM adds extra text like 'Here is the json:'.
        """
        try:
            # Look for { ... } structure across multiple lines
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                candidate = match.group(0)
                try:
                    # Try standard JSON (double quotes)
                    return json.loads(candidate)
                except:
                    # Try Python Literal (single quotes - common in 3B models)
                    try:
                        return ast.literal_eval(candidate)
                    except:
                        pass
            return {}
        except:
            return {}

    def check_sufficiency(self, history: List[Dict], query: str) -> bool:
        """
        Decides if we need to search the database.
        """
        # 1. HEURISTIC OVERRIDE (The "Paranoid" Check)
        # If history is empty, ALWAYS search unless it's a greeting/statement.
        # 3B models are bad at judging 'sufficiency' on turn 0.
        if not history:
            # If it looks like a question, force search
            if "?" in query or any(w in query.lower() for w in ["what", "who", "where", "when", "how", "remind", "recall"]):
                logger.info("First turn question detected. Forcing search.")
                return False # False means "Not Sufficient" -> Search DB

        # 2. LLM CHECK (For ongoing conversation)
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-4:]])
        
        prompt = (
            f"Chat History:\n{history_text}\n\n"
            f"User Input: '{query}'\n\n"
            f"Task: Does the Chat History above ALREADY contain the answer to the User's input?\n"
            f"Reply 'YES' if the answer is right there in the text.\n"
            f"Reply 'NO' if you need to check long-term memory or external knowledge.\n"
            f"Answer (YES/NO):"
        )
        
        output = self.llm.create_completion(prompt, max_tokens=3, temperature=0.0)
        decision = output['choices'][0]['text'].strip().upper()
        
        logger.debug(f"Sufficiency Decision: {decision}")
        
        # Only skip search if the model is 100% sure it has the answer in RAM
        if "YES" in decision:
            return True
        return False

    def extract_facts(self, user_input: str) -> Dict[str, str]:
        """Extracts User Facts (Name, Job) or Tasks as JSON."""
        prompt = (
            f"Input: '{user_input}'\n"
            f"Task: Extract User Facts (name, job, preferences) or Tasks/Reminders as JSON.\n"
            f"If no new facts found, return {{}}.\n"
            f"Format: {{\"key\": \"value\"}}\n"
            f"JSON:"
        )
        output = self.llm.create_completion(prompt, max_tokens=60, temperature=0.0)
        text = output['choices'][0]['text'].strip()
        logger.debug(f"Raw Extraction Output: {text}")
        return self.extract_json_from_text(text)

    def generate_response(self, system_context: str, history: List[Dict], query: str) -> str:
        """Generates the final answer."""
        messages = [{"role": "system", "content": system_context}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        return output['choices'][0]['message']['content']

def main():
    print("=== Booting Up Memory System v2.2 (Final) ===")
    
    # Initialize components
    try:
        router = SemanticRouter(EMBEDDING_MODEL)
        memory = MemoryManager(DB_PATH, PROFILE_PATH, CHAT_LOG_PATH, router)
        brain = Brain(MODEL_PATH, CTX_SIZE, GPU_LAYERS)
    except Exception as e:
        logger.critical(f"Startup Failed: {e}")
        return

    short_term_history = [] 

    print("\n>>> Model Ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")
            if user_input.lower() in ['exit', 'quit']:
                print("Saving and shutting down...")
                break

            # --- STEP 1: CONTEXT CHECK ---
            # Does the AI know enough from Short-Term RAM?
            # We check this even on the first turn to catch "Statements" vs "Questions"
            is_sufficient = brain.check_sufficiency(short_term_history, user_input)
            
            # --- STEP 2: CONDITIONAL RETRIEVAL ---
            long_term_context = ""
            if not is_sufficient:
                print("\033[90m[Thinking: Searching Long-Term Memory...]\033[0m")
                long_term_context = memory.search_episodic(user_input)
            
            persistent_profile = memory.get_profile()

            # --- STEP 3: GENERATION ---
            system_prompt = (
                f"You are an intelligent assistant with persistent memory.\n"
                f"{persistent_profile}\n"
                f"{long_term_context}\n"
                f"INSTRUCTIONS: Answer based on the Context and History above. "
                f"If you don't know, say you don't know."
            )

            response = brain.generate_response(system_prompt, short_term_history, user_input)
            print(f"\033[92mAI:\033[0m {response}")

            # --- STEP 4: UPDATE & CONSOLIDATE ---
            # A. Update Short Term (RAM)
            short_term_history.append({"role": "user", "content": user_input})
            short_term_history.append({"role": "assistant", "content": response})
            
            # Keep RAM lean (Rolling Window)
            if len(short_term_history) > HISTORY_LIMIT:
                short_term_history = short_term_history[-HISTORY_LIMIT:]

            # B. Save to Long Term (Disk)
            memory.save_interaction(user_input, response)

            # C. Extract Facts (Silent Background Process)
            new_facts = brain.extract_facts(user_input)
            if new_facts:
                memory.update_profile(new_facts)

        except KeyboardInterrupt:
            print("\nShutdown.")
            break
        except Exception as e:
            logger.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()