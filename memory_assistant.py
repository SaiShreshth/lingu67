import os
import sys
import json
import logging
import hashlib
import re  # Added for robust JSON parsing
from datetime import datetime
from typing import List, Dict, Any

# Third-party imports
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- CONFIGURATION ---
MODEL_PATH = "qwen2.5-3b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./memory_db"
PROFILE_PATH = "./user_profile.json"
CHAT_LOG_PATH = "./conversation_log.txt"
CTX_SIZE = 4096
GPU_LAYERS = 35
HISTORY_LIMIT = 10  # Short-Term Memory Size

# --- LOGGING ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("MemorySystem")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

class SemanticRouter:
    """CPU-based Router for converting text to vectors."""
    def __init__(self, model_name: str):
        logger.info(f"Loading Router ({model_name})...")
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
            return f"User Profile Facts:\n{json.dumps(data, indent=2)}"
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
        logger.info(f"Updated Profile with {len(new_facts)} new facts.")

    def search_episodic(self, query: str, n_results: int = 3) -> str:
        """Searches Long-Term Memory (Vector DB)."""
        try:
            vector = self.router.encode(query)
            results = self.collection.query(query_embeddings=[vector], n_results=n_results)
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            # Simple deduplication
            unique_memories = list(set(results['documents'][0]))
            return "Retrieved Long-Term Memories:\n" + "\n".join([f"- {m}" for m in unique_memories])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ""

    def save_interaction(self, user_input: str, ai_response: str):
        """Saves to Episodic DB and Raw Log file."""
        # 1. Save to Vector DB
        text_blob = f"User: {user_input} | AI: {ai_response}"
        vector = self.router.encode(text_blob)
        doc_id = hashlib.md5(text_blob.encode()).hexdigest()
        self.collection.add(documents=[text_blob], embeddings=[vector], ids=[doc_id])
        
        # 2. Save to Raw Text Log
        with open(self.log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\nUser: {user_input}\nAI: {ai_response}\n{'-'*20}\n")

class Brain:
    """The Intelligence (runs on GPU)."""
    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int):
        if not os.path.exists(model_path):
            logger.critical("Model not found! Download the .gguf file.")
            sys.exit(1)
        
        logger.info("Loading Brain on RTX 3050...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            verbose=False # Keep it quiet
        )

    def check_sufficiency(self, history: List[Dict], query: str) -> bool:
        """
        Step 1: The 'Reflective' Check. 
        Decides if Short-Term memory is enough or if we need Long-Term retrieval.
        """
        # Create a mini-prompt for the check
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        prompt = (
            f"Chat History:\n{history_text}\n"
            f"User's New Query: '{query}'\n\n"
            f"Task: Does the Chat History above contain enough context to answer the query? "
            f"Or does the query refer to old/unknown information?\n"
            f"Reply with exactly one word: 'SUFFICIENT' or 'SEARCH'."
        )
        
        output = self.llm.create_completion(prompt, max_tokens=5, temperature=0.1)
        decision = output['choices'][0]['text'].strip().upper()
        
        logger.debug(f"Context Check Decision: {decision}")
        return "SUFFICIENT" in decision

    def extract_facts(self, user_input: str) -> Dict[str, str]:
        """
        Step 4: The 'Profiling' Step.
        Extracts facts like names, projects, or preferences into JSON.
        """
        prompt = (
            f"Analyze this sentence: '{user_input}'\n"
            f"If the user mentions a specific fact (name, preference, project, location), "
            f"extract it as a valid JSON object. If no fact, return {{}}.\n"
            f"Example: 'My name is Bob' -> {{\"user_name\": \"Bob\"}}\n"
            f"JSON:"
        )
        output = self.llm.create_completion(prompt, max_tokens=128, temperature=0.1)
        text = output['choices'][0]['text'].strip()
        
        # Debug log to see what the model actually output
        logger.debug(f"Fact Extraction Raw Output: {text}")

        try:
            # Robust Regex to find JSON object { ... }
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON Parsing failed: {e}")
        
        return {}

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
    print("=== Booting Up Conditional Memory Model ===")
    
    # Initialize components
    router = SemanticRouter(EMBEDDING_MODEL)
    memory = MemoryManager(DB_PATH, PROFILE_PATH, CHAT_LOG_PATH, router)
    brain = Brain(MODEL_PATH, CTX_SIZE, GPU_LAYERS)
    
    # Short-Term Memory (RAM)
    short_term_history = [] 

    print("\n>>> Model Ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")
            if user_input.lower() in ['exit', 'quit']:
                break

            # --- STEP 1: CONTEXT CHECK ---
            # FIX: Default to False (SEARCH) if history is empty
            is_sufficient = False
            if len(short_term_history) > 0:
                is_sufficient = brain.check_sufficiency(short_term_history, user_input)
            
            long_term_context = ""
            
            # --- STEP 2: CONDITIONAL RETRIEVAL ---
            if not is_sufficient:
                print("\033[90m[Thinking: Recalling past memories...]\033[0m")
                long_term_context = memory.search_episodic(user_input)
            else:
                pass 

            persistent_profile = memory.get_profile()

            # --- STEP 3: GENERATION ---
            # Construct the "Brain's Workspace"
            system_prompt = (
                f"You are an intelligent assistant with persistent memory.\n"
                f"{persistent_profile}\n"
                f"{long_term_context}\n"
                f"INSTRUCTIONS: Use the Profile and Retrieved Memories to answer. "
                f"Prioritize the Short-Term Chat History below for immediate context."
            )

            response = brain.generate_response(system_prompt, short_term_history, user_input)
            print(f"\033[92mAI:\033[0m {response}")

            # --- STEP 4: MEMORY UPDATE (Consolidation) ---
            # A. Update Short Term (Rolling Window)
            short_term_history.append({"role": "user", "content": user_input})
            short_term_history.append({"role": "assistant", "content": response})
            if len(short_term_history) > HISTORY_LIMIT:
                short_term_history = short_term_history[-HISTORY_LIMIT:]

            # B. Save to Long Term (Disk)
            memory.save_interaction(user_input, response)

            # C. Intelligent Profiling (Extract Facts)
            print("\033[90m[Thinking: Updating profile...]\033[0m")
            new_facts = brain.extract_facts(user_input)
            if new_facts:
                memory.update_profile(new_facts)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()