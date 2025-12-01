import os
import sys
import json
import logging
import hashlib
import re
import ast
from datetime import datetime
from typing import List, Dict, Any, Set

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
HISTORY_LIMIT = 15

# --- LOGGING ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("MemorySystem")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

class SemanticRouter:
    def __init__(self, model_name: str):
        print(f"Loading Router ({model_name})...")
        self.model = SentenceTransformer(model_name, device='cpu')

    def encode(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class Brain:
    """The Intelligence (runs on GPU)."""
    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int):
        if not os.path.exists(model_path):
            logger.critical(f"Model not found at {model_path}!")
            sys.exit(1)
        logger.info("Loading Brain on RTX 3050...")
        self.llm = Llama(model_path=model_path, n_ctx=ctx_size, n_gpu_layers=gpu_layers, verbose=False)

    def extract_json_from_text(self, text: str) -> Dict:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(0))
                except: return ast.literal_eval(match.group(0))
            return {}
        except: return {}

    def extract_keywords(self, text_chunk: str) -> str:
        """Layer 2/3 Helper: Extracts search terms from found memories."""
        prompt = (
            f"Text: \"{text_chunk[:500]}\"\n" # Limit chunk size for speed
            f"Task: Extract 3-5 key entities, codes, or specific terms from the text above to use as a search query.\n"
            f"Output ONLY the keywords separated by spaces."
        )
        output = self.llm.create_completion(prompt, max_tokens=32, temperature=0.0)
        return output['choices'][0]['text'].strip()

    def check_sufficiency(self, history: List[Dict], query: str) -> bool:
        if not history:
            if "?" in query or any(w in query.lower() for w in ["what", "who", "where", "code", "value", "secret"]):
                return False 
            return True

        prompt = (
            f"User Input: '{query}'\n"
            f"Task: Does the AI need to search its long-term database to answer this?\n"
            f"- If it's a greeting or statement -> NO.\n"
            f"- If the answer is in the recent chat history -> NO.\n"
            f"- If it asks about old facts, codes, or history -> YES.\n"
            f"Answer (YES/NO):"
        )
        output = self.llm.create_completion(prompt, max_tokens=3, temperature=0.0)
        if "YES" in output['choices'][0]['text'].strip().upper():
            return False 
        return True 

    def extract_facts(self, user_input: str) -> Dict[str, str]:
        sys_prompt = "You are a Memory Manager. Extract user facts (name, job) as JSON. If none, return {}."
        output = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_input}],
            max_tokens=128, temperature=0.0
        )
        return self.extract_json_from_text(output['choices'][0]['message']['content'])

    def generate_response(self, system_context: str, history: List[Dict], query: str) -> str:
        messages = [{"role": "system", "content": system_context}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        output = self.llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7)
        return output['choices'][0]['message']['content']

class MemoryManager:
    def __init__(self, db_path: str, profile_path: str, log_path: str, router: SemanticRouter, brain: Brain):
        self.router = router
        self.brain = brain
        self.profile_path = profile_path
        self.log_path = log_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="episodic_memory")
        
        if not os.path.exists(self.profile_path):
            with open(self.profile_path, 'w') as f: json.dump({}, f)

    def get_profile(self) -> str:
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
            if not data: return "User Profile: Empty."
            return f"User Facts & Reminders:\n{json.dumps(data, indent=2)}"
        except: return ""

    def update_profile(self, new_facts: Dict[str, str]):
        data = {}
        try:
            with open(self.profile_path, 'r') as f: data = json.load(f)
        except: pass
        data.update(new_facts)
        with open(self.profile_path, 'w') as f: json.dump(data, f, indent=2)
        print(f"\033[90m[Memory: Learned new fact: {new_facts}]\033[0m")

    def search_layer(self, query_text: str, n_results: int = 5) -> List[str]:
        """Helper: Performs a single vector search."""
        try:
            vector = self.router.encode(query_text)
            results = self.collection.query(query_embeddings=[vector], n_results=n_results)
            if results['documents'] and results['documents'][0]:
                return results['documents'][0]
        except Exception as e:
            logger.error(f"Layer Search Error: {e}")
        return []

    def search_episodic(self, user_query: str) -> str:
        """
        3-Layer Deep Search Algorithm
        """
        unique_memories: Set[str] = set()
        
        # --- Layer 1: Direct User Query ---
        print("\033[90m  -> Layer 1: Searching User Input...\033[0m")
        layer1_results = self.search_layer(user_query, n_results=5)
        unique_memories.update(layer1_results)
        
        # Combine L1 results to extract keywords
        if layer1_results:
            l1_context = " ".join(layer1_results)
            
            # --- Layer 2: Derived Keywords ---
            keywords_l2 = self.brain.extract_keywords(l1_context)
            print(f"\033[90m  -> Layer 2: Searching Keywords '{keywords_l2}'...\033[0m")
            layer2_results = self.search_layer(keywords_l2, n_results=5)
            unique_memories.update(layer2_results)
            
            if layer2_results:
                l2_context = " ".join(layer2_results)
                
                # --- Layer 3: Deep Dive ---
                keywords_l3 = self.brain.extract_keywords(l2_context)
                print(f"\033[90m  -> Layer 3: Searching Keywords '{keywords_l3}'...\033[0m")
                layer3_results = self.search_layer(keywords_l3, n_results=5)
                unique_memories.update(layer3_results)

        # Final Processing
        print(f"\033[93m[Deep Search: Found {len(unique_memories)} unique memories]\033[0m")
        
        if not unique_memories:
            return "No relevant memories found."
            
        return "Deep Search Memories:\n" + "\n".join([f"- {m}" for m in unique_memories])

    def recall_recent(self, n_last: int = 5) -> str:
        if not os.path.exists(self.log_path): return ""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            interactions = [i.strip() for i in content.split('-'*20) if i.strip()]
            return "Recent Conversation History (Chronological):\n" + "\n---\n".join(interactions[-n_last:])
        except: return ""

    def save_interaction(self, user_input: str, ai_response: str):
        text_blob = f"User: {user_input} | AI: {ai_response}"
        vector = self.router.encode(text_blob)
        doc_id = hashlib.md5(text_blob.encode()).hexdigest()
        self.collection.add(documents=[text_blob], embeddings=[vector], ids=[doc_id])
        try: self.client.heartbeat() 
        except: pass
        with open(self.log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\nUser: {user_input}\nAI: {ai_response}\n{'-'*20}\n")

def main():
    print("=== Booting Up Memory System v2.8 (3-Layer Deep Search) ===")
    try:
        router = SemanticRouter(EMBEDDING_MODEL)
        brain = Brain(MODEL_PATH, CTX_SIZE, GPU_LAYERS)
        # Pass brain to memory manager for keyword extraction
        memory = MemoryManager(DB_PATH, PROFILE_PATH, CHAT_LOG_PATH, router, brain)
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

            is_sufficient = brain.check_sufficiency(short_term_history, user_input)
            
            long_term_context = ""
            if not is_sufficient:
                print("\033[90m[Thinking: Starting 3-Layer Deep Search...]\033[0m")
                vector_memories = memory.search_episodic(user_input)
                recent_memories = memory.recall_recent(n_last=5)
                long_term_context = f"{vector_memories}\n\n{recent_memories}"
            
            persistent_profile = memory.get_profile()

            system_prompt = (
                f"You are a helpful assistant.\n"
                f"--- LONG TERM MEMORY ---\n{long_term_context}\n"
                f"--- USER PROFILE ---\n{persistent_profile}\n"
                f"--- INSTRUCTIONS ---\n"
                f"1. Use the Deep Search Memories to find specific codes or facts.\n"
                f"2. Prioritize the Chat History below if it contradicts older memories."
            )

            response = brain.generate_response(system_prompt, short_term_history, user_input)
            print(f"\033[92mAI:\033[0m {response}")

            short_term_history.append({"role": "user", "content": user_input})
            short_term_history.append({"role": "assistant", "content": response})
            if len(short_term_history) > HISTORY_LIMIT:
                short_term_history = short_term_history[-HISTORY_LIMIT:]

            memory.save_interaction(user_input, response)

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