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

class MemoryManager:
    def __init__(self, db_path: str, profile_path: str, log_path: str, router: SemanticRouter):
        self.router = router
        self.profile_path = profile_path
        self.log_path = log_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="episodic_memory")

    def get_profile(self) -> str:
        if not os.path.exists(self.profile_path):
            return "User Profile: No known facts yet."
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
            return f"User Facts & Reminders:\n{json.dumps(data, indent=2)}"
        except:
            return ""

    def update_profile(self, new_facts: Dict[str, str]):
        data = {}
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
            except: pass
        
        data.update(new_facts)
        
        with open(self.profile_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\033[90m[Memory: Learned new fact: {new_facts}]\033[0m")

    def search_episodic(self, query: str, n_results: int = 5) -> str:
        """Searches Vector DB for SEMANTICALLY similar memories."""
        try:
            vector = self.router.encode(query)
            results = self.collection.query(query_embeddings=[vector], n_results=n_results)
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            unique_memories = list(set(results['documents'][0]))
            return "Relevant Past Memories (Vector Search):\n" + "\n".join([f"- {m}" for m in unique_memories])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ""

    def recall_recent(self, n_last: int = 5) -> str:
        """Reads the raw log file to get the CHRONOLOGICAL last 5 interactions."""
        if not os.path.exists(self.log_path):
            return ""
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by the separator line
            interactions = content.split('-'*20)
            # Filter empty strings
            interactions = [i.strip() for i in interactions if i.strip()]
            
            # Get last N
            recent = interactions[-n_last:]
            
            if not recent:
                return ""
                
            return "Recent Conversation History (Chronological):\n" + "\n---\n".join(recent)
        except Exception as e:
            logger.error(f"Log read failed: {e}")
            return ""

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

class Brain:
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
                candidate = match.group(0)
                try: return json.loads(candidate)
                except:
                    try: return ast.literal_eval(candidate)
                    except: pass
            return {}
        except:
            return {}

    def check_sufficiency(self, history: List[Dict], query: str) -> bool:
        if not history:
            # Always search on startup to prevent amnesia
            if "?" in query or any(w in query.lower() for w in ["what", "who", "where", "remind", "recall"]):
                return False 
            return True

        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-4:]])
        prompt = (
            f"Chat History:\n{history_text}\n"
            f"User Input: '{query}'\n\n"
            f"Task: Do we need to search the Database to answer this?\n"
            f"- If the answer is ALREADY in the Chat History above, reply NO.\n"
            f"- If the user asks for old information NOT in history, reply YES.\n\n"
            f"Search Database? (YES/NO):"
        )
        output = self.llm.create_completion(prompt, max_tokens=4, temperature=0.0)
        decision = output['choices'][0]['text'].strip().upper()
        
        if "NO" in decision: return True
        return False

    def extract_facts(self, user_input: str) -> Dict[str, str]:
        prompt = (
            f"Input: '{user_input}'\n"
            f"Task: Extract User Facts (name, job) or Tasks/Reminders as JSON.\n"
            f"Format: {{\"key\": \"value\"}}\n"
            f"JSON:"
        )
        output = self.llm.create_completion(prompt, max_tokens=60, temperature=0.0)
        return self.extract_json_from_text(output['choices'][0]['text'].strip())

    def generate_response(self, system_context: str, history: List[Dict], query: str) -> str:
        messages = [{"role": "system", "content": system_context}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        output = self.llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7)
        return output['choices'][0]['message']['content']

def main():
    print("=== Booting Up Memory System v2.4 (Hybrid Recall) ===")
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

            is_sufficient = brain.check_sufficiency(short_term_history, user_input)
            
            long_term_context = ""
            if not is_sufficient:
                print("\033[90m[Thinking: Searching Long-Term + Recent Logs...]\033[0m")
                # 1. Get Semantic Matches (Vector)
                vector_memories = memory.search_episodic(user_input)
                # 2. Get Chronological Matches (Last 5 lines from file)
                recent_memories = memory.recall_recent(n_last=5)
                
                long_term_context = f"{vector_memories}\n\n{recent_memories}"
            
            persistent_profile = memory.get_profile()

            system_prompt = (
                f"You are an intelligent assistant with persistent memory.\n"
                f"{persistent_profile}\n"
                f"{long_term_context}\n"
                f"INSTRUCTIONS: Answer based on the Context and History above. "
                f"If the answer is in 'Recent Conversation History', use that."
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