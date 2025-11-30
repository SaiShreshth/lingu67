from llama_cpp import Llama
import os

MODEL_PATH = "qwen2.5-3b-instruct-q4_k_m.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"File not found: {MODEL_PATH}")
else:
    print(f"File found. Size: {os.path.getsize(MODEL_PATH)} bytes")
    try:
        print("Attempting to load model with verbose=True...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=35, # Try with GPU
            verbose=True
        )
        print("Success!")
    except Exception as e:
        print(f"Error loading model: {e}")
