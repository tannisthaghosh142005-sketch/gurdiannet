import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def debug_client():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    model = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"DEBUG: Initializing client with provider='hf-inference'")
    client = InferenceClient(provider="hf-inference", token=token)
    
    try:
        print("DEBUG: Calling text_generation...")
        res = client.text_generation(model=model, prompt="hi", max_new_tokens=1)
        print(f"DEBUG: Result: {res}")
    except Exception as e:
        print(f"DEBUG: FAILED: {e}")

if __name__ == "__main__":
    debug_client()
