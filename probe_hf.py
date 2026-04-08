import os
import sys
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def probe_model():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found.")
        return

    model = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Probing model: {model}")
    
    # Try without explicit provider first to see what HF recommends
    client_auto = InferenceClient(token=token)
    try:
        # Just a very small call
        print("Attempting with provider='auto'...")
        res = client_auto.text_generation(model=model, prompt="hi", max_new_tokens=1)
        print(f"Success with auto! Response: {res}")
    except Exception as e:
        print(f"Failed with auto: {e}")

    # Try with hf-inference
    client_hf = InferenceClient(provider="hf-inference", token=token)
    try:
        print("Attempting with provider='hf-inference'...")
        res = client_hf.text_generation(model=model, prompt="hi", max_new_tokens=1)
        print(f"Success with hf-inference! Response: {res}")
    except Exception as e:
        print(f"Failed with hf-inference: {e}")

if __name__ == "__main__":
    probe_model()
