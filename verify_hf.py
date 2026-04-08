import os
import sys
from dotenv import load_dotenv

# Ensure local imports work
sys.path.append(os.getcwd())

from inference import generate_text

def test_inference():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[SKIP] No HF_TOKEN found in .env. Skipping real inference test.")
        return

    print(f"Testing HF Inference with token: {token[:4]}...")
    try:
        prompt = "Hello, how are you today?"
        response = generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("[SUCCESS] Inference completed successfully.")
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")

if __name__ == "__main__":
    test_inference()
