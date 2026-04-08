import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

def deploy():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in environment. Please set it in .env first.")
        return

    api = HfApi(token=token)
    
    # Get current user/org
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"Error authenticating with Hugging Face: {e}")
        return

    repo_id = f"{username}/GuardianNet-AI"
    print(f"Target Space: https://huggingface.co/spaces/{repo_id}")

    # Create space if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            private=False,
            exist_ok=True
        )
        print(f"Successfully created or confirmed Space: {repo_id}")
    except Exception as e:
        print(f"Error creating Space: {e}")
        return

    # Upload files
    print("Uploading project files to Spaces...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[".venv", "__pycache__", ".git", "*.log", ".env", "*.pyc", "venv"]
        )
        print("Deployment successful! Your Space is building.")
        print(f"Visit: https://huggingface.co/spaces/{repo_id}")
        print("\nIMPORTANT: Remember to add 'HF_TOKEN' to your Space Settings -> Variables and Secrets.")
    except Exception as e:
        print(f"Error during file upload: {e}")

if __name__ == "__main__":
    deploy()
