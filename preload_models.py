"""
Preload script for GuardianNet models (optional).
Used during Docker build to cache models for the media classifier.
This is not required for the core OpenEnv evaluation.
"""
import sys
import os

def preload():
    print("Preloading GuardianNet models...")
    # Attempt to load NSFW detector (public, no authentication needed)
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print("Loading NSFW detector...")
        processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        print("NSFW detector loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not preload NSFW detector: {e}")
        print("The media classifier will still work (fallback to normal classification).")
    
    # Deepfake detection is disabled (no reliable public model)
    print("Preload finished (optional models).")

if __name__ == "__main__":
    preload()
