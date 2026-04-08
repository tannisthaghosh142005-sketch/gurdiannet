from typing import Dict
from cae_env.types import Message, HarmCategory
import random

class MediaClassifier:
    """Advanced simulation of image/video safety analysis for forensics dashboard."""
    
    def analyze_media(self, filename: str, content_type: str = "image") -> Dict[str, float]:
        """Classifies media based on simulated heuristics for dashboard demo."""
        low_file = filename.lower()
        
        # Initializing Detection Scores
        scores = {
            "benign": 0.05,
            "synthetic": 0.05,
            "illegal": 0.05,
            "deepfake": 0.05,
            "violence": 0.05,
            "nsfw": 0.05
        }
        
        # 1. Detection: Synthetic / AI-Generated / Deepfake
        synthetic_keywords = ["fake", "ai", "gen", "midjourney", "stable", "dall-e", "synthetic"]
        if any(kw in low_file for kw in synthetic_keywords):
            scores["synthetic"] = 0.96
            scores["deepfake"] = 0.89
            scores["benign"] = 0.02
            
        # 2. Detection: Illegal / High-Risk (CSAM, Violence, NSFW)
        illegal_keywords = ["illegal", "gore", "weapon", "nsfw", "blood", "abuse", "threat"]
        elif any(kw in low_file for kw in illegal_keywords):
            scores["illegal"] = 0.99
            scores["violence"] = 0.94
            scores["nsfw"] = 0.88
            scores["benign"] = 0.01
            
        # 3. Default: Normal / Benign
        else:
            scores["benign"] = 0.94
            
        return scores

    def analyze_image(self, message: Message) -> Dict[str, float]:
        """Gym environment simulation using ground truth (for evaluation)."""
        is_high_risk = message.ground_truth_type in (HarmCategory.CSAM, HarmCategory.DEEPFAKE)
        confidence = 0.92 if is_high_risk else 0.12
        return {
            "nsfw_score": confidence if message.ground_truth_type == HarmCategory.CSAM else 0.05,
            "deepfake_score": confidence if message.ground_truth_type == HarmCategory.DEEPFAKE else 0.05
        }

    def analyze_video(self, message: Message) -> Dict[str, float]:
        """Gym environment simulation using ground truth (for evaluation)."""
        return self.analyze_image(message)
