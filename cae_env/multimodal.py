import numpy as np
from typing import Dict

def extract_harm_probs(obs: np.ndarray, num_users: int) -> Dict[str, float]:
    """
    Extracts harm probabilities from the OmniAlignEnv observation vector.
    
    Observation Layout:
    current_emb (128) + trust_weights (num_users) + history (640) 
    + media (4) + lang (15) + harm (11) + scalars (3) + tail (5)
    """
    emb_dim = 128
    # Calculate offset
    # 128 (current) + num_users + 640 (history) + 4 (media) + 15 (lang) = 787 + num_users
    harm_start = emb_dim + num_users + 5 * emb_dim + 4 + 15
    harm_probs = obs[harm_start: harm_start + 11]
    
    harm_cats = [
        "hate_speech", "harassment", "misinformation", "csam", "radicalization",
        "self_harm", "spam", "deepfake", "doxxing", "benign", "ambiguous"
    ]
    
    return {c: float(p) for c, p in zip(harm_cats, harm_probs)}
