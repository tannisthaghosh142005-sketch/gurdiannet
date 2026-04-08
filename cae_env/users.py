from __future__ import annotations
import numpy as np
import random
from typing import List, Optional
from cae_env.types import UserProfile, Language, HarmCategory

class User:
    """User agent with behavioral logic and harm sensitivity."""
    def __init__(self, profile: UserProfile, rng: Optional[np.random.RandomState] = None):
        self.profile = profile
        self._rng = rng or np.random.RandomState()

    def perceived_harm(self, embedding: List[float]) -> float:
        """Predict harm using preference vector dot product."""
        emb = np.array(embedding[:11], dtype=np.float32)
        pref = np.array(self.profile.preference_vector[:11], dtype=np.float32)
        return float(np.clip(np.dot(pref, emb) / (np.linalg.norm(pref) * np.linalg.norm(emb) + 1e-8), 0, 1))

def build_users(num_users: int, rng: np.random.RandomState) -> List[User]:
    users = []
    roles = ["admin", "moderator", "regular", "regular", "new"]
    for i in range(num_users):
        role = roles[i % len(roles)]
        profile = UserProfile(
            user_id=i,
            role=role,
            trust_weight=float(rng.uniform(0.4, 0.9)),
            preference_vector=rng.uniform(0.2, 0.8, 11).tolist(),
            consistency_score=float(rng.uniform(0.6, 0.95))
        )
        users.append(User(profile, rng))
    return users