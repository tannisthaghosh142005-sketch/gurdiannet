from __future__ import annotations
import numpy as np
from typing import List
from cae_env.users import User
from cae_env.types import Message

class NegotiationEngine:
    """Predicts group disagreement and facilitates weighted voting."""
    def predict_disagreement(self, message: Message, users: List[User]) -> float:
        harms = [u.perceived_harm(message.embedding) for u in users]
        return float(np.std(harms))

    def weighted_vote(self, message: Message, users: List[User]) -> int:
        total_trust = sum(u.profile.trust_weight for u in users)
        weighted_harm = sum(u.perceived_harm(message.embedding) * u.profile.trust_weight for u in users) / (total_trust + 1e-8)
        if weighted_harm > 0.8: return 2
        if weighted_harm > 0.4: return 1
        return 0
