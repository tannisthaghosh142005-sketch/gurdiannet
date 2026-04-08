from __future__ import annotations
import numpy as np
import random
from typing import List, Optional
from cae_env.types import (
    Message, MediaContent, MediaType, Language, HarmCategory
)

def generate_message(
    msg_id: int,
    sender_id: int,
    task: str,
    user_count: int,
    rng: np.random.RandomState,
    category: Optional[HarmCategory] = None,
    difficulty: str = "easy"
) -> Message:
    if category is None:
        category = random.choice(list(HarmCategory))
        
    embedding = rng.randn(128).astype(np.float32)
    sig = np.zeros(11)
    cat_list = [c.value for c in HarmCategory]
    idx = cat_list.index(category.value)
    
    # Sharp embeddings for accurate rule-based detection
    sig[idx] = 1.0
    embedding[:11] += sig * (5.0 if difficulty == "easy" else 2.5)
    
    true_harm_per_user = [float(np.clip(0.5 + rng.normal(0, 0.1), 0.0, 1.0)) for _ in range(user_count)]
    
    media_content = MediaContent(
        media_type=MediaType.TEXT,
        language=Language.EN,
        text=f"Sample {category.value} content."
    )
    
    return Message(
        msg_id=msg_id,
        sender_id=sender_id,
        timestamp=float(msg_id * 10),
        contents=[media_content],
        embedding=embedding.tolist(),
        true_harm_per_user=true_harm_per_user,
        ground_truth_type=category,
        is_adversarial=(task == "adversarial"),
        context_dependent=(task == "context"),
        difficulty=difficulty,
        primary_strength=5.0 if difficulty == "easy" else 2.5
    )

def shuffle_episode_categories(task: str, max_steps: int, seed: int) -> List[HarmCategory]:
    rd = random.Random(seed)
    return [rd.choice(list(HarmCategory)) for _ in range(max_steps)]

def difficulty_for_step(step_index: int, seed: int) -> str:
    return "easy" if (step_index + seed) % 2 == 0 else "medium"