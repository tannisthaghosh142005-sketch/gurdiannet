import numpy as np
from typing import Dict, Any, List
from cae_env.types import HarmCategory

def compute_group_health(conflict_rate: float, retention_rate: float, escalation_rate: float, silence_cost: float) -> float:
    health = (retention_rate * 0.4) + (1.0 - conflict_rate * 0.3) + (1.0 - escalation_rate * 0.2) - (silence_cost * 0.1)
    return float(np.clip(health, 0.0, 1.0))

def compute_reward(action: int, gt_type: HarmCategory, group_health_delta: float, friction_occurred: bool, was_overridden: bool) -> float:
    reward = group_health_delta * 10
    if gt_type == HarmCategory.BENIGN:
        reward += 1.0 if action == 0 else -1.0
    elif gt_type != HarmCategory.AMBIGUOUS:
        reward += 1.0 if action > 0 else -2.0
    else:
        reward += 0.5 if action == 1 else 0.0
    if friction_occurred: reward -= 0.2
    if was_overridden: reward -= 0.5
    return float(reward)
