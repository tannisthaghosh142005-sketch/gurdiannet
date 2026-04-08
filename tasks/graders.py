from __future__ import annotations
import numpy as np
from typing import Dict, Any

HARMFUL_GT = {"hate_speech", "harassment", "misinformation", "csam", "radicalization", "self_harm", "spam", "deepfake", "doxxing"}

def grade_basic(episode_data: Dict[str, Any]) -> float:
    steps = episode_data.get("steps", [])
    if not steps: return 0.0
    correct = 0
    total = 0
    for s in steps:
        gt = s.get("ground_truth_type")
        action = int(s.get("action", 0))
        if gt == "benign":
            if action == 0: correct += 1
            total += 1
        elif gt in HARMFUL_GT:
            if action > 0: correct += 1
            total += 1
    return float(correct / total if total > 0 else 1.0)

def grade_context(episode_data: Dict[str, Any]) -> float:
    steps = episode_data.get("steps", [])
    if not steps: return 0.0
    amb_steps = [s for s in steps if s.get("ground_truth_type") == "ambiguous"]
    if not amb_steps: return 1.0
    cautious = sum(1 for s in amb_steps if int(s.get("action", 0)) == 1)
    return float(cautious / len(amb_steps))

def grade_adversarial(episode_data: Dict[str, Any]) -> float:
    steps = episode_data.get("steps", [])
    if not steps: return 0.0
    harmful = [s for s in steps if s.get("ground_truth_type") in HARMFUL_GT]
    h_red = sum(1 for s in harmful if int(s.get("action", 0)) > 0) / (len(harmful) + 1e-8)
    benign = [s for s in steps if s.get("ground_truth_type") == "benign"]
    fpr = sum(1 for s in benign if int(s.get("action", 0)) > 0) / (len(benign) + 1e-8)
    health = float(episode_data.get("final_group_health", 0.5))
    score = (h_red * 0.4) + ((1.0 - fpr) * 0.2) + (health * 0.4)
    return float(np.clip(score, 0.0, 1.0))
