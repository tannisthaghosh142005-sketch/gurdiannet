import os
import sys
import re
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import huggingface_hub
from openai import OpenAI
from inference import generate_text

# Dashboard Specific Constants
ACTION_NAMES = ["allow", "flag", "delete", "warn", "quarantine", "escalate"]
ACTION_COLOR = ["#2ecc71", "#f1c40f", "#e74c3c", "#3498db", "#9b59b6", "#e67e22"]

# ─────────────────────────────────────────────
# Core Utilities
# ─────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_embeddings(text: str = None, image=None, audio=None, video=None) -> np.ndarray:
    dim = 64
    embeds = []
    if text:
        h = hash(text) % 1000000
        rng = np.random.RandomState(h)
        embeds.append(rng.randn(dim))
    if image is not None:
        embeds.append(np.random.randn(dim))
    if audio is not None:
        embeds.append(np.random.randn(dim))
    if video is not None:
        embeds.append(np.random.randn(dim))
    if not embeds:
        return np.zeros(dim)
    return np.mean(embeds, axis=0)

# ─────────────────────────────────────────────
# Dashboard Helpers
# ─────────────────────────────────────────────

def _gt_str(gt: Any) -> str:
    if gt is None: return "benign"
    if hasattr(gt, "value"): return str(gt.value)
    return str(gt)

def risk_numeric(risk_str: str) -> float:
    mapping = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.9}
    return mapping.get(str(risk_str).upper(), 0.1)

def calculate_risk(probs: Dict[str, float]) -> str:
    max_val = max(probs.values(), default=0)
    if max_val > 0.8: return "HIGH"
    if max_val > 0.4: return "MEDIUM"
    return "LOW"

def get_router_config():
    return {
        "base_url": os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"),
        "model": os.getenv("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
        "token": os.getenv("HF_TOKEN") or (huggingface_hub.get_token() if "huggingface_hub" in sys.modules else None)
    }

def validate_hf_token(token: str = None) -> Tuple[bool, str]:
    token = token or os.getenv("HF_TOKEN")
    if not token or len(token) < 10:
        return False, "HF_TOKEN not found or invalid in environment."
    try:
        from openai import OpenAI
        base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
        model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
        client = OpenAI(base_url=base_url, api_key=token)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        return True, ""
    except Exception as e:
        return False, str(e)

def keyword_threat_detection(text: str) -> Tuple[int, str]:
    from cae_env.text_threat import THREAT_PATTERNS
    text_lower = text.lower()
    for cat, pattern in THREAT_PATTERNS.items():
        if re.search(pattern, text_lower):
            return 2, f"Keyword match: {cat}"
    return 0, "No keywords found"

def episode_to_trace_list(episode_log: List[Dict]) -> List[Dict]:
    traces = []
    for log in episode_log:
        trace = {
            "step": log.get("step", 0),
            "action": ACTION_NAMES[min(int(log.get("action", 0)), 5)],
            "reason": log.get("reason", "Standard policy"),
            "ground_truth": _gt_str(log.get("ground_truth_type", "benign")),
            "reward": round(float(log.get("reward", 0.0)), 3),
            "risk": log.get("risk_level", "LOW"),
            "harm_probs": log.get("harm_probabilities", {})
        }
        traces.append(trace)
    return traces

def format_step_trace_console(
    step: int,
    inputs: Dict,
    all_probs: Dict,
    risk: str,
    ai_act: str,
    ai_expl: str,
    safe_act: str,
    safe_expl: str,
    final_act: str,
    reward: float,
    llm_used: bool
) -> str:
    """Produces a rich console-style trace for the dashboard."""
    top_cat = max(all_probs, key=all_probs.get) if all_probs else "unknown"
    top_p = all_probs.get(top_cat, 0.0)
    
    trace = [
        f"─── STEP {step:02d} ─────────────────────────────────",
        f"INPUT    | Category: {top_cat.upper()} ({top_p:.4f})",
        f"ANALYSIS | Risk: {risk.upper()} | LLM Used: {llm_used}",
        f"AI DECIDE| Action: {ai_act.upper()} | Why: {ai_expl[:60]}...",
        f"SAFETY   | Action: {safe_act.upper()} | Why: {safe_expl[:60]}...",
        f"FINAL    | Action: {final_act.upper()} | Reward: {reward:+.3f}",
        "─────────────────────────────────────────────────────"
    ]
    return "\n".join(trace)

def build_confusion_labels(ep_log: List[Dict] = None):
    # Dummy implementation for dashboard compatibility
    return ["allow", "flag", "delete"], np.eye(3), 1.0

def step_record_from_episode(env_step_log: Dict) -> Dict:
    return {
        "action": int(env_step_log.get("action", 0)),
        "ground_truth": _gt_str(env_step_log.get("ground_truth_type", "benign")),
        "reward": float(env_step_log.get("reward", 0.0))
    }

def process_single_message(text, media_type="text", sender_id=0, num_users=5, task="basic", use_llm=False, manual_probs=None):
    """
    Moderates a single message using keyword detection and optionally LLM.
    Returns a dictionary with decision and trace data.
    """
    from cae_env.environment import OmniAlignEnv
    env = OmniAlignEnv(num_users=num_users, max_steps=1, task=task)
    obs, _ = env.reset(seed=42)
    
    from cae_env.multimodal import extract_harm_probs
    probs_dict = extract_harm_probs(obs, num_users)
    
    if manual_probs:
        probs_dict.update(manual_probs)
        
    top_cat = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_cat]
    
    action, reason = keyword_threat_detection(text)
    risk = calculate_risk(probs_dict)
    
    trace = {
        "top_cat": top_cat,
        "top_prob": top_prob,
        "all_probs": probs_dict,
        "risk_level": risk,
        "llm_used": False
    }
    
    if action == 0 and use_llm:
        try:
            res_text = generate_text(f"Moderate message: '{text}'\nHarm Analysis: {top_cat}\nAction Index (0=allow, 1=flag, 2=delete):")
            match = re.search(r'[0-2]', res_text)
            if match:
                action = int(match.group())
                reason = "AI moderation decision"
                trace["llm_used"] = True
                trace["ai_explanation"] = res_text
        except Exception as e:
            trace["llm_error"] = str(e)
    
    if action == 0 and reason == "No keywords found": 
        reason = "Clean content"
        
    return {
        "decision": {
            "final_action": action,
            "final_action_str": ACTION_NAMES[min(action, 5)],
            "ai_explanation": reason,
            "llm_used": trace["llm_used"]
        },
        "trace": trace
    }

def run_episode(env, task_name=None, max_steps=30, verbose_trace=False, seed=42, use_llm=False):
    # This is a dummy for app.py batch processing
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    for step in range(max_steps):
        action = 0 # Dummy action
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc: break
    return {"score": total_reward / (step + 1), "traces": [], "task": task_name or "basic"}