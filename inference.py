"""
GuardianNet – FINAL PHASE 2 SURPASS VERSION
"""

import os
import sys
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ----------------------------------------------------------------------
# API Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

if GROQ_API_KEY:
    API_BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_NAME   = "llama-3.1-8b-instant"
    API_KEY      = GROQ_API_KEY
elif HF_TOKEN:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
    API_KEY      = HF_TOKEN
else:
    API_BASE_URL = None
    MODEL_NAME   = None
    API_KEY      = None

client = None
if API_KEY:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except:
        client = None

# ----------------------------------------------------------------------
# Imports
try:
    from cae_env.environment import OmniAlignEnv
    from tasks.graders import grade_basic, grade_context, grade_adversarial
except ImportError:
    print("[START] error=import")
    print("[END] success=false steps=0 score=0.00 rewards=")
    sys.exit(0)

# ----------------------------------------------------------------------
def extract_digit(text):
    m = re.search(r'\b[0-2]\b', text)
    if m:
        return int(m.group())
    for ch in text:
        if ch in '012':
            return int(ch)
    return None

# ----------------------------------------------------------------------
# 🔥 OPTIMIZED FALLBACK (KEY FIX)
def rule_based_action(obs, num_users):
    emb_dim = 128
    harm_start = emb_dim + num_users + 5*emb_dim + 4 + 15
    harm_probs = obs[harm_start:harm_start+11]

    cats = ["hate_speech","harassment","misinformation","csam","radicalization",
            "self_harm","spam","deepfake","doxxing","benign","ambiguous"]

    probs = {c: float(p) for c,p in zip(cats, harm_probs)}

    severe = {"csam","radicalization","doxxing"}
    max_cat = max(probs, key=probs.get)
    max_prob = probs[max_cat]

    # 🚨 DELETE only extreme
    if max_prob > 0.95 and max_cat in severe:
        return 2

    # ⚖️ FLAG most harmful (NOT delete)
    if max_prob > 0.7:
        return 1

    # 🤔 ambiguous
    if probs["ambiguous"] > 0.6:
        return 1

    return 0

# ----------------------------------------------------------------------
# 🔥 LLM DECISION (SAFE)
def get_llm_action(obs, num_users, task, user_text=""):
    if client is None:
        return None, "LLM unavailable"

    try:
        emb_dim = 128
        harm_start = emb_dim + num_users + 5*emb_dim + 4 + 15
        harm_probs = obs[harm_start:harm_start+11]

        cats = ["hate_speech","harassment","misinformation","csam","radicalization",
                "self_harm","spam","deepfake","doxxing","benign","ambiguous"]

        probs = {c: float(p) for c,p in zip(cats, harm_probs)}
        top_cat = max(probs, key=probs.get)
        top_prob = probs[top_cat]

        prompt = f"""
You are a moderation system.
0 = allow
1 = flag
2 = delete
Guidelines:
- Delete ONLY extreme harmful content
- Flag uncertain or moderate harm
- Allow safe content
Content: {user_text}
Top category: {top_cat} ({top_prob:.2f})
Return ONLY 0, 1, or 2.
"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            max_tokens=3,
            temperature=0,
        )

        raw = resp.choices[0].message.content.strip()
        digit = extract_digit(raw)

        if digit is not None:
            return digit, None

        return None, "Invalid output"

    except Exception as e:
        return None, str(e)

# ----------------------------------------------------------------------
def run_episode(env, task_name, max_steps=30):
    obs, _ = env.reset(seed=42)
    rewards = []
    actions_taken = []
    done = False
    step = 0

    action_names = ["allow","flag","delete","warn","quarantine","escalate"]

    while not done and step < max_steps:

        text = ""
        if hasattr(env, "pending_message") and env.pending_message:
            try:
                text = env.pending_message.contents[0].text or ""
            except:
                pass

        llm_action, err = get_llm_action(obs, env.num_users, task_name, text)

        if llm_action is not None:
            action = min(llm_action, 2)

            # 🔥 SOFT SAFETY (less aggressive)
            emb_dim = 128
            harm_start = emb_dim + env.num_users + 5*emb_dim + 4 + 15
            harm_probs = obs[harm_start:harm_start+11]

            if max(harm_probs) > 0.95 and action == 0:
                action = 1

        else:
            action = rule_based_action(obs, env.num_users)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        rewards.append(reward)
        actions_taken.append(action)

        print(f"[STEP] step={step} action={action_names[action]} reward={reward:.2f} done={str(done).lower()} error={err if err else 'null'}")
        sys.stdout.flush()

        step += 1

    # ------------------------------------------------------------------
    # Correct episode_data
    episode_data = {"steps": []}

    if hasattr(env, 'episode_log') and env.episode_log:
        for i, log in enumerate(env.episode_log[:step]):
            episode_data["steps"].append({
                "step": i,
                "action": actions_taken[i],
                "ground_truth_type": log.get("ground_truth_type", "unknown")
            })

    episode_data["final_group_health"] = info.get("group_health", 0.5)

    # ------------------------------------------------------------------
    # Scoring
    if task_name == "basic_moderation":
        score = grade_basic(episode_data)
    elif task_name == "context_aware":
        score = grade_context(episode_data)
    else:
        score = grade_adversarial(episode_data)

    # 🔥 REQUIRED FIX
    score = max(0.01, min(0.99, score))

    return step, rewards, score

# ----------------------------------------------------------------------
def main():
    tasks = [
        ("basic_moderation", OmniAlignEnv(num_users=5, max_steps=30, task="basic")),
        ("context_aware", OmniAlignEnv(num_users=5, max_steps=30, task="context")),
        ("adversarial_highstakes", OmniAlignEnv(num_users=5, max_steps=30, task="adversarial"))
    ]

    for task_name, env in tasks:
        try:
            print(f"[START] task={task_name}")

            steps, rewards, score = run_episode(env, task_name)

            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            success = score >= 0.8

            print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

            env.close()

        except Exception:
            print("[END] success=false steps=0 score=0.00 rewards=")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
