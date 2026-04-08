"""
GuardianNet – app.py
Complete Streamlit dashboard: login, live chat moderation, media analysis,
batch episode runner, user management, Telegram bot controls, analytics.
"""

from __future__ import annotations
import os, sys, re, time, json, base64, sqlite3, subprocess, tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "GuardianNet",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  /* Header */
  .gn-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    border: 1px solid #334155; color: white;
  }
  .gn-header h1 { margin:0; font-size:2rem; font-weight:600; color:#f8fafc; }
  .gn-header p  { margin:0; color:#94a3b8; font-size:0.95rem; }

  /* Message bubbles */
  .msg-bubble {
    border-radius: 12px; padding: 0.75rem 1rem; margin: 0.4rem 0;
    border-left: 4px solid; font-size: 0.93rem; line-height: 1.6;
  }
  .msg-allow  { background:#f0fdf4; border-color:#22c55e; }
  .msg-flag   { background:#fffbeb; border-color:#f59e0b; }
  .msg-delete { background:#fef2f2; border-color:#ef4444; }

  /* Verdict badge */
  .badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:600; letter-spacing:0.5px;
  }
  .badge-allow  { background:#dcfce7; color:#166534; }
  .badge-flag   { background:#fef9c3; color:#854d0e; }
  .badge-delete { background:#fee2e2; color:#991b1b; }
  .badge-low    { background:#dbeafe; color:#1e40af; }
  .badge-medium { background:#fef9c3; color:#854d0e; }
  .badge-high   { background:#fee2e2; color:#991b1b; }

  /* Metric card */
  .metric-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
    padding:1rem 1.25rem; text-align:center;
  }
  .metric-card .val { font-size:2rem; font-weight:700; color:#0f172a; }
  .metric-card .lbl { font-size:0.8rem; color:#64748b; margin-top:2px; }

  /* Login card */
  .login-wrapper {
    max-width:380px; margin:8vh auto; background:#ffffff;
    border:1px solid #e2e8f0; border-radius:20px; padding:2.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  }

  /* Trace box */
  .trace-box {
    background:#0f172a; color:#e2e8f0; border-radius:10px;
    padding:1rem; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
    line-height:1.7; white-space:pre-wrap;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────
def _init_session_defaults():
    defaults = {
        "logged_in":    False,
        "chat_history": [],         # [{sender, text, action, scores, risk}]
        "sim_history":  [],         # [{step, action, reward, ml_verdict}]
        "user_viols":   {u: 0 for u in ["Alice","Bob","Charlie","Dana","Evan"]},
        "user_blocked": {u: False for u in ["Alice","Bob","Charlie","Dana","Evan"]},
        "bot_proc":     None,
        "last_obs":     None,
        "last_scores":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_defaults()

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
CATS = ["hate_speech","harassment","misinformation","csam","radicalization",
        "self_harm","spam","deepfake","doxxing","benign","ambiguous"]
CRITICAL = {"hate_speech","deepfake","doxxing","radicalization","csam"}
MEDIUM   = {"harassment","misinformation","self_harm","spam"}

SLANG_MAP = {
    "unalive":      "suicide",  "sewer slide":  "suicide",
    "grape":        "rape",     "segs":         "sex",
    "pdf file":     "pedophile","corn":         "pornography",
    "le dollar bean":"lesbian", "spicy eggplant":"sex",
    "accountant":   "drug dealer",
}
GREETINGS = re.compile(
    r'^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|yo|sup)\b',
    re.IGNORECASE
)
THREAT_KW = re.compile(
    r'\b(kill|murder|die|attack|bomb|shoot|stab|rape|destroy|hurt you|'
    r'find you|watch your back|you\'re dead|come for you)\b',
    re.IGNORECASE
)

# ──────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────
def expand_slang(text: str) -> str:
    t = text.lower()
    for slang, real in SLANG_MAP.items():
        t = t.replace(slang, real)
    return t

def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def translate_to_english(text: str, lang: str) -> str:
    if lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # graceful fallback

# ──────────────────────────────────────────────────────────────
# Rule-based classifier (text)
# ──────────────────────────────────────────────────────────────
def rule_classify_text(text: str) -> tuple[str, float, dict]:
    """Returns (category, confidence, all_scores)."""
    scores = {c: 0.0 for c in CATS}

    expanded = expand_slang(text)

    if GREETINGS.match(text):
        scores["benign"] = 0.97
        return "benign", 0.97, scores

    if THREAT_KW.search(expanded):
        scores["hate_speech"] = 0.0
        scores["harassment"]  = 0.9
        return "harassment", 0.90, scores

    # Pattern-based scoring
    hate_re  = re.compile(r'\b(all \w+ are|go back|sub-human|vermin|infestation|\w+ scum)\b', re.I)
    misinfo  = re.compile(r'\b(5g|vaccines cause|flat earth|crisis actor|deep state|chemtrail|covid hoax)\b', re.I)
    spam_re  = re.compile(r'\b(click here|buy now|free money|earn \$|limited offer|act now)\b', re.I)
    slang_re = re.compile(r'\b(lol|lmao|bruh|fr fr|no cap|bussin|slay|lowkey|based|mid|vibe)\b', re.I)

    if hate_re.search(expanded):   scores["hate_speech"]    = 0.88
    if misinfo.search(expanded):   scores["misinformation"] = 0.85
    if spam_re.search(expanded):   scores["spam"]           = 0.80
    if slang_re.search(expanded):
        hits = len(slang_re.findall(expanded))
        scores["benign"] = max(scores["benign"], min(0.4 + hits*0.08, 0.85))

    total_harm = sum(v for k,v in scores.items() if k not in ("benign","ambiguous"))
    if total_harm < 0.1:
        scores["benign"] = max(scores["benign"], 0.88)

    top = max(scores, key=scores.get)
    return top, scores[top], scores

# ──────────────────────────────────────────────────────────────
# HuggingFace & Groq Inference API
# ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

if GROQ_API_KEY:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
    API_KEY      = GROQ_API_KEY
elif HF_TOKEN:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-1B-Instruct")
    API_KEY      = HF_TOKEN
else:
    API_BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_NAME   = "llama-3.1-8b-instant"
    API_KEY      = ""

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def _hf_post(url: str, payload, timeout: int = 20):
    import requests
    try:
        r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_resource
def get_media_classifier():
    """Try to load cae_env media classifier, else return None."""
    try:
        from cae_env.media_classifier import MediaClassifier
        return MediaClassifier()
    except Exception:
        return None

def classify_text_full(raw_text: str, use_llm: bool = True) -> dict:
    """Full pipeline: slang expansion → rule → optional LLM → verdict."""
    lang     = detect_language(raw_text)
    trans    = translate_to_english(raw_text, lang)
    expanded = expand_slang(trans)

    # Rule-based baseline
    cat, conf, scores = rule_classify_text(expanded)

    # Optional LLM boost via Groq/HF
    if use_llm and API_KEY:
        try:
            from openai import OpenAI
            llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role":"system","content":(
                        "You are a content moderation classifier. "
                        "Return ONLY valid JSON: {\"category\": \"<cat>\", \"confidence\": <0-1>} "
                        "where category is one of: " + ", ".join(CATS)
                    )},
                    {"role":"user","content":f"Classify: {expanded[:300]}"},
                ],
                max_tokens=60, temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            llm_cat  = data.get("category", cat)
            llm_conf = float(data.get("confidence", conf))
            if llm_cat in scores and llm_conf > conf:
                scores[llm_cat] = max(scores[llm_cat], llm_conf)
                cat, conf = llm_cat, llm_conf
        except Exception:
            pass  # silently fall back to rule-based

    sev_map = {"benign":0,"ambiguous":1,"spam":2,"misinformation":3,
               "harassment":4,"self_harm":4,"hate_speech":5,
               "radicalization":5,"deepfake":5,"doxxing":5,"csam":6}
    severity = sev_map.get(cat, 3)
    action   = 2 if severity >= 5 else 1 if severity >= 3 else 0
    risk     = "high" if severity >= 5 else "medium" if severity >= 3 else "low"

    return {
        "raw_text":  raw_text,
        "language":  lang,
        "translated": trans,
        "expanded":  expanded,
        "category":  cat,
        "confidence":conf,
        "scores":    scores,
        "severity":  severity,
        "action":    ["allow","flag","delete"][action],
        "risk":      risk,
        "action_int":action,
    }

# ──────────────────────────────────────────────────────────────
# Image classifier
# ──────────────────────────────────────────────────────────────
IMAGE_CATS = {
    "natural_photo":   {"label":"Natural Photo",   "color":"#22c55e","action":"allow","severity":0},
    "synthetic_ai":    {"label":"AI Generated",    "color":"#3b82f6","action":"flag", "severity":2},
    "deepfake":        {"label":"Deepfake",        "color":"#f97316","action":"flag", "severity":4},
    "nsfw_adult":      {"label":"NSFW / Adult",    "color":"#ef4444","action":"delete","severity":5},
    "violent_graphic": {"label":"Violent / Graphic","color":"#dc2626","action":"delete","severity":6},
    "weapon_illegal":  {"label":"Weapon / Illegal","color":"#7f1d1d","action":"delete","severity":7},
}

def classify_image(img: Image.Image) -> dict:
    """Classify image using HF NSFW API + pixel heuristics."""
    scores = {k: 0.0 for k in IMAGE_CATS}
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    # HF NSFW model
    import requests
    r = None
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/Falconsai/nsfw_image_detection",
            headers=HF_HEADERS, data=img_bytes, timeout=20
        )
        if r and r.status_code == 200:
            for item in r.json():
                lbl = item.get("label","").lower()
                sc  = float(item.get("score",0))
                if any(w in lbl for w in ("nsfw","explicit","hentai","porn","sexy")):
                    scores["nsfw_adult"] = max(scores["nsfw_adult"], sc)
                elif any(w in lbl for w in ("normal","safe","neutral")):
                    scores["natural_photo"] = max(scores["natural_photo"], sc)
    except Exception:
        pass

    # Pixel heuristics: variance → synthetic detection
    arr = np.array(img.convert("RGB"), dtype=float)
    lap_var = float(np.var(np.diff(arr, axis=0)))
    if lap_var < 40:
        scores["synthetic_ai"] = max(scores["synthetic_ai"], 0.65)
    elif lap_var > 600:
        scores["natural_photo"] = max(scores["natural_photo"], 0.72)

    # Colour entropy for flesh-tone detection (NSFW heuristic)
    hsv = np.array(img.convert("HSV"))
    flesh = ((hsv[:,:,0] > 5) & (hsv[:,:,0] < 25) &
             (hsv[:,:,1] > 40) & (hsv[:,:,2] > 60))
    flesh_ratio = flesh.sum() / flesh.size
    if flesh_ratio > 0.30:
        scores["nsfw_adult"] = max(scores["nsfw_adult"], 0.55)

    if sum(scores.values()) < 0.2:
        scores["natural_photo"] = 0.80

    top = max(scores, key=scores.get)
    return {"category": top, "confidence": scores[top], "scores": scores, **IMAGE_CATS[top]}

def classify_video(file_bytes: bytes) -> dict:
    """Sample frames from video and return most severe classification."""
    try:
        import cv2, tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, total-1, min(10, total), dtype=int)
        frames, results = [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        _os.unlink(tmp_path)

        for frm in frames:
            results.append(classify_image(frm))

        if not results:
            return {"category":"natural_photo","confidence":0.5,"action":"allow","severity":0,"frames":0}

        worst = max(results, key=lambda x: x["severity"])
        worst["frames"] = len(frames)
        return worst
    except ImportError:
        return {"category":"natural_photo","confidence":0.5,
                "action":"allow","severity":0,"frames":0,
                "warning":"opencv not installed – pip install opencv-python-headless"}
    except Exception as exc:
        return {"category":"natural_photo","confidence":0.5,
                "action":"allow","severity":0,"frames":0,"error":str(exc)}

# ──────────────────────────────────────────────────────────────
# Telegram bot helpers
# ──────────────────────────────────────────────────────────────
def get_telegram_logs(n: int = 20) -> list[dict]:
    db = "telegram_logs.db"
    if not os.path.exists(db):
        return []
    try:
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT timestamp, user, text, action, category FROM logs ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        con.close()
        return [{"time":r[0],"user":r[1],"text":r[2],"action":r[3],"category":r[4]} for r in rows]
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────
# ██  LOGIN PAGE
# ──────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    st.markdown("""
    <div class="login-wrapper">
      <div style="text-align:center; margin-bottom:1.5rem;">
        <span style="font-size:3rem">🛡️</span>
        <h2 style="margin:0.5rem 0 0.25rem; font-size:1.5rem; color:#0f172a">GuardianNet</h2>
        <p style="color:#64748b; font-size:0.9rem">Content Moderation Intelligence</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_pad, col_form, col_pad2 = st.columns([1.5, 2, 1.5])
    with col_form:
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••")
        login_btn = st.button("Sign In →", use_container_width=True, type="primary")

        if login_btn:
            if username == "admin" and password == "admin":
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / admin")

    st.stop()

# ──────────────────────────────────────────────────────────────
# ██  MAIN DASHBOARD (post-login)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="gn-header">
  <h1>🛡️ GuardianNet</h1>
  <p>Real-time Multi-Modal Content Moderation · Collective Alignment Engine</p>
</div>
""", unsafe_allow_html=True)

TABS = st.tabs([
    "💬 Live Chat", "🖼️ Media Analysis",
    "🎮 Simulation", "👥 User Management",
    "🤖 Telegram Bot", "📊 Analytics"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE CHAT
# ══════════════════════════════════════════════════════════════
with TABS[0]:
    chat_col, monitor_col = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Chat frontend ──────────────────────────────────
    with chat_col:
        st.subheader("Group Chat Simulator")

        # Display history
        chat_box = st.container(height=420)
        with chat_box:
            for entry in st.session_state.chat_history[-30:]:
                action  = entry.get("action", "allow")
                cls_map = {"allow":"msg-allow","flag":"msg-flag","delete":"msg-delete"}
                bdg_map = {"allow":"badge-allow","flag":"badge-flag","delete":"badge-delete"}
                css_cls = cls_map.get(action, "msg-allow")
                bdg_cls = bdg_map.get(action, "badge-allow")
                cat     = entry.get("category","benign")
                conf    = entry.get("confidence",0)
                st.markdown(f"""
                <div class="msg-bubble {css_cls}">
                  <strong>{entry['sender']}</strong>
                  <span class="badge {bdg_cls}" style="float:right">{action.upper()}</span>
                  <br>{entry['text']}
                  <br><small style="color:#64748b">Category: {cat} ({conf*100:.0f}%)</small>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Input controls
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sender  = st.selectbox("Sender", ["Alice","Bob","Charlie","Dana","Evan"], label_visibility="collapsed")
        with c2:
            task_sel = st.selectbox("Task", ["basic","context","adversarial"], label_visibility="collapsed")
        with c3:
            use_llm = st.toggle("Use LLM", value=True)

        msg_text = st.text_area("Message", height=80, placeholder="Type a message…", label_visibility="collapsed")

        send_btn = st.button("📤 Send", type="primary", use_container_width=True)

        # Quick injection examples
        with st.expander("🧪 Inject Example Messages"):
            examples = [
                ("✅ Greeting",     "Hey everyone! Good morning 😊"),
                ("💬 Slang",        "bruh this is lowkey bussin no cap fr fr lmaooo"),
                ("📰 Misinfo",      "5G towers spread COVID, the government won't admit it!"),
                ("😤 Harassment",   "You are absolutely pathetic and should quit already."),
                ("⚠️ Threat",       "I will find you and you are going to regret this."),
                ("🚫 Hate speech",  "All those people are vermin and should leave this country."),
                ("💀 CSAM trigger", "Kids these days need more supervision, pdf files everywhere."),
                ("📧 Spam",         "CLICK HERE to earn $5000 per day — limited offer, act NOW!"),
            ]
            cols = st.columns(2)
            for i, (lbl, txt) in enumerate(examples):
                if cols[i % 2].button(lbl, key=f"inj_{i}", use_container_width=True):
                    st.session_state["inject_text"] = txt
                    st.rerun()

        if "inject_text" in st.session_state:
            msg_text = st.session_state.pop("inject_text")

        if send_btn and msg_text.strip():
            with st.spinner("Analysing…"):
                result = classify_text_full(msg_text.strip(), use_llm=use_llm)

            entry = {
                "sender":     sender,
                "text":       msg_text.strip(),
                "action":     result["action"],
                "category":   result["category"],
                "confidence": result["confidence"],
                "scores":     result["scores"],
                "risk":       result["risk"],
                "language":   result["language"],
                "translated": result["translated"],
                "expanded":   result["expanded"],
            }
            st.session_state.chat_history.append(entry)
            st.session_state.last_scores = result

            # Track violations
            if result["action"] == "delete":
                st.session_state.user_viols[sender] = st.session_state.user_viols.get(sender,0) + 1
            st.rerun()

    # ── RIGHT: Backend Monitor ────────────────────────────────
    with monitor_col:
        st.subheader("Decision Monitor")

        if st.session_state.last_scores:
            r = st.session_state.last_scores
            cat    = r["category"]
            conf   = r["confidence"]
            action = r["action"]
            risk   = r["risk"]

            # Risk badge
            risk_cols = {"low":"badge-low","medium":"badge-medium","high":"badge-high"}
            act_cols  = {"allow":"badge-allow","flag":"badge-flag","delete":"badge-delete"}
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:1rem;">
              <span class="badge {act_cols.get(action,'badge-allow')}" style="font-size:1rem;padding:6px 16px">
                {action.upper()}
              </span>
              <span class="badge {risk_cols.get(risk,'badge-low')}" style="font-size:1rem;padding:6px 16px">
                RISK: {risk.upper()}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Summary table
            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Top Category | `{cat}` |
            | Confidence | {conf*100:.1f}% |
            | Language | {r.get('language','en')} |
            | Severity | {r.get('severity',0)} / 6 |
            """)

            # Harm probability bar chart
            scores_df = pd.DataFrame([
                {"Category": k.replace("_"," ").title(), "Score": round(v*100,1)}
                for k, v in sorted(r["scores"].items(), key=lambda x: -x[1])
            ])
            fig = px.bar(
                scores_df, x="Score", y="Category", orientation="h",
                color="Score",
                color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                range_x=[0,100],
                labels={"Score":"Confidence (%)","Category":""}
            )
            fig.update_layout(
                height=300, showlegend=False,
                margin=dict(l=0,r=10,t=10,b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Decision trace
            with st.expander("🔍 Full Decision Trace"):
                trace = (
                    f"INPUT:       {r['raw_text'][:100]}\n"
                    f"LANGUAGE:    {r.get('language','en')}\n"
                    f"TRANSLATED:  {r.get('translated','')[:100]}\n"
                    f"EXPANDED:    {r.get('expanded','')[:100]}\n"
                    f"{'─'*40}\n"
                    f"TOP CATEGORY:{cat}\n"
                    f"CONFIDENCE:  {conf*100:.1f}%\n"
                    f"SEVERITY:    {r.get('severity',0)}/6\n"
                    f"ACTION:      {action.upper()}\n"
                    f"RISK LEVEL:  {risk.upper()}\n"
                )
                st.markdown(f'<div class="trace-box">{trace}</div>', unsafe_allow_html=True)
        else:
            st.info("Send a message to see the analysis here.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — MEDIA ANALYSIS
# ══════════════════════════════════════════════════════════════
with TABS[1]:
    st.subheader("Multi-Modal Media Classification")
    st.caption("Images → natural / AI-synthetic / deepfake / NSFW / violent / illegal  ·  Videos → frame sampling")

    media_l, media_r = st.columns(2, gap="large")

    with media_l:
        media_src = st.radio("Source", ["Upload File","Image URL"], horizontal=True)
        media_obj = None
        media_type = None

        if media_src == "Upload File":
            up = st.file_uploader(
                "Upload image or video",
                type=["jpg","jpeg","png","webp","mp4","mov","avi","webm"]
            )
            if up:
                ext = up.name.rsplit(".",1)[-1].lower()
                if ext in ("mp4","mov","avi","webm"):
                    media_type = "video"
                    media_obj  = up.read()
                    st.video(BytesIO(media_obj))
                else:
                    media_type = "image"
                    media_obj  = Image.open(up).convert("RGB")
                    st.image(media_obj, use_column_width=True)
        else:
            url = st.text_input("Image URL")
            if url:
                try:
                    import requests
                    media_obj  = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
                    media_type = "image"
                    st.image(media_obj, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not load: {e}")

        analyse_media = st.button("🔍 Classify Media", type="primary", use_container_width=True,
                                   disabled=(media_obj is None))

    with media_r:
        if analyse_media and media_obj is not None:
            with st.spinner("Running classification…"):
                if media_type == "image":
                    res = classify_image(media_obj)
                else:
                    res = classify_video(media_obj)

            info = IMAGE_CATS.get(res["category"], IMAGE_CATS["natural_photo"])
            sev  = info["severity"]
            bg   = "#f0fdf4" if sev < 2 else "#fffbeb" if sev < 4 else "#fef2f2"

            st.markdown(f"""
            <div style="background:{bg};border:2px solid {info['color']};
                 border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
              <p style="margin:0;font-size:1.3rem;font-weight:700;color:{info['color']}">
                {info['label']}
              </p>
              <p style="margin:4px 0 0;color:{info['color']}">
                Confidence: {res['confidence']*100:.1f}%
                &nbsp;·&nbsp; Action:
                <strong>{info['action'].upper()}</strong>
              </p>
              {f"<p style='margin:4px 0 0;color:#64748b;font-size:0.85rem'>Frames analysed: {res.get('frames',1)}</p>" if media_type=="video" else ""}
            </div>
            """, unsafe_allow_html=True)

            # Radar chart
            if "scores" in res:
                cats_r = [k.replace("_"," ").title() for k in res["scores"]]
                vals_r = list(res["scores"].values())
                fig_r  = go.Figure(go.Scatterpolar(
                    r=vals_r+[vals_r[0]], theta=cats_r+[cats_r[0]],
                    fill="toself", line_color=info["color"]
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                    showlegend=False, height=280,
                    margin=dict(l=20,r=20,t=20,b=20)
                )
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Upload an image or video and click Classify.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — SIMULATION
# ══════════════════════════════════════════════════════════════
with TABS[2]:
    st.subheader("Batch Episode Runner")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_task  = sim_c1.selectbox("Task", ["basic","context","adversarial"], key="sim_task")
    sim_steps = sim_c2.slider("Steps", 5, 30, 20, key="sim_steps")
    sim_seed  = sim_c3.number_input("Seed", value=42, key="sim_seed")

    run_sim = st.button("▶ Run Episode", type="primary")

    if run_sim:
        try:
            from cae_env.environment import OmniAlignEnv
            env = OmniAlignEnv(num_users=5, max_steps=sim_steps, task=sim_task)
            obs, _ = env.reset(seed=int(sim_seed))
            sim_data = []
            done = False
            step = 0
            progress = st.progress(0, text="Running…")

            while not done and step < sim_steps:
                text = ""
                if hasattr(env,"pending_message") and env.pending_message:
                    try: text = env.pending_message.contents[0].text or ""
                    except: pass

                result = classify_text_full(text, use_llm=False)
                action = result["action_int"]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                sim_data.append({
                    "Step": step, "Action": result["action"],
                    "Category": result["category"],
                    "Reward": reward, "Done": done,
                    "Message": text[:60]+"…" if len(text)>60 else text,
                })
                step += 1
                progress.progress(step / sim_steps, text=f"Step {step}/{sim_steps}")

            env.close()
            progress.empty()
            df_sim = pd.DataFrame(sim_data)

            # Metrics
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Steps",  len(df_sim))
            m2.metric("Total Reward", f"{df_sim['Reward'].sum():.2f}")
            m3.metric("Avg Reward",   f"{df_sim['Reward'].mean():.2f}")
            deletes = (df_sim["Action"]=="delete").sum()
            m4.metric("Deletions",    int(deletes))

            # Reward trend
            fig_sim = px.line(df_sim, x="Step", y="Reward",
                              title="Reward Trend", markers=True)
            fig_sim.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_sim, use_container_width=True)

            # Action timeline + confusion
            col_a, col_b = st.columns(2)
            with col_a:
                act_counts = df_sim["Action"].value_counts().reset_index()
                act_counts.columns = ["Action","Count"]
                fig_act = px.bar(act_counts, x="Action", y="Count",
                                  color="Action",
                                  color_discrete_map={"allow":"#22c55e","flag":"#f59e0b","delete":"#ef4444"},
                                  title="Action Distribution")
                st.plotly_chart(fig_act, use_container_width=True)
            with col_b:
                cat_counts = df_sim["Category"].value_counts().head(8).reset_index()
                cat_counts.columns = ["Category","Count"]
                fig_cat = px.pie(cat_counts, values="Count", names="Category",
                                  title="Category Distribution")
                st.plotly_chart(fig_cat, use_container_width=True)

            st.dataframe(df_sim, use_container_width=True, hide_index=True)

        except ImportError:
            st.warning("cae_env not found. Install the GuardianNet environment package.")
        except Exception as exc:
            st.error(f"Simulation error: {exc}")

# ══════════════════════════════════════════════════════════════
# TAB 4 — USER MANAGEMENT
# ══════════════════════════════════════════════════════════════
with TABS[3]:
    st.subheader("User Profiles & Violation Tracker")

    users = ["Alice","Bob","Charlie","Dana","Evan"]
    avatars = {"Alice":"👩","Bob":"👨","Charlie":"🧑","Dana":"👩‍💻","Evan":"👨‍💼"}

    um_cols = st.columns(len(users))
    for col, u in zip(um_cols, users):
        viols   = st.session_state.user_viols.get(u, 0)
        blocked = st.session_state.user_blocked.get(u, False)
        status  = "🔴 Blocked" if blocked else ("⚠️ At Risk" if viols >= 2 else "🟢 Active")

        with col:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                 padding:1rem;text-align:center;">
              <div style="font-size:2.2rem">{avatars[u]}</div>
              <p style="font-weight:600;margin:4px 0">{u}</p>
              <p style="color:#64748b;font-size:0.8rem;margin:2px 0">{viols} violations</p>
              <p style="font-size:0.8rem">{status}</p>
            </div>
            """, unsafe_allow_html=True)
            if blocked:
                if st.button("Unblock", key=f"unb_{u}", use_container_width=True):
                    st.session_state.user_blocked[u] = False
                    st.session_state.user_viols[u]   = 0
                    st.rerun()
            elif viols >= 3:
                if st.button("Block", key=f"blk_{u}", use_container_width=True, type="primary"):
                    st.session_state.user_blocked[u] = True
                    st.rerun()

    st.divider()
    st.subheader("Harm Sensitivity Radar")
    st.caption("Shows distribution of detected harm categories per user from current session.")

    user_cats = {u: {c:0.0 for c in CATS} for u in users}
    for entry in st.session_state.chat_history:
        u = entry.get("sender")
        if u in user_cats and "scores" in entry:
            for cat, sc in entry["scores"].items():
                user_cats[u][cat] = max(user_cats[u][cat], sc)

    sel_user = st.selectbox("Select user for radar", users)
    ud = user_cats[sel_user]
    fig_radar = go.Figure(go.Scatterpolar(
        r    = list(ud.values()) + [list(ud.values())[0]],
        theta= [c.replace("_"," ").title() for c in ud.keys()] + [list(ud.keys())[0].replace("_"," ").title()],
        fill = "toself", name = sel_user, line_color="#6366f1"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False, height=380, margin=dict(l=30,r=30,t=30,b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — TELEGRAM BOT
# ══════════════════════════════════════════════════════════════
with TABS[4]:
    st.subheader("Telegram Bot Management")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN","")
    if bot_token:
        st.success(f"✅ Bot token configured: `{bot_token[:8]}…{bot_token[-4:]}`")
    else:
        st.warning("⚠️ TELEGRAM_BOT_TOKEN not set in .env")

    bc1, bc2, bc3 = st.columns(3)
    if bc1.button("▶ Start Bot", type="primary", use_container_width=True):
        if st.session_state.bot_proc is None:
            try:
                st.session_state.bot_proc = subprocess.Popen(
                    [sys.executable, "telegram_bot.py"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                st.success("Bot started!")
            except Exception as e:
                st.error(f"Failed to start: {e}")

    if bc2.button("⏹ Stop Bot", use_container_width=True):
        proc = st.session_state.bot_proc
        if proc:
            proc.terminate()
            st.session_state.bot_proc = None
            st.info("Bot stopped.")

    running = st.session_state.bot_proc is not None and st.session_state.bot_proc.poll() is None
    bc3.metric("Status", "🟢 Running" if running else "🔴 Stopped")

    st.divider()
    st.subheader("Recent Moderation Logs")
    logs = get_telegram_logs(20)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    else:
        st.info("No logs yet. Start the bot and send messages to a Telegram group.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with TABS[5]:
    st.subheader("Moderation Analytics")

    hist = st.session_state.chat_history
    if not hist:
        st.info("Send some messages in the Live Chat tab to populate analytics.")
    else:
        df_h = pd.DataFrame(hist)

        # KPIs
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Messages Analysed", len(df_h))
        k2.metric("Allowed",   int((df_h["action"]=="allow").sum()))
        k3.metric("Flagged",   int((df_h["action"]=="flag").sum()))
        k4.metric("Deleted",   int((df_h["action"]=="delete").sum()))
        k5.metric("Avg Confidence", f"{df_h['confidence'].mean()*100:.0f}%")

        cl, cr = st.columns(2)
        with cl:
            pie = px.pie(df_h, names="category", title="Category Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(pie, use_container_width=True)

        with cr:
            act_df = df_h["action"].value_counts().reset_index()
            act_df.columns = ["action","count"]
            bar = px.bar(act_df, x="action", y="count", title="Actions Taken",
                         color="action",
                         color_discrete_map={"allow":"#22c55e","flag":"#f59e0b","delete":"#ef4444"})
            st.plotly_chart(bar, use_container_width=True)

        # Sender breakdown
        send_df = df_h.groupby("sender")["action"].value_counts().unstack(fill_value=0).reset_index()
        fig_s = px.bar(send_df, x="sender",
                       y=[c for c in ["allow","flag","delete"] if c in send_df.columns],
                       title="Actions by Sender", barmode="stack",
                       color_discrete_map={"allow":"#22c55e","flag":"#f59e0b","delete":"#ef4444"})
        st.plotly_chart(fig_s, use_container_width=True)

        # Confidence distribution
        fig_c = px.histogram(df_h, x="confidence", nbins=20,
                              title="Confidence Score Distribution",
                              color_discrete_sequence=["#6366f1"])
        st.plotly_chart(fig_c, use_container_width=True)

        # Full table + export
        st.dataframe(df_h[["sender","text","action","category","confidence","risk","language"]],
                     use_container_width=True, hide_index=True)
        csv = df_h.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "guardiannet_analytics.csv", "text/csv")
