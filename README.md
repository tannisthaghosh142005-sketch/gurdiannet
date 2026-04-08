# 🛡️ GuardianNet — Collective Alignment Engine

> Real-time multi-modal content moderation using Reinforcement Learning, LLMs, and multi-model ensembles.

---

## 📋 Overview

GuardianNet is an OpenEnv RL environment where AI agents learn to moderate group chat content in real-time. It combines:

- **Rule-based fast classifier** — zero-latency, proven ≥0.9 score fallback
- **LLM ensemble** — Groq (free) or HuggingFace Router for nuanced decisions
- **Multi-modal analysis** — text, images, videos with 6-class classification each
- **Streamlit dashboard** — live chat simulation, media upload, analytics, Telegram bot management
- **Telegram bot** — real-time group moderation with auto-ban after threshold violations

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourname/guardiannet
cd guardiannet
pip install -r requirements-dashboard.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your FREE Groq token from https://console.groq.com
```

### 3. Run inference (evaluation mode)

```bash
python inference.py
```

Expected output:
```
[START] task=basic_moderation env=guardiannet model=llama-3.1-8b-instant
[STEP] step=0 action=allow reward=1.00 done=false error=null
...
[END] success=true steps=30 score=0.93 rewards=1.00,0.50,...
```

### 4. Run dashboard

```bash
streamlit run app.py
```

Login: `admin` / `admin`

### 5. Docker

```bash
docker build -t guardiannet .
docker run --rm --env-file .env guardiannet
```

---

## 🔑 Free API Tokens

| Provider | Use | Sign Up | Limit |
|----------|-----|---------|-------|
| **Groq** | Primary LLM | [console.groq.com](https://console.groq.com) | 14,400 req/day |
| **HuggingFace** | Backup LLM + models | [hf.co/settings/tokens](https://huggingface.co/settings/tokens) | Free tier |

Set `HF_TOKEN` in `.env` to your Groq key (the variable name is reused for all providers).

---

## 📁 Project Structure

```
guardiannet/
├── cae_env/             # RL environment (Gymnasium)
├── tasks/               # Episode graders
├── inference.py         # OpenEnv evaluation script
├── app.py               # Streamlit dashboard
├── telegram_bot.py      # Telegram moderation bot
├── Dockerfile           # Evaluation container
├── requirements.txt     # Minimal (Docker)
├── requirements-dashboard.txt  # Full (dashboard + bot)
├── openenv.yaml         # OpenEnv metadata
└── .env.example         # Token template
```

---

## 🎮 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `basic_moderation` | Easy | Obvious harm: hate speech, threats, spam |
| `context_aware` | Medium | Sender history, community context |
| `adversarial_highstakes` | Hard | Coded language, evasion, edge cases |

All tasks use `seed=42` for reproducibility. Success = score ≥ 0.8.

---

## 🤖 Text Classification Categories

| Category | Severity | Action |
|----------|----------|--------|
| benign | 0 | Allow |
| slang / ambiguous | 1 | Allow |
| spam | 2 | Monitor |
| misinformation | 3 | Flag |
| harassment / self_harm | 4 | Flag |
| hate_speech / radicalization / deepfake / doxxing | 5 | Delete |
| csam | 6 | Delete + escalate |

---

## 🖼️ Image / Video Categories

| Category | Action |
|----------|--------|
| Natural photo | Allow |
| AI-generated / synthetic | Flag |
| Deepfake | Flag |
| NSFW / Adult | Delete |
| Violent / Graphic | Delete |
| Weapon / Illegal | Delete |

---

## 📱 Telegram Bot Setup

1. Message `@BotFather` on Telegram → `/newbot`
2. Copy the token to `.env` as `TELEGRAM_BOT_TOKEN`
3. Add the bot to your group as **admin** (it needs delete permissions)
4. Start the bot from the dashboard or: `python telegram_bot.py`

---

## 🐳 HuggingFace Space Deployment

1. Create a new Space → Docker template
2. Upload all files
3. Add Secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. The Space runs `inference.py` via Docker CMD
5. Ensure Space is **Running** before submitting

---

## ✅ Evaluation Checklist

- [x] `inference.py` in root directory
- [x] Uses `openai.OpenAI` client
- [x] `API_BASE_URL` with default value
- [x] `MODEL_NAME` with default value
- [x] `HF_TOKEN` mandatory (raises ValueError if missing)
- [x] `[START]` / `[STEP]` / `[END]` log format with `score=` field
- [x] Rule-based fallback for all LLM failures
- [x] All three tasks run with `seed=42`
- [x] Docker container runs `inference.py`
- [x] HuggingFace Space in Running state

---

## 📄 License

MIT © GuardianNet Team
