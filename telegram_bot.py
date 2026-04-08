"""
GuardianNet – telegram_bot.py
Real-time content moderation bot using python-telegram-bot v20+.
Reads TELEGRAM_BOT_TOKEN from .env.
Logs all actions to telegram_logs.db (SQLite).
Auto-blocks users after BLOCK_THRESHOLD violations.
"""

from __future__ import annotations
import os, re, json, sqlite3, logging, asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BLOCK_THRESHOLD    = int(os.getenv("BLOCK_THRESHOLD", "3"))
HF_TOKEN           = os.getenv("HF_TOKEN", "")
API_BASE_URL       = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME         = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SQLite logging
# ──────────────────────────────────────────────────────────────
DB_PATH = "telegram_logs.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chat_id   INTEGER,
            user      TEXT,
            text      TEXT,
            action    TEXT,
            category  TEXT,
            confidence REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            user_id   INTEGER PRIMARY KEY,
            username  TEXT,
            count     INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()

def log_action(chat_id, user, text, action, category, confidence):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO logs (timestamp, chat_id, user, text, action, category, confidence) "
        "VALUES (?,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), chat_id, user, text[:500], action, category, float(confidence))
    )
    con.commit()
    con.close()

def get_violations(user_id: int) -> int:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT count FROM violations WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    return row[0] if row else 0

def increment_violations(user_id: int, username: str) -> int:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO violations (user_id, username, count) VALUES (?,?,1) "
        "ON CONFLICT(user_id) DO UPDATE SET count=count+1",
        (user_id, username)
    )
    con.commit()
    row = con.execute("SELECT count FROM violations WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    return row[0] if row else 1

# ──────────────────────────────────────────────────────────────
# Moderation logic (mirrors inference.py)
# ──────────────────────────────────────────────────────────────
CATS = ["hate_speech","harassment","misinformation","csam","radicalization",
        "self_harm","spam","deepfake","doxxing","benign","ambiguous"]

GREETINGS = re.compile(
    r'^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|yo|sup)\b',
    re.IGNORECASE
)
THREAT_KW = re.compile(
    r'\b(kill|murder|die|attack|bomb|shoot|stab|rape|destroy|hurt you|'
    r'find you|watch your back|you\'re dead)\b',
    re.IGNORECASE
)
HATE_RE  = re.compile(r'\b(all \w+ are|go back|sub-human|vermin|infestation)\b', re.I)
MISINFO  = re.compile(r'\b(5g|vaccines cause|flat earth|crisis actor|deep state)\b', re.I)
SPAM_RE  = re.compile(r'\b(click here|buy now|free money|earn \$|limited offer)\b', re.I)

SLANG_MAP = {
    "unalive": "suicide", "sewer slide": "suicide",
    "grape": "rape", "pdf file": "pedophile", "corn": "pornography",
}

def expand_slang(text: str) -> str:
    t = text.lower()
    for s, r in SLANG_MAP.items():
        t = t.replace(s, r)
    return t

def classify_message(text: str) -> tuple[str, float, int]:
    """Returns (category, confidence, action_int 0/1/2)."""
    expanded = expand_slang(text)

    if GREETINGS.match(text):
        return "benign", 0.97, 0

    if THREAT_KW.search(expanded):
        return "harassment", 0.92, 2

    if HATE_RE.search(expanded):
        return "hate_speech", 0.88, 2

    if MISINFO.search(expanded):
        return "misinformation", 0.82, 1

    if SPAM_RE.search(expanded):
        return "spam", 0.78, 1

    # Optional LLM call
    if HF_TOKEN:
        try:
            from openai import OpenAI
            llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role":"system","content":(
                        "You are a content moderation classifier. "
                        "Return ONLY valid JSON: {\"category\":\"<cat>\",\"action\":<0|1|2>} "
                        "where category is one of: "+", ".join(CATS)+". "
                        "0=allow, 1=flag, 2=delete."
                    )},
                    {"role":"user","content":f"Classify: {expanded[:300]}"}
                ],
                max_tokens=60, temperature=0
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = re.sub(r"```json|```","",raw).strip()
            data = json.loads(raw)
            cat  = data.get("category","benign")
            act  = int(data.get("action", 0))
            conf = 0.85
            return cat, conf, act
        except Exception:
            pass

    return "benign", 0.80, 0

# ──────────────────────────────────────────────────────────────
# Telegram bot handlers
# ──────────────────────────────────────────────────────────────
async def handle_message(update, context):
    msg  = update.message
    if msg is None:
        return

    text     = msg.text or ""
    user     = msg.from_user
    user_id  = user.id
    username = user.username or user.first_name or str(user_id)
    chat_id  = msg.chat_id

    category, confidence, action_int = classify_message(text)
    action = ["allow","flag","delete"][action_int]

    logger.info(f"[{action.upper()}] @{username}: {text[:80]} → {category} ({confidence:.2f})")
    log_action(chat_id, username, text, action, category, confidence)

    if action_int == 2:
        # Delete the message
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg.message_id)
        except Exception as e:
            logger.warning(f"Could not delete message: {e}")

        viols = increment_violations(user_id, username)
        warn_text = (
            f"⚠️ @{username} — Message removed ({category.replace('_',' ')}). "
            f"Violation {viols}/{BLOCK_THRESHOLD}."
        )
        await context.bot.send_message(chat_id=chat_id, text=warn_text)

        if viols >= BLOCK_THRESHOLD:
            try:
                await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚫 @{username} has been removed after {viols} violations."
                )
                logger.info(f"Banned user @{username} (id={user_id})")
            except Exception as e:
                logger.warning(f"Could not ban user: {e}")

    elif action_int == 1:
        # Flag only — send private warning to admins (or group notice)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🚩 Message flagged for review: [{category.replace('_',' ')}]"
            )
        except Exception:
            pass

async def handle_start(update, context):
    await update.message.reply_text(
        "🛡️ GuardianNet is active and monitoring this chat.\n"
        "All messages will be moderated in real-time."
    )

async def handle_stats(update, context):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT action, COUNT(*) as n FROM logs GROUP BY action"
    ).fetchall()
    con.close()
    if not rows:
        await update.message.reply_text("No moderation data yet.")
        return
    lines = ["📊 *GuardianNet Stats*"]
    for action, n in rows:
        lines.append(f"  • {action.upper()}: {n}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

def main():
    from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters

    init_db()
    logger.info("Starting GuardianNet Telegram Bot…")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",  handle_start))
    app.add_handler(CommandHandler("stats",  handle_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot polling…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
