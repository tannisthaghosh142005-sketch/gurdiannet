import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from cae_env.types import UserProfile, Language

DB_PATH = "user_profiles.db"

def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            role TEXT,
            trust_weight REAL,
            harm_sensitivity TEXT,
            consistency_score REAL,
            flags_made INTEGER DEFAULT 0,
            flags_received INTEGER DEFAULT 0,
            messages_sent INTEGER DEFAULT 0,
            reports_validated INTEGER DEFAULT 0,
            false_report_rate REAL DEFAULT 0.0,
            violation_count INTEGER DEFAULT 0,
            is_blocked INTEGER DEFAULT 0,
            warnings TEXT DEFAULT '[]',
            left_group INTEGER DEFAULT 0,
            language TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_user(user_id: int) -> Optional[UserProfile]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        data = dict(row)
        data["harm_sensitivity"] = json.loads(data["harm_sensitivity"])
        data["warnings"] = json.loads(data["warnings"])
        data["is_blocked"] = bool(data["is_blocked"])
        data["left_group"] = bool(data["left_group"])
        data["language"] = Language(data["language"])
        return UserProfile(**data)
    return None

def save_user(profile: UserProfile):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = profile.model_dump()
    data["harm_sensitivity"] = json.dumps(data["harm_sensitivity"])
    data["warnings"] = json.dumps(data["warnings"])
    data["is_blocked"] = 1 if data["is_blocked"] else 0
    data["left_group"] = 1 if data["left_group"] else 0
    data["language"] = data["language"].value
    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (profile.user_id,))
    if cursor.fetchone():
        update_data = {k: v for k, v in data.items() if k != "user_id"}
        fields = ", ".join([f"{k} = ?" for k in update_data.keys()])
        cursor.execute(f"UPDATE users SET {fields} WHERE user_id = ?", list(update_data.values()) + [profile.user_id])
    else:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data.keys()])
        cursor.execute(f"INSERT INTO users ({columns}) VALUES ({placeholders})", list(data.values()))
    conn.commit()
    conn.close()

def get_all_users() -> List[UserProfile]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    conn.close()
    users = []
    for row in rows:
        data = dict(row)
        data["harm_sensitivity"] = json.loads(data["harm_sensitivity"])
        data["warnings"] = json.loads(data["warnings"])
        data["is_blocked"] = bool(data["is_blocked"])
        data["left_group"] = bool(data["left_group"])
        data["language"] = Language(data["language"])
        users.append(UserProfile(**data))
    return users

def unblock_user(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_blocked = 0, violation_count = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def increment_violations(user_id: int, threshold: int = 3) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET violation_count = violation_count + 1 WHERE user_id = ?", (user_id,))
    cursor.execute("SELECT violation_count FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    count = row[0] if row else 0
    if count >= threshold:
        cursor.execute("UPDATE users SET is_blocked = 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return count
