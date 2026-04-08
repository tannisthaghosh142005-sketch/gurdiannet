from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MediaType(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class Language(str, Enum):
    EN = "en"
    HI = "hi"
    ZH = "zh"
    AR = "ar"
    ES = "es"
    FR = "fr"
    PT = "pt"

class HarmCategory(str, Enum):
    HATE_SPEECH      = "hate_speech"
    HARASSMENT       = "harassment"
    MISINFORMATION   = "misinformation"
    CSAM             = "csam"
    RADICALIZATION   = "radicalization"
    SELF_HARM        = "self_harm"
    SPAM             = "spam"
    DEEPFAKE         = "deepfake"
    DOXXING          = "doxxing"
    BENIGN           = "benign"
    AMBIGUOUS        = "ambiguous"

class Action(int, Enum):
    ALLOW      = 0
    FLAG       = 1
    DELETE     = 2
    WARN       = 3
    QUARANTINE = 4
    ESCALATE   = 5

class MediaContent(BaseModel):
    media_type: MediaType
    language: Language = Language.EN
    text: Optional[str] = None
    feature_vector: Optional[List[float]] = None
    transcript: Optional[str] = None
    ocr_text: Optional[str] = None

class Message(BaseModel):
    msg_id: int
    sender_id: int
    timestamp: float
    contents: List[MediaContent]
    embedding: List[float]
    true_harm_per_user: List[float]
    ground_truth_type: HarmCategory
    language: Language = Language.EN
    is_adversarial: bool = False
    context_dependent: bool = False
    difficulty: str = "easy"
    primary_strength: float = 0.8

class UserProfile(BaseModel):
    user_id: int
    role: str = "regular"
    trust_weight: float = 0.5
    preference_vector: List[float] = Field(default_factory=lambda: [0.5]*11)
    consistency_score: float = 0.5
    flags_made: int = 0
    flags_received: int = 0
    messages_sent: int = 0
    violation_count: int = 0
    is_blocked: bool = False
    language: Language = Language.EN