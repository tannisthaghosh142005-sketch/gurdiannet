from typing import Tuple, List
import re

# Expanded Algospeak and Slang Map
# Drawn from real-world online platforms (Reddit, Quora, TikTok)
# Each entry: (category, risk_score)
SLANG_MAP = {
    # Self-Harm & Suicide (Bypassing filters)
    "unalive": ("self_harm", 0.95),
    "sewer slide": ("self_harm", 0.92),
    "suicide": ("self_harm", 0.98),
    "commit": ("self_harm", 0.65),
    "final exit": ("self_harm", 0.88),
    
    # Harassment & Hate Speech
    "grape": ("harassment", 0.95), # Algospeak for rape
    "rape": ("harassment", 0.98),
    "snowflake": ("harassment", 0.70),
    "retard": ("hate_speech", 0.95),
    
    # CSAM & Grooming (Coded)
    "pdf file": ("csam", 0.95), # Algospeak for pedophile
    "pedophile": ("csam", 0.98),
    "corn": ("csam", 0.90),
    "loli": ("csam", 0.95),
    
    # Radicalization & Violence
    "pew pew": ("radicalization", 0.85),
    "boom": ("radicalization", 0.40),
    "based": ("radicalization", 0.30), # Contextual
    "redpill": ("radicalization", 0.65),
    "bluepill": ("radicalization", 0.40),
    "insurrection": ("radicalization", 0.85),
    
    # Misinformation & Spam
    "fake news": ("misinformation", 0.70),
    "plandemic": ("misinformation", 0.85),
    "microchip": ("misinformation", 0.60),
    "flat earth": ("misinformation", 0.30),
    "win money": ("spam", 0.90),
    "claim prize": ("spam", 0.95),
    "free crypto": ("spam", 0.90),
}

# Regex patterns for more complex detection
REGEX_PATTERNS = {
    "self_harm": [r"kill mys.*lf", r"want to die", r"end it all"],
    "hate_speech": [r"i hate (all )?[\w\s]+", r"[\w\s]+ are subhuman"],
    "radicalization": [r"destroy the (government|system)", r"join the fight"],
}

def detect_slang_threats(text: str) -> Tuple[bool, str, float]:
    """Detects threats based on expanded slang map and regex."""
    low_text = text.lower()
    
    # Check slang map
    for slang, (cat, conf) in SLANG_MAP.items():
        if slang in low_text:
            return True, cat, conf
            
    # Check regex
    for cat, patterns in REGEX_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, low_text):
                return True, cat, 0.90
                
    return False, "none", 0.0

def detect_logical_contradictions(text: str) -> float:
    """Mock-up for detecting logical inconsistencies in misinformation."""
    contradictions = [("orphan", "parents"), ("square", "circle"), ("free", "$"), ("guaranteed", "risk")]
    for w1, w2 in contradictions:
        if w1 in text.lower() and w2 in text.lower():
            return 0.85
    return 0.0

def keyword_threat_detection(text: str) -> List[str]:
    """Returns list of categories triggered by keywords."""
    found = []
    low_text = text.lower()
    for slang, (cat, _) in SLANG_MAP.items():
        if slang in low_text:
            found.append(cat)
    return list(set(found))
