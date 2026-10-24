"""Environment + tunable thresholds (config/thresholds.json)."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # repo root
API_DIR = ROOT / "api"
load_dotenv(ROOT / ".env")
load_dotenv(API_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_data_dir = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR = _data_dir if _data_dir.is_absolute() else (ROOT / _data_dir).resolve()
DEV_DATA_DIR = ROOT / "data_dev"      # synthetic stand-in, used ONLY if DATA_DIR has no organiser files
CONFIG_DIR = API_DIR / "config"
REPORTS_DIR = ROOT / "reports"

THRESHOLDS_PATH = CONFIG_DIR / "thresholds.json"
THRESHOLDS: dict = {}


def reload_thresholds() -> dict:
    THRESHOLDS.clear()
    THRESHOLDS.update(json.loads(THRESHOLDS_PATH.read_text()))
    return THRESHOLDS


reload_thresholds()

SAMPLE_LOCATION = {"lat": 28.54, "lng": 77.27}
