import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

os.environ["HF_HOME"] = str(PROJECT_ROOT / "artifacts" / "models")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(PROJECT_ROOT / "artifacts" / "models")

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import yaml

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

SETTINGS = load_config()

RAW_DATA_DIR = PROJECT_ROOT / SETTINGS["paths"]["raw_data"]
PROCESSED_DATA_DIR = PROJECT_ROOT / SETTINGS["paths"]["processed_data"]
VECTOR_DB_DIR = PROJECT_ROOT / SETTINGS["paths"]["vector_db"]
MODELS_CACHE_DIR = PROJECT_ROOT / SETTINGS["paths"]["models_cache"]
EVALUATION_DIR = PROJECT_ROOT / SETTINGS["paths"]["evaluation"]

NLTK_DIR = MODELS_CACHE_DIR / "nltk_data"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

NLTK_DIR.mkdir(parents=True, exist_ok=True)

(PROJECT_ROOT / "artifacts" / "phoenix").mkdir(parents=True, exist_ok=True)