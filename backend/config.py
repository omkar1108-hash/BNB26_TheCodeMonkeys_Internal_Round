from pathlib import Path
import os
import time

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DATASETS_DIR = Path(os.getenv("DATASET_ROOT", str(DATA_DIR / "datasets")))

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration (Ollama only)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

_OLLAMA_CACHE = {"online": None, "last_check": 0.0}

def is_ollama_online(timeout: float = 0.5) -> bool:
    """Checks if local Ollama daemon is actively reachable (cached for 10 seconds)."""
    now = time.time()
    if _OLLAMA_CACHE["online"] is not None and (now - _OLLAMA_CACHE["last_check"] < 10.0):
        return _OLLAMA_CACHE["online"]
        
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            res = client.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags")
            online = (res.status_code == 200)
    except Exception:
        online = False
        
    _OLLAMA_CACHE["online"] = online
    _OLLAMA_CACHE["last_check"] = now
    return online

# Hardware / Device Configuration
def get_device() -> str:
    """Return 'cuda' if GPU is available and torch is installed, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

# Fusion & Verdict Target Classes
TARGET_CLASSES = [
    "authentic",
    "manipulated",
    "coordinated_synthetic",
    "insufficient_evidence"
]

# Default Confidence & Abstention thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
ENTROPY_ABSTAIN_THRESHOLD = float(os.getenv("ENTROPY_ABSTAIN_THRESHOLD", "1.25"))
