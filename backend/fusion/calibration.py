import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration.json"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class TemperatureScaler:
    """
    Temperature scaling: p_cal = softmax(log(p) / T).
    T is fitted on held-out data by minimising negative log-likelihood
    (see ``fit_temperature``) instead of being hard-coded.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.05, float(temperature))

    def calibrate_probabilities(self, raw_probabilities: Dict[str, float]) -> Dict[str, float]:
        labels = list(raw_probabilities.keys())
        probs = np.clip(np.array([raw_probabilities[k] for k in labels], dtype=np.float64), 1e-12, 1.0)
        cal = _softmax(np.log(probs) / self.temperature)
        return {label: float(cal[i]) for i, label in enumerate(labels)}

    def calibrate_matrix(self, probs: np.ndarray) -> np.ndarray:
        return _softmax(np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature)


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Fit T by 1-D minimisation of NLL on a calibration set."""
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        p = _softmax(np.log(np.clip(probs, 1e-12, 1.0)) / t)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(res.x)


def conformal_qhat(probs: np.ndarray, y_idx: np.ndarray, alpha: float = 0.1) -> float:
    """
    Split-conformal threshold. Nonconformity score = 1 - p(true class).
    A class enters the prediction set when 1 - p <= qhat, i.e. p >= 1 - qhat.
    Under exchangeability the set contains the true class with prob >= 1 - alpha.
    """
    n = len(y_idx)
    scores = 1.0 - probs[np.arange(n), y_idx]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def load_calibration(path: Path = CALIBRATION_PATH) -> Dict[str, object]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"temperature": 1.2, "conformal_qhat": None, "alpha": None, "fitted": False}
