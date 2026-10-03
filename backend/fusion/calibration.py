from typing import Dict, List
import numpy as np


class TemperatureScaler:
    """
    Temperature Scaling calibration.
    Rescales class logits to align predicted confidence with true empirical accuracy.
    """
    def __init__(self, temperature: float = 1.20):
        self.temperature = max(0.1, float(temperature))

    def calibrate_probabilities(
        self,
        raw_probabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Converts probability distribution to temperature-scaled probabilities.
        """
        labels = list(raw_probabilities.keys())
        probs = np.array([raw_probabilities[k] for k in labels], dtype=np.float64)
        
        # Avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        # Convert back to pseudo-logits
        logits = np.log(probs)
        
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Stable Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)
        
        return {
            label: float(calibrated_probs[i])
            for i, label in enumerate(labels)
        }

