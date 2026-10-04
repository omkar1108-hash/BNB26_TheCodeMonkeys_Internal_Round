from pathlib import Path
from typing import Optional, Tuple
from backend.learned import registry


def verify_claimed_speaker(
    audio_path: Optional[str],
    claimed_speaker: Optional[str]
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Evaluates acoustic consistency between the audio recording and claimed speaker identity.
    Returns: (speaker_similarity [0, 1], is_fallback, fallback_reason)
    """
    if not audio_path or not claimed_speaker or not Path(audio_path).exists():
        return None, False, None

    if not registry.enabled():
        return None, False, None

    try:
        from speechbrain.inference.speaker import SpeakerRecognition  # lazy import
        # SpeechBrain ECAPA-TDNN would compare the audio embedding against an enrolled reference voice
        return None, True, "SpeechBrain ECAPA-TDNN requires an enrolled reference voice profile"
    except ImportError:
        return None, True, "SpeechBrain ECAPA-TDNN not installed; acoustic speaker verification unavailable"
    except Exception as e:
        return None, True, f"Speaker verification error: {str(e)}"


