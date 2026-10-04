import json
import re
from typing import Optional, List
import httpx

from backend.config import OLLAMA_HOST, OLLAMA_MODEL, is_ollama_online
from backend.models.bundle import ExtractedClaims


def extract_claims_rule_based(text: str) -> ExtractedClaims:
    """
    Deterministic rule-based fallback extractor for short fields:
    entities, dates, locations, claimed_speaker.
    """
    if not text:
        return ExtractedClaims()

    # 1. Dates
    date_patterns = [
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    ]
    dates = []
    for pat in date_patterns:
        dates.extend(re.findall(pat, text, re.IGNORECASE))
    dates = list(dict.fromkeys(dates))  # deduplicate

    # 2. Locations
    locations = []
    loc_match = re.search(r'\b(?:at|in|near)\s+([A-Z][a-zA-Z\s,]+?)(?:,\s+on\s+|\s+on\s+|\s+dated|\.|\n|$)', text)
    if loc_match:
        loc_cand = loc_match.group(1).strip().rstrip(".,")
        if len(loc_cand) > 2:
            locations.append(loc_cand)

    # 3. Entities / Organizers
    entities = []
    org_match = re.search(r'\b(?:organized by|organised by|announced by|issued by)\s+([A-Z][a-zA-Z\s]+?)(?:\.|\n|$)', text, re.IGNORECASE)
    if org_match:
        entities.append(org_match.group(1).strip().rstrip(".,"))

    # 4. Claimed Speaker
    claimed_speaker = None
    spk_match = re.search(
        r'(?i:\b(?:speaker|spokesperson|addressed by|statement by|speech by|said by))\s*:?\s*'
        r'((?:(?:Dr|Prof|Mr|Mrs|Ms|Shri|Smt)\.?\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        text
    )
    if spk_match:
        claimed_speaker = spk_match.group(1).strip().rstrip(".,")

    return ExtractedClaims(
        entities=entities,
        dates=dates,
        locations=locations,
        claimed_speaker=claimed_speaker,
        is_fallback=True,
        fallback_reason="Extracted via rule-based regex fallback"
    )


def extract_claims(text: Optional[str]) -> ExtractedClaims:
    """
    Extracts short factual fields (entities, dates, locations, claimed_speaker)
    using local Ollama with JSON schema enforcement, temperature 0, 1 retry,
    and rule-based fallback.
    """
    if not text or not text.strip():
        return ExtractedClaims()

    if not is_ollama_online():
        return extract_claims_rule_based(text)

    prompt = (
        "Extract factual claim fields from the text. Return only short entities, "
        "dates, locations, and claimed speaker if mentioned.\n\n"
        f"TEXT:\n{text.strip()}\n"
    )

    schema = ExtractedClaims.model_json_schema()
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0.0
        }
    }

    # Attempt with 1 retry
    for attempt in range(2):
        try:
            timeout_cfg = httpx.Timeout(read=10.0, connect=1.0)
            with httpx.Client(timeout=timeout_cfg) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    raw_data = res.json().get("response", "{}")
                    parsed = json.loads(raw_data)
                    claims = ExtractedClaims.model_validate(parsed)
                    claims.is_fallback = False
                    claims.fallback_reason = None
                    return claims
        except Exception:
            if attempt == 1:
                break

    # If Ollama unavailable or failed twice, use deterministic fallback
    return extract_claims_rule_based(text)