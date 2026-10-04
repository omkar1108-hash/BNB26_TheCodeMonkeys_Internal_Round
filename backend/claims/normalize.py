"""Deterministic normalisation helpers used by the contradiction engine.

Everything here is plain Python (no LLM) so that cross-modal contradictions are
reproducible and cannot be hallucinated.
"""
import re
from datetime import date
from typing import Optional

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
    )
}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def parse_date(value: str) -> Optional[date]:
    """Parse the date formats TrustLayer extracts (and EXIF) into a ``date``."""
    if not value:
        return None
    v = str(value).strip()

    # EXIF "2026:09:15 10:00:00" or ISO "2026-09-15" / "2026/09/15"
    m = re.match(r"^(\d{4})[:\-/](\d{1,2})[:\-/](\d{1,2})", v)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # "15 September 2026"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})\b", v)
    if m and m.group(2).lower() in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))

    # "September 15, 2026"
    m = re.search(r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", v)
    if m and m.group(1).lower() in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))

    # "15/09/2026" (day first, as is standard in India / UK)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", v)
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    return None


def _safe_date(y: int, mo: int, d: int) -> Optional[date]:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().replace(",", " , ").split()).replace(" ,", ",")


def _city(loc: str) -> str:
    """Last comma-separated component, e.g. 'Convention Center, Pune' -> 'pune'."""
    parts = [p.strip() for p in str(loc).lower().split(",") if p.strip()]
    return parts[-1] if parts else ""


def locations_compatible(a: str, b: str) -> bool:
    """True if two location strings could refer to the same place.

    'Mumbai' and 'Aditya Institute, Mumbai' are compatible; 'Mumbai' and 'Pune'
    are not.
    """
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return True
    if na == nb or na in nb or nb in na:
        return True
    ca, cb = _city(a), _city(b)
    return bool(ca) and ca == cb


_HONORIFICS = {"dr", "prof", "professor", "mr", "mrs", "ms", "shri", "smt", "sir", "spokesperson"}


def _name_tokens(name: str) -> set:
    toks = re.findall(r"[a-z]+", str(name).lower())
    return {t for t in toks if t not in _HONORIFICS}


def speakers_compatible(a: str, b: str) -> bool:
    """'Dr. Rajesh Sharma' and 'Rajesh Sharma' match; 'Rajesh Sharma' and 'Anita Desai' do not."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return True
    return ta <= tb or tb <= ta
