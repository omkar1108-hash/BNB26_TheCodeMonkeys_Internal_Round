import re


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_dates(text):
    patterns = [
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
    ]

    dates = []

    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))

    return dates


def extract_location(text):
    """
    Basic location extraction.

    Example:
    'held at ABC Institute, Mumbai, on 15 September 2026'
    """

    match = re.search(
        r'\bat\s+(.+?)(?:,\s+on\s+|\s+on\s+)',
        text,
        re.IGNORECASE
    )

    if match:
        location = match.group(1).strip()
        return location.rstrip("., ")

    return None


def extract_organizer(text):
    match = re.search(
        r'(?:organized by|organised by)\s+(.+?)(?:\.|$)',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None


def extract_claims(text, file_id=None):
    claims = []

    sentences = split_into_sentences(text)

    for index, sentence in enumerate(sentences, start=1):

        dates = extract_dates(sentence)

        organizer = extract_organizer(sentence)
        location = extract_location(sentence)

        claim = {
            "claim_id": f"{file_id}_claim_{index}" if file_id else f"claim_{index}",
            "text": sentence,
            "who": organizer,
            "what": sentence,
            "when": dates[0] if dates else None,
            "where": location,
            "file_id": file_id
        }

        claims.append(claim)

    return claims