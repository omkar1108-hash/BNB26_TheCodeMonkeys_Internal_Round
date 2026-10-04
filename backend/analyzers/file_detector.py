from pathlib import Path


SUPPORTED_TYPES = {
    # Images
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",

    # Text & Documents
    ".txt": "text",
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",

    # Audio
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".m4a": "audio",

    # Metadata
    ".json": "metadata"
}


def detect_file_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return SUPPORTED_TYPES.get(extension, "unknown")