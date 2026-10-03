from pathlib import Path


SUPPORTED_TYPES = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",

    ".txt": "text",

    ".pdf": "document",
    ".docx": "document",
    ".doc": "document"
}


def detect_file_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    return SUPPORTED_TYPES.get(extension, "unknown")