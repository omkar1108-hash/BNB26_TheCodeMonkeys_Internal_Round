import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body, BackgroundTasks
from backend.models.bundle import BundleInput, VerdictResult
from backend.verdict.engine import analyze_bundle
from backend.config import UPLOADS_DIR

router = APIRouter(prefix="/api/bundle", tags=["Bundle Verification"])

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ALLOWED_DOC_EXTS = {".txt", ".pdf", ".docx", ".doc"}


def _validate_and_save(upload: UploadFile, dest_path: Path, allowed_exts: set[str], expected_type: str) -> None:
    filename = upload.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}' for {expected_type} '{filename}'. Allowed: {sorted(allowed_exts)}"
        )
    ct = (upload.content_type or "").lower()
    if ct and ct != "application/octet-stream":
        if expected_type == "image" and not ct.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid MIME type '{ct}' for image upload.")
        elif expected_type == "audio" and not (ct.startswith("audio/") or "audio" in ct or ct == "video/ogg"):
            raise HTTPException(status_code=400, detail=f"Invalid MIME type '{ct}' for audio upload.")
        elif expected_type == "document" and not (ct.startswith("text/") or "pdf" in ct or "word" in ct or "document" in ct):
            raise HTTPException(status_code=400, detail=f"Invalid MIME type '{ct}' for document upload.")

    total_size = 0
    chunk_size = 64 * 1024
    with open(dest_path, "wb") as f:
        while True:
            chunk = upload.file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                f.close()
                if dest_path.exists():
                    dest_path.unlink()
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{filename}' exceeds maximum allowed size of 25 MB"
                )
            f.write(chunk)


@router.post("/analyze", response_model=VerdictResult)
async def analyze_bundle_json(bundle: BundleInput = Body(...)):
    """
    Direct JSON payload endpoint for automated evaluation pipelines and programmatic clients.
    """
    try:
        verdict = analyze_bundle(bundle)
        return verdict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bundle analysis failed: {str(e)}")


@router.post("/analyze-upload", response_model=VerdictResult)
async def analyze_bundle_multipart(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None),
    extra_documents: Optional[list[UploadFile]] = File(None),
    caption: Optional[str] = Form(None),
    metadata_json: Optional[str] = Form(None),
    claimed_speaker: Optional[str] = Form(None)
):
    """
    Multipart upload endpoint for the Streamlit web interface and end-user uploads.
    Accepts arbitrary combinations of image, audio, document, text caption, and metadata.
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)

    image_path = None
    audio_path = None
    extracted_text = caption or ""
    documents: dict = {}

    try:
        # Save image
        if image and image.filename:
            p = session_dir / f"image_{Path(image.filename).name}"
            _validate_and_save(image, p, ALLOWED_IMAGE_EXTS, "image")
            image_path = str(p)

        # Save audio
        if audio and audio.filename:
            p = session_dir / f"audio_{Path(audio.filename).name}"
            _validate_and_save(audio, p, ALLOWED_AUDIO_EXTS, "audio")
            audio_path = str(p)

        # Save and extract document text
        if document and document.filename:
            p = session_dir / f"doc_{Path(document.filename).name}"
            _validate_and_save(document, p, ALLOWED_DOC_EXTS, "document")
            from backend.analyzers.document_analyzer import extract_document_text
            doc_text = extract_document_text(str(p))
            if doc_text:
                documents["document"] = doc_text

        # Additional independent text sources, each kept separate so they can be cross-checked
        for k, extra in enumerate(extra_documents or [], start=1):
            if extra and extra.filename:
                p = session_dir / f"extra{k}_{Path(extra.filename).name}"
                _validate_and_save(extra, p, ALLOWED_DOC_EXTS, "document")
                from backend.analyzers.document_analyzer import extract_document_text
                extra_text = extract_document_text(str(p))
                if extra_text:
                    documents[Path(extra.filename).stem or f"extra{k}"] = extra_text

        # Parse metadata JSON
        meta_dict = None
        if metadata_json and metadata_json.strip():
            try:
                meta_dict = json.loads(metadata_json)
            except Exception:
                meta_dict = {"raw_metadata_text": metadata_json}

        bundle = BundleInput(
            bundle_id=session_id,
            image_path=image_path,
            text=extracted_text if extracted_text else None,
            documents=documents or None,
            audio_path=audio_path,
            metadata=meta_dict,
            claimed_speaker=claimed_speaker if claimed_speaker else None
        )

        result = analyze_bundle(bundle)
        return result

    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Multimodal bundle ingestion failed: {str(e)}")
