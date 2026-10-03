import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from backend.models.bundle import BundleInput, VerdictResult
from backend.verdict.engine import analyze_bundle
from backend.config import UPLOADS_DIR

router = APIRouter(prefix="/api/bundle", tags=["Bundle Verification"])


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
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None),
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

    image_path = None
    audio_path = None
    extracted_text = caption or ""

    try:
        # Save image
        if image and image.filename:
            image_path = str(session_dir / f"image_{Path(image.filename).name}")
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

        # Save audio
        if audio and audio.filename:
            audio_path = str(session_dir / f"audio_{Path(audio.filename).name}")
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

        # Save and extract document text
        if document and document.filename:
            doc_path = str(session_dir / f"doc_{Path(document.filename).name}")
            with open(doc_path, "wb") as f:
                shutil.copyfileobj(document.file, f)

            from backend.analyzers.document_analyzer import extract_document_text
            doc_text = extract_document_text(doc_path)
            if doc_text:
                extracted_text = f"{extracted_text}\n\n{doc_text}".strip()

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
            audio_path=audio_path,
            metadata=meta_dict,
            claimed_speaker=claimed_speaker if claimed_speaker else None
        )

        result = analyze_bundle(bundle)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal bundle ingestion failed: {str(e)}")
