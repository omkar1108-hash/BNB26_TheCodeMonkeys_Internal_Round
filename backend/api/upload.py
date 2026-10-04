from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil

from backend.analyzers.file_detector import detect_file_type
from backend.config import UPLOADS_DIR as UPLOAD_DIR


router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    file_id = str(uuid.uuid4())

    safe_filename = Path(file.filename).name

    modality = detect_file_type(safe_filename)

    if modality == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    file_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

    return {
        "success": True,
        "file_id": file_id,
        "filename": safe_filename,
        "modality": modality,
        "path": str(file_path)
    }