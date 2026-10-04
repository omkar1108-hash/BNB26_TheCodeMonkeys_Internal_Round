from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import uuid
import shutil

from backend.analyzers.file_detector import detect_file_type
from backend.config import UPLOADS_DIR as UPLOAD_DIR

router = APIRouter(prefix="/upload", tags=["Upload"])

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


@router.post("/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
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
            detail=f"Unsupported file type '{Path(safe_filename).suffix}'"
        )

    session_dir = UPLOAD_DIR / file_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / safe_filename
    background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)

    try:
        total_size = 0
        chunk_size = 64 * 1024
        with file_path.open("wb") as buffer:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    buffer.close()
                    shutil.rmtree(session_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{safe_filename}' exceeds maximum allowed size of 25 MB"
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
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