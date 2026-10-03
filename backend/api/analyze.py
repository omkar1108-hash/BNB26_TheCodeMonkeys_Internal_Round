from fastapi import APIRouter, HTTPException
from pathlib import Path

from backend.analyzers.file_detector import detect_file_type
from backend.analyzers.document_analyzer import extract_document_text
from backend.claims.extractor import extract_claims
from backend.claims.conflict import find_conflicts
from backend.claims.evidence_graph import build_evidence_graph
from backend.verdict.engine import generate_verdict
from backend.config import UPLOADS_DIR


router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"]
)


def analyze_file(
    file_path: Path,
    file_id: str,
    filename: str
):
    """
    Analyze a single uploaded file.

    Currently supports:
    - Text files
    - PDF documents
    - DOCX documents

    Image/audio analysis will be added later.
    """

    modality = detect_file_type(filename)

    if modality == "unknown":
        raise ValueError(
            f"Unsupported file type: {filename}"
        )

    extracted_text = ""
    claims = []

    # ---------------------------------------------------------
    # TEXT / DOCUMENT
    # ---------------------------------------------------------

    if modality in ("document", "text"):

        extracted_text = extract_document_text(
            str(file_path)
        )

        claims = extract_claims(
            extracted_text,
            file_id
        )

    # ---------------------------------------------------------
    # ATTACH FILE ID
    # ---------------------------------------------------------

    for claim in claims:
        claim["file_id"] = file_id

    return {
        "file_id": file_id,
        "filename": filename,
        "modality": modality,
        "text_length": len(extracted_text),
        "claims": claims
    }


@router.post("/")
async def analyze_files(file_ids: list[str]):
    """
    Analyze multiple previously uploaded files.

    Request body:

    [
        "file-id-1",
        "file-id-2"
    ]
    """

    # ---------------------------------------------------------
    # VALIDATE INPUT
    # ---------------------------------------------------------

    if not file_ids:
        raise HTTPException(
            status_code=400,
            detail="No file IDs provided"
        )

    upload_dir = UPLOADS_DIR

    if not upload_dir.exists():
        raise HTTPException(
            status_code=500,
            detail="Upload directory does not exist"
        )

    analyzed_files = []
    all_claims = []

    # ---------------------------------------------------------
    # ANALYZE EACH FILE
    # ---------------------------------------------------------

    for file_id in file_ids:

        matching_files = list(
            upload_dir.glob(f"{file_id}_*")
        )

        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_id}"
            )

        file_path = matching_files[0]

        # Remove UUID prefix to recover original filename
        filename = file_path.name[
            len(file_id) + 1:
        ]

        try:

            result = analyze_file(
                file_path=file_path,
                file_id=file_id,
                filename=filename
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to analyze {filename}: {str(e)}"
                )
            )

        analyzed_files.append({
            "file_id": file_id,
            "filename": filename,
            "modality": result["modality"]
        })

        all_claims.extend(
            result["claims"]
        )

    # ---------------------------------------------------------
    # FIND CONFLICTS
    # ---------------------------------------------------------

    conflicts = find_conflicts(
        all_claims
    )

    # ---------------------------------------------------------
    # BUILD EVIDENCE GRAPH
    # ---------------------------------------------------------

    graph = build_evidence_graph(
        analyzed_files,
        all_claims,
        conflicts
    )

    # ---------------------------------------------------------
    # GENERATE VERDICT
    # ---------------------------------------------------------

    verdict = generate_verdict(
        analyzed_files,
        all_claims,
        conflicts
    )

    # ---------------------------------------------------------
    # FINAL RESPONSE
    # ---------------------------------------------------------

    return {
        "success": True,

        "verdict": verdict,

        "files": analyzed_files,

        "claims": all_claims,

        "conflicts": conflicts,

        "evidence_graph": graph
    }