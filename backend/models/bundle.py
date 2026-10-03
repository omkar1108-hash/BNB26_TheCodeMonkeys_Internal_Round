from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class BundleInput(BaseModel):
    bundle_id: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    claimed_speaker: Optional[str] = None


class ModalityEvidence(BaseModel):
    modality: str
    present: bool = True
    score: float = Field(0.0, ge=0.0, le=1.0, description="Synthetic / manipulation anomaly score (0=clean, 1=synthetic/manipulated)")
    evidence: str = ""
    features: Dict[str, float] = Field(default_factory=dict)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class ExtractedClaims(BaseModel):
    entities: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    claimed_speaker: Optional[str] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class ContradictionItem(BaseModel):
    field: str
    value_a: str
    value_b: str
    source_a: str
    source_b: str
    description: str
    severity: str = "medium"  # low, medium, high


class CrossModalEvidence(BaseModel):
    image_text_similarity: Optional[float] = None
    speaker_similarity: Optional[float] = None
    metadata_conflicts: List[str] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class EvidenceItem(BaseModel):
    rank: int
    category: str  # modality, cross_modal, metadata
    modality: Optional[str] = None
    description: str
    importance: float = 0.0
    is_fallback: bool = False


class VerdictResult(BaseModel):
    bundle_id: str
    label: str  # authentic, manipulated, coordinated_synthetic, insufficient_evidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    abstain_reason: Optional[str] = None
    ranked_evidence: List[EvidenceItem] = Field(default_factory=list)
    report: str = ""
    report_verified: bool = True
    is_fallback: bool = False
    modality_results: Dict[str, ModalityEvidence] = Field(default_factory=dict)
    cross_modal_results: Optional[CrossModalEvidence] = None
