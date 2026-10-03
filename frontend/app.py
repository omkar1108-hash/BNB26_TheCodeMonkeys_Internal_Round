import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import io
import json
import os
import requests
import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="TrustLayer Multimodal Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for modern styling
st.markdown("""
<style>
    .verdict-box {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
        font-weight: 700;
    }
    .verdict-authentic { background: linear-gradient(135deg, #10b981, #059669); }
    .verdict-manipulated { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .verdict-coordinated_synthetic { background: linear-gradient(135deg, #ef4444, #b91c1c); }
    .verdict-insufficient_evidence { background: linear-gradient(135deg, #6b7280, #4b5563); }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🛡️ TrustLayer: Multimodal Verification Engine")
st.caption("Cross-modal forensic analysis for detecting manipulated artifacts & coordinated synthetic disinformation campaigns")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=1.0)
        if res.status_code == 200:
            st.success(f"FastAPI Backend: Online ({BACKEND_URL})")
        else:
            st.warning("FastAPI Backend: Degraded")
    except Exception:
        st.info("FastAPI Backend: Offline (Operating with In-Process Engine)")

    st.divider()
    st.header("📂 Demo Case Loader")
    demo_option = st.selectbox(
        "Select a pre-built demo bundle:",
        [
            "Custom Upload",
            "Demo 1: Authentic Content",
            "Demo 2: Single-Artifact Manipulated",
            "Demo 3: Coordinated Synthetic Campaign"
        ]
    )

# Pre-load demo variables
default_text = ""
default_meta = ""
default_speaker = ""

if demo_option == "Demo 1: Authentic Content":
    default_text = (
        "TechFest 2026 was held at Aditya Institute of Management Studies and Research, "
        "Mumbai, on 15 September 2026. The event was organized by the Computer Department."
    )
    default_meta = json.dumps({"DateTimeOriginal": "2026:09:15 10:00:00", "Make": "Sony", "Model": "Alpha A7 IV"}, indent=2)
    default_speaker = "Dr. Rajesh Sharma"

elif demo_option == "Demo 2: Single-Artifact Manipulated":
    default_text = (
        "TechFest 2026 took place at Aditya Institute of Management Studies and Research, "
        "Mumbai, on 18 September 2026. The Computer Department organized the event."
    )
    default_meta = json.dumps({"Software": "Adobe Photoshop 2024 (Macintosh)", "DateTime": "2026:09:20 18:30:00"}, indent=2)
    default_speaker = ""

elif demo_option == "Demo 3: Coordinated Synthetic Campaign":
    default_text = (
        "TechFest 2026 was held at Aditya Institute of Management Studies and Research, "
        "Pune, on 15 September 2026. The Computer Department organized the event."
    )
    default_meta = json.dumps({"DateTimeOriginal": "2026:09:18 14:00:00", "Make": "Canon", "Model": "EOS R5"}, indent=2)
    default_speaker = "Unknown Spokesperson"

# Multimodal Ingestion Interface
st.subheader("📦 Multimodal Bundle Ingestion")
col_img, col_aud = st.columns(2)

with col_img:
    uploaded_image = st.file_uploader("🖼️ Upload Image Artifact", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_image:
        st.image(uploaded_image, caption="Image Preview", use_container_width=True)

with col_aud:
    uploaded_audio = st.file_uploader("🎙️ Upload Audio Artifact", type=["wav", "mp3", "flac"])
    if uploaded_audio:
        st.audio(uploaded_audio)

col_txt, col_meta = st.columns(2)

with col_txt:
    caption_text = st.text_area(
        "📝 Text Caption / Article / Transcript",
        value=default_text,
        height=140,
        placeholder="Enter caption or narrative text..."
    )
    uploaded_doc = st.file_uploader("Or upload Document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

with col_meta:
    meta_json = st.text_area(
        "🏷️ Metadata JSON (EXIF, Provenance, Tags)",
        value=default_meta,
        height=140,
        placeholder='{"Software": "...", "DateTimeOriginal": "..."}'
    )
    claimed_speaker = st.text_input("Claimed Speaker Identity (Optional)", value=default_speaker)

# Trigger Analysis
analyze_btn = st.button("🚀 Analyze Multimodal Bundle", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner("Executing 4-layer verification pipeline..."):
        result_data = None

        # 1. Attempt API call
        try:
            files_dict = {}
            if uploaded_image:
                files_dict["image"] = (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
            if uploaded_audio:
                files_dict["audio"] = (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)
            if uploaded_doc:
                files_dict["document"] = (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type)

            data_payload = {
                "caption": caption_text,
                "metadata_json": meta_json,
                "claimed_speaker": claimed_speaker
            }

            res = requests.post(
                f"{BACKEND_URL}/api/bundle/analyze-upload",
                files=files_dict if files_dict else None,
                data=data_payload,
                timeout=15.0
            )
            if res.status_code == 200:
                result_data = res.json()
        except Exception:
            pass

        # 2. In-process fallback if backend is unreachable
        if not result_data:
            from backend.models.bundle import BundleInput
            from backend.verdict.engine import analyze_bundle
            
            meta_dict = None
            if meta_json.strip():
                try:
                    meta_dict = json.loads(meta_json)
                except Exception:
                    meta_dict = {"raw": meta_json}

            # Save temporary files if present
            uploads_dir = _REPO_ROOT / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            temp_img_path = None
            temp_aud_path = None
            if uploaded_image:
                temp_img_path = str(uploads_dir / f"temp_{uploaded_image.name}")
                with open(temp_img_path, "wb") as f:
                    f.write(uploaded_image.getvalue())
            if uploaded_audio:
                temp_aud_path = str(uploads_dir / f"temp_{uploaded_audio.name}")
                with open(temp_aud_path, "wb") as f:
                    f.write(uploaded_audio.getvalue())

            bundle_obj = BundleInput(
                image_path=temp_img_path,
                text=caption_text if caption_text.strip() else None,
                audio_path=temp_aud_path,
                metadata=meta_dict,
                claimed_speaker=claimed_speaker if claimed_speaker.strip() else None
            )
            verdict_res = analyze_bundle(bundle_obj)
            result_data = verdict_res.model_dump()

    # Render Results Dashboard
    if result_data:
        st.divider()
        label = result_data["label"]
        confidence = result_data["confidence"]
        abstained = result_data.get("abstained", False)
        abstain_reason = result_data.get("abstain_reason")
        is_fallback = result_data.get("is_fallback", False)

        # Verdict Header Banner
        clean_label = label.replace("_", " ").upper()
        st.markdown(
            f"""
            <div class="verdict-box verdict-{label}">
                <h1 style="margin: 0; color: white;">{clean_label}</h1>
                <h3 style="margin: 8px 0 0 0; color: white;">Calibrated Confidence: {confidence:.1%}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if abstained:
            st.warning(f"⚠️ **Abstention Triggered**: {abstain_reason}")

        if is_fallback:
            st.info("ℹ️ Note: One or more detectors utilized lightweight or mock fallbacks for this run.")

        # Class Probabilities Distribution
        st.subheader("📊 Calibrated Class Probabilities")
        probs = result_data.get("probabilities", {})
        if probs:
            prob_df = pd.DataFrame({
                "Class": [k.replace("_", " ").title() for k in probs.keys()],
                "Probability": [v for v in probs.values()]
            })
            st.bar_chart(prob_df.set_index("Class"))

        # Layer 1 & Layer 2 Breakdown
        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.subheader("🔬 Layer 1: Per-Modality Anomaly Scores")
            mod_results = result_data.get("modality_results", {})
            for mod, data in mod_results.items():
                if data.get("present"):
                    score = data.get("score", 0.0)
                    evidence = data.get("evidence", "")
                    st.markdown(
                        f"""
                        <div class="card">
                            <b>{mod.upper()}</b> — Anomaly Score: <code>{score:.2f}</code><br>
                            <small>{evidence}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="card" style="opacity: 0.6;">
                            <b>{mod.upper()}</b>: <i>Artifact not provided (imputed)</i>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with col_l2:
            st.subheader("🔗 Layer 2: Cross-Modal Coherence")
            cross_modal = result_data.get("cross_modal_results") or {}
            
            sim = cross_modal.get("image_text_similarity")
            if sim is not None:
                st.metric("Image-Caption Semantic Similarity", f"{sim:.2f}")
                
            contradictions = cross_modal.get("contradictions", [])
            meta_conflicts = cross_modal.get("metadata_conflicts", [])

            if contradictions:
                st.error(f"❌ {len(contradictions)} Cross-Modal Factual Contradiction(s) Flagged:")
                for c in contradictions:
                    st.write(f"- **{c.get('field', 'Conflict').upper()}**: {c.get('description')}")
            else:
                st.success("✅ No direct cross-modal factual contradictions detected.")

            if meta_conflicts:
                st.warning("⚠️ Metadata Conflicts:")
                for mc in meta_conflicts:
                    st.write(f"- {mc}")

        # Layer 4: Ranked Evidence & LLM Report
        st.subheader("📋 Layer 4: Ranked Forensic Evidence")
        ranked_ev = result_data.get("ranked_evidence", [])
        if ranked_ev:
            ev_df = pd.DataFrame([
                {
                    "Rank": e["rank"],
                    "Category": e["category"].title(),
                    "Modality": (e.get("modality") or "N/A").upper(),
                    "Finding": e["description"],
                    "Importance": f"{e['importance']:.2f}"
                }
                for e in ranked_ev
            ])
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

        st.subheader("📝 Grounded Forensic Intelligence Report")
        report_text = result_data.get("report", "")
        verified = result_data.get("report_verified", True)
        
        if verified:
            st.success("✅ Report Verified: Strictly grounded against structured findings.")
        else:
            st.info("ℹ️ Standard Fallback Template Report:")

        st.text_area("Forensic Summary", value=report_text, height=220)
