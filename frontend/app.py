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
            "Demo 3: Coordinated Synthetic Campaign",
            "Demo 4: Insufficient Evidence",
        ]
    )
    st.caption("Demos run the full pipeline on the synthetic bundles in `data/demos/` (learned detectors are off for them; they apply to your own uploads).")

    st.divider()
    st.header("🧠 Learned detectors")
    from backend.learned import registry as _learned
    if not _learned.enabled():
        st.info("Disabled (TRUSTLAYER_LEARNED=off).")
    else:
        for _k, _v in _learned.status().items():
            _icon = "✅" if _v["cached"] else "⬇️"
            st.markdown(f"{_icon} **{_k.replace('_', ' ')}**  \n`{_v['model_id']}`")
            if _v.get("last_error"):
                st.caption(f"load error: {_v['last_error']}")
        if not all(v["cached"] for v in _learned.status().values()):
            st.caption("Run `python -m backend.learned.download` once to enable the ⬇️ models.")

DEMO_FOLDERS = {
    "Demo 1: Authentic Content": "case1_authentic",
    "Demo 2: Single-Artifact Manipulated": "case2_manipulated",
    "Demo 3: Coordinated Synthetic Campaign": "case3_coordinated_fake",
    "Demo 4: Insufficient Evidence": "case4_insufficient",
}
demo_folder = DEMO_FOLDERS.get(demo_option)
demo_bundle = None
if demo_folder:
    from evaluation.run_demos import load_case, DEMOS
    demo_dir = DEMOS / demo_folder
    demo_bundle = load_case(demo_dir)

tab_analyze, tab_eval = st.tabs(["🔍 Investigate a bundle", "📈 Evaluation & generalisation"])

with tab_analyze:
    # Multimodal Ingestion Interface
    st.subheader("📦 Multimodal Bundle Ingestion")
    if demo_bundle is not None:
        st.success(f"Loaded **{demo_option}** from `data/demos/{demo_folder}` — press Analyze.")
        d1, d2 = st.columns(2)
        with d1:
            if demo_bundle.image_path:
                st.image(demo_bundle.image_path, caption="Image artifact", width="stretch")
            if demo_bundle.audio_path:
                st.audio(demo_bundle.audio_path)
        with d2:
            for name, text in (demo_bundle.documents or {}).items():
                st.markdown(f"**📝 {name}**")
                st.caption(text)
            if demo_bundle.metadata:
                st.json(demo_bundle.metadata)
            if demo_bundle.claimed_speaker:
                st.caption(f"Claimed speaker: **{demo_bundle.claimed_speaker}**")
        uploaded_image = uploaded_audio = uploaded_doc = None
        extra_docs = []
        caption_text, meta_json, claimed_speaker = "", "", ""
    else:
        col_img, col_aud = st.columns(2)
        with col_img:
            uploaded_image = st.file_uploader("🖼️ Upload Image Artifact", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image:
                st.image(uploaded_image, caption="Image Preview", width="stretch")
        with col_aud:
            uploaded_audio = st.file_uploader("🎙️ Upload Audio Artifact", type=["wav", "mp3", "flac"])
            if uploaded_audio:
                st.audio(uploaded_audio)

        col_txt, col_meta = st.columns(2)
        with col_txt:
            caption_text = st.text_area(
                "📝 Text Caption / Article / Transcript",
                value="",
                height=140,
                placeholder="Enter caption or narrative text..."
            )
            docs_uploaded = st.file_uploader(
                "Or upload one or more text sources (.txt, .pdf, .docx) — each is cross-checked against the others",
                type=["txt", "pdf", "docx"], accept_multiple_files=True,
            )
            uploaded_doc = docs_uploaded[0] if docs_uploaded else None
            extra_docs = docs_uploaded[1:] if docs_uploaded else []

        with col_meta:
            meta_json = st.text_area(
                "🏷️ Metadata JSON (EXIF, Provenance, Tags)",
                value="",
                height=140,
                placeholder='{"DateTimeOriginal": "2026:09:15 10:00:00", "Location": "Mumbai", "Software": "..."}'
            )
            claimed_speaker = st.text_input("Claimed Speaker Identity (Optional)", value="")

    # Trigger Analysis
    analyze_btn = st.button("🚀 Analyze Multimodal Bundle", type="primary", width="stretch")

    if analyze_btn:
        with st.spinner("Executing 4-layer verification pipeline..."):
            result_data = None

            # Demo bundles are analysed in-process (they live on this machine's disk)
            if demo_bundle is not None:
                from backend.verdict.engine import analyze_bundle
                from backend.learned import registry as _reg
                with _reg.disabled():  # demo bundles are synthetic stand-ins, not real photos/voices
                    result_data = analyze_bundle(demo_bundle).model_dump()
            else:
                # 1. Attempt API call
                try:
                    files_list = []
                    if uploaded_image:
                        files_list.append(("image", (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)))
                    if uploaded_audio:
                        files_list.append(("audio", (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)))
                    if uploaded_doc:
                        files_list.append(("document", (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type)))
                    for extra in extra_docs:
                        files_list.append(("extra_documents", (extra.name, extra.getvalue(), extra.type)))

                    data_payload = {
                        "caption": caption_text,
                        "metadata_json": meta_json,
                        "claimed_speaker": claimed_speaker
                    }
                    res = requests.post(
                        f"{BACKEND_URL}/api/bundle/analyze-upload",
                        files=files_list if files_list else None,
                        data=data_payload,
                        timeout=30.0
                    )
                    if res.status_code == 200:
                        result_data = res.json()
                except Exception:
                    pass

                # 2. In-process fallback if backend is unreachable
                if not result_data:
                    from backend.models.bundle import BundleInput
                    from backend.verdict.engine import analyze_bundle
                    from backend.analyzers.document_analyzer import extract_document_text

                    meta_dict = None
                    if meta_json.strip():
                        try:
                            meta_dict = json.loads(meta_json)
                        except Exception:
                            meta_dict = {"raw": meta_json}

                    uploads_dir = _REPO_ROOT / "data" / "uploads"
                    uploads_dir.mkdir(parents=True, exist_ok=True)

                    def _save(f, prefix):
                        path = uploads_dir / f"{prefix}_{f.name}"
                        path.write_bytes(f.getvalue())
                        return str(path)

                    documents = {}
                    for k, f in enumerate([d for d in ([uploaded_doc] + list(extra_docs)) if d], start=1):
                        txt = extract_document_text(_save(f, "doc"))
                        if txt:
                            documents[Path(f.name).stem or f"doc{k}"] = txt

                    bundle_obj = BundleInput(
                        image_path=_save(uploaded_image, "temp") if uploaded_image else None,
                        text=caption_text if caption_text.strip() else None,
                        documents=documents or None,
                        audio_path=_save(uploaded_audio, "temp") if uploaded_audio else None,
                        metadata=meta_dict,
                        claimed_speaker=claimed_speaker if claimed_speaker.strip() else None
                    )
                    result_data = analyze_bundle(bundle_obj).model_dump()

        # Render Results Dashboard
        if result_data:
            st.divider()
            label = result_data["label"]
            confidence = result_data["confidence"]
            abstained = result_data.get("abstained", False)
            abstain_reason = result_data.get("abstain_reason")
            model_info = result_data.get("model_info") or {}
            conformal_set = result_data.get("conformal_set") or []

            clean_label = label.replace("_", " ").upper()
            st.markdown(
                f"""
                <div class="verdict-box verdict-{label}">
                    <h1 style="margin: 0; color: white;">{clean_label}</h1>
                    <h3 style="margin: 8px 0 0 0; color: white;">Calibrated confidence: {confidence:.1%}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if conformal_set:
                names = ", ".join(c.replace("_", " ") for c in conformal_set)
                st.caption(f"Conformal prediction set (90% coverage target): **{{{names}}}** — one class means the evidence singles out a verdict.")

            if abstained:
                st.warning(f"⚠️ **Abstained — not enough reliable evidence.** {abstain_reason}")

            # What would settle it?
            next_steps = result_data.get("next_steps") or []
            if next_steps:
                st.markdown("#### 🧭 What would strengthen or settle this verdict")
                for step in next_steps:
                    st.markdown(f"- {step}")

            if model_info.get("type") == "rule_based_fallback":
                st.error("The trained fusion model was not found — verdict came from the rule-based fallback. Run `python -m evaluation.train`.")
            elif result_data.get("is_fallback"):
                st.info("ℹ️ Some components used lightweight fallbacks this run "
                        f"(claim extractor: {model_info.get('claim_extractor')}, report: {model_info.get('report')}).")

            # Evidence graph
            graph = result_data.get("evidence_graph") or {}
            if graph.get("nodes"):
                from backend.claims.evidence_graph import graph_to_dot
                st.subheader("🕸️ Evidence graph")
                st.caption("Boxes are artifacts (green = clean, amber = suspicious, red = anomalous). Blue ovals are claims; a claim shared by several artifacts is one node. **Red edges are contradictions.**")
                st.graphviz_chart(graph_to_dot(graph), width="stretch")

            # Counterfactuals
            cfs = result_data.get("counterfactuals") or []
            if cfs:
                st.subheader("🔀 Counterfactuals: what if an artifact were removed?")
                cf_df = pd.DataFrame([
                    {"Remove": c["removed"].upper(),
                     "New verdict": c["label"].replace("_", " "),
                     "Confidence": f"{c['confidence']:.0%}",
                     "Changes verdict?": "YES" if c["flips_verdict"] else "no"}
                    for c in cfs
                ])
                st.dataframe(cf_df, width="stretch", hide_index=True)
                critical = [c["removed"] for c in cfs if c["flips_verdict"]]
                if critical:
                    st.caption("Verdict depends on: " + ", ".join(m.upper() for m in critical))

            # Class probabilities
            st.subheader("📊 Calibrated Class Probabilities")
            probs = result_data.get("probabilities", {})
            if probs:
                prob_df = pd.DataFrame({
                    "Class": [k.replace("_", " ").title() for k in probs.keys()],
                    "Probability": [v for v in probs.values()]
                })
                st.bar_chart(prob_df.set_index("Class"))

            col_l1, col_l2 = st.columns(2)

            with col_l1:
                st.subheader("🔬 Layer 1: Per-Modality Anomaly Scores")
                for mod, data in result_data.get("modality_results", {}).items():
                    if data.get("present"):
                        st.markdown(
                            f"""
                            <div class="card">
                                <b>{mod.upper()}</b> — Anomaly Score: <code>{data.get('score', 0.0):.2f}</code>
                                {' <span title="heuristic fallback">⚠️ fallback</span>' if data.get('is_fallback') else ''}<br>
                                <small>{data.get('evidence', '')}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="card" style="opacity: 0.6;">
                                <b>{mod.upper()}</b>: <i>Artifact not provided</i>
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
                st.caption(f"Independent text sources compared: {cross_modal.get('n_text_sources', 0)}")

                contradictions = cross_modal.get("contradictions", [])
                meta_conflicts = cross_modal.get("metadata_conflicts", [])
                if contradictions:
                    st.error(f"❌ {len(contradictions)} cross-source contradiction(s):")
                    for c in contradictions:
                        st.write(f"- **{c.get('field', 'Conflict').upper()}**: {c.get('description')}")
                else:
                    st.success("✅ No contradictions between text sources.")
                if meta_conflicts:
                    st.warning("⚠️ Capture metadata vs. claims:")
                    for mc in meta_conflicts:
                        st.write(f"- {mc}")

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
                st.dataframe(ev_df, width="stretch", hide_index=True)

            st.subheader("📝 Forensic Report")
            if result_data.get("report_verified", True):
                st.success("✅ Report verified: grounded against the structured findings.")
            else:
                st.info("ℹ️ Deterministic template report (local LLM unavailable or its output failed grounding checks).")
            st.text_area("Forensic Summary", value=result_data.get("report", ""), height=220)


with tab_eval:
    st.subheader("How well does it generalise to manipulations it has not seen?")
    st.caption("Synthetic, technique-tagged benchmark. Each technique is held out of training and calibration, then tested. "
               "Reproduce with `python -m evaluation.train` and `python -m evaluation.run_eval`.")
    results_path = _REPO_ROOT / "evaluation" / "results" / "eval_results.json"
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python -m evaluation.run_eval`.")
    else:
        ev = json.loads(results_path.read_text())
        pooled = ev["loto"]["pooled_unseen"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unseen-technique accuracy", f"{pooled['accuracy']:.1%}")
        m2.metric("Macro-F1", f"{pooled['macro_f1']:.3f}")
        m3.metric("AUROC (OvR)", f"{pooled['auroc_ovr']:.3f}")
        m4.metric("False alarms on authentic", f"{pooled['authentic_false_alarm_rate']:.1%}")

        rows = []
        for t, r in ev["loto"]["per_technique"].items():
            rows.append({"Technique": t, "Seen": r["seen"]["correct"], "Unseen (held out)": r["unseen"]["correct"]})
        st.markdown("**Correct verdicts: technique seen in training vs. held out**")
        st.bar_chart(pd.DataFrame(rows).set_index("Technique"))
        det = pd.DataFrame([
            {"Technique": t, "Class": r["label"], "Unseen correct": f"{r['unseen']['correct']:.0%}",
             "Abstained": f"{r['unseen']['abstained']:.0%}", "Wrong & confident": f"{r['unseen']['wrong_confident']:.0%}"}
            for t, r in ev["loto"]["per_technique"].items()
        ])
        st.dataframe(det, width="stretch", hide_index=True)

        st.markdown("**Why reason across modalities? Single detector vs. fused (coordinated-class recall on held-out techniques)**")
        abl = pd.DataFrame([{"Model": k, "Coordinated recall": v["loto_coordinated_recall"], "Macro-F1": v["loto_macro_f1"]}
                            for k, v in ev["ablation"].items()])
        st.dataframe(abl, width="stretch", hide_index=True)

        st.markdown("**Selective accuracy: answering only when confident**")
        cov = pd.DataFrame(pooled["coverage_vs_accuracy"])
        if not cov.empty:
            st.line_chart(cov.set_index("coverage")[["selective_accuracy"]])

        st.markdown("**Robustness to image laundering (re-compression / resize / screenshot)**")
        rob = pd.DataFrame([{"Transform": k, **v} for k, v in ev["robustness"].items()])
        st.dataframe(rob, width="stretch", hide_index=True)
