import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import html
import io
import json
import os
import requests
import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="TrustLayer - Can you trust this digital content?",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
NOT_AVAILABLE = "Not available from the current analysis."

# ---------------------------------------------------------------------------
# Styling: clean light theme, blue/indigo accent, soft shadows
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    :root { color-scheme: light; }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: #f8fafc !important; color: #0f172a; }
    [data-testid="stHeader"] { background: #f8fafc !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #f1f5f9 !important; }
    .block-container { max-width: 1150px; padding-top: 1.6rem; padding-bottom: 4rem; }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: #0f172a !important; }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li { color: #0f172a; }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p { color: #475569 !important; }
    [data-testid="stWidgetLabel"] p { color: #0f172a !important; font-size: 1.05rem; font-weight: 600; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #0f172a; }
    [data-testid="stMetric"] label p, [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #0f172a !important; }

    textarea, input { background: #ffffff !important; color: #0f172a !important; font-size: 1.05rem !important; }
    [data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
        background: #ffffff !important; border-color: #cbd5e1 !important;
    }
    [data-baseweb="select"] * { color: #0f172a !important; }
    [data-baseweb="popover"], [data-baseweb="menu"] { background: #ffffff !important; }
    [data-baseweb="menu"] li, [data-baseweb="menu"] * { color: #0f172a !important; }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important; border: 2px dashed #93c5fd !important;
        border-radius: 12px; padding: 18px;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #334155 !important; }
    [data-testid="stFileUploaderDropzone"] button { background: #ffffff !important; border: 1px solid #4f46e5 !important; }
    [data-testid="stFileUploaderDropzone"] button * { color: #4f46e5 !important; font-weight: 700; }
    [data-testid="stFileUploaderFile"] * { color: #0f172a !important; }

    [data-testid="stExpander"] { background: #ffffff; border: 1px solid #e2e8f0 !important; border-radius: 12px; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
        color: #0f172a !important; font-weight: 700; font-size: 1.05rem;
    }
    button[data-baseweb="tab"] p { color: #334155; font-size: 1.05rem; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #4f46e5 !important; }

    /* Buttons: professional indigo (primary), white with indigo border (secondary) */
    div.stButton > button, button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-secondary"] {
        min-height: 3rem; font-size: 1.1rem; font-weight: 700; border-radius: 10px; padding: 0.5rem 2rem;
    }
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: #4f46e5 !important; border: 1px solid #4f46e5 !important; color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(79,70,229,0.25);
    }
    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background: #4338ca !important; border-color: #4338ca !important;
    }
    div.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {
        color: #ffffff !important; font-size: 1.1rem; font-weight: 700;
    }
    div.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
        background: #ffffff !important; border: 1px solid #4f46e5 !important; color: #4f46e5 !important;
    }
    div.stButton > button[kind="secondary"] p, button[data-testid="stBaseButton-secondary"] p {
        color: #4f46e5 !important; font-size: 1.1rem; font-weight: 700;
    }
    div.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
        background: #eef2ff !important;
    }

    /* Hero */
    .tl-hero {
        background: #ffffff; border: 1px solid #dbeafe; border-radius: 16px;
        padding: 28px 30px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }
    .tl-hero .tl-hero-title { font-size: 2.3rem; font-weight: 900; color: #0f172a; line-height: 1.15; }
    .tl-hero .tl-hero-tag { font-size: 1.5rem; font-weight: 700; color: #4f46e5; margin: 6px 0 12px 0; }
    .tl-hero .tl-hero-text { font-size: 1.12rem; line-height: 1.6; color: #1e293b; max-width: 800px; }

    .tl-status { margin-bottom: 8px; }
    .tl-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.92rem; border: 1px solid; }
    .tl-pill-ok { background: #ecfdf5; color: #065f46; border-color: #6ee7b7; }
    .tl-pill-warn { background: #fffbeb; color: #92400e; border-color: #fcd34d; }
    .tl-pill-info { background: #eff6ff; color: #1e40af; border-color: #93c5fd; }

    .tl-step { display: flex; align-items: center; gap: 14px; margin: 32px 0 12px 0; }
    .tl-step-num {
        flex: 0 0 auto; width: 42px; height: 42px; border-radius: 50%;
        background: #4f46e5; color: #ffffff; font-size: 1.25rem; font-weight: 800;
        display: flex; align-items: center; justify-content: center;
    }
    .tl-step-kicker { font-size: 0.85rem; letter-spacing: 2px; font-weight: 800; color: #4f46e5; }
    .tl-step-title { font-size: 1.55rem; font-weight: 800; line-height: 1.2; color: #0f172a; }
    .tl-step-sub { font-size: 1.02rem; color: #475569; margin-top: 2px; }

    .tl-section { font-size: 1.5rem; font-weight: 800; margin: 28px 0 4px 0; color: #0f172a; }
    .tl-section-sub { font-size: 1.02rem; color: #475569; margin-bottom: 12px; }
    .tl-group { font-size: 1.1rem; font-weight: 800; margin: 18px 0 8px 0; color: #334155; }

    .tl-upload-head { font-size: 1.2rem; font-weight: 800; color: #0f172a; }
    .tl-upload-sub { font-size: 1rem; color: #475569; margin-bottom: 6px; }

    /* Verdict card */
    .verdict-box { padding: 28px 26px; border-radius: 16px; margin: 6px 0 10px 0; border: 3px solid; color: #0f172a; box-shadow: 0 3px 10px rgba(15,23,42,0.08); }
    .verdict-box .v-kicker { font-size: 0.95rem; letter-spacing: 2px; font-weight: 800; color: #334155; }
    .verdict-box .v-label { font-size: 2.4rem; font-weight: 900; margin: 4px 0 10px 0; line-height: 1.2; }
    .verdict-box .v-text { font-size: 1.25rem; line-height: 1.55; max-width: 840px; color: #0f172a; font-weight: 600; }
    .verdict-box .v-extra { font-size: 1.08rem; line-height: 1.55; max-width: 840px; color: #1e293b; margin-top: 10px; }
    .verdict-consistent { background: #ecfdf5; border-color: #059669; }
    .verdict-consistent .v-label { color: #065f46; }
    .verdict-needs_verification { background: #fffbeb; border-color: #d97706; }
    .verdict-needs_verification .v-label { color: #92400e; }
    .verdict-high_risk { background: #fef2f2; border-color: #dc2626; }
    .verdict-high_risk .v-label { color: #991b1b; }
    .verdict-insufficient { background: #f1f5f9; border-color: #64748b; }
    .verdict-insufficient .v-label { color: #334155; }
    .verdict-unknown { background: #eff6ff; border-color: #3b82f6; }
    .verdict-unknown .v-label { color: #1e3a8a; }

    /* Finding cards */
    .finding {
        display: flex; gap: 14px; align-items: flex-start;
        background: #ffffff; color: #0f172a;
        border: 1px solid #e2e8f0; border-left-width: 6px;
        border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
        font-size: 1.1rem; line-height: 1.5; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    }
    .finding .f-icon { font-size: 1.4rem; line-height: 1.3; }
    .finding .f-tag { display: block; font-weight: 800; font-size: 0.88rem; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 2px; }
    .finding .f-detail { display: block; font-size: 0.98rem; color: #334155; margin-top: 4px; }
    .finding-bad { border-left-color: #dc2626; }
    .finding-bad .f-tag { color: #991b1b; }
    .finding-warn { border-left-color: #d97706; }
    .finding-warn .f-tag { color: #92400e; }
    .finding-ok { border-left-color: #059669; }
    .finding-ok .f-tag { color: #065f46; }
    .finding-info { border-left-color: #4f46e5; }
    .finding-info .f-tag { color: #4338ca; }

    .todo {
        display: flex; gap: 14px; align-items: flex-start;
        background: #eef2ff; color: #0f172a; border: 1px solid #c7d2fe;
        border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
        font-size: 1.12rem; line-height: 1.5; box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }
    .todo .t-icon { font-size: 1.4rem; line-height: 1.3; }

    /* Professional report card */
    .tl-pro {
        background: #ffffff; border: 1px solid #c7d2fe; border-left: 6px solid #4f46e5;
        border-radius: 14px; padding: 20px 22px; margin: 26px 0 12px 0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.07);
    }
    .tl-pro .p-badge { display: inline-block; font-size: 0.8rem; font-weight: 800; letter-spacing: 1.5px; color: #4338ca; background: #eef2ff; border-radius: 999px; padding: 3px 10px; margin-bottom: 8px; }
    .tl-pro .p-title { font-size: 1.45rem; font-weight: 800; color: #0f172a; }
    .tl-pro .p-sub { font-size: 1.05rem; color: #475569; margin-top: 4px; line-height: 1.5; }

    /* Technical cards (inside the detailed report) */
    .card { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }
    .card small { color: #334155; }
    .card code { background: #e2e8f0; color: #0f172a; padding: 1px 6px; border-radius: 6px; }

    /* Evaluation metric cards */
    .tl-metric { background: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; border-top: 4px solid #4f46e5; border-radius: 14px; padding: 18px; height: 100%; box-shadow: 0 1px 4px rgba(15,23,42,0.06); }
    .tl-metric .m-val { font-size: 2.2rem; font-weight: 900; line-height: 1.1; color: #0f172a; }
    .tl-metric .m-name { font-size: 1.05rem; font-weight: 800; margin-top: 4px; color: #0f172a; }
    .tl-metric .m-desc { font-size: 0.95rem; color: #475569; margin-top: 6px; line-height: 1.4; }

    .tl-footer { font-size: 0.95rem; color: #64748b; margin-top: 36px; text-align: center; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers (presentation only)
# ---------------------------------------------------------------------------
def esc(value) -> str:
    """Escape any text before placing it into HTML."""
    return html.escape("" if value is None else str(value))


def fmt_score(value, spec=".2f") -> str:
    """Formats a raw number for the detailed report only. Never interpreted as a risk."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    return "n/a"


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_status(url: str) -> str:
    """Returns 'online', 'degraded' or 'offline'. Cached briefly to avoid slowing reruns."""
    try:
        res = requests.get(f"{url}/health", timeout=1.0)
        if res.status_code == 200:
            return "online"
        return "degraded"
    except Exception:
        return "offline"


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="tl-section">{esc(title)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="tl-section-sub">{esc(sub)}</div>', unsafe_allow_html=True)


def group_title(title: str):
    st.markdown(f'<div class="tl-group">{esc(title)}</div>', unsafe_allow_html=True)


def step_header(number: int, title: str, subtitle: str = ""):
    sub_html = f'<div class="tl-step-sub">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="tl-step">
            <div class="tl-step-num">{number}</div>
            <div>
                <div class="tl-step-kicker">STEP {number}</div>
                <div class="tl-step-title">{esc(title)}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Technical verdict -> user-facing verdict (display mapping only; the backend decides the verdict)
VERDICT_MAP = {
    "authentic": "consistent",
    "consistent": "consistent",
    "manipulated": "high_risk",
    "coordinated_synthetic": "high_risk",
    "high_risk": "high_risk",
    "needs_verification": "needs_verification",
    "insufficient_evidence": "insufficient",
    "not_enough_information": "insufficient",
}

VERDICT_DISPLAY = {
    "consistent": {
        "icon": "✅",
        "label": "CONSISTENT",
        "text": "Available evidence agrees. This does not prove the content is genuine.",
        "extra": "Agreement alone is not proof, so confirming the original source is still a good idea.",
    },
    "needs_verification": {
        "icon": "⚠️",
        "label": "NEEDS VERIFICATION",
        "text": "Some evidence is missing, unclear, or does not agree. More verification is recommended.",
        "extra": ("This does not automatically mean that a file is fake. The conflicting or missing "
                  "information should be independently checked."),
    },
    "high_risk": {
        "icon": "🚨",
        "label": "HIGH RISK",
        "text": "Strong warning signals were found. Treat this content with caution until it has been verified.",
        "extra": "This is a risk warning, not proof.",
    },
    "insufficient": {
        "icon": "❓",
        "label": "NOT ENOUGH INFORMATION",
        "text": "There is not enough usable evidence to make a reliable assessment.",
        "extra": "Adding more evidence, such as a photo, a recording, or a document, may help.",
    },
    "unknown": {
        "icon": "ℹ️",
        "label": "RESULT",
        "text": "Please review the details below. Further verification is recommended.",
        "extra": "",
    },
}

# Extra plain-language sentence for the two backend labels that both map to HIGH RISK.
# Backend labels themselves are never shown on the main screen.
LABEL_EXTRA = {
    "manipulated": "At least one item shows warning signs that it may have been altered. This is a risk warning, not proof.",
    "coordinated_synthetic": ("Several items appear to support the same story while showing warning signs. "
                              "This is a risk warning, not proof."),
}

ACTIONS = {
    "consistent": [
        ("📍", "Check the original source. A consistent result is not proof that the content is genuine."),
        ("🔎", "Look for another independent source if the content is important."),
    ],
    "needs_verification": [
        ("📍", "Check the original source of the content."),
        ("🔎", "Verify the conflicting or missing information using an independent, trusted source before relying on it."),
        ("⏸️", "Avoid sharing the content until it is verified."),
    ],
    "high_risk": [
        ("📍", "Find and check the original source."),
        ("🔎", "Look for another independent, trusted source that confirms the details."),
        ("⏸️", "Avoid sharing the content until it is verified."),
        ("🙋", "If it is important, ask a trusted person or an official source to confirm the details."),
    ],
    "insufficient": [
        ("➕", "Add more evidence (a photo, a recording, a document, or a short description) and check again."),
        ("📍", "Try to find the original source of the content."),
        ("⏸️", "Avoid drawing conclusions until more information is available."),
    ],
    "unknown": [
        ("📍", "Check the original source of the content."),
        ("🔎", "Look for another independent source."),
        ("⏸️", "Avoid sharing the content until it is verified."),
    ],
}

MODALITY_FRIENDLY = {
    "image": "Photo",
    "audio": "Recording",
    "text": "Text / document",
    "document": "Document",
    "metadata": "File details",
}

MISSING_TEXT = {
    "image": "No photo was provided.",
    "audio": "No recording was provided.",
    "text": "No text or document was provided.",
    "document": "No document was provided.",
    "metadata": "No file details were available.",
}


def get_verdict_key(label) -> str:
    key = str(label or "").strip().lower().replace(" ", "_").replace("-", "_")
    return VERDICT_MAP.get(key, "unknown")


def friendly_modality(mod) -> str:
    return MODALITY_FRIENDLY.get(str(mod).lower(), str(mod).replace("_", " ").title())


def finding_card(kind: str, tag: str, icon: str, text: str, detail: str = ""):
    detail_html = f'<span class="f-detail">{esc(detail)}</span>' if detail else ""
    st.markdown(
        f"""
        <div class="finding finding-{kind}">
            <div class="f-icon">{icon}</div>
            <div><span class="f-tag">{esc(tag)}</span>{esc(text)}{detail_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def action_card(icon: str, text: str):
    st.markdown(
        f"""
        <div class="todo">
            <div class="t-icon">{icon}</div>
            <div>{esc(text)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _extra_detail(c: dict) -> str:
    """Any extra simple fields the backend attached to a contradiction (nothing is invented)."""
    parts = []
    for k, v in c.items():
        if k in ("field", "description") or v in (None, "", [], {}):
            continue
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            parts.append(f"{str(k).replace('_', ' ').title()}: {v}")
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float)) for x in v):
            parts.append(f"{str(k).replace('_', ' ').title()}: {', '.join(str(x) for x in v)}")
    return " | ".join(parts)


def build_findings(result_data: dict, verdict_key: str = "unknown") -> list:
    """Turns the backend result into plain-language findings for the simple view.

    Returns a list of (group, kind, tag, icon, text, detail). Groups: checked, compare, suspicious, notes.
    Presentation only: it uses explicit text returned by the backend (abstention reason, contradictions,
    conflicts, evidence descriptions). It never converts a numeric score into a finding.
    """
    findings = []
    seen = set()

    def add(group, kind, tag, icon, text, detail=""):
        text = str(text).strip()
        key = (group, text)
        if text and key not in seen:
            seen.add(key)
            findings.append((group, kind, tag, icon, text, detail))

    abstained = result_data.get("abstained", False)
    abstain_reason = result_data.get("abstain_reason")
    is_fallback = result_data.get("is_fallback", False)

    cross_modal = result_data.get("cross_modal_results") or {}
    contradictions = cross_modal.get("contradictions", []) or []
    meta_conflicts = cross_modal.get("metadata_conflicts", []) or []
    mod_results = result_data.get("modality_results", {}) or {}
    ranked = result_data.get("ranked_evidence", []) or []

    # What was checked / not provided
    for mod, data in mod_results.items():
        if not isinstance(data, dict):
            continue
        name = friendly_modality(mod)
        if data.get("present"):
            add("checked", "ok", "Checked", "✔️", f"{name} was checked.")
        else:
            add("checked", "info", "Not provided", "➖",
                MISSING_TEXT.get(str(mod).lower(), f"No {name.lower()} was provided."))

    # Agree or conflict
    for c in contradictions:
        if isinstance(c, dict):
            field = str(c.get("field", "Detail")).replace("_", " ").title()
            desc = c.get("description") or "Two pieces of evidence give different information."
            add("compare", "bad", f"{field} conflict", "🔴", desc, _extra_detail(c))
        else:
            add("compare", "bad", "Conflict", "🔴", str(c))

    for mc in meta_conflicts:
        add("compare", "warn", "File details do not match", "⚠️", mc if isinstance(mc, str) else json.dumps(mc))

    if cross_modal and not contradictions and not meta_conflicts:
        add("compare", "ok", "No conflicts reported", "✅",
            "No conflicting details were reported by the comparisons that were run.")

    # What looks suspicious: backend-written evidence descriptions, only when a warning verdict was returned
    if verdict_key in ("needs_verification", "high_risk"):
        for e in ranked[:3]:
            if isinstance(e, dict) and e.get("description"):
                add("suspicious", "warn", "Warning sign", "⚠️", e["description"])

    # Notes
    if abstained:
        reason = abstain_reason or "the available evidence was not strong enough"
        add("notes", "warn", "Held back", "⏸️", f"TrustLayer chose not to give a firm conclusion because: {reason}")

    any_modality_fallback = False
    for mod, data in mod_results.items():
        if isinstance(data, dict) and data.get("present") and data.get("is_fallback"):
            any_modality_fallback = True
            add("notes", "info", "Limited analysis", "ℹ️",
                f"{friendly_modality(mod)} was checked with a simpler backup method.")
    if is_fallback and not any_modality_fallback:
        add("notes", "info", "Limited analysis", "ℹ️", "Some checks used a simpler backup method.")

    if not findings:
        add("notes", "info", "Summary", "ℹ️", "No specific findings were returned for this check.")

    return findings


def render_simple_result(result_data: dict) -> str:
    """Normal user view: verdict, what we found, what to do. Returns the user-facing verdict key."""
    label = result_data.get("label", "")
    verdict_key = get_verdict_key(label)
    disp = dict(VERDICT_DISPLAY[verdict_key])
    raw = str(label or "").strip().lower().replace(" ", "_").replace("-", "_")
    if raw in LABEL_EXTRA:
        disp["extra"] = LABEL_EXTRA[raw]

    extra_html = f'<div class="v-extra">{esc(disp["extra"])}</div>' if disp.get("extra") else ""
    st.markdown(
        f"""
        <div class="verdict-box verdict-{verdict_key}">
            <div class="v-kicker">YOUR RESULT</div>
            <div class="v-label">{disp['icon']} {esc(disp['label'])}</div>
            <div class="v-text">{esc(disp['text'])}</div>
            {extra_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # WHAT WE FOUND
    section("What we found", "Only details reported for the evidence you provided.")
    findings = build_findings(result_data, verdict_key)
    group_titles = [
        ("checked", "What was checked"),
        ("compare", "Where the information agrees or conflicts"),
        ("suspicious", "What looks suspicious"),
        ("notes", "Good to know"),
    ]
    for gkey, gtitle in group_titles:
        items = [f for f in findings if f[0] == gkey]
        if not items:
            continue
        group_title(gtitle)
        for _g, kind, tag, icon, text, detail in items:
            finding_card(kind, tag, icon, text, detail)

    # WHAT SHOULD I DO?
    section("What should I do?")
    for icon, text in ACTIONS.get(verdict_key, ACTIONS["unknown"]):
        action_card(icon, text)
    next_steps = result_data.get("next_steps") or []
    if next_steps:
        group_title("What would strengthen or settle this result")
        for step in next_steps:
            action_card("➡️", str(step))

    st.caption("TrustLayer is a decision-support tool, not a certification. A result shows what the available "
               "evidence suggests, not absolute proof.")
    return verdict_key


def render_detailed_report(result_data: dict):
    """Professional report: all technical detail, using only data returned by the backend."""
    label = result_data.get("label", "")
    confidence = result_data.get("confidence")
    model_info = result_data.get("model_info") or {}
    abstained = result_data.get("abstained", False)
    abstain_reason = result_data.get("abstain_reason")

    t_over, t_mod, t_claims, t_graph, t_ev, t_cf, t_rep = st.tabs([
        "📊 Overview",
        "🧩 Evidence details",
        "⚠️ Claims & contradictions",
        "🔗 Evidence relationships",
        "📚 Ranked evidence",
        "🔀 Counterfactuals",
        "📝 Forensic report",
    ])

    # Overview
    with t_over:
        o1, o2 = st.columns(2)
        with o1:
            if isinstance(confidence, (int, float)):
                st.metric("Calibrated confidence (reported by the engine)", f"{confidence:.1%}")
            else:
                st.caption(f"Confidence: {NOT_AVAILABLE}")
        with o2:
            st.metric("Engine verdict", str(label).replace("_", " ").title() if label else "n/a")

        if abstained:
            st.warning(f"⚠️ **Abstention Triggered**: {abstain_reason}")
        if model_info.get("type") == "rule_based_fallback":
            st.error("The trained fusion model was not found, so the verdict came from the rule-based "
                     "fallback. Run `python -m evaluation.train`.")
        elif result_data.get("is_fallback"):
            st.info("ℹ️ One or more components used lightweight fallbacks for this run.")

        st.markdown("#### Calibrated class probabilities")
        probs = result_data.get("probabilities", {}) or {}
        if probs:
            prob_df = pd.DataFrame({
                "Class": [str(k).replace("_", " ").title() for k in probs.keys()],
                "Probability": [v for v in probs.values()]
            })
            st.bar_chart(prob_df.set_index("Class"))
        else:
            st.caption(NOT_AVAILABLE)

    # Evidence details (per-modality)
    with t_mod:
        st.markdown("#### Per-modality detector results")
        st.caption("Raw heuristic detector scores. They are not probabilities and have no universal meaning "
                   "on their own. The final verdict comes from the backend.")
        mod_results = result_data.get("modality_results", {}) or {}
        if not mod_results:
            st.caption(NOT_AVAILABLE)
        score_rows = []
        for mod, data in mod_results.items():
            if not isinstance(data, dict):
                continue
            if data.get("present"):
                score = data.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    score_rows.append({"Modality": str(mod).upper(), "Detector score": score})
                fb = ' <span title="heuristic fallback">⚠️ fallback</span>' if data.get("is_fallback") else ""
                st.markdown(
                    f"""
                    <div class="card">
                        <b>{esc(str(mod).upper())}</b> — Detector score: <code>{esc(fmt_score(score))}</code>{fb}<br>
                        <small>{esc(data.get('evidence') or NOT_AVAILABLE)}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="card" style="opacity: 0.75;">
                        <b>{esc(str(mod).upper())}</b>: <i>Artifact not provided</i>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        if score_rows:
            st.markdown("#### Detector scores compared")
            st.bar_chart(pd.DataFrame(score_rows).set_index("Modality"))
            st.caption("Raw heuristic scores returned by the backend, shown side by side. Not probabilities.")

    # Claims & contradictions (cross-modal)
    with t_claims:
        st.markdown("#### Cross-modal coherence")
        cross_modal = result_data.get("cross_modal_results") or {}
        if not cross_modal:
            st.caption(NOT_AVAILABLE)
        else:
            sim = cross_modal.get("image_text_similarity")
            if sim is not None:
                st.metric("Image-caption semantic similarity", fmt_score(sim))
            st.caption(f"Independent text sources compared: {cross_modal.get('n_text_sources', 0)}")
            contradictions = cross_modal.get("contradictions", []) or []
            meta_conflicts = cross_modal.get("metadata_conflicts", []) or []
            if contradictions:
                st.error(f"❌ {len(contradictions)} cross-source contradiction(s):")
                for c in contradictions:
                    if isinstance(c, dict):
                        extra = _extra_detail(c)
                        st.write(f"- **{str(c.get('field', 'Conflict')).upper()}**: {c.get('description')}"
                                 + (f"  \n  _{extra}_" if extra else ""))
                    else:
                        st.write(f"- {c}")
            else:
                st.success("✅ No contradictions between text sources.")
            st.markdown("#### 🕒 Metadata and timestamp signals")
            if meta_conflicts:
                st.warning("⚠️ Capture metadata vs. claims:")
                for mc in meta_conflicts:
                    st.write(f"- {mc}")
            else:
                st.caption("No metadata conflicts were reported.")

    # Evidence relationships (graph)
    with t_graph:
        graph = result_data.get("evidence_graph") or {}
        if graph.get("nodes"):
            from backend.claims.evidence_graph import graph_to_dot
            st.caption("Boxes are pieces of evidence (green = clean, amber = suspicious, red = anomalous). "
                       "Blue ovals are claims; a claim shared by several sources is one node. "
                       "**Red lines are contradictions.**")
            st.graphviz_chart(graph_to_dot(graph), width="stretch")
        else:
            st.caption(NOT_AVAILABLE)

    # Ranked evidence
    with t_ev:
        st.markdown("#### Ranked forensic evidence")
        ranked_ev = result_data.get("ranked_evidence", []) or []
        if ranked_ev:
            ev_df = pd.DataFrame([
                {
                    "Rank": e.get("rank"),
                    "Category": str(e.get("category", "")).title(),
                    "Modality": (e.get("modality") or "N/A").upper(),
                    "Finding": e.get("description"),
                    "Importance": fmt_score(e.get("importance")),
                }
                for e in ranked_ev if isinstance(e, dict)
            ])
            st.dataframe(ev_df, width="stretch", hide_index=True)
        else:
            st.caption(NOT_AVAILABLE)

    # Counterfactuals
    with t_cf:
        st.markdown("#### What if an artifact were removed?")
        cfs = result_data.get("counterfactuals") or []
        if cfs:
            cf_df = pd.DataFrame([
                {"Remove": str(c.get("removed", "")).upper(),
                 "New verdict": str(c.get("label", "")).replace("_", " "),
                 "Confidence": (f"{c['confidence']:.0%}" if isinstance(c.get("confidence"), (int, float)) else "n/a"),
                 "Changes verdict?": "YES" if c.get("flips_verdict") else "no"}
                for c in cfs if isinstance(c, dict)
            ])
            st.dataframe(cf_df, width="stretch", hide_index=True)
            critical = [c.get("removed", "") for c in cfs if isinstance(c, dict) and c.get("flips_verdict")]
            if critical:
                st.caption("Verdict depends on: " + ", ".join(str(m).upper() for m in critical))
        else:
            st.caption(NOT_AVAILABLE)

    # Forensic report
    with t_rep:
        st.markdown("#### Forensic explanation")
        report_text = result_data.get("report", "")
        if report_text:
            if result_data.get("report_verified", True):
                st.success("✅ Report verified: grounded against the structured findings.")
            else:
                st.info("ℹ️ Deterministic template report (local LLM unavailable or its output failed grounding checks).")
            st.text_area("Forensic summary", value=report_text, height=240)
        else:
            st.caption(NOT_AVAILABLE)


def render_result(result_data: dict):
    # 1) Simple result first
    render_simple_result(result_data)

    # 2) Optional professional report (secondary)
    st.markdown(
        """
        <div class="tl-pro">
            <div class="p-badge">PROFESSIONAL</div>
            <div class="p-title">📰 Detailed Verification Report</div>
            <div class="p-sub">For journalists, fact-checkers and professional verification. Includes evidence
            comparison, evidence relationships, contradictions, metadata signals, detector results and a
            forensic explanation.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.get("tl_show_report", False):
        if st.button("📰 Generate Detailed Report", key="tl_btn_report"):
            st.session_state["tl_show_report"] = True
            st.rerun()
    else:
        st.info("Detailed verification reports are designed for professional users such as journalists "
                "and fact-checkers.")
        render_detailed_report(result_data)
        if st.button("Hide detailed report", key="tl_btn_hide_report"):
            st.session_state["tl_show_report"] = False
            st.rerun()


def metric_card(value: str, name: str, desc: str):
    st.markdown(
        f"""
        <div class="tl-metric">
            <div class="m-val">{esc(value)}</div>
            <div class="m-name">{esc(name)}</div>
            <div class="m-desc">{esc(desc)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Demo configuration (which backend demo is loaded is unchanged; only wording differs)
# ---------------------------------------------------------------------------
DEMO_LABELS = {
    "Custom Upload": "✏️ Use my own evidence",
    "Demo 1: Authentic Content": "Example 1: Consistent Evidence",
    "Demo 2: Single-Artifact Manipulated": "Example 2: Possible Manipulation",
    "Demo 3: Coordinated Synthetic Campaign": "Example 3: Conflicting / Coordinated Evidence",
    "Demo 4: Insufficient Evidence": "Example 4: Not Enough Information",
}

DEMO_FOLDERS = {
    "Demo 1: Authentic Content": "case1_authentic",
    "Demo 2: Single-Artifact Manipulated": "case2_manipulated",
    "Demo 3: Coordinated Synthetic Campaign": "case3_coordinated_fake",
    "Demo 4: Insufficient Evidence": "case4_insufficient",
}

# ---------------------------------------------------------------------------
# Sidebar (minimal)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ TrustLayer")
    st.markdown("**SYSTEM STATUS**")
    _status = check_backend_status(BACKEND_URL)
    if _status == "online":
        st.success("🟢 Analysis Engine Online")
    elif _status == "degraded":
        st.warning("🟡 Analysis Engine Running Slowly")
    else:
        st.info("⚪ Backend offline. Using the built-in engine.")

    st.divider()
    with st.expander("🧠 Learned detectors (for reviewers)"):
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

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tl-hero">
    <div class="tl-hero-title">🛡️ TrustLayer</div>
    <div class="tl-hero-tag">Can you trust this digital content?</div>
    <div class="tl-hero-text">Upload a photo, recording, or document. TrustLayer compares the available evidence and looks for signs of inconsistency or manipulation.</div>
</div>
""", unsafe_allow_html=True)

_main_status = check_backend_status(BACKEND_URL)
if _main_status == "online":
    _pill = '<span class="tl-pill tl-pill-ok">● System online</span>'
elif _main_status == "degraded":
    _pill = '<span class="tl-pill tl-pill-warn">● System running slowly</span>'
else:
    _pill = '<span class="tl-pill tl-pill-info">● Running in offline mode (built-in engine)</span>'
st.markdown(f'<div class="tl-status">{_pill}</div>', unsafe_allow_html=True)

tab_analyze, tab_eval = st.tabs(["🔍 Investigate a bundle", "📊 Evaluation & generalisation"])

# ---------------------------------------------------------------------------
# INVESTIGATE TAB
# ---------------------------------------------------------------------------
with tab_analyze:
    section("Try an example", "Not sure where to start? Load an example, or choose your own evidence.")
    demo_option = st.selectbox(
        "Try an example",
        list(DEMO_LABELS.keys()),
        format_func=lambda x: DEMO_LABELS.get(x, x),
        label_visibility="collapsed",
    )

    demo_folder = DEMO_FOLDERS.get(demo_option)
    demo_bundle = None
    if demo_folder:
        from evaluation.run_demos import load_case, DEMOS
        demo_dir = DEMOS / demo_folder
        demo_bundle = load_case(demo_dir)

    # STEP 1
    step_header(1, "Add Your Evidence", "Add whatever evidence you have. You do not need everything.")

    if demo_bundle is not None:
        st.success(f"Example loaded: **{DEMO_LABELS.get(demo_option, demo_option)}**. "
                   "Press the check button below.")
        d1, d2 = st.columns(2)
        with d1:
            if demo_bundle.image_path:
                st.markdown("**📷 Image**")
                st.image(demo_bundle.image_path, caption="Image in this example", width="stretch")
            if demo_bundle.audio_path:
                st.markdown("**🎧 Audio**")
                st.audio(demo_bundle.audio_path)
        with d2:
            for name, text in (demo_bundle.documents or {}).items():
                st.markdown(f"**📄 {name}**")
                st.caption(text)
        uploaded_image = uploaded_audio = uploaded_doc = None
        extra_docs = []
        caption_text, meta_json, claimed_speaker = "", "", ""
    else:
        c_img, c_aud, c_doc = st.columns(3)
        with c_img:
            st.markdown('<div class="tl-upload-head">📷 Image</div>'
                        '<div class="tl-upload-sub">Upload a photo or image</div>', unsafe_allow_html=True)
            uploaded_image = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"],
                                              label_visibility="collapsed")
            if uploaded_image:
                st.image(uploaded_image, caption="Your image", width="stretch")
        with c_aud:
            st.markdown('<div class="tl-upload-head">🎧 Audio</div>'
                        '<div class="tl-upload-sub">Upload an audio recording</div>', unsafe_allow_html=True)
            uploaded_audio = st.file_uploader("Audio", type=["wav", "mp3", "flac"],
                                              label_visibility="collapsed")
            if uploaded_audio:
                st.audio(uploaded_audio)
        with c_doc:
            st.markdown('<div class="tl-upload-head">📄 Document</div>'
                        '<div class="tl-upload-sub">Upload a PDF, DOCX or text file. You can add several; '
                        'each is cross-checked against the others.</div>', unsafe_allow_html=True)
            docs_uploaded = st.file_uploader(
                "Document", type=["txt", "pdf", "docx"], accept_multiple_files=True,
                label_visibility="collapsed",
            )
            uploaded_doc = docs_uploaded[0] if docs_uploaded else None
            extra_docs = docs_uploaded[1:] if docs_uploaded else []

    # STEP 2
    step_header(2, "Add Optional Information", "A few words help us understand what we are looking at.")

    if demo_bundle is not None:
        st.caption("This example already includes its own details.")
        if demo_bundle.claimed_speaker:
            st.markdown(f"🗣️ **Speaker in this example:** {demo_bundle.claimed_speaker}")
        if demo_bundle.metadata:
            with st.expander("⚙️ Technical Details (Optional)"):
                if isinstance(demo_bundle.metadata, dict):
                    for _k, _v in demo_bundle.metadata.items():
                        st.markdown(f"- **{_k}**: {_v}")
                else:
                    st.write(demo_bundle.metadata)
    else:
        caption_text = st.text_area(
            "📝 What is this content about? (a caption, a short description, or the text itself)",
            value="",
            height=140,
            placeholder="Example: A photo from our school festival held last month..."
        )
        claimed_speaker = st.text_input(
            "🗣️ Who is speaking? (optional)",
            value="",
            placeholder="Example: Dr. Rajesh Sharma"
        )
        with st.expander("⚙️ Technical Details (Optional)"):
            st.caption("For advanced users: file information such as camera, software, location, or dates.")
            meta_json = st.text_area(
                "File details (Metadata JSON: EXIF, provenance, tags)",
                value="",
                height=140,
                placeholder='{"DateTimeOriginal": "2026:09:15 10:00:00", "Location": "Mumbai", "Software": "..."}'
            )

    # STEP 3
    step_header(3, "Check Your Content", "Press the button and we will compare everything for you.")
    analyze_btn = st.button("🔍 CHECK THIS CONTENT", type="primary")

    if analyze_btn:
        with st.spinner("Checking the uploaded evidence..."):
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

        if result_data:
            st.session_state["tl_result"] = result_data
            st.session_state["tl_result_demo"] = demo_option
            st.session_state["tl_show_report"] = False  # new check: detailed report starts closed
        else:
            st.session_state.pop("tl_result", None)
            st.error("We could not complete the check. Please try again or add more evidence.")

    # Render results (kept across reruns; cleared when the example changes)
    if st.session_state.get("tl_result") and st.session_state.get("tl_result_demo") == demo_option:
        st.divider()
        render_result(st.session_state["tl_result"])

# ---------------------------------------------------------------------------
# EVALUATION TAB
# ---------------------------------------------------------------------------
with tab_eval:
    section("📊 MODEL EVALUATION",
            "How reliably does TrustLayer handle manipulation techniques it has not seen during training?")
    st.caption("Synthetic, technique-tagged benchmark. Each technique is held out of training and calibration, then tested. "
               "Reproduce with `python -m evaluation.train` and `python -m evaluation.run_eval`.")
    results_path = _REPO_ROOT / "evaluation" / "results" / "eval_results.json"
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python -m evaluation.run_eval`.")
    else:
        ev = json.loads(results_path.read_text())
        pooled = ev["loto"]["pooled_unseen"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card(f"{pooled['accuracy']:.1%}", "Unseen-technique accuracy",
                        "Share of correct verdicts on manipulation techniques held out of training.")
        with m2:
            metric_card(f"{pooled['macro_f1']:.3f}", "Macro-F1",
                        "Balance of precision and recall, averaged across all verdict classes.")
        with m3:
            metric_card(f"{pooled['auroc_ovr']:.3f}", "AUROC (one-vs-rest)",
                        "How well the model's class probabilities rank the right class above the others.")
        with m4:
            metric_card(f"{pooled['authentic_false_alarm_rate']:.1%}", "False alarms on authentic",
                        "Share of authentic cases that were wrongly flagged.")

        section("Performance by technique",
                "Correct verdicts when a technique was seen in training vs. held out.")
        rows = []
        for t, r in ev["loto"]["per_technique"].items():
            rows.append({"Technique": t, "Seen": r["seen"]["correct"], "Unseen (held out)": r["unseen"]["correct"]})
        st.bar_chart(pd.DataFrame(rows).set_index("Technique"))

        section("Detailed results, abstentions, and confident mistakes",
                "Abstained = the system declined to answer. Wrong & confident = a wrong verdict given with high confidence.")
        det = pd.DataFrame([
            {"Technique": t, "Class": r["label"], "Unseen correct": f"{r['unseen']['correct']:.0%}",
             "Abstained": f"{r['unseen']['abstained']:.0%}", "Wrong & confident": f"{r['unseen']['wrong_confident']:.0%}"}
            for t, r in ev["loto"]["per_technique"].items()
        ])
        st.dataframe(det, width="stretch", hide_index=True)

        section("Why reason across modalities?",
                "Single detector vs. fused model (coordinated-class recall on held-out techniques).")
        abl = pd.DataFrame([{"Model": k, "Coordinated recall": v["loto_coordinated_recall"], "Macro-F1": v["loto_macro_f1"]}
                            for k, v in ev["ablation"].items()])
        st.dataframe(abl, width="stretch", hide_index=True)

        section("Answering only when confident",
                "Selective accuracy as the system answers fewer, more confident cases.")
        cov = pd.DataFrame(pooled["coverage_vs_accuracy"])
        if not cov.empty:
            st.line_chart(cov.set_index("coverage")[["selective_accuracy"]])

        section("Robustness to image laundering",
                "Re-compression, resizing, and screenshots.")
        rob = pd.DataFrame([{"Transform": k, **v} for k, v in ev["robustness"].items()])
        st.dataframe(rob, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="tl-footer">
        🛡️ TrustLayer is a hackathon prototype. It reports what the available evidence suggests and
        never claims a file is definitely real or definitely fake.
    </div>
    """,
    unsafe_allow_html=True
)