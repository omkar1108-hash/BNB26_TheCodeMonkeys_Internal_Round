from typing import List, Dict


def build_evidence_graph(
    files: List[Dict],
    claims: List[Dict],
    conflicts: List[Dict]
) -> Dict:
    """
    Build a graph representation of files, claims and conflicts.

    Nodes:
        - file
        - claim

    Edges:
        - contains
        - conflicts
    """

    nodes = []
    edges = []

    # -------------------------
    # File nodes
    # -------------------------

    for file in files:

        nodes.append({
            "id": file["file_id"],
            "type": "file",
            "label": file["filename"],
            "modality": file.get("modality")
        })

    # -------------------------
    # Claim nodes
    # -------------------------

    for claim in claims:

        nodes.append({
            "id": claim["claim_id"],
            "type": "claim",
            "label": claim["text"],
            "when": claim.get("when"),
            "where": claim.get("where"),
            "who": claim.get("who")
        })

    # -------------------------
    # File → Claim edges
    # -------------------------

    for claim in claims:

        file_id = claim.get("file_id")

        if file_id:

            edges.append({
                "source": file_id,
                "target": claim["claim_id"],
                "type": "contains"
            })

    # -------------------------
    # Conflict edges
    # -------------------------

    for conflict in conflicts:

        edges.append({
            "source": conflict["claim_a"],
            "target": conflict["claim_b"],
            "type": "conflict",
            "field": conflict.get("field"),
            "severity": conflict.get("severity", "medium"),
            "description": conflict.get("description")
        })

    return {
        "nodes": nodes,
        "edges": edges
    }


# ---------------------------------------------------------------------------
# Bundle-level evidence graph (artifacts -> claims, with contradiction edges)
# ---------------------------------------------------------------------------
from typing import Any, Optional
from backend.claims.normalize import parse_date, locations_compatible, normalize_text


def _claim_key(field: str, value: str) -> str:
    if field == "date":
        d = parse_date(value)
        return f"claim:date:{d.isoformat()}" if d else f"claim:date:{normalize_text(value)}"
    return f"claim:{field}:{normalize_text(value)}"


def build_bundle_graph(
    claims_map: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    modality_results: Dict[str, Any],
    claimed_speaker: Optional[str],
    contradictions: List[Any],
) -> Dict:
    """
    Nodes : artifacts (image / audio / metadata / each text source) and the
            factual claims they assert (date, location, speaker, entity).
    Edges : 'asserts' (artifact -> claim) and 'conflict' (claim <-> claim).
    A claim asserted by several artifacts is ONE node with several incoming
    edges, so agreement is visible as shared nodes and disagreement as red edges.
    """
    from backend.cross_modal.metadata_rules import get_metadata_date, get_metadata_location

    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    def add_artifact(node_id: str, label: str, modality: str):
        ev = modality_results.get(modality)
        nodes[node_id] = {
            "id": node_id, "type": "artifact", "label": label, "modality": modality,
            "score": round(float(ev.score), 3) if ev is not None and ev.present else None,
        }

    def add_claim(artifact_id: str, field: str, value: str):
        cid = _claim_key(field, value)
        shown = parse_date(value).isoformat() if field == "date" and parse_date(value) else str(value)
        nodes.setdefault(cid, {"id": cid, "type": "claim", "field": field, "label": f"{field}: {shown}"})
        edges.append({"source": artifact_id, "target": cid, "type": "asserts"})
        return cid

    if modality_results.get("image") is not None and modality_results["image"].present:
        add_artifact("art:image", "Image", "image")

    for src, claims in claims_map.items():
        if src == "audio":
            continue
        aid = f"art:text:{src}"
        add_artifact(aid, f"Text: {src}", "text")
        for d in claims.dates:
            add_claim(aid, "date", d)
        for loc in claims.locations:
            add_claim(aid, "location", loc)
        for ent in claims.entities:
            add_claim(aid, "entity", ent)
        if claims.claimed_speaker:
            add_claim(aid, "speaker", claims.claimed_speaker)

    if modality_results.get("audio") is not None and modality_results["audio"].present:
        add_artifact("art:audio", "Audio", "audio")
        if claimed_speaker:
            add_claim("art:audio", "speaker", claimed_speaker)

    meta_date = get_metadata_date(metadata)
    meta_loc = get_metadata_location(metadata)
    if modality_results.get("metadata") is not None and modality_results["metadata"].present:
        add_artifact("art:metadata", "Metadata (EXIF)", "metadata")
        if meta_date:
            add_claim("art:metadata", "date", meta_date.isoformat())
        if meta_loc:
            add_claim("art:metadata", "location", meta_loc)

    seen = set()

    def add_conflict(a: str, b: str, field: str, desc: str):
        key = tuple(sorted((a, b))) + (field,)
        if a == b or key in seen or a not in nodes or b not in nodes:
            return
        seen.add(key)
        edges.append({"source": a, "target": b, "type": "conflict", "field": field,
                      "severity": "high", "description": desc})

    for c in contradictions:
        add_conflict(_claim_key(c.field, c.value_a), _claim_key(c.field, c.value_b), c.field, c.description)

    # metadata vs text claims
    if meta_date:
        mk = _claim_key("date", meta_date.isoformat())
        for src, claims in claims_map.items():
            for d in claims.dates:
                if parse_date(d) and parse_date(d) != meta_date:
                    add_conflict(mk, _claim_key("date", d), "date",
                                 f"Metadata date {meta_date.isoformat()} vs {src} date '{d}'")
    if meta_loc:
        lk = _claim_key("location", meta_loc)
        for src, claims in claims_map.items():
            for loc in claims.locations:
                if not locations_compatible(meta_loc, loc):
                    add_conflict(lk, _claim_key("location", loc), "location",
                                 f"Metadata location '{meta_loc}' vs {src} location '{loc}'")

    return {"nodes": list(nodes.values()), "edges": edges}


def graph_to_dot(graph: Dict) -> str:
    """Graphviz DOT for st.graphviz_chart. Red edges = contradictions."""
    def colour(score):
        if score is None:
            return "#e5e7eb"
        if score < 0.30:
            return "#bbf7d0"
        if score < 0.55:
            return "#fde68a"
        return "#fecaca"

    esc = lambda t: str(t).replace('"', "'")
    lines = ['digraph G {', 'rankdir=LR;', 'node [fontname="Helvetica", fontsize=11];',
             'edge [fontname="Helvetica", fontsize=9];']
    for n in graph.get("nodes", []):
        if n["type"] == "artifact":
            sc = n.get("score")
            label = n["label"] + (f"\\nanomaly {sc:.2f}" if sc is not None else "")
            lines.append(f'"{esc(n["id"])}" [shape=box, style="filled,rounded", fillcolor="{colour(sc)}", label="{esc(label)}"];')
        else:
            lines.append(f'"{esc(n["id"])}" [shape=ellipse, style=filled, fillcolor="#e0e7ff", label="{esc(n["label"])}"];')
    for e in graph.get("edges", []):
        if e["type"] == "conflict":
            lines.append(f'"{esc(e["source"])}" -> "{esc(e["target"])}" [color="#dc2626", penwidth=2.5, dir=both, '
                         f'label="CONFLICT: {esc(e.get("field", ""))}", fontcolor="#dc2626", style=bold];')
        else:
            lines.append(f'"{esc(e["source"])}" -> "{esc(e["target"])}" [color="#9ca3af"];')
    lines.append("}")
    return "\n".join(lines)
