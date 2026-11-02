"""Shared hit formatting for rag_search.py and rag_server.py (identical output either way)."""
from __future__ import annotations

try:
    from .retrieval import node_locator, node_source
except ImportError:  # direct script execution
    from retrieval import node_locator, node_source


def format_hits(query, nodes, engine, corpus, seconds) -> str:
    out = [f"Q: {query}   ({seconds:.2f}s, top {engine.top_k}, corpus={corpus}, index={engine.persist})"]
    cache = {}
    for i, n in enumerate(nodes):
        md = n.node.metadata or {}
        txt = " ".join((n.node.get_content() or "")[:220].split())
        score = round(n.score, 3) if n.score is not None else ""
        out.append(f"  [{i+1}] {score}  ({md.get('source_type', '?')}) {node_source(n)}\n"
                   f"      at {node_locator(n, cache)}\n      {txt}")
    if not nodes:
        out.append("  (no hits" + (f" in corpus '{corpus}'" if corpus != "all" else "") + ")")
    return "\n".join(out) + "\n\n"
