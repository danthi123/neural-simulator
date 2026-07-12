"""Query a SOMA sbert-indexed bundle (works around the CLI load() bug via the Python API). Usage: python soma_search.py "<query>" [k]"""
import os, io, sys, logging
os.environ["TRANSFORMERS_VERBOSITY"]="error"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"; logging.disable(logging.WARNING)
from contextlib import redirect_stderr
BUNDLE = os.environ.get("SOMA_BUNDLE", r"E:\Documents\Projects\soma_bundles\sim_findings")
with redirect_stderr(io.StringIO()):
    from soma.memory import MemoryLayer
    mem = MemoryLayer.load_with_sbert(BUNDLE)
q = sys.argv[1] if len(sys.argv) > 1 else "off-diagonal red herring"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
for i, h in enumerate(mem.retrieve(q, k=k)):
    meta = getattr(h, "metadata", None) or (h.get("metadata") if isinstance(h, dict) else {}) or {}
    src = os.path.basename(str(meta.get("source") or meta.get("path") or meta.get("title") or "")) if isinstance(meta, dict) else ""
    sc = getattr(h, "score", None) or (h.get("score") if isinstance(h, dict) else "")
    txt = getattr(h, "text", None) or (h.get("text") if isinstance(h, dict) else "")
    sys.stdout.buffer.write(f"[{i+1}] {round(sc,3) if isinstance(sc,float) else sc}  {src}\n    {' '.join(str(txt)[:160].split())}\n".encode("utf-8","replace"))
