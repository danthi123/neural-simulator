"""rag_server — keep the RAG index loaded so a record search takes seconds, not minutes.

WHY (2026-11-02). Every `rag_search.py` call re-loaded the 290 MB index and rebuilt BM25: ~1-2 min per query on a
loaded workstation, and on the GPU it died with CUDA OOM whenever the local model held the card, so
`tools/before_you_build.sh` printed "rag unavailable" and searched nothing. Nobody consults a record that slow.

Protocol: one JSON line in ({"query", "top_k", "corpus"}), the formatted rag_search text out, on a Unix socket
(default $XDG_RUNTIME_DIR/sim-rag.sock, override SIM_RAG_SOCKET). The index is reloaded when its docstore changes
(the autoupdate timer rebuilds it). CPU only (retrieval.py). Run as the user unit tools/rag/sim-rag-server.service.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_paths import resolve_paths  # noqa: E402
from retrieval import RagRetriever  # noqa: E402
from rag_format import format_hits  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def socket_path() -> str:
    return os.environ.get("SIM_RAG_SOCKET") or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "sim-rag.sock")


def _stamp(persist) -> float:
    try:
        return os.path.getmtime(os.path.join(str(persist), "docstore.json"))
    except OSError:
        return 0.0


def main() -> None:
    paths = resolve_paths(_REPO)
    engine = RagRetriever(paths, corpus="all", top_k=5)
    stamp = _stamp(engine.persist)
    for c in ("finding", "catalog", "kandel", "doc", "plan"):  # build each corpus's BM25 once, up front
        engine.retrieve("warm up", corpus=c)
    path = socket_path()
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(8)
    print(f"rag_server: ready on {path} ({engine.node_count} nodes)", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                raw = b""
                while not raw.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    raw += chunk
                req = json.loads(raw.decode("utf-8") or "{}")
                if _stamp(engine.persist) != stamp:  # the autoupdate rebuilt the index: reload once
                    engine = RagRetriever(paths, corpus="all", top_k=5)
                    stamp = _stamp(engine.persist)
                engine.top_k = int(req.get("top_k", 5))
                started = time.time()
                nodes, _ = engine.retrieve(req["query"], corpus=req.get("corpus", "all"))
                text = format_hits(req["query"], nodes, engine, req.get("corpus", "all"), time.time() - started)
            except Exception as exc:  # one bad request never kills the server
                text = f"rag_server error: {exc!r}\n"
            conn.sendall(text.encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
