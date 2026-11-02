"""rag_search — "check our own memory FIRST" locate-accelerator for the research gate.

One-liner over the LlamaIndex index of the project's PROSE knowledge base (hybrid vector+BM25 RRF fusion ->
cross-encoder rerank). Surfaces the docs that already answer a research-gate question, so a gate does not re-derive
what the record holds. It LOCATES the passage; the discipline is still to open the cited doc and read it.

Broadened corpus (source_type shown per hit; build via tools/rag/build_llamaindex_full.py):
  finding : research/findings/*.md          -> "have we already CONCLUDED / tried X?"
  plan    : docs/plans/*.md                 -> "did we already DESIGN X?"
  doc     : docs/*.md + CLAUDE/ROADMAP/README
  catalog : sim-catalog/references/*.md      -> "is there a CATALOG ENTRY for X?"
  kandel  : Kandel 6e full text              -> "how does the BIOLOGY do X?"
  paper   : the specialty texts/papers/books -> "what does the ORIGINAL SOURCE say?" (Marr 1969, Albus 1971,
            Buzsaki "Rhythms of the Brain", O'Keefe-Nadel "Cognitive Map", Schultz, Sutton-Barto, Tepper/Bolam BG)
  coord   : research/coordination/*.md, the runbook split per ruling -> "what did a RULING / packet / RESULT decide?"
  draft   : docs/drafts/**/*.md          -> "is there a DRAFT plan for X?" (drafts are not approved plans)

Run with the isolated RAG venv (has llama-index; base sim env untouched -- installing llama-index into the sim
venv would churn its pinned torch/cupy CUDA stack):
    .venv-rag/bin/python tools/rag/rag_search.py "<question>" [top_k] [--corpus TYPE]
  --corpus one of {all(default), finding, plan, doc, catalog, kandel, paper, coord, draft} -- target one corpus (e.g. --corpus kandel for biology).

Index location resolves through Git's common checkout, so linked worktrees use the canonical sibling
``rag_index`` and ``sim-catalog``. ``SIM_RAG_ROOT`` remains an explicit override. See docs/RAG_COMPARISON.md."""
import os, io, sys
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rag_paths import resolve_paths
from retrieval import RagRetriever

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATHS = resolve_paths(_REPO)

argv = sys.argv[1:]
corpus = "all"
if "--corpus" in argv:
    i = argv.index("--corpus"); corpus = argv[i + 1]; del argv[i:i + 2]
valid_corpora = {"all", "finding", "plan", "doc", "catalog", "kandel", "paper", "coord", "draft"}
if corpus not in valid_corpora:
    sys.stderr.write(f"unknown corpus {corpus!r}; choose one of {sorted(valid_corpora)}\n")
    raise SystemExit(2)
if not argv or not argv[0].strip():
    sys.stdout.write('usage: rag_search.py "<question>" [top_k] [--corpus finding|plan|doc|catalog|kandel|paper|coord|draft|all]\n')
    raise SystemExit(2)
query = argv[0]
top_k = int(argv[1]) if len(argv) > 1 else 5


def _via_server():
    """The resident index (tools/rag/rag_server.py) answers in seconds; None when it is not running."""
    import json, socket
    path = os.environ.get("SIM_RAG_SOCKET") or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "sim-rag.sock")
    if not os.path.exists(path):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(float(os.environ.get("SIM_RAG_SERVER_TIMEOUT", "60")))
            s.connect(path)
            s.sendall((json.dumps({"query": query, "top_k": top_k, "corpus": corpus}) + "\n").encode())
            s.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        text = data.decode("utf-8", "replace")
        return None if text.startswith("rag_server error") or not text else text
    except OSError:
        return None


text = _via_server()
if text is None:
    import time
    from rag_format import format_hits
    try:
        engine = RagRetriever(PATHS, corpus=corpus, top_k=top_k)
    except FileNotFoundError as exc:
        sys.stderr.write(f"rag_search: {exc}\n")
        raise SystemExit(1)
    started = time.time()
    nodes, _ = engine.retrieve(query)
    text = format_hits(query, nodes, engine, corpus, time.time() - started)
sys.stdout.buffer.write(text.encode("utf-8", "replace"))
