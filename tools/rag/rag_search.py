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

Run with the isolated RAG venv (has llama-index; base sim env untouched -- installing llama-index into the sim
venv would churn its pinned torch/cupy CUDA stack):
    .venv-rag/bin/python tools/rag/rag_search.py "<question>" [top_k] [--corpus TYPE]
  --corpus one of {all(default), finding, plan, doc, catalog, kandel, paper} -- target one corpus (e.g. --corpus kandel for biology).

Index location resolves through Git's common checkout, so linked worktrees use the canonical sibling
``rag_index`` and ``sim-catalog``. ``SIM_RAG_ROOT`` remains an explicit override. See docs/RAG_COMPARISON.md."""
import os, io, sys, time
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import redirect_stderr
from rag_paths import choose_index, resolve_paths

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATHS = resolve_paths(_REPO)

argv = sys.argv[1:]
corpus = "all"
if "--corpus" in argv:
    i = argv.index("--corpus"); corpus = argv[i + 1]; del argv[i:i + 2]
valid_corpora = {"all", "finding", "plan", "doc", "catalog", "kandel", "paper"}
if corpus not in valid_corpora:
    sys.stderr.write(f"unknown corpus {corpus!r}; choose one of {sorted(valid_corpora)}\n")
    raise SystemExit(2)
if not argv or not argv[0].strip():
    sys.stdout.write('usage: rag_search.py "<question>" [top_k] [--corpus finding|plan|doc|catalog|kandel|paper|all]\n')
    raise SystemExit(2)
query = argv[0]
top_k = int(argv[1]) if len(argv) > 1 else 5
try:
    persist = choose_index(PATHS, corpus)
except FileNotFoundError as exc:
    sys.stderr.write(f"rag_search: {exc}\n")
    raise SystemExit(1)

with redirect_stderr(io.StringIO()):
    from llama_index.core import StorageContext, load_index_from_storage, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(persist)))
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_k)
    t0 = time.time()
    if corpus != "all":
        # filter DURING retrieval (vector + a source_type metadata filter) so a small corpus (catalog/plan) is not
        # crowded out of the rerank window by the big corpora (Kandel/findings). BM25 can't filter, so vector-only here.
        flt = MetadataFilters(filters=[MetadataFilter(key="source_type", value=corpus, operator=FilterOperator.EQ)])
        retr = index.as_retriever(similarity_top_k=max(top_k * 4, 20), filters=flt)
        cand = retr.retrieve(query)
    else:
        vec = index.as_retriever(similarity_top_k=12)
        bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=12)
        fusion = QueryFusionRetriever([vec, bm25], num_queries=1, mode="reciprocal_rerank",
                                      similarity_top_k=12, use_async=False)
        cand = fusion.retrieve(query)
    nodes = reranker.postprocess_nodes(cand, query_str=query)[:top_k]

buf = io.StringIO()
buf.write(f'Q: {query}   ({time.time()-t0:.2f}s, top {top_k}, corpus={corpus}, index={persist})\n')
for i, n in enumerate(nodes):
    md = n.node.metadata or {}
    stype = md.get("source_type", "?")
    src = md.get("source") or os.path.basename(str(n.node.ref_doc_id or ""))
    txt = " ".join((n.node.get_content() or "")[:220].split())
    score = round(n.score, 3) if n.score is not None else ""
    buf.write(f"  [{i+1}] {score}  ({stype}) {src}\n      {txt}\n")
if not nodes:
    buf.write("  (no hits" + (f" in corpus '{corpus}'" if corpus != "all" else "") + ")\n")
sys.stdout.buffer.write(buf.getvalue().encode("utf-8", "replace"))
sys.stdout.buffer.write(b"\n")
