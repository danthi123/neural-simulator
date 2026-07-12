"""rag_search — "check our own findings FIRST" locate-accelerator for the research gate.

One-liner over the LlamaIndex index of research/findings/ (hybrid vector+BM25 RRF fusion ->
cross-encoder rerank, same embedder+reranker as SOMA). Surfaces the finding docs that already
answer a "have we already concluded X?" question, so a research gate does not re-derive what the
record holds. It LOCATES the passage; the discipline is still to open the cited finding and read it.

Run with the isolated RAG venv (has llama-index; base sim env untouched):
    E:/Documents/Projects/rag_compare_env/Scripts/python.exe tools/rag/rag_search.py "<question>" [top_k]

SOMA CLI fallback (owner's project, sbert-load bug fixed on branch fix/cli-load-sbert-bundle):
    soma search "<question>" --bundle E:/Documents/Projects/soma_bundles/sim_findings

See docs/RAG_COMPARISON.md for the head-to-head (LlamaIndex 7/7 exact, primary; SOMA 5/7, fallback)."""
import os, io, sys, time
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import redirect_stderr

PERSIST = r"E:\Documents\Projects\rag_compare\llamaindex_findings"

if len(sys.argv) < 2 or not sys.argv[1].strip():
    sys.stdout.write('usage: rag_search.py "<question>" [top_k]\n')
    raise SystemExit(2)
query = sys.argv[1]
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with redirect_stderr(io.StringIO()):
    from llama_index.core import StorageContext, load_index_from_storage, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.postprocessor import SentenceTransformerRerank

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST))
    vec = index.as_retriever(similarity_top_k=10)
    bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    fusion = QueryFusionRetriever([vec, bm25], num_queries=1, mode="reciprocal_rerank",
                                  similarity_top_k=10, use_async=False)
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_k)
    t0 = time.time()
    nodes = reranker.postprocess_nodes(fusion.retrieve(query), query_str=query)

# utf-8 to stdout (findings contain emoji/unicode; avoid cp1252 crash on Windows console)
buf = io.StringIO()
buf.write(f'Q: {query}   ({time.time()-t0:.2f}s, top {top_k})\n')
for i, n in enumerate(nodes[:top_k]):
    src = n.node.metadata.get("file_name") or n.node.metadata.get("file_path") or n.node.ref_doc_id or ""
    src = os.path.basename(str(src))
    txt = " ".join((n.node.get_content() or "")[:220].split())
    score = round(n.score, 3) if n.score is not None else ""
    buf.write(f"  [{i+1}] {score}  {src}\n      {txt}\n")
sys.stdout.buffer.write(buf.getvalue().encode("utf-8", "replace"))
sys.stdout.buffer.write(b"\n")
