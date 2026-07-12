"""Query the LlamaIndex index with a SOMA-analogous hybrid pipeline (vector + BM25 fusion -> cross-encoder rerank),
on the SAME query set, and write results to a utf-8 file for the head-to-head comparison doc."""
import os, io, time
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import redirect_stderr

with redirect_stderr(io.StringIO()):
    from llama_index.core import StorageContext, load_index_from_storage, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.postprocessor import SentenceTransformerRerank

    PERSIST = r"E:\Documents\Projects\rag_compare\llamaindex_findings"
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST))
    vec = index.as_retriever(similarity_top_k=10)
    bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    fusion = QueryFusionRetriever([vec, bm25], num_queries=1, mode="reciprocal_rerank",
                                  similarity_top_k=10, use_async=False)          # RRF hybrid, no LLM query-gen
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)  # same reranker as SOMA

QUERIES = [
    "have we tested the dendrite for multi-attribute binding",
    "off-diagonal decorrelation is a red herring for generalization",
    "long-range language is input-representation bound on a fixed reservoir",
    "CA3 has no feedback inhibition sparse coding pattern separation",
    "no-confab moat false accepts abstention gate",
    "dual-timescale eligibility was an effective learning rate artifact",
    "reservoir beats full backprop by learning only the input",
]
out = open(r"E:\Documents\Projects\rag_compare\_llamaindex_results.txt", "w", encoding="utf-8")
tq = time.time()
for q in QUERIES:
    t0 = time.time()
    nodes = fusion.retrieve(q)
    nodes = reranker.postprocess_nodes(nodes, query_str=q)
    out.write("=" * 95 + f"\nQ: {q}   ({time.time()-t0:.2f}s)\n")
    for i, n in enumerate(nodes[:3]):
        src = n.node.metadata.get("file_name") or n.node.metadata.get("file_path") or n.node.ref_doc_id or ""
        src = os.path.basename(str(src))
        txt = " ".join((n.node.get_content() or "")[:150].split())
        out.write(f"  [{i+1}] {round(n.score, 3) if n.score is not None else ''}  {src}\n      {txt}\n")
out.write(f"\n[total retrieval {time.time()-tq:.1f}s for {len(QUERIES)} queries]\n")
out.close()
print("COMPARE_DONE", flush=True)
