"""rag_eval — scored, ACCUMULATING head-to-head of the two RAG engines (SOMA vs LlamaIndex) on OUR use case.

Why: docs/RAG_COMPARISON.md is a static, hand-graded snapshot. This runs the SAME labelled query set through BOTH
engines' PRODUCTION retrieval paths, scores them against known-correct findings (hit@1 / hit@3 / MRR + latency), and
APPENDS one timestamped record per run to tools/rag/rag_eval_history.jsonl + a row to docs/RAG_EVAL_HISTORY.md. Run it
periodically (or after a rebuild / a corpus jump / a config change) so we can TRACK how each engine performs over time
as the corpus grows, and draw real conclusions -- not re-eyeball a static table.

Grow the query set: add lines to tools/rag/rag_eval_queries.jsonl as real research gates run
  {"q": "<a real 'have we already ...?' question>", "relevant": ["<distinctive basename substring>", ...]}
A hit counts as relevant iff any `relevant` substring appears (case-insensitively) in the hit's source basename.

Run with the isolated RAG venv (has llama-index + soma):
  E:/Documents/Projects/rag_compare_env/Scripts/python.exe tools/rag/rag_eval.py [--top-k 5] [--ts ISO8601] [--note "..."]
`--ts` stamps the record (pass one for reproducibility; defaults to now). See docs/RAG_COMPARISON.md."""
import os, io, sys, csv, json, glob, time, argparse
from contextlib import redirect_stderr
from datetime import datetime

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SIM = r"E:\Documents\Projects\sim"
LLAMA_FULL = r"E:\Documents\Projects\rag_compare\llamaindex_full"
LLAMA_FINDINGS = r"E:\Documents\Projects\rag_compare\llamaindex_findings"
SOMA_BUNDLE = r"E:\Documents\Projects\soma_bundles\sim_kb"
SOMA_MANIFEST = r"E:\Documents\Projects\soma_bundles\.soma_kb_manifest.json"
QUERIES = os.path.join(SIM, "tools", "rag", "rag_eval_queries.jsonl")
HISTORY_JSONL = os.path.join(SIM, "tools", "rag", "rag_eval_history.jsonl")
HISTORY_MD = os.path.join(SIM, "docs", "RAG_EVAL_HISTORY.md")


def load_queries():
    qs = []
    with open(QUERIES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def _first_relevant_rank(basenames, relevant):
    """1-based rank of the first hit whose basename contains any `relevant` substring; 0 if none in the list."""
    rl = [r.lower() for r in relevant]
    for i, b in enumerate(basenames):
        bl = (b or "").lower()
        if any(r in bl for r in rl):
            return i + 1
    return 0


def score_one(basenames, relevant, top_k):
    rank = _first_relevant_rank(basenames, relevant)
    return {"first_rel_rank": rank,
            "hit@1": int(rank == 1),
            "hit@3": int(1 <= rank <= 3),
            "hit@5": int(1 <= rank <= min(5, top_k)),
            "rr": (1.0 / rank) if rank else 0.0,
            "top": basenames[:3]}


# ------------------------- engines (production retrieval paths) -------------------------

def build_llamaindex(top_k):
    with redirect_stderr(io.StringIO()):
        from llama_index.core import StorageContext, load_index_from_storage, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.postprocessor import SentenceTransformerRerank
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = None
        persist = LLAMA_FULL if os.path.isdir(LLAMA_FULL) else LLAMA_FINDINGS
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist))
        reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_k)
        vec = index.as_retriever(similarity_top_k=12)
        bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=12)
        fusion = QueryFusionRetriever([vec, bm25], num_queries=1, mode="reciprocal_rerank",
                                      similarity_top_k=12, use_async=False)
    n_nodes = len(index.docstore.docs)

    def run(query):
        with redirect_stderr(io.StringIO()):
            t0 = time.time()
            cand = fusion.retrieve(query)
            nodes = reranker.postprocess_nodes(cand, query_str=query)[:top_k]
            dt = (time.time() - t0) * 1000.0
        names = []
        for n in nodes:
            md = n.node.metadata or {}
            names.append(md.get("source") or os.path.basename(str(n.node.ref_doc_id or "")))
        return names, dt
    return run, n_nodes


def build_soma(top_k):
    with redirect_stderr(io.StringIO()):
        from soma.memory.api import MemoryLayer
        mem = MemoryLayer.load_with_sbert(SOMA_BUNDLE)
    n_chunks = None
    try:
        man = json.load(open(SOMA_MANIFEST))
        n_chunks = sum(len(v.get("ids", [])) for v in man.values() if isinstance(v, dict))
    except Exception:
        pass

    def run(query):
        with redirect_stderr(io.StringIO()):
            t0 = time.time()
            hits = mem.retrieve(query, k=top_k)
            dt = (time.time() - t0) * 1000.0
        names = []
        for h in hits:
            md = getattr(h, "metadata", None) or {}
            names.append(md.get("path") or (os.path.basename(md.get("src", "")) if md.get("src") else ""))
        return names, dt
    return run, n_chunks


def aggregate(rows):
    n = len(rows)
    if not n:
        return {}
    keys = ("hit@1", "hit@3", "hit@5", "rr")
    agg = {k: round(sum(r["score"][k] for r in rows) / n, 4) for k in keys}
    lat = sorted(r["latency_ms"] for r in rows)
    agg["mrr"] = agg.pop("rr")
    agg["mean_latency_ms"] = round(sum(lat) / n, 1)
    agg["median_latency_ms"] = round(lat[n // 2], 1)
    agg["n_queries"] = n
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ts", default=None, help="ISO8601 timestamp to stamp this run (default: now)")
    ap.add_argument("--note", default="", help="free-text note for this run (what changed)")
    args = ap.parse_args()
    ts = args.ts or datetime.now().isoformat(timespec="seconds")

    queries = load_queries()
    n_findings = len([f for f in glob.glob(os.path.join(SIM, "research", "findings", "*.md"))
                      if os.path.basename(f) not in ("AUTONOMOUS_STATE.md", "AUTONOMOUS_STATE_ARCHIVE.md")])

    engines = {}
    try:
        engines["llamaindex"] = build_llamaindex(args.top_k)
    except Exception as e:
        print(f"[rag_eval] llamaindex unavailable: {e}", flush=True)
    try:
        engines["soma"] = build_soma(args.top_k)
    except Exception as e:
        print(f"[rag_eval] soma unavailable: {e}", flush=True)

    result = {"ts": ts, "top_k": args.top_k, "note": args.note,
              "corpus": {"n_findings": n_findings}, "engines": {}, "per_query": []}
    per_engine_rows = {name: [] for name in engines}

    for q in queries:
        entry = {"q": q["q"], "relevant": q["relevant"]}
        for name, (run, size) in engines.items():
            names, dt = run(q["q"])
            sc = score_one(names, q["relevant"], args.top_k)
            entry[name] = {"score": sc, "latency_ms": round(dt, 1)}
            per_engine_rows[name].append({"score": sc, "latency_ms": dt})
        result["per_query"].append(entry)

    for name, (run, size) in engines.items():
        result["corpus"]["llama_nodes" if name == "llamaindex" else "soma_chunks"] = size
        result["engines"][name] = aggregate(per_engine_rows[name])

    # append the structured record (the accumulating over-time log)
    with open(HISTORY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    # human summary
    print(f"\n=== RAG eval  {ts}  (top_k={args.top_k}, {len(queries)} queries, {n_findings} findings) ===")
    hdr = f"{'engine':<12} {'hit@1':>6} {'hit@3':>6} {'MRR':>6} {'lat_ms':>8}  corpus"
    print(hdr); print("-" * len(hdr))
    for name in engines:
        a = result["engines"][name]
        size = result["corpus"].get("llama_nodes" if name == "llamaindex" else "soma_chunks")
        print(f"{name:<12} {a['hit@1']:>6} {a['hit@3']:>6} {a['mrr']:>6} {a['mean_latency_ms']:>8}  {size} chunks/nodes")

    # append a running markdown row
    _append_md_row(ts, args.top_k, len(queries), n_findings, result, args.note)
    print(f"\n[rag_eval] appended -> {os.path.relpath(HISTORY_JSONL, SIM)} + {os.path.relpath(HISTORY_MD, SIM)}")


def _append_md_row(ts, top_k, nq, n_findings, result, note):
    header = ("# RAG eval history (SOMA vs LlamaIndex, over time)\n\n"
              "Auto-appended by `tools/rag/rag_eval.py` — each row is one scored run over the labelled query set in\n"
              "`tools/rag/rag_eval_queries.jsonl` (structured records in `tools/rag/rag_eval_history.jsonl`). Higher\n"
              "hit@1 / hit@3 / MRR = better; lower latency = better. Grow the query set as real research gates run.\n\n"
              "| date | queries | findings | engine | hit@1 | hit@3 | MRR | lat ms | note |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
    if not os.path.exists(HISTORY_MD):
        with open(HISTORY_MD, "w", encoding="utf-8") as f:
            f.write(header)
    with open(HISTORY_MD, "a", encoding="utf-8") as f:
        for name in result["engines"]:
            a = result["engines"][name]
            f.write(f"| {ts} | {nq} | {n_findings} | {name} | {a['hit@1']} | {a['hit@3']} | "
                    f"{a['mrr']} | {a['mean_latency_ms']} | {note} |\n")


if __name__ == "__main__":
    main()
