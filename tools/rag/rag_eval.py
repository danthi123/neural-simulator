"""rag_eval — scored, ACCUMULATING head-to-head of the two RAG engines (SOMA vs LlamaIndex) on OUR use case.

Why: docs/RAG_COMPARISON.md is a static, hand-graded snapshot. This runs the SAME labelled query set through BOTH
engines' PRODUCTION retrieval paths, scores them against known-correct findings (hit@1 / hit@3 / MRR + latency), and
APPENDS one timestamped record per run to tools/rag/rag_eval_history.jsonl + a row to docs/RAG_EVAL_HISTORY.md. Run it
periodically (or after a rebuild / a corpus jump / a config change) so we can TRACK how each engine performs over time
as the corpus grows, and draw real conclusions -- not re-eyeball a static table.

Grow the query set: add lines to tools/rag/rag_eval_queries.jsonl as real research gates run
  {"q": "<a real 'have we already ...?' question>", "relevant": ["<distinctive basename substring>", ...]}
A hit counts as relevant iff any `relevant` substring appears (case-insensitively) in the hit's source basename.

Run with the canonical checkout's isolated RAG venv:
  bash tools/rag/eval.sh [--top-k 5] [--ts ISO8601] [--note "..."]
`--ts` stamps the record (pass one for reproducibility; defaults to now). See docs/RAG_COMPARISON.md."""
import os, io, sys, json, glob, time, argparse
from contextlib import redirect_stderr
from datetime import datetime

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_paths import resolve_paths
from retrieval import RagRetriever, node_source

PATHS = resolve_paths(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SIM = str(PATHS.repo)
SOMA_BUNDLE = str(PATHS.projects_root / "soma_bundles" / "sim_kb")
SOMA_MANIFEST = str(PATHS.projects_root / "soma_bundles" / ".soma_kb_manifest.json")
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


def _first_relevant_rank(hits, relevant, must_contain=(), must_not_contain=()):
    """Return the first source-and-passage match, not merely a matching filename."""
    rl = [r.lower() for r in relevant]
    required = [term.lower() for term in must_contain]
    forbidden = [term.lower() for term in must_not_contain]
    for i, hit in enumerate(hits):
        source = (hit.get("source") or "").lower()
        passage = (hit.get("text") or "").lower()
        if (not rl or any(r in source for r in rl)) \
                and all(term in passage for term in required) \
                and not any(term in passage for term in forbidden):
            return i + 1
    return 0


def score_one(hits, relevant, top_k, must_contain=(), must_not_contain=()):
    rank = _first_relevant_rank(hits, relevant, must_contain, must_not_contain)
    return {"first_rel_rank": rank,
            "hit@1": int(rank == 1),
            "hit@3": int(1 <= rank <= 3),
            "hit@5": int(1 <= rank <= min(5, top_k)),
            "rr": (1.0 / rank) if rank else 0.0,
            "top": [hit.get("source", "") for hit in hits[:3]]}


# ------------------------- engines (production retrieval paths) -------------------------

def build_llamaindex(top_k):
    engine = RagRetriever(PATHS, corpus="all", top_k=top_k)

    def run(query, corpus="finding"):
        nodes, dt = engine.retrieve(query, corpus=corpus)
        hits = [
            {"source": node_source(node), "text": node.node.get_content() or ""}
            for node in nodes
        ]
        return hits, dt
    return run, engine.node_count


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
            raw_hits = mem.retrieve(query, k=top_k)
            dt = (time.time() - t0) * 1000.0
        hits = []
        for h in raw_hits:
            md = getattr(h, "metadata", None) or {}
            source = md.get("path") or (os.path.basename(md.get("src", "")) if md.get("src") else "")
            hits.append({"source": source, "text": getattr(h, "text", "") or ""})
        return hits, dt
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
    ap.add_argument("--no-write", action="store_true", help="score without appending history")
    ap.add_argument("--min-hit3", type=float, default=1.0,
                    help="fail if LlamaIndex hit@3 is below this floor (default: 1.0)")
    ap.add_argument("--min-mrr", type=float, default=0.90,
                    help="fail if LlamaIndex MRR is below this floor (default: 0.90)")
    args = ap.parse_args()
    ts = args.ts or datetime.now().isoformat(timespec="seconds")

    queries = load_queries()
    n_findings = len([f for f in glob.glob(os.path.join(SIM, "research", "findings", "**", "*.md"), recursive=True)
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
        corpus = q.get("corpus", "finding")
        entry = {
            "q": q["q"],
            "corpus": corpus,
            "relevant": q["relevant"],
            "required_rank": q.get("required_rank", 3),
        }
        for name, (run, size) in engines.items():
            if name == "soma" and corpus != "finding":
                continue
            hits, dt = run(q["q"], corpus) if name == "llamaindex" else run(q["q"])
            sc = score_one(
                hits,
                q["relevant"],
                args.top_k,
                q.get("must_contain", ()),
                q.get("must_not_contain", ()),
            )
            entry[name] = {"score": sc, "latency_ms": round(dt, 1)}
            per_engine_rows[name].append({"score": sc, "latency_ms": dt})
        result["per_query"].append(entry)

    for name, (run, size) in engines.items():
        result["corpus"]["llama_nodes" if name == "llamaindex" else "soma_chunks"] = size
        result["engines"][name] = aggregate(per_engine_rows[name])

    if not engines:
        print("[rag_eval] no retrieval engine is available", file=sys.stderr)
        return 1

    if not args.no_write:
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
    for entry in result["per_query"]:
        llama_entry = entry.get("llamaindex")
        if llama_entry and not llama_entry["score"]["hit@3"]:
            print(f"MISS@3 ({entry['corpus']}): {entry['q']}")
            print(f"  expected: {entry['relevant']}")
            print(f"  returned: {llama_entry['score']['top']}")
        elif llama_entry and not llama_entry["score"]["hit@1"]:
            print(f"RANK>1 ({entry['corpus']}): {entry['q']}")
            print(f"  first relevant rank: {llama_entry['score']['first_rel_rank']}")
            print(f"  returned: {llama_entry['score']['top']}")

    if not args.no_write:
        _append_md_row(ts, args.top_k, len(queries), n_findings, result, args.note)
        print(f"\n[rag_eval] appended -> {os.path.relpath(HISTORY_JSONL, SIM)} + {os.path.relpath(HISTORY_MD, SIM)}")

    llama = result["engines"].get("llamaindex")
    rank_violations = []
    for entry in result["per_query"]:
        scored = entry.get("llamaindex", {}).get("score", {})
        rank = scored.get("first_rel_rank", 0)
        if not rank or rank > entry["required_rank"]:
            rank_violations.append((entry["q"], rank, entry["required_rank"]))
    if llama is None or llama["hit@3"] < args.min_hit3 \
            or llama["mrr"] < args.min_mrr or rank_violations:
        actual_hit3 = "unavailable" if llama is None else llama["hit@3"]
        actual_mrr = "unavailable" if llama is None else llama["mrr"]
        print(
            f"RAG_QUALITY_BLOCKED: hit@3={actual_hit3} required={args.min_hit3}; "
            f"MRR={actual_mrr} required={args.min_mrr}; rank_violations={rank_violations}",
            file=sys.stderr,
        )
        return 1
    print(f"RAG_QUALITY_READY: hit@3={llama['hit@3']} MRR={llama['mrr']}")
    return 0


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
    raise SystemExit(main())
