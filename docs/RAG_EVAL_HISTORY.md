# RAG eval history (SOMA vs LlamaIndex, over time)

Auto-appended by `tools/rag/rag_eval.py` — each row is one scored run over the labelled query set in
`tools/rag/rag_eval_queries.jsonl` (structured records in `tools/rag/rag_eval_history.jsonl`). Higher
hit@1 / hit@3 / MRR = better; lower latency = better. Grow the query set as real research gates run.

| date | queries | findings | engine | hit@1 | hit@3 | MRR | lat ms | note |
|---|---|---|---|---|---|---|---|---|
| 2026-07-12T07:54:14 | 7 | 1508 | llamaindex | 0.8571 | 1.0 | 0.9048 | 356.3 | seed baseline (broadened corpus, node-id incremental) |
| 2026-07-12T07:54:14 | 7 | 1508 | soma | 0.4286 | 0.7143 | 0.5595 | 77.7 | seed baseline (broadened corpus, node-id incremental) |
