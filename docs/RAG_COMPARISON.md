# RAG retrieval systems — head-to-head for the research workflow (SOMA vs LlamaIndex)

**Purpose.** We keep re-deriving conclusions our own `research/findings/` already record (the dendrite recommendation, the off-diagonal red herring, etc.). A retrieval index over our findings — surfaced at the top of research gates as a "check our own memory first" step — directly prevents that. This doc tracks two local, free RAG systems on the SAME corpus + query set so we can pick (or keep both). **Running doc — updated as we use them.**

## Current Linux status (2026-08-03)

LlamaIndex is the required production path and is working from the canonical checkout and linked worktrees. The `main` post-commit hook performs a detached incremental refresh for committed project prose. A user-level `sim-rag-autoupdate.timer` also checks the complete corpus every five minutes, covering papers and textbooks added directly to the sibling catalog. Both paths defer while indexed project Markdown is uncommitted, then build a candidate index, run `bash tools/rag/eval.sh --no-write` against that candidate, and atomically publish only after it passes. A failed candidate therefore leaves the prior index and manifest live. If files change during the refresh, the lock holder repeats before marking the manifest current. Targeted searches fuse filtered BM25 and vector candidates before cross-encoder reranking for every corpus, rather than dropping lexical retrieval whenever `--corpus` was used.

The portable evaluation currently scores **13/13 top-three hits, 11/13 rank-1 hits, MRR 0.923** across eight real project-history questions plus passage-level catalog, Kandel, and specialty-paper checks. Scientific labels require both the expected source and required passage text, and may reject misleading neighboring text; this prevents `feature-catalog.md` or Kandel's `full-book.txt` from passing merely because every result shares that filename. Explicitly named sources and the newest Gate B v4 retirement are required at rank 1. Search output includes the source's absolute path and a best-effort line number. SOMA is not installed on this Linux host (`No module named 'soma'`); this is nonfatal because LlamaIndex is the maintained required engine, but it is reported rather than hidden.

**Why not fine-tuning:** a fine-tune injects lossy paraphrases (hallucination-prone, un-citable, stale) — the opposite of our "read the load-bearing source, cite chapter/page" discipline. Retrieval surfaces the actual passage to read. See the 2026-07-12 discussion.

## The two systems (configured apples-to-apples)
| | **SOMA** (`E:\Documents\Projects\SOMA`) | **LlamaIndex** (venv `E:\Documents\Projects\rag_compare_env`) |
|---|---|---|
| What | Local-first agent-memory / RAG, hybrid retrieval | The most popular open RAG framework |
| Embedder | sentence-transformers `all-MiniLM-L6-v2` | **same** (`all-MiniLM-L6-v2`, HuggingFaceEmbedding) |
| Retrieval | BM25 + cosine, **cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`) | vector + BM25 **reciprocal-rank fusion** → **same cross-encoder rerank** |
| Chunking | its own (~20-25 chunks/doc, 38 829 total) | LlamaIndex default `SentenceSplitter` (~1024-token) |
| Interface | CLI (`soma index/search`) + Python API | Python (LlamaIndex API) |
| LLM | none (retrieval only) | none (`Settings.llm=None`) |

**Corpus (identical):** `E:\Documents\Projects\soma_bundles\_findings_md` — 1502 findings `.md` (staged copy of `research/findings/*.md`, raw JSON excluded).

## Index build
| | SOMA | LlamaIndex |
|---|---|---|
| Docs / nodes | 1502 / **38 829 chunks** (~25/doc) | 1502 / **7 201 nodes** (~1024-token chunks) |
| Build time | a few min (GPU-assisted) | **45 s** |
| Bundle | `soma_bundles/sim_findings` | `rag_compare/llamaindex_findings` |

## Retrieval quality — same 7 workflow queries, top hits
Legend: ✅ surfaced the exact/right finding · 🟡 relevant-adjacent · ❌ missed.

| # | Query (a real "have we already…?") | SOMA top hit | LlamaIndex top hit |
|---|---|---|---|
| 1 | dendrite tested for multi-attribute binding? | ✅ `dendritic-binding-toy-derisk` (the NEGATIVE, #2) + phase-3 scoping #1 | ✅ `dendritic-binding-toy-derisk` (the NEGATIVE, #2); AUTONOMOUS_STATE #1 |
| 2 | off-diagonal decorrelation a red herring? | ✅ `off-diagonal-red-herring-ppmi…` (#2) | ✅ `off-diagonal-red-herring-ppmi…` **(#1)** |
| 3 | long-range = input-representation bound on a fixed reservoir? | 🟡 content-addressable arc + AUTONOMOUS_STATE (not the R3 doc #1) | ✅ **`R3-REFRAME-…frozen-reservoir-beats-full-BPTT` (#1 & #2)** |
| 4 | CA3 feedback inhibition / sparsity? | ✅ riii-CA3 arc + NIGHT-ARC "CA3 SATURATED 94.6%" | ✅ **`riii-ca3-feedback-inhibition-sparsifies-but-nonselective` (#1)** |
| 5 | no-confab moat false accepts? | ✅ burndown moat + fluid-conversation moat | ✅ DA-salience-gate-moat + harden-320-moat |
| 6 | dual-timescale eligibility = LR artifact? | 🟡 2026-04-26 surprise-LR (adjacent — the one MISS) | ✅ **`R2b-dual-timescale-eligibility…` — the exact retraction (#1)** |
| 7 | reservoir beats full backprop via input? | ✅ `R3-REFRAME-…` (#1) | ✅ `R3-REFRAME-…` (#1) |

**Score:** **SOMA 5 ✅ / 2 🟡** · **LlamaIndex 7 ✅ / 0 🟡** — both are genuinely useful; **LlamaIndex was more precise**, nailing the exact finding #1 on the *two hardest queries SOMA got only adjacent* (3 = the R3 reframe; 6 = the dual-timescale retraction). Likely cause: LlamaIndex's larger ~1024-token chunks preserve each finding's headline/context, so the exact doc reranks to #1, whereas SOMA's ~25-chunks-per-doc fragments the key phrasing. **Caveat:** LlamaIndex floats the giant `AUTONOMOUS_STATE.md` into results (big-doc dominance, e.g. Q1 #1); SOMA's finer chunks avoid that.

## Speed (retrieval, warm)
| | SOMA | LlamaIndex |
|---|---|---|
| index build | a few min | **45 s** |
| ~per query | not separately timed (same embedder+reranker; comparable) | **~0.2 s** (1.6 s / 7 queries, warm) |

## Setup notes / bugs found
- **SOMA:** installed editable from the clone; models cached (MiniLM + the cross-encoder). **Bug found (being fixed + pushed):** the CLI read commands (`search`/`stats`/`chat`/`forget`) call plain `MemoryLayer.load()`, which cannot reload an sbert-indexed bundle (`"saved with a custom embed_fn"`); the Python API `load_with_sbert()` works. Fix = auto-detect the saved embed config on load.
- **LlamaIndex:** installed in an isolated venv (`--system-site-packages` to reuse base torch/transformers; base sim env verified untouched — torch 2.10 / numpy 2.4.6). llama-index 0.14.20 + `-embeddings-huggingface` + `-retrievers-bm25`. Build/query scripts: `E:\Documents\Projects\rag_compare\build_llamaindex.py`, `compare_llamaindex.py`.

## Verdict (first pass) / how we use them
- **LlamaIndex is the primary "check our own findings first" tool** — 7/7 exact on this query set (incl. the 2 SOMA got only adjacent), ~10× faster indexing (45 s vs a few min), ~0.2 s/query. Fix the big-doc-dominance by chunking `AUTONOMOUS_STATE.md` or excluding it from the index (it duplicates the findings anyway).
- **SOMA stays** — a genuinely good local-first alternative (5/7 exact) with a real CLI, and it's the owner's project (dogfooding surfaced + fixed the sbert-load bug). Once its CLI bug lands, `soma search` is the most ergonomic one-liner.
- **Neither replaces reading the source** — both SURFACE the passage; the discipline is still to open the cited finding and read it. That is the correct role (a locate-accelerator), not a paraphrase layer.
- **Workflow wiring (next):** a one-line helper (`rag_search "<question>"`) at the top of every research gate — "check our own memory before deep-researching externally." Default to LlamaIndex; keep SOMA as the CLI fallback.

## Corpus BROADENED beyond findings (2026-07-12) — a whole-knowledge-base + biology locator
The a-1 gate has two needs: "have we CONCLUDED/DESIGNED X?" AND "how does the BIOLOGY do X?". The findings-only index served only the first. The index now covers the whole PROSE knowledge base with `source_type` metadata (`tools/rag/build_llamaindex_full.py` → `rag_compare/llamaindex_full`, **10,125 nodes**, ~150 s embed):

| source_type | corpus | serves |
|---|---|---|
| `finding` | `research/findings/*.md` (1502) | "have we CONCLUDED / tried X?" |
| `plan` | `docs/plans/*.md` (281) | "did we already DESIGN X?" |
| `doc` | CLAUDE / ROADMAP / README / docs (15) | project state / architecture |
| `catalog` | `sim-catalog/references/*.md` (feature-catalog etc.) | **"is there a CATALOG ENTRY for X?"** |
| `kandel` | Kandel 6e full text (8.7 MB) | **"how does the BIOLOGY do X?"** |

`rag_search.py` gains `--corpus finding|plan|doc|catalog|kandel|all` (default `all`) and shows `(source_type)` per hit. **Filtering is applied DURING retrieval** (a `source_type` metadata filter on the vector retriever, not a post-filter) so a small corpus (catalog/plan) is not crowded out of the rerank window by the big corpora (Kandel/findings) — the bug the first version hit. **Verified:** `--corpus kandel "dentate gyrus CA3 pattern separation/completion"` → the exact Kandel DG/CA3 section (score 4.19); `--corpus catalog "dendritic plateau NMDA apical"` → the feature-catalog dendritic-computation entry (Larkum two-layer). Code/tests are deliberately EXCLUDED (Grep is better for code). Rebuild with `build_llamaindex_full.py` when the corpus drifts. Cold-start note: each `rag_search` invocation reloads the index + reranker (~a few s); batch queries sparingly.

### Honest scope of this comparison
One corpus (our findings), one 7-query set, hand-graded exact-vs-adjacent — indicative, not a benchmark. The two use the SAME embedder + reranker, so the delta is chunking + fusion strategy, not the embedding model. A fairer future pass: match chunk sizes, add the catalog/Kandel/PDF corpus, and grade blind. Updated as we actually use them in the loop.

## Automated + ACCUMULATING scored eval (2026-07-12) — track performance over time

The table above is a static hand-grade. `tools/rag/rag_eval.py` makes it a **living, scored, over-time** record so we can draw real conclusions as the corpus grows and configs change: it runs a **labelled query set** (`tools/rag/rag_eval_queries.jsonl` — each query tagged with the basename(s) of the finding that correctly answers it) through **both engines' production retrieval paths**, scores hit@1 / hit@3 / MRR + latency against that ground truth, and **appends one timestamped record per run** to `tools/rag/rag_eval_history.jsonl` (structured) + a row to `docs/RAG_EVAL_HISTORY.md` (human table).

```bash
E:/Documents/Projects/rag_compare_env/Scripts/python.exe tools/rag/rag_eval.py --note "what changed this run"
```

**Grow it as we use it:** every time a real research gate asks a "have we already…?" question, add the query + the correct finding's basename to `rag_eval_queries.jsonl`. The query set becomes more representative of our actual use case over time, and the history JSONL lets us plot each engine's precision/latency trend.

**Seed baseline (2026-07-12, 7 queries, 1508 findings, top-k 5):** LlamaIndex **hit@1 0.857 / hit@3 1.00 / MRR 0.905** @ ~356 ms/query; SOMA **hit@1 0.429 / hit@3 0.714 / MRR 0.560** @ ~78 ms/query. Confirms the hand-grade quantitatively — **LlamaIndex is more precise** (nails the exact finding in the top-3 on all 7), **SOMA is ~4.6× faster per query**. Run it again after a corpus jump / rebuild / retrieval-config change and compare the rows.

---

## 2026-07-16 — LINUX REDEPLOY: the index is rebuilt local + portable, and the specialty PAPERS/BOOKS are now in it

The Linux migration broke a-1 outright, and the fix turned up a corruption the old box had been carrying silently.

**What was broken.** `rag_search.py` / `build_llamaindex_full.py` / `update_indexes.py` all hardcoded
`E:\Documents\Projects\...`; `llama_index` was absent; `soma` was absent; `sim-catalog` had never been migrated.
So gate step a-1 ("check our own record FIRST") and step (a) ("read the source") could not run at all on this box.

**The carried-over index was CORRUPT — at the source, not from the copy.** `llamaindex_full/docstore.json` threw
`JSONDecodeError: Extra data` at char 51,629,090 with ~12 KB of trailing bytes. Source and copy are byte-identical
(51,641,153 both), and there are no NULs — the signature of a **non-atomic overwrite**: a shorter new docstore written
over a longer old one without truncating, almost certainly the auto-update hook caught mid-write during the migration
(`_autoupdate.log` is stamped Jul 16 04:57). A salvage (truncate to the valid prefix) was rejected in favour of a local
rebuild, which fixes three things at once: the corruption, the stale doc ids, and the frozen corpus.

**Paths are now portable** (no rebuild against `/mnt/projects` — that read-only backup is not a runtime dependency):
`$SIM_RAG_ROOT` → `<parent-of-repo>/rag_index` → the legacy Windows path as a last-resort fallback; likewise
`$SIM_CATALOG` / `$SIM_REPO`. The index + `sim-catalog` were copied off the backup to `~/Projects/rag_index` and
`~/Projects/sim-catalog`.

**llama-index lives in an isolated `.venv-rag`, NOT the sim `.venv`** — preserving the original "base sim env
untouched" design for a concrete reason: the sim venv pins `torch 2.11.0+cu128` alongside cupy, and llama-index's
dependency resolution would churn that CUDA stack. Always run the RAG tools with `.venv-rag/bin/python`, and
`CUDA_VISIBLE_DEVICES=""` when a GPU experiment is live (embedding is CPU-fine and must not contend for VRAM).

**NEW corpus `paper` — the specialty texts/papers/books are searchable for the first time.** Previously only Kandel's
`full-book.txt` was indexed, so every OTHER source the gate cites by name was un-locatable.
`tools/rag/extract_reference_pdfs.py` (idempotent; never overwrites the hand-made `.txt` siblings that already shipped
for Schultz + the Tepper/Bolam BG reviews) extracted 7 more with real text layers:

| newly searchable | chars | pages |
|---|---|---|
| Buzsaki, *Rhythms of the Brain* (2006) | 1,377,463 | 465 |
| O'Keefe & Nadel, *The Hippocampus as a Cognitive Map* (1978) | 1,648,490 | 297 |
| Moore (ed.), *Neuroscientist's Guide to Classical Conditioning* (2002) | 981,513 | 338 |
| Sutton & Barto, *RL* 2nd ed. | 737,728 | 352 |
| Albus 1971 / Marr 1969 / Hesslow 2013 | ~203,000 | 80 |

`--corpus paper` joins `finding|plan|doc|catalog|kandel|all`. `load_docs()` now **dedupes by path across source types**
(a `Document`'s `id_` IS its path, so a file matched by two globs would emit duplicate ids); `paper` is ordered AFTER
`kandel` so `full-book.txt` stays `source_type=kandel`.

**Index: 11,325 nodes** (9,232 before the papers), ~271 s CPU rebuild. **Verified live:**
`--corpus paper "cerebellar cortex granule cell codon expansion recoding"` → **Albus-1971** at score 2.72, returning
the actual recoding/pattern-recognition passage — i.e. the Marr-Albus codon source the EMERGE-35 arc is built on,
locatable at last. `--corpus all "homeostatic threshold adaptation degrading learning when neurons are idle"` → the
current co-training finding at 3.3 s.

**Auto-update is schema-gated and worktree-aware.** Repository-relative document ids replace checkout-specific absolute
paths, allowing the shared canonical index to refresh from the linked `main` worktree without deleting and re-embedding
the corpus solely because its checkout path changed. `tools/rag/check_workflow.py` verifies the executable repo hooks,
canonical interpreter/index/catalog, and `.rag_schema.json`; the post-commit hook logs missing dependencies, legacy
schema, and feature-branch skips rather than silently claiming a refresh. A one-time `update_indexes.py --rebuild` is
required when migrating an older absolute-path index.

**VERIFIED end-to-end, not assumed** (2026-07-16). (1) The hook fires: a `docs/` commit triggered it and it correctly
hit the lock held by a manual rebuild — `another update is running; skip (it will pick up these changes)`. (2) The
INCREMENTAL path is correct on the new ids — the load-bearing check, since a stale-id index is what silently wipes:
```
llamaindex refreshed (1/1924 new-or-changed, 0 deleted)   [28 s]
```
**`0 deleted` is the proof.** With foreign (E:) ids this run would instead have reported ~1924 deleted + ~1924
re-embedded. It found exactly the ONE file just edited and finished in 28 s vs 271 s for a full rebuild ⇒ the index
is genuinely incremental, correct, and self-maintaining. Re-run `update_indexes.py --force` (no `--rebuild`) to
re-check this cheaply after any corpus/path change.

**SOMA is NOT restored** (neither the module nor the bundles migrated). Its call site in `update_indexes.py` is already
`try/except`-wrapped, so it degrades cleanly — the log line `SOMA rebuild failed (LlamaIndex still updated): No module
named 'soma'` is EXPECTED on this box, not a fault. Consequence: `tools/rag/rag_eval.py`'s two-engine comparison is
LlamaIndex-only here until SOMA is redeployed; the head-to-head table above is historical (old box).
