# RAG retrieval systems — head-to-head for the research workflow (SOMA vs LlamaIndex)

**Purpose.** We keep re-deriving conclusions our own `research/findings/` already record (the dendrite recommendation, the off-diagonal red herring, etc.). A retrieval index over our findings — surfaced at the top of research gates as a "check our own memory first" step — directly prevents that. This doc tracks two local, free RAG systems on the SAME corpus + query set so we can pick (or keep both). **Running doc — updated as we use them.**

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

### Honest scope of this comparison
One corpus (our findings), one 7-query set, hand-graded exact-vs-adjacent — indicative, not a benchmark. The two use the SAME embedder + reranker, so the delta is chunking + fusion strategy, not the embedding model. A fairer future pass: match chunk sizes, add the catalog/Kandel/PDF corpus, and grade blind. Updated as we actually use them in the loop.
