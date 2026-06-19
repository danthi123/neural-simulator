# Latency arc — OneBrainComposer query-invariant CSR cache = GO (10.6–19.5× faster, answer-identical, no-confab moat intact) (2026-06-19, CYCLE 229/230)

**Chapter:** the latency / per-op-cost arc (owner-chosen at the CYCLE-225 next-chapter fork: speed is the universal enabler → real-time, locally viable at small-LLM scale). This is the top increment from the CYCLE-227 profile-first scoping, now built + verified.

## What it is

The production conversational composer (`OneBrainComposer`, the one-brain spiking pipeline) rebuilt its
**unbind + cleanup complex-weight CSRs from scratch on every query** — a nested-loop tuple list (~100k–240k
entries) → two fresh `cupyx`/`scipy` `csr_matrix` constructions → host→device copy, repeated identically every
`query_patient`. The CYCLE-227 profile measured this weight-rebuild at **62–72% of a query** (the resonate loop
was already solved by the A5 masked megakernel, so the bottleneck had moved here).

Those CSRs are **query-invariant**: they depend only on `(n_facts, the role/concept codebooks, the fixed block
layout)`, never on the stored fact content (which lives in `store_conns`). So build them ONCE and reuse the
device matrices.

**The cache (`enable_csr_cache`, default ON; composer-layer, NO `sim/` edit):**
- `_csr_cache[n_facts]` → the batched unbind + cleanup CSR pairs, built once per store size `n`.
- `_store_csr` → the `store_conns` CSR pair, rebuilt only when `_store_dirty`.
- `_write_block()` (the SINGLE `store_conns` mutation point) sets `_store_dirty` → both an initial `store` AND a
  reconsolidation in-place rewrite invalidate the store CSR.
- `_read_all_blocks` installs the cached CSRs by direct `cp_rf_w_re/im` assignment; the cached and stock paths
  share the refactored `_decode_batched_mem` → **answer-identical by construction**. `enable_csr_cache=False` =
  the byte-identical stock path (the A/B baseline).

## Verification (all three gates GREEN)

1. **Answer-identity — by construction + empirical.** Byte-review: the cached and stock paths differ ONLY in
   *when* the CSRs are built (cached once per `n`), never in their VALUES; the matvec/dynamics/decode are
   byte-unchanged. Empirical: `tests/test_one_brain_composer_agent.py` **13/13 PASS** with the cache default-ON
   — the full who/what matrix, the `is None` no-confab abstentions, batched==per-block parity, clause,
   reconsolidation, grounded-codes drop-in, and multi-turn. **The no-confab moat is intact** (the abstention
   decisions are bit-identical).
2. **Invalidation correctness.** A new `store` grows `n` → a new unbind/clean cache key (rebuild); a
   reconsolidation in-place rewrite keeps `n` → REUSES the (content-independent) unbind/clean operators,
   rebuilds only the store CSR → the update is reflected and the other fact's read is uncorrupted. Covered by
   the reconsolidation + multi-turn CI tests and the subagent's `_csr_cache_answer_identity.py` gates (2)+(3).
3. **Speedup ≥4× — PASS decisively** (`_csr_cache_answer_identity.py` §5, GPU, RTX 3090):

   | K (facts) | stock (rebuild/query) | cached | speedup |
   |---|---|---|---|
   | 8  | 102.2 ms | 9.7 ms | **10.6×** |
   | 16 | 131.2 ms | 9.4 ms | **14.0×** |
   | 32 | 189.6 ms | 9.7 ms | **19.5×** |

   The cached read is **flat ~9.5 ms regardless of `K`** — the cache removes the per-query CSR rebuild that
   scaled with the knowledge-base size, so the win GROWS with `K` (19.5× at the production K=32). A sub-10ms
   composer query is real-time-viable on local hardware.

## Status + next

- **GO + shipped default-ON** (`research/runners/one_brain_composer.py`, commit `e63cfba5`; controller
  byte-reviewed + CI-confirmed). `rf`/`rate` composers and the numpy-CPU path are unaffected.
- **The latency chapter's top increment is landed.** Per the CYCLE-227 scoping, the remaining increments
  (persistent cross-op CUDA-graph) target the now-tiny resonate (already megakernel'd) and the now-cached CSR
  rebuild — low marginal value. The chapter can PAUSE here; the dominant per-op cost is removed. Re-profile if
  a future workload re-surfaces a bottleneck.
- Reuse: `_csr_cache_answer_identity.py` is the standing A/B + speedup gate (`SIM_BACKEND=cupy python -u -m
  research.runners._csr_cache_answer_identity`).
