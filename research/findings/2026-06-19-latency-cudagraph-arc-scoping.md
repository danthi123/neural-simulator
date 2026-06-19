# Latency / CUDA-graph arc — SCOPING (profiling-grounded): the bottleneck MOVED off the resonate loop

**Date:** 2026-06-19 · **Type:** scoping (profile + ranked next increments + cheap-first de-risk). NO build yet.
**Owner priority:** reduce per-op LATENCY toward real-time so the conversational sim is locally viable at small-LLM
scale (memory `feedback_prioritize_orchestration_overhead`). **Substrate:** `SIM_BACKEND=cupy`, RTX 3090.

## TL;DR (the headline — and it contradicts the "resonate loop is 98%" thesis)

The owner's earlier profile (one composer op ≈160 ms, **97.7 % the 208-step resonate loop**) was the PRE-megakernel
picture. The A5 megakernel (`cfg.enable_rf_cudagraph`, CYCLE 185, GO) already collapsed that loop **6–31×** and IS
**ON BY DEFAULT in the production `OneBrainComposer`**. **So the bottleneck has MOVED.** Measured NOW, on a full
`OneBrainComposer.query_patient` with the megakernel ON:

| scale | full query | resonate (megakernel) | **per-op WEIGHT REBUILD** (`rf_set_complex_weights`) |
|---|---|---|---|
| D=128, K=8 | 80.9 ms | 33.1 ms (~40 %) | **~59 ms (≈72 % of the read)** |
| D=128, K=16 | ~99 ms | ~3 ms (~3 %) | **~71 ms (≈72 %)** |
| D=128, K=32 | 174.1 ms | 10.3 ms (~6 %) | **~116 ms (≈62 %)** |

**The residual bottleneck is the per-op reconstruction of the unbind + cleanup COMPLEX-WEIGHT MATRICES** — not the
resonate. Every query rebuilds ~100k–240k `(post, pre, complex_w)` Python tuples in nested loops, then constructs two
fresh `cupyx` CSR matrices + H2D transfer, **from scratch, identically, every time.** The megakernel made each
resonate step 12.5 µs (208 steps ≈ 2.7 ms, fixed per-call setup ~0.07 ms) — the resonate is *solved*.

**The fix is cheap, and DE-RISKED GO this session:** those unbind/cleanup CSRs are **query-INVARIANT** (they depend
only on `(n_facts, vocab, fixed block layout)`, never on stored fact content — that lives in `store_conns`). Building
them ONCE and reusing the device CSRs is **answer-IDENTICAL** (all blocks + all query cues bit-for-bit) and already
**6.0× on the batched read** (100.7 → 16.9 ms) in a monkeypatch PoC.

## 1. Current CUDA-graph / megakernel state (what's already done — assessed, not re-invented)

**`sim/bridge.py` `rf_megastep` RawKernel + `_rf_resonate_steps_megakernel`, gated `cfg.enable_rf_cudagraph`
(`sim/config.py:658`-ish), dispatched in `rf_resonate_steps` (bridge.py ~5607).** One thread/neuron does the WHOLE
resonate step (complex CSR matvec + rotate/decay + zero-crossing + writes), collapsing the launch-bound ~15-kernel/step
loop into **1 kernel/step**. Double-buffered `(re,im)`; honors the co-residence `_rf_neuron_mask` (A5 lever 3) so the
masked co-resident composer uses it too.

- **Status:** GO, adoption gate PASSED (`research/findings/2026-06-17-rf-megakernel-resonate-GO.md`): full
  conversational suite **8/8 answer-identical** (megakernel == loop == ground truth, incl. embedded clause + multi-hop
  + the no-confab abstention). Clean quiet-GPU **24×** on a query (856.5 → 36.3 ms) was the banked number.
- **Default-on where it counts:** `OneBrainComposer.__init__` defaults `enable_rf_cudagraph=True`
  (`one_brain_composer.py:74`) → **the production "one brain" composer already runs the megakernel.** Bridge-level
  default stays OFF (so tests/numpy are byte-identical); `RFPhasorComposer` exposes it as default-OFF opt-in
  (`enable_rf_cudagraph=False`, the rf composer is the test oracle / numpy-CPU path).
- **My re-measurement of the megakernel alone (this session, K=16, D=128):** resonate 208 steps = **2.72 ms** with the
  megakernel vs **212–321 ms** for the loop → **6.4× (K=8) … 31.2× (K=32)** (the speedup grows with N because the loop
  is purely launch-bound while the megakernel is compute-bound). **The megakernel works as advertised.**

## 2. The MEASURED residual per-op breakdown (the load-bearing measurement)

Profilers (committed): `research/runners/_latency_profile_onebrain.py`, `_latency_profile_sections.py`,
`_latency_cache_poc.py`. CuPy `Stream.null.synchronize()`-bracketed medians, 5–7 reps, warmup.

**`OneBrainComposer._read_all_blocks` (the batched query hot path) at K=16, D=128 — total ≈ 98 ms:**

| section | time | % | what it is |
|---|---|---|---|
| **clean-list Python build** | **28.8 ms** | 29 % | 108,544 `(post,pre,w)` tuples in a triple-nested host loop (`K × 3V × D`) |
| **`rf_set_complex_weights(clean)`** | **42.5 ms** | 43 % | H2D + **2× `csr_matrix` construction** for those 108k conns |
| unbind-list build + set | ~4 ms | 4 % | `K × 4 × D` = 8,192 conns |
| store_conns set + 3 resonates | ~3 ms | 3 % | the megakernel resonates — **negligible now** |
| readout (D2H `cp_membrane` + argmax) | small | — | one `to_host` + numpy argmax |

⇒ **the cleanup weight rebuild (Python tuple-gen + 2 CSR builds + H2D) ≈ 71 ms ≈ 72 % of the query.** Both halves
matter: the **host tuple-iteration** (`np.fromiter` over generator comprehensions of Python `complex()` calls) AND the
**GPU CSR construction** scale with `n_conns` (108k @ K=16 → 217k @ K=32). The resonate — the thing the owner's thesis
named as 98 % — is now **3–6 %.**

**Why the bottleneck moved:** the megakernel removed the launch-bound resonate, exposing the next layer. The composer
was written reuse-by-import with the convention "build the complex weights FRESH each op → replaces" (a correctness-
first choice, see `rf_set_complex_weights` docstring + `RFPhasorComposer._resonate` "(c-opt) builds fresh each op").
That was fine when the resonate dwarfed it; now it IS the cost.

**Kernel-launch count is no longer the issue for the resonate** (1/step), but the CSR constructors + H2D copies are
themselves multi-kernel/multi-copy host-synchronizing operations issued 3× per query (store/unbind/clean).

**Cross-check — `RFPhasorComposer` (GPU, megakernel ON) query = 27.8 ms** at the same K/D, i.e. ~3.5× faster than
`OneBrainComposer`'s 98 ms, because the rf composer does smaller per-op resonates and does NOT rebuild a giant
all-K-blocks cleanup set in one shot. So the onebrain **batched** cleanup-weight set is the *specific* hot spot
(the batched scan, A5 lever 1, traded resonate count for a bigger one-shot weight build).

## 3. Ranked next increments (leverage × cheapness, against the MEASURED bottleneck)

1. **★ Cache the query-invariant unbind + cleanup CSRs (build once, reuse the device matrices).** Directly kills the
   ~72 % weight-rebuild cost. The cleanup codebook conj-phasors + the block-diagonal layout are fixed per
   `(vocab, n_facts)`; only `store_conns` changes when facts change. **DE-RISKED GO (§4): answer-identical, 6.0×.**
   *Cheapest, highest leverage, byte-safe.*
2. **Cache cleanup ONCE for `k_max` (not per-n), slice the readout to `n`.** The batched cleanup is block-diagonal;
   build the full `k_max`-block CSR a single time, fire only the first `n` triggers, read only `n` blocks. Removes the
   per-n rebuild entirely (one build per composer lifetime). Folds into #1.
3. **Avoid rebuilding `store_conns` CSR when facts are unchanged.** A query doesn't change the store; only `store`/
   `update_on_mismatch` do. Mark the store CSR dirty on write, reuse otherwise. Small (store set is `n×D`, the
   smallest of the three) but free once #1 lands.
4. **Build the connection sets ON-DEVICE (vectorized), not as host Python tuples.** If any weight set must be rebuilt
   (e.g. `store_conns` after a write), construct `rows/cols/data` with cupy arange/broadcast instead of `np.fromiter`
   over Python generators — removes the 29 ms host tuple-gen. Lower priority once #1+#3 make rebuilds rare.
5. **Persistent CUDA graph across the 3 resonate windows of a query.** The residual ~3 ms resonate is already tiny;
   graph-capture is a *final* increment, not the lever. (The A5 GO doc already flags graph-capture as the smaller
   follow-on.) **De-prioritized — the profile says weights, not launches, dominate now.**
6. **Index the fact store (the owner's lever c).** Relevant to *scaling* K (skip non-matching blocks) but the batched
   read already settles all blocks in 3 windows; with #1 the per-query cost is dominated by the (now-cached) cleanup,
   so indexing is a scaling lever for very large K, not the current real-time lever.

**The profile says #1 (+#2/#3 as the same edit) matters MOST.** It targets the measured 72 %.

## 4. Recommended cheap-first de-risk + GO bar (top increment = cached query-invariant CSRs)

**Already executed this session as a monkeypatch PoC** (`research/runners/_latency_cache_poc.py`): a cached
`_read_all_blocks` that builds the unbind + cleanup CSRs once (keyed by `n_facts`) and installs them by direct
`cp_rf_w_re/im` assignment instead of rebuilding from a fresh tuple list each query.

**Result (RTX 3090, K=16, D=128):**
- **Answer-IDENTITY:** the cached batched read == the stock read **for all 16 blocks bit-for-bit**, AND
  `query_patient` is **identical across all 16 query cues.** (Caching reuses the *same* CSR values; the only thing
  that changed is *when* they're built.)
- **Speed:** `_read_all_blocks` **100.7 → 16.9 ms = 6.0×.** Residual is store_conns rebuild + 3 resonates + readout
  (further removable via increments #2/#3).

**GO bar for the real build (to land it in `OneBrainComposer`, default-on, reuse-by-import, NO `sim/` edit needed —
this is a composer-layer cache):**
1. **Answer-identity (the anti-regression gate, MUST pass byte-level):** on GPU, a cached-composer and a stock
   composer at the same seed, hearing the same facts (incl. a recursive embedded clause + chain facts + a NEGATE +
   an unstored cue), return **identical** answers across the FULL who/what matrix — `query_patient` / `query_agent` /
   `ask_yes_no` / `render_fact` / `query_chain` — **including the `is None` no-confab abstentions.** Speed must not
   change behavior. (Reuse the existing `tests/test_one_brain_composer_agent.py` 11-test matrix + the A5
   `_phaseB_megakernel_conversation_validation.py` harness pattern, run with caching ON.)
2. **Cache invalidation correctness:** after `store` / `update_on_mismatch` (which change `n_facts` or `store_conns`),
   the cache rebuilds the affected CSR (the store CSR always; the unbind/cleanup CSR only when `n_facts` grows). A
   reconsolidation in-place rewrite (same `n_facts`) must reuse the cleanup CSR but rebuild the store CSR → still
   answer-identical (the `test_reconsolidation_*` tests must pass with caching on).
3. **Speedup ≥ 4× on the batched read at K∈{8,16,32}, D=128** (the PoC shows 6.0× at K=16) AND a measurable
   end-to-end `query_patient` improvement (target: ~98 → ~25 ms at K=16, i.e. into the rf-reference ballpark).
4. **No memory blow-up:** the cached CSRs are the same size as the per-op ones, just persistent (1 set per
   `n_facts`, or 1 total with increment #2). Bounded by `k_max`.

**Anti-regression gate (load-bearing, per `feedback_moat_not_hard_lossy_memory_ok` — the moat stays where it's free):**
the no-confab abstention + answer-identity are the hard gate for THIS speed work — a latency optimization must be
**behaviorally transparent.** (This is distinct from the owner's standing "trade the moat for scaling/dev-speed where
biological" note, which is about *representation* changes, not a kernel/cache speedup. A cache that changed an answer
would be a bug, not a trade.)

## 5. Technical note (the CuPy pattern for the measured bottleneck)

The lever is ordinary engineering, not an exotic kernel: **stop reconstructing static device CSRs per call.** Build
`cupyx.scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n,n))` ONCE for the query-invariant unbind/cleanup
operators and hold the device handles; reuse via direct assignment. Where a set genuinely must be rebuilt
(`store_conns` after a write), prefer constructing `rows/cols/data` with on-device cupy `arange`/broadcast over
`np.fromiter` host generators (the 29 ms host-tuple cost). The CSR `data/indices/indptr` are 1-D device arrays
([CuPy docs](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html)); the matvec line
(`W @ z`) is unchanged, so this is purely a build-time/reuse change — the dynamics (and the megakernel matvec) are
byte-identical.

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._latency_profile_onebrain    # per-op breakdown, megakernel ON
SIM_BACKEND=cupy python -u -m research.runners._latency_profile_sections    # build vs set vs resonate vs readout
SIM_BACKEND=cupy python -u -m research.runners._latency_cache_poc           # the de-risk: answer-identical, 6.0x
```

## Files

- Profilers/PoC: `research/runners/_latency_profile_onebrain.py`, `_latency_profile_sections.py`,
  `_latency_cache_poc.py`.
- Megakernel (existing): `sim/bridge.py` (`_RF_MEGASTEP_SRC` + `_rf_resonate_steps_megakernel` + `rf_resonate_steps`
  dispatch), `sim/config.py` (`enable_rf_cudagraph`). Findings: `2026-06-17-rf-megakernel-resonate-GO.md`.
- Production composer (the cache target): `research/runners/one_brain_composer.py`
  (`_read_all_blocks` / `_read_block` / `_decode_clause`), reuse-by-import of `RFPhasorComposer`.

Sources: [CuPy csr_matrix](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html).
