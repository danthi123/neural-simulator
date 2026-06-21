# Megakernel-revisit optimization audit (2026-06-22)

**Owner directive (CYCLE 405):** revisit the existing megakernel / CUDA-graph optimization work, because the
sim has grown enormously since it landed — much is likely now unoptimized. This is the **read-only scope/audit
step** that maps the CURRENT optimization state against the CURRENT hot paths so the latency optimization that
follows is well-targeted. No code edits, no experiments. All claims verified against the actual code, not memory.

**Empirical anchor (the in-flight latency diagnostic, `e1e8159e` / `c3875a9d`):** the steady-state per-query
RESONATE path is FAST (~14 ms — the existing RF megakernel works). The visible costs have **moved off the
optimized resonate loop** onto newer paths: the one-time BUILD (~41 s at V=64, partly CUDA warmup), the fact
STORE (~0.6 s/fact), ELABORATE (~10 s), and the `integrated_loop` K-way SEQUENCER per query at V=320 (the
flagged ~244,580-neuron bridge; the path the diagnostic was timing). The bottleneck is exactly where the owner
predicted.

---

## Terms (defined once)

- **Fused kernel** — multiple element-wise math ops compiled into ONE GPU kernel launch (`@fuse()` / `@cp.fuse()`).
  Removes per-op launch overhead. `sim/kernels.py`.
- **Megakernel** — one hand-written CUDA `RawKernel` doing a WHOLE simulation step (matvec + dynamics + crossing +
  writeback), one thread per neuron, collapsing ~15-20 CuPy kernels into 1 launch. `sim/bridge.py` `rf_megastep`.
- **CSR / SpMV** — Compressed Sparse Row matrix; sparse matrix-vector product. Cost is O(nnz) (non-zeros), not
  O(N²). The synaptic propagation and the RF complex-synapse matvec are both CSR SpMV.
- **Launch-bound / sync-bound** — wall-clock dominated by GPU kernel-launch count or device→host transfer
  stalls (`float(x)`, `.to_host()`), NOT by arithmetic. The RF megakernel exists precisely to cut launch count.
- **Build-vs-per-op split** — a slow ONE-TIME build amortizes for a persistent agent (acceptable); a slow
  PER-OP cost recurs every turn (the real-time wall). The whole audit hinges on which bucket each cost lands in.
- **H2D** — host→device (CPU→GPU) memory transfer.

---

## 1. The existing optimization machinery — what's already fast, and how

The project already has a complete, well-engineered optimization toolkit. These are the patterns to reapply, not
reinvent.

### 1a. Fused element-wise kernels (`sim/kernels.py`, `@fuse()`)
Every neuron-dynamics + synaptic-conductance + plasticity op is a `@fuse()` kernel (`cp.fuse()` on CuPy, no-op on
NumPy): `fused_izhikevich2007_dynamics_update`, `fused_hodgkin_huxley_dynamics_update`, `fused_adex_dynamics_update`,
`fused_conductance_decay_and_current` (`kernels.py:208`), `fused_nmda_update_and_current`, `fused_stp_decay_recovery`,
`fused_stdp_weight_update`, `fused_homeostasis_update`, `fused_eligibility_trace_decay`, the two dendritic-plateau
kernels. **Pattern:** collapse a chain of element-wise array ops into one launch. Fully covers the Izhikevich/HH/AdEx
per-neuron step.

### 1b. The RF resonate MEGAKERNEL (`cfg.enable_rf_cudagraph`, `sim/bridge.py:5740-5813`)
The headline existing optimization and the diagnostic's confirmed success. `rf_resonate_steps` (`bridge.py:5707`)
**deliberately bypasses the whole `_run_one_simulation_step` machinery** (conductance / plasticity / recording /
engram / gate-couplings / stats — none of which the FHRR substrate uses; docstring `bridge.py:5708-5711`), looping a
tiny O(nnz) complex CSR matvec over a ~2D-neuron op-bridge. With `enable_rf_cudagraph` ON (default ON for OneBrain,
`one_brain_composer.py:177`), the WHOLE step is ONE `rf_megastep` `RawKernel` launch (`_RF_MEGASTEP_SRC`,
`bridge.py:5740-5769`): one thread per neuron does the complex CSR matvec + rotate/decay + Im zero-crossing + double-
buffered writeback. **"A5 lever 3" masking** (`bridge.py:5763`, `use_mask`): the kernel writes back only the masked
(RF) neurons so a co-resident Izhikevich slice's v/u is untouched — `use_mask==0` short-circuits to a byte-identical
no-mask path. This is why the resonate is ~14 ms and not the bottleneck.

### 1c. The sparse CSR synaptic propagation, already optimized (`sim/bridge.py:6017-6133`)
The main-step synaptic matvec is a **sparse CSR transpose-SpMV** `effective_connections_matrix.T @ fired`
(`bridge.py:6068-6071`), O(nnz). Optimizations already present and verified:
- The E/I split is a **single batched 2-column matmul** (`cp.stack([exc, inh], axis=1)` then one `@`, `bridge.py:6057`),
  reusing one CSR index traversal instead of two separate SpMVs.
- The inhibitory mask is **cached** (`self._cached_inhibitory_mask`, `bridge.py:6044-6051`) — traits don't change.
- `_prev_any = bool(self.cp_prev_firing_states.any())` is **cached once per step** (`bridge.py:5831`) and gates the
  whole propagation/STP/Hebbian block, so a silent step skips the matvec entirely.
- The per-type STP parameter arrays are **cached** (`_cached_stp_per_type`, `bridge.py:5856`), and the COO view is
  cached (`_get_cached_coo`, `bridge.py:5882`, "avoids 40-400ms tocoo() per step").

### 1d. Backend abstraction (`sim/backend.py`)
`get_backend()` / `fuse()` / `synchronize()` / `to_host()` / `get_sparse_module()` — one code path runs on CuPy
(GPU) or NumPy (CPU). `fuse()` is `cp.fuse()` on CuPy, a no-op on NumPy. The optimization patterns are backend-portable.

### 1e. The vectorized gate-coupling path (`enable_vectorized_gate_couplings`, `bridge.py:3253-3275`)
A segment-sum (`cp.add.reduceat`) collapses N per-coupling `cp_firing_states[idx].mean()` GPU reductions (each a
device→host sync) into ONE kernel + one transfer. Byte-identical for boolean firing (exact integer sum / count).
Built for the K-way sequencer (`_phaseB_onebrain_sequencerK_derisk.py:96` opts in). **Caveat — incomplete:** see §2.

### 1f. Composer-layer CSR cache (`OneBrainComposer.enable_csr_cache`, `one_brain_composer.py:178-188`, default ON)
Caches the query-INVARIANT unbind + cleanup complex-weight CSRs (keyed by `n_facts` + the fixed block layout) and the
store CSR (keyed by a `_store_dirty` flag), so the batched read reuses device matrices instead of rebuilding
~100k-240k tuples + two fresh `csr_matrix` constructions + H2D EVERY query (the measured "~72%-of-a-query cost").
Answer-identical. **This is the one place `rf_set_complex_weights`'s rebuild cost is already amortized — and ONLY on
the OneBrain batched path** (§2 / §3).

**Pattern summary — the toolbox to reapply:** (1) `@fuse()` element-wise kernels; (2) one-thread-per-neuron
`RawKernel` megakernel that bypasses the full step machinery for a special substrate; (3) batched multi-column SpMV;
(4) cache-and-reuse of query-invariant CSRs + masks + parameter arrays; (5) segment-sum vectorization of per-element
host loops; (6) `_prev_any`-style "skip if silent" gating.

---

## 2. Hot-path × {optimized? how? / unoptimized — why} table

Production scale: D=128 (merged-agent default), D=2048 (documented 320-concept composer scale), V=320, K up to 32.
"per-op" = recurs every conversational turn; "one-time" = paid once per persistent agent.

| Hot path | File:line | Bucket | Optimized? | Cost shape |
|---|---|---|---|---|
| RF resonate loop | `bridge.py:5707-5813` | per-op | **YES — megakernel** (1c) | ~208 steps × 1 `rf_megastep` launch over ~256 neurons. **~14 ms. NOT the bottleneck.** |
| Main-step synaptic SpMV | `bridge.py:6017-6133` | per-op (nav/parser) | **YES — batched sparse SpMV + cached masks + `_prev_any` gate** (1c) | O(nnz) CSR transpose-SpMV. Dominant cost of a LARGE Izhikevich bridge, but already optimal in form. |
| Fused neuron/synapse kernels | `kernels.py` | per-op | **YES — `@fuse()`** (1a) | one launch per element-wise op chain. |
| `OneBrainComposer` batched read | `one_brain_composer.py:497-558` | per-op | **YES — CSR cache + batched 3-window read** (1f, A5 lever 1) | reads ALL K blocks in 3 resonate windows; query-invariant CSRs cached. Answer-identical, 7.3×. |
| **integrated_loop K-way SEQUENCER (per query)** | `_phaseB_onebrain_sequencerK_derisk.py` + `one_brain_composer.py:585-632` | **per-op** | **NO — naive: a fresh ~244K–836K-neuron Izhikevich bridge driven 80 full steps/query** | **THE bottleneck.** §3. |
| Sequencer per-step gate loop | `bridge.py:3242-3251` | per-op (inside sequencer) | **PARTIAL — reduceat removes the syncs, but the EMA/gate-write loop stays Python per-coupling** (1e) | ~20,512 Python iterations × 80 steps/query at K=32, V=320. §3. |
| `block_cleanup_scores` (sequencer drive seed) | `_phaseB_onebrain_sequencer_derisk.py:64-93` | one-time per battery (per K-grow) | **NO — rebuilds a ~123K-tuple cleanup CSR + ~417 resonate steps PER BLOCK, ×K; bypasses the CSR cache** | ~32 × (417 resonate + 123K tuples) ≈ 13.3K resonate steps + ~3.9M tuple constructions at K=32. |
| `OneBrainComposer.__init__` build | `one_brain_composer.py:103-238` | one-time | acceptable bucket, but heavy: 54K-neuron bridge build + 25.2K-step parser train | **the ~41 s the diagnostic flagged.** §4. |
| Merged bridge build | `nav_conv_merged_bridge.py:447-1049` | one-time | acceptable bucket; **two full CSR injections** + 25.2K-step parser pass + **O(V²·ps²) dlPFC edge build** | one-time but a V² scaling cliff if vocab grows. §4. |
| **`elaborate` (dialogue planning)** | `nav_conv_merged_bridge.py:1565-1600` + `content_selection_spiking.py:389-414` | **per-op** | **NO — 60 step-loop iters, each with a per-concept host-sync reduction (`for c in vocab: to_host(fs[...]).sum()`)** | **~60·V ≈ 19,200 GPU→host syncs per call ≈ the ~10 s.** §3. Control object IS cached; the inner read is not batched. |
| `_assoc_graph` Hebbian dict | `nav_conv_merged_bridge.py:1551-1563` | per-op (cheap) | recomputed every `elaborate`, but O(K·9) pure Python; gates the cached Control | NOT a hot spot. |
| Fact STORE — OneBrain | `one_brain_composer.py:304-349` | per-op | reuses ONE persistent bridge + lazy store CSR (good); residual = 2 bind/bundle CSR rebuilds + 416 resonate steps/fact | the ~0.6 s/fact on the onebrain path. §3. |
| Fact STORE — RFComposer substrate | `rf_phasor_composer.py:521-534` | per-op | **NO — `_build_rf_bridge` (full `_initialize_simulation_data`) PER FACT** | a whole bridge construction per stored fact (only when `enable_substrate_store=True`). |
| `rf_set_complex_weights` | `bridge.py:5649-5666` | primitive | **NO internal cache — 4× `np.fromiter` + 4 H2D + 2× `csr_matrix(N,N)`, FRESH every call** | 40K–240K tuples + dual CSR build + H2D per call at V=320/D=2048. Amortized ONLY by OneBrain's `enable_csr_cache`; the `rf` path (`_resonate:164`) rebuilds fresh every op. |

---

## 3. The bottleneck, dissected — the `integrated_loop` sequencer (per query)

This is the single highest-leverage target. The production demo `consolidated_320_conversation_demo` **defaults
`integrated_loop=True`** (`consolidated_320_conversation_demo.py:242`), so the 320-concept production conversation
DOES build and run the K-way sequencer per query. Three compounding costs:

**(a) A fresh, enormous Izhikevich control bridge, driven 80 full steps per query.** `_ensure_sequencer`
(`one_brain_composer.py:585`) builds `build_sequencerK_bridge(seed, V=self.V, K=K)` — at the composer's FULL vocab
`self.V`. The region math (`_phaseB_onebrain_sequencerK_derisk.py:98-156`): `2·V` cue word-lines + `K·2·V` decoded
word-lines + `K·2·V` gated-match lines, each `n_word=20`, plus `(4K+1)+K` pools of `n_pool=30`. Computed scaling:

| V | K | sequencer neurons | gate couplings (K·V·2+K) |
|---|---|---|---|
| 22 (sequencer's own test VOCAB) | 32 | 62,030 | 1,440 |
| 72 | 32 | 192,030 | 4,640 |
| **320 (production composer vocab)** | **32** | **836,830** | **20,512** |

The diagnostic's flagged **244,580** sits between V=72 and V=92 at K=32 — i.e. the sequencer is built at the
composer's vocab size, and at the true production V=320 it is an ~837K-neuron bridge. Each query runs
`run_sequencerK_with_drive` → `reset_sequencerK_state` (20 drain steps) + `settle=60` steps = **80 FULL
`_run_one_simulation_step` calls on that ~244K–837K-neuron bridge** (`_phaseB_onebrain_sequencerK_derisk.py:179,215`).
Unlike the RF resonate, this runs the entire step machinery (conductance kernel + the CSR SpMV over all those
neurons + STP/plasticity/engram/stats). The matvec FORM is fine (sparse); the cost is the sheer neuron count ×
full-step machinery × 80 steps × every query.

**(b) The per-coupling Python gate loop survives the vectorization.** Even with `enable_vectorized_gate_couplings`
ON (the sequencer opts in), the `reduceat` only collapses the GPU reductions; the EMA + gate-write loop
(`bridge.py:3242-3251`) still iterates ALL couplings in Python every step. At K=32, V=320: **20,512 couplings × 80
steps/query ≈ 1.64M Python iterations per query.**

**(c) The drive seed (`block_cleanup_scores`, once per K-grow) is a large uncached precompute.** Seeding the
sequencer's per-block decoded-line drives runs the FULL composer reconstruct→unbind→cleanup PER BLOCK: ~417 resonate
steps + a **~123K-tuple cleanup-codebook CSR rebuilt from scratch** (`_phaseB_onebrain_sequencer_derisk.py:84-88`),
×K = ~13.3K resonate steps + ~3.9M tuple constructions at K=32. This is once-per-battery, but it bypasses
`OneBrainComposer._build_complex_csr`'s cache (it calls `bridge.rf_set_complex_weights` directly).

**The `elaborate` path (the ~10 s) has the SAME failure mode as (b):** a 60-step spiking loop whose inner read is an
un-batched per-concept host sync — `for c in self._vocab: to_host(fs[...]).sum()` inside each of 60 steps
(`content_selection_spiking.py:410-413`) = ~60·V ≈ 19,200 device→host syncs per `elaborate`. Launch/sync-bound, not
compute-bound. The Control + graph are cached (good); only the read is naive.

---

## 4. The build costs (one-time bucket — amortizes for a persistent agent)

The ~41 s build the diagnostic flagged is **`OneBrainComposer.__init__`** (NOT `RFPhasorComposer.__init__`, which is
pure-numpy codebook generation, microseconds — `rf_phasor_composer.py:135`). It splits as:
- the **54K-neuron co-resident bridge** build (`build_coresident_bridge`, `one_brain_composer.py:232`; `n_total`
  scales with `k_max·V`) + CUDA context warmup;
- the **parser train**: `BridgeParser._train(n_epochs=30, train_steps=120)` = 30 × 6 conjunctions × (20 reset + 120
  train) ≈ **25,200 full `_run_one_simulation_step` calls** on the full bridge (`brain_conversational_agent.py:110-121`),
  most of it first-call CUDA JIT/warmup.

The **merged bridge build** (`build_merged_nav_conv_bridge`) similarly is one-time but pays **two full CSR injections**
(the framework auto-inject at init, then a second `inject_explicit_wiring` to include the hand-built `dlpfc_loop`,
`nav_conv_merged_bridge.py:992`) + the 25.2K-step parser pass + an **O(V²·ps²) eager dlPFC edge build**
(`_build_dlpfc_loop_population`, `nav_conv_merged_bridge.py:188-229`): at V=320, ps=50 that is 320²·2500 ≈ 256M edges
— cheap at the V=16 default, a scaling cliff at V=320.

These are acceptable as one-time costs **for a persistent agent**, but they dominate any short/benchmark run and gate
iteration speed. Lower priority than the per-op sequencer/elaborate, but flagged.

---

## 5. Ranked optimization opportunities

Cross-referenced to the diagnostic where numbers exist. "Leverage" = expected wall-clock win on a production turn;
"effort" is rough.

| # | Path | Current cost-shape | Technique | Leverage | Effort |
|---|---|---|---|---|---|
| **1** | **`elaborate` inner read** (`content_selection_spiking.py:410-413`) | 60 steps × V per-concept `to_host().sum()` = ~19.2K syncs/call (~10 s) | **Batch the read:** one device segment-reduction over all V assemblies per step (or accumulate on-device, sync once after the loop). Reapply pattern 1e/1c. | **HIGH** — collapses ~19.2K syncs → ~60 (or 1). Targets the whole ~10 s. | **LOW** — localized; no `sim/` edit; the spreading dynamics are unchanged. |
| **2** | **integrated_loop sequencer per-query (a)** | 80 full steps on a 244K–837K-neuron bridge/query | **Megakernel/fast-loop the sequencer step** (the §1b pattern applied to the Izhikevich control fabric) OR shrink the fabric (build at a reduced CUE vocab, not full V; the sequencer only needs the words that actually appear) OR cap K/V scaling. | **HIGH** — the flagged primary; per-query and scales as V·K. | **MED–HIGH** — fast-loop is the §1b template (matvec + Izhikevich step); the vocab-shrink is the cheapest sub-lever. |
| **3** | **Sequencer gate loop (b)** (`bridge.py:3242-3251`) | ~20,512 Python iters × 80 steps/query | **Vectorize the EMA + gate-write** alongside the existing `reduceat`: compute the EMA update + threshold compare as array ops; write only changed gates via a masked batch. Completes 1e. | **MED** — removes ~1.64M Python iters/query; compounds with #2. | **LOW–MED** — extends the existing vectorized path in the same method. |
| **4** | **`block_cleanup_scores` drive seed** (`_phaseB_onebrain_sequencer_derisk.py:84-88`) | ~123K-tuple CSR rebuild × K (~3.9M tuples) per K-grow | **Route through `OneBrainComposer._build_complex_csr` cache** (the cleanup codebook is query-invariant). Reapply 1f. | **MED** — once-per-battery, but large; helps every K change / restart. | **LOW** — the cache already exists; point this call site at it. |
| **5** | **`rf_set_complex_weights` on the `rf` path** (`bridge.py:5649` / `rf_phasor_composer.py:164`) | 4× `np.fromiter` + 4 H2D + 2× `csr_matrix(N,N)` FRESH every op | **Cache query-invariant CSRs** the way OneBrain does (or memoize by a content hash). The `rf` path rebuilds every op; OneBrain already proves the win. | **MED** — the `rf` test-oracle/CPU path; lower priority since onebrain is the production default. | **MED** — mirror `enable_csr_cache` onto the rf path. |
| **6** | **STORE — OneBrain** (`one_brain_composer.py:304-349`) | 2 bind/bundle CSR rebuilds + 416 resonate steps/fact (~0.6 s) | Pre-size + reuse the bind/bundle CSR buffers; the resonate is already the megakernel. | **LOW–MED** — per-fact, but stores are bursty (ingest), not per-turn. | **MED**. |
| **7** | **Build double-injection + dlPFC V² edges** (`nav_conv_merged_bridge.py:992,188`) | 2 full CSR injects + O(V²·ps²) eager edges | Single injection where possible; lazy/sparse dlPFC edges. | **LOW** (one-time) — but removes a V² cliff at vocab scale-up. | **MED–HIGH**. |

---

## 6. Recommended first target (highest-leverage, cheapest-first)

**Opportunity #1 — batch the `elaborate` inner per-concept read** (`content_selection_spiking.py:389-414`,
specifically the `for c in self._vocab: to_host(fs[...]).sum()` at lines 410-413).

**Why first:**
- **Cheapest-first + highest ROI.** It is a localized, low-risk change (a host-loop → one on-device segment
  reduction) that targets the entire ~10 s `elaborate` cost the diagnostic measured. No `sim/` edit; the spreading
  dynamics and the selection result are unchanged (it's purely HOW the firing is read out — answer-identical by
  construction, the §1e segment-sum is exact).
- **It is the cleanest instance of the dominant anti-pattern**, which recurs in the bottleneck: an un-batched
  per-element host sync inside a spiking settle loop. Fixing it here both wins the ~10 s AND validates the exact
  technique (device-side reduction, sync-once) that opportunity #3 then applies to the sequencer gate loop — so it's
  a de-risking beachhead for the bigger #2/#3 sequencer work.
- The bigger prize (#2, the sequencer) is real but higher-effort and needs a design choice (fast-loop the fabric vs
  shrink its vocab). Land #1 first for an immediate, safe, large win; then take #2/#3 with the read-batching pattern
  already proven.

**Technique:** replace the V separate `to_host(cp_firing_states[idx]).sum()` calls with one segment-sum over a
cached concat of the per-concept assembly indices (mirror `_gate_coupling_rates_vectorized`, `bridge.py:3253-3275`),
synchronizing to host ONCE per step (or accumulating on-device across the 60 steps and syncing once at the end).
Expected: ~19,200 syncs/call → ~60 (or 1). **Verify** answer-identical relevance ranking on the `elaborate` GO test
before/after.

---

## Provenance

Verified directly (Read/Grep, this audit): `sim/kernels.py` (full); `sim/bridge.py` `rf_set_complex_weights`
(5649-5666), `_rf_advance_one` / `rf_resonate_steps` / `_rf_resonate_steps_megakernel` (5668-5813),
`_apply_gate_couplings` + `_gate_coupling_rates_vectorized` (3224-3275), the synaptic propagation matvec
(6017-6133); `research/runners/one_brain_composer.py` (full); `research/runners/_phaseB_onebrain_sequencerK_derisk.py`
(full, + the neuron-count scaling computed from its region math); `research/runners/rf_phasor_composer.py` `_resonate`
(156-167); `research/runners/consolidated_320_conversation_demo.py` (the `integrated_loop=True` default, line 242).
The merged-build / `elaborate` / `_assoc_graph` / store / `RFPhasorComposer.__init__` / gate-coupling-scaling /
`block_cleanup_scores` / divnorm-sequencer cost shapes were traced by two parallel read-only subagents with file:line
evidence and independently cross-checked against the directly-verified primitives above. Empirical numbers (~14 ms
resonate, ~41 s build, ~0.6 s/fact, ~10 s elaborate, the ~244,580-neuron sequencer flag) are from the in-flight
latency diagnostic (`e1e8159e`, `c3875a9d`).
