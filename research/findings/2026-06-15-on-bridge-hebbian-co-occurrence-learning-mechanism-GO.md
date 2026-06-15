# On-bridge co-occurrence learning: RATE-HEBBIAN is the matched rule (STDP is measured-negative) — mechanism GO (2026-06-15, CYCLE 95)

## One-line

The last genuinely-new on-bridge piece of the biology-faithful stream cortex — *does spiking
plasticity, fed co-active word scenes, learn the co-occurrence matrix M that the numpy online
cortex builds?* — is **GO on the mechanism**: on-bridge **rate-based Hebbian** plasticity learns the
co-occurrence (correlation of the learned weights with the true counts, `corr(M,C)`, climbs to
**+0.705** at seed 42, monotonic in the co-activation budget, permuted-control clean). The wrong
rule — **STDP** — is **measured-negative** (656,206 weight-update events, **0** weight change),
because symmetric co-occurrence has no consistent pre→post order and lands at the STDP kernel's
`delta_t ≈ 0` (exactly zero update). The single-neuron-per-concept *read-out* plateaus at ~50% of
the host reference, which is the project's already-documented single-neuron rate-code SNR wall; the
validated **population code** (CYCLE 91, ~94%) is the established lift.

## Why this matters / where it sits

CYCLE 94 shipped the **biology-faithful online stream cortex** in numpy (a cortex that hears the
stream word-by-word — online Hebbian co-occurrence + a running word-frequency estimate +
log-double-centring — reaching **+0.513**, matching batch-PPMI **+0.502**, no whole-corpus
preprocessing). The on-bridge realization of that cortex composes three pieces, each validated on the
real `SimulationBridge` spiking substrate:

| Piece | On-bridge status | Source |
|---|---|---|
| The **representation** (the graded code carried by spikes) | population code reaches ~94% of host | CYCLE 91 |
| The **normalization** (log + double-centring circuit) | +0.285, scales with population | CYCLE 93b |
| The **learning** (build M from co-active scenes) | **THIS finding — mechanism GO** | CYCLE 95 |

This was the one remaining unbuilt piece. It is now de-risked.

## The decisive measurements (seed 42)

Setup: a two-region bridge — `hub` (300 context-word neurons) → `target` (64 concept neurons), a
fully-connected plastic pathway. Each concept's "scene" co-activates the target plus its
co-occurring context hubs (hub drive **graded** by the true co-occurrence row `C[t]`, so a
stronger-co-occurring hub fires more → more plasticity → a higher learned weight). After training,
the learned `hub→target` weight block is read directly from `cp_connections` as `M`, and compared to
the true co-occurrence `C` and to the host reference `double_center(log1p(C·100))`.

**STDP (the wrong rule) — measured-negative.** With the spike-timing rule enabled, the firing
diagnostics confirmed neurons fire (hubs and target both spike), the clock advances, and the
plasticity block runs **656,206 times** — yet **`n changed 0 / 19200`** synapses, `corr(M,C) = −0.006`.
Root cause, read straight from the kernel: the bridge STDP update is
`delta_t = t_post − t_pre`, with `delta_t > 0 → LTP`, `< 0 → LTD`, and **exactly `0 → zero update`**.
Co-occurrence is symmetric — co-driven hub and target fire on the *same* steps — so every event lands
at `delta_t ≈ 0` and the kernel returns zero. STDP encodes *causal sequence*, not *symmetric
correlation*; it is the wrong tool for a co-occurrence count.

**Rate-Hebbian (the matched rule) — GO.** Switching to the bridge's Hebbian rule (potentiate a
synapse whose pre fired at `t−1` AND post fired at `t` — a built-in one-step coincidence detector;
soft-bound `Δw = rate·(w_max − w)`) makes the co-occurrence appear and grow monotonically with the
co-activation budget:

| Co-activation budget | plasticity events | `corr(M,C)` | single-neuron normalized code | permuted |
|---|---|---|---|---|
| 3 epochs × 12 steps | 2,233 | **+0.386** | +0.036 (11% of ref) | +0.016 |
| 8 epochs × 20 steps | 14,394 | **+0.620** | +0.153 (45% of ref) | −0.000 |
| 20 epochs × 40 steps | 64,828 | **+0.705** | +0.175 (51% of ref) | −0.008 |

Host reference (the ceiling for this read-out): `+0.344`. The Hebbian rule genuinely learns the
co-occurrence — `corr(M,C)` rises with more co-activation, and the permuted-label control stays at
~0 (the structure is **learned**, not wired in by the small uniform initialization, which
double-centring removes).

## The honest residual: single-neuron read-out is wall-bounded (not a mechanism failure)

`corr(M,C)` climbs to 0.705 while the *normalized code* plateaus near 0.17 (~50% of the host
reference). The gap is **not** a learning failure — it is the project's already-documented
**single-neuron rate-code SNR wall**: here each concept is **one** neuron, so `M` is a single noisy
spiking estimate of each count, and the per-synapse noise survives the log-double-centring and
dilutes the cosine structure. The fix is the established **population code** (CYCLE 91: ~32
neurons/dimension recovers ~94% of host) — exactly the documented lift for this wall. Demonstrating
the population lift *inside this runner* is the immediate next compose step.

## Verdict

- **Mechanism GO:** on-bridge spiking **rate-Hebbian** plasticity learns the online co-occurrence
  matrix the numpy stream cortex builds (`corr(M,C)` +0.705 seed 42, monotonic, permuted-clean).
- **Clean mechanistic by-product:** STDP is the wrong rule for symmetric co-occurrence (measured: 0
  weight change at `delta_t ≈ 0`). Rule must match computation — STDP = causal sequence, Hebbian =
  symmetric correlation.
- **Honest residual:** single-neuron read-out is bounded by the documented rate-code wall; the
  validated population code is the lift.
- **Compose:** Hebbian co-occurrence learning (this) + population code (CYCLE 91) + log-domain
  normalization (CYCLE 93b) = the full on-bridge biology-faithful stream cortex.

## Multi-seed mechanism confirmation (seeds 42–47) — GO

The mechanism holds robustly across all 6 seeds (15 epochs × 30 steps): **mean corr(M,C) +0.686**
(range [+0.658, +0.718]), **permuted-clean (mean −0.009)**. The single-neuron normalized code stays
in the wall-bounded ~0.19–0.26 band (mean +0.207, 60% of host-ref), as expected — the population lift
below is what carries it to host.

## The population lift — CONFIRMED: the composition reaches host fidelity (added same cycle)

Composing the two validated on-bridge pieces — **Hebbian co-occurrence learning + the population
code** (CYCLE 91) — on one bridge: each concept gets `n_per` neurons; drive the whole
concept-population, average the `n_per × n_per` learned-weight sub-block per concept-pair → `M_pop`.
Population averaging cancels the per-synapse spiking noise that bounds the single-neuron read-out.
The lift is decisive and saturates at host fidelity by ~8 neurons/concept (seed 42, 12 epochs × 25
steps):

| neurons / concept | `corr(M,C)` | normalized code | % of host-ref (+0.344) |
|---|---|---|---|
| 1 (single) | +0.674 | +0.162 | 47% |
| 8 | +0.900 | +0.345 | **100%** |
| 16 | +0.932 | +0.354 | **103%** |
| 32 | +0.947 | +0.371 | **108%** |

So the single-neuron ~50% plateau was indeed the documented rate-code SNR wall, and the population
code lifts the **Hebbian-learned** cortex to the host reference — now demonstrated in the *learning*
setting (CYCLE 91 showed the population lift only for a *fixed* host-PPMI drive; this shows it for a
cortex that **learned** its codes by Hebbian co-occurrence). permuted stays ~0 throughout.

## What this completes

All three pieces of the on-bridge biology-faithful stream cortex are now validated on the real
spiking substrate, and the two that interact (learning + representation) are shown to **compose to
host fidelity**:

1. **Learning** — rate-Hebbian co-occurrence (corr(M,C) +0.674 single / +0.93 population).
2. **Representation** — population code (lifts the read-out to 100–103% of host).
3. **Normalization** — log-domain double-centring circuit (+0.285, CYCLE 93b).

## The capstone — the fully-faithful stream cortex ON the bridge (GO)

`_phaseB_onbridge_stream_cortex_derisk.py` streams the **actual** TinyStories corpus window-by-window
into the population bridge — each window just co-activates the populations of the words that co-occur
in it (**no precomputed counts in the drive**) — and the bridge's own population Hebbian synapses
accumulate the co-occurrence `M`; read via population block-mean + log-double-centring. Seed 42, n_per
16, 30000 windows (521 s):

| metric | value |
|---|---|
| `corr(M,C_stream)` (learning fidelity) | **+0.885** |
| normalized code | +0.170 (**65% of host-ref +0.263**) |
| generalization (held-out) | 0.45 |
| permuted | −0.011 |

This is a **GO** (the script's gate is code ≥ 0.60 × host-ref): the spiking bridge, hearing the
corpus stream, learns the cortex faithfully (`corr(M,C) +0.885`) and the read-out generalizes,
permuted-clean. **Honest nuance:** the *absolute* fidelity (65%) is bounded by the 30000-window budget,
not the substrate — that stream subset's own ceiling (host-ref +0.263) is below the full corpus's
+0.344, and `corr(M,C) 0.885` shows the learning is faithful; more stream (more windows) raises the
ceiling and the code together. The compressed per-target graded presentation (the mechanism de-risk)
reaches full fidelity faster (one scene per target = the whole accumulated co-occurrence), but the
fully-faithful stream is the biology-faithful version (no precomputed co-occurrence in the drive) and
it works.

⇒ the CYCLE-94 numpy milestone (a cortex that learns from the conversation stream, no preprocessing)
is **realized on the real spiking substrate**: representation (population) + learning (Hebbian
co-occurrence) + normalization (log-double-centring), composed end-to-end.

## The conversation on the stream-learned codes — GO, the moat holds

`_phaseB_onbridge_stream_conversation_derisk.py` runs the **exact CYCLE-90 conversational pipeline**
(multi-role HRR SVO binding → who/what recall → the no-confab abstention moat) on the codes the bridge
**learned from the stream** (not curated, not PPMI). Seed 42, n_per 16, 30000 windows (306 s solo):

Multi-seed (42/43/44, n_per 16, 30000 windows each, ~290 s/seed):

| metric | stream-learned codes (3-seed) | CYCLE-90 PPMI baseline |
|---|---|---|
| who-Q&A recall (present) | **1.00** all 3 seeds (0 within-cat err) | 0.88–1.00 |
| no-confab abstain (absent) | **0.96** (1 false-accept, at seed 43) | 1.00 (0) |
| familiarity gap | present +0.437 ≫ absent +0.075 | present +0.42–0.52 ≫ absent +0.03–0.10 |

**GO** — the stream-learned on-bridge cortex carries the full conversational capability **end-to-end**:
perfect who/what recall across all 3 seeds, and the no-confab moat holds strongly (abstain 0.96).
Notably this works even though the stream codes were only ~65% of the host-ref read-out fidelity: HRR
binding + cleanup tolerates moderate code quality (the codes just need to be distinguishable).

**Honest caveat (surfaced, not hidden — the moat is a load-bearing bar):** the PPMI baseline had **0**
false-accepts; the stream codes had **1** (a single tail confabulation on seed 43, the lowest-fidelity
seed — its absent-match +0.093 was closest to the gate 0.25). This is the **code-fidelity cost**, not a
weakening of the moat *mechanism* (the gate threshold and conjunctive cue are unchanged and were not
tuned on the test). The lever to restore the perfect moat is **better codes** (more stream → wider
familiarity gap → fewer tail matches) — never a looser gate.

**Confirmed:** re-running seed 43 at **70000 windows** (vs 30000) restores the moat to **abstain 1.00
(0 false-accepts)** with recall still 1.00 — the absent-match drops +0.093 → +0.065 (wider gap, present
+0.449). So the single 30000-window false-accept was indeed the **code-fidelity cost, not a
moat-mechanism weakness**: more stream widens the familiarity gap and the moat holds perfectly. The
no-confab bar is intact end-to-end on the stream-learned cortex.

⇒ the complete chain closes on the real spiking substrate: **the bridge hears the conversation stream
→ its population Hebbian synapses learn the cortex → the codes bind into facts → who/what recall +
the abstention moat.** No preprocessing, no curated concepts.

## Next-arc cheap-first: 320-concept stream-scaling feasibility

A cheap CPU de-risk of scaling the stream cortex from 64 to the 320-concept production tier: **feasible
on frequency, gated on the semantic taxonomy.** TinyStories has 7948 distinct word types; **633 content
words at freq ≥ 200, 284 of the existing g20 320-vocab (89%) above the proven learnability floor**
(freq ≥ 50; the 64-tier's least-frequent concept fired 48×); the g20 vocab is 98% corpus-present (only
4 never appear: false/narrow/north/south). So frequency is **not** the limiter. The limiter is a
**320-word *semantic* taxonomy** for the `S_true` ground truth: the 8×8 taxonomy's categories are
semantic (animal/food/…), but g20's are grammatical (noun/verb/adj/spatial/functional) — a different
structure. ⇒ scaling needs a curated 40×8 corpus-grounded *semantic* taxonomy (the categorizable
subset of the ~633 frequent content words), then the (long) GPU stream-run.

**320-SCALING VALIDATED (seed 42, GO).** The 40×8 = 320-word semantic taxonomy was curated
(`research/runners/stream_taxonomy_320.py`, all freq ≥ 50, clean semantic categories, independently
verified) and the stream cortex learned all 320 concepts from the corpus on a 9920-neuron /
24.6M-synapse bridge (150000 windows, ~96 min, ~469 windows/concept = the same per-concept budget as
the 64-run). At the production 320-concept tier: who-Q&A recall **1.00** (8/8, 0 within-cat), no-confab
abstain **1.00 (0 false-accepts)**, familiarity gap present +0.468 ≫ absent +0.050 — the full
conversational capability holds **perfectly at 5× the concepts** (cleaner than the reduced-budget
64-run). ⇒ the biology-faithful stream cortex **scales to the production tier on the spiking
substrate**. Multi-seed (the project standard) is a ~5 hr commitment (~96 min/seed); seed 42 is a clean
GO. Codes cached at `_phaseB_stream_codes_320_seed42.npy` (HRR re-tests instant).

Runner: `research/runners/_phaseB_stdp_cooccurrence_derisk.py` (GPU; CuPy; `--n-per` for the
population). Raw: `research/findings/raw/_phaseB_stdp_cooccurrence.json`,
`_phaseB_hebbian_cooc_6seed.log`, `_phaseB_hebbian_pop_sweep.log`.
