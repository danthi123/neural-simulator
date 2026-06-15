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

The capstone — streaming the actual corpus window-by-window into these populations on the bridge
(`_phaseB_onbridge_stream_cortex_derisk.py`) — is the end-to-end realization of the CYCLE-94 numpy
milestone on the substrate.

Runner: `research/runners/_phaseB_stdp_cooccurrence_derisk.py` (GPU; CuPy; `--n-per` for the
population). Raw: `research/findings/raw/_phaseB_stdp_cooccurrence.json`,
`_phaseB_hebbian_cooc_6seed.log`, `_phaseB_hebbian_pop_sweep.log`.
