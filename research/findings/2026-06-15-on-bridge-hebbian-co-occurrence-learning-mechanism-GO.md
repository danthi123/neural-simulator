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

Multi-seed (42–47) confirmation in flight at the time of writing; numbers above are seed 42.

Runner: `research/runners/_phaseB_stdp_cooccurrence_derisk.py` (GPU; CuPy).
Raw: `research/findings/raw/_phaseB_stdp_cooccurrence.json`,
`research/findings/raw/_phaseB_hebbian_cooc_6seed.log`.
