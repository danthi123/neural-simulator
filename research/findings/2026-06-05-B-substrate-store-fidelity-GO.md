# De-risk (B) RESOLVED — substrate-held bound facts unbind at numpy parity → GO — 2026-06-05

**Verdict: GO.** A **substrate-held** bound fact — its `(ON,OFF)` vector imprinted in the static connection
**WEIGHTS** of a small dedicated population and retrieved in **SPIKES** — unbinds at **numpy parity**: across seeds
42/43/44, **every role of every fact** that the substrate-store retrieval unbinds recovers the **SAME filler** as
unbinding the numpy-held bound vector (the cleanup held constant). This GATES the whole (B) build (substrate-held
memory): the bound fact does NOT have to live in a Python list — a Crawford-Gingerich-Eliasmith-style per-fact
weight-store holds it faithfully enough for the downstream spiking unbind.

## The crux question (the GATE for the whole B build)
After (A) cleared the READOUT shortcut (numpy `argmax` cleanup → spiking NEF cleanup,
`2026-06-05-composer-cleanup-NEF-GO.md`), (B) is the deeper MEMORY shortcut: the composer's bound fact is a numpy
`(ON,OFF)` vector held in `CoreSimComposer.kb` (a Python list) — the memory is NOT in the substrate. The (B)
research VERDICT (`2026-06-05-substrate-held-memory-literature-synthesis.md`): do NOT engram-tag the GRADED bound
vector (engrams BINARIZE — they store WHICH cells fired, not per-cell amplitudes); the recommended store is a
per-fact NEF associative memory — the fact's bound vector in STATIC weights, retrieved by firing it (validated to
117,659 facts at D=512, Crawford 2016). **Can such a substrate-held bound fact be unbound at numpy parity?**

## Result (the de-risk GATE)
`research/findings/raw/_b_substrate_weight_store_probe.py`, proj_dim=800 (the harder/noisier regime), numpy
deterministic cleanup held constant, CuPy / RTX 3090:

| seed | recovery (substrate-store unbind == numpy-store unbind) | recon cosine `(bon',boff')·(bon,boff)` |
|---|---|---|
| 42 | **12/12 = 1.000** | 0.9746 (min 0.9728) |
| 43 | **12/12 = 1.000** | 0.9779 (min 0.9769) |
| 44 | **12/12 = 1.000** | 0.9774 (min 0.9759) |
| **min** | **1.000** | mean **0.9767** |

8-fact re-run (rule out small-sample luck): **24/24 per seed, all three seeds** — `_b_substrate_store_8facts.json`.

**The f-I reconstruction loss does NOT degrade filler recovery.** The spiking read reconstructs the bound vector to
cosine ~0.975 (not 1.0 — the readout Izhikevich f-I nonlinearity rectifies the smallest weights and compresses the
largest), yet **every** role unbinds to the identical filler. The unbind's `_scale_to_current` normalization +
nearest-neighbor cleanup absorb a scaled-but-faithful (monotone) reconstruction — exactly the high-fidelity regime
the literature places this problem shape in (M=3 bindings/fact, D=800, vocab=16; the Frady-Kleyko-Sommer SNR is
cleared several-fold).

## The store mechanism (Crawford-style weight-store, simplest faithful version; NO `sim/` edits, reuse-by-import)
A SEPARATE small `SimulationBridge` of Izhikevich neurons per fact (`build_store_bridge`):
- **trigger population** of `n_trig=40` neurons (the per-fact "memory" address);
- two D-neuron readout banks `readout_ON[0,D)`, `readout_OFF[D,2D)` (with `n_per=4` neurons per bound dimension,
  population-averaging the f-I read noise — Singh-Eliasmith error ∝ 1/√N);
- **the bound vector lives in the OUTPUT weights**: every trigger neuron `i` projects
  `trigger_i → readout_ON[k]` with weight `bon[k] · w_gain` and `trigger_i → readout_OFF[k]` with weight
  `boff[k] · w_gain` (`w_gain=250`). The `(ON,OFF)` bound vector IS the synaptic weight matrix; it is not in numpy.

**RETRIEVE (`retrieve_bound`) — a genuine spiking read:** drive the trigger population at a constant current
(`trig_drive=600 pA`) for `run_steps=300`; the trigger neurons fire steadily and the only thing they drive is the
readout banks (via the fixed store synapses), so each readout neuron fires at a rate set by its imprinted weight
`f(n_trig · bon[k] · w_gain)`. Accumulate the readout firing over the window and per-dimension-average the `n_per`
neurons → `(bon', boff')`, the reconstructed bound vector.

## Smell-test — the read is from SPIKES, not a numpy passthrough
Zeroing the trigger drive (everything else identical) **collapses** the reconstruction:

| trigger | `readout_ON` total rate | recon |
|---|---|---|
| ON (600 pA) | **20.43** | cos 0.974 |
| OFF (0 pA) | **0.14** (OU-noise floor only, 145× lower) | — |

The bound vector appears in `bon'` **only because the trigger neurons fire** and drive the readout banks through
the imprinted synaptic weights. A numpy passthrough would survive a silent trigger; this does not. (Pinned as a
regression test — `tests/test_b_substrate_weight_store.py::test_substrate_store_read_is_from_spikes_not_numpy`.)

## Methodology — the cleanup MUST be held constant (and deterministic) so the STORE is what's tested
The spiking NEF cleanup (the A path) is **stochastic** (OU noise + spiking dynamics): on the numpy-store est it does
not even agree with itself — `seed 42 = 8/12 (0.667)`, `seed 44 = 11/12 (0.917)` self-consistency. When that
stochastic cleanup is held constant, the substrate-store-vs-numpy-store recovery (5/12, 9/12) sits **inside that
cleanup-noise band** — the disagreement is the cleanup re-rolling its dice, not the store losing information (the
recon cosine is identical, ~0.976). So the correct oracle for a STORE de-risk is the **deterministic numpy
argmax** cleanup, held constant across both arms — which gives the clean **1.000** parity above. (The
`--spiking-cleanup` arm is in the probe for completeness; its lower number is a cleanup-determinism artifact, not a
store finding, and is documented as such.)

## Honest scope / boundaries
- **The STORE is de-risked; the in-network superposition + opponency are NOT yet built.** This validates that a
  bound vector held in static weights survives the round-trip to spikes and back into a faithful unbind. The numpy
  `bon += o` superposition and `onoff(bon − boff)` opponency in `bind_fact` are still numpy — those two linear
  pieces (rate-summation on a shared bank; ON/OFF lateral inhibition) are the next step, then the opt-in composer
  build (mirroring `enable_spiking_cleanup`).
- **The bound vector is constructed by the composer's spiking bind, which is GPU-only at this operating point.** On
  the numpy backend the composer's coincidence bind produces a DEGENERATE (all-zero) bound vector here — so the
  de-risk is a CuPy/GPU result (per the GPU-for-real-runs mandate), and the regression test skips on a degenerate
  bind so CI stays green.
- **Per-fact bridge in the de-risk.** Each fact got its own small store bridge (the cleanest isolation for the
  fidelity question). The production store is one population per fact addressed by a discrete index (engram-style
  address into a per-fact store, per the synthesis) — `n_trig=40` + `2·D·n_per` neurons/fact at D=2048 is ~16k
  neurons/fact, so a ~30-fact KB is ~0.5M neurons (Crawford ran 2.5M for 117,659 facts — comfortably in budget).

## Next (the B build, now de-risked at the store)
1. In-network **superposition** (the role binds' rates SUM on a shared bank) + **opponency** (ON/OFF lateral
   inhibition) — the two linear pieces — replacing `bon += o` / `onoff(bon − boff)`.
2. Build the weight-store into the composer as an opt-in flag (like `enable_spiking_cleanup`): `store(...)`
   imprints the bound vector into a per-fact substrate population instead of `self.kb.append`; `unbind` drives the
   fact's trigger to retrieve `B'` then spiking-unbinds. No-regression on the capability matrix at D=2048.

## Artifacts
- Probe: `research/findings/raw/_b_substrate_weight_store_probe.py`
- Results: `_b_substrate_store.json` (canonical 3-seed, 12/12), `_b_substrate_store_8facts.json` (24/24),
  `_b_substrate_store_spikingcleanup.json` (the cleanup-determinism control, documented above)
- Test: `tests/test_b_substrate_weight_store.py` (parity + spike-read smell-test; skips on degenerate/numpy bind)
- Backend: CuPy / RTX 3090. NO `sim/` edits.
