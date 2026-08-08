---
type: finding
status: contributing
date: 2026-08-08
mechanism: forward-model-spiking-synaptic-readout-neural-wta
lane: world-model
---

# BIOLOGIZING the world-model READ-OUT (host ridge/argmax → a spiking synaptic read-out + neural WTA): the read-out RESOLUTION WALL, quantified (single-seed SMOKE, honest NO-GO, 2026-08-08)

The wired world-model content decode (`fm_decode` in `_stageA_full_integration_derisk.py`; identical in shape to `_forward_model_reservoir_derisk`'s host read-out) is a DECLARED host shortcut — `argmax(spikecounts @ Ws)`, a ridge least-squares map + a host argmax over the reservoir spike-counts. This arc BUILDS the biologized replacement — a genuine SPIKING SYNAPTIC read-out co-resident with the reservoir on ONE `SimulationBridge`, whose winner is a NEURAL winner-take-all — and MEASURES it against the ridge on the toy 5×5 world. **Verdict: honest NO-GO.** The fully-neural read-out is well-formed and provably neural (reservoir active, seeded/byte-identical, no host ridge/argmax in the content path, matched-sham specificity control valid) but **does NOT match the ridge held-out**: held-out per coordinate = **0.20 / 0.20 = chance** (joint 0.04) while the SAME reservoir feature under the host ridge decodes held-out **0.80** (joint). The reservoir feature is fine; the SPIKING read-out under-resolves the decision margin. This reproduces, for the world-model read-out, the exact read-out-resolution wall `_rungB1c` hit for role selection — and NAMES the surpass.

**Runner:** `research/runners/_fm_spiking_synaptic_readout_derisk.py` (reuse-by-import: the 5×5 world + local delta rule from `_forward_model_reservoir_derisk`; reservoir statistics + wash-out from `_emerge82`/`_emerge61`; the Ws_shifted-synapse + mutual-inhibition-WTA + neural-winner recipe from `_rungB1c`). **numpy; NO `sim/` edit** — a standalone bridge built via public APIs (`inject_explicit_wiring` / `set_pathway_weights`); the shared substrate is untouched. **Artifact:** `research/findings/raw/_fm_spiking_synaptic_readout_smoke.json`.

## What was BUILT (the biologized read-out — genuinely neural, on ONE bridge)
The whole predict-s' path runs on neurons + synapses:
- **Reservoir (NEURAL, kept):** a recurrent Izhikevich `reservoir` region driven per (state,action) token; the read-out feature is the EXCITATORY population's real `cp_firing_states` spike-count (the inhibitory subset shapes E/I dynamics but does not project to the read-out — Dale: an inhibitory neuron's read-out synapse would invert its feature contribution).
- **Read-out weights (LOCAL rule, kept):** trained by the normalized-LMS delta rule (post-synaptic error × pre-synaptic reservoir activity — a three-factor local rule; no BPTT), then realized as **Dale-legal EXCITATORY synapses** reservoir(exc) → G output ensembles per coordinate block (`Ws_shifted = Ws − Ws.min()`; the per-block uniform offset preserves the argmax — the rungB1c insight).
- **Winner (NEURAL WTA, replaces the host argmax):** G output ensembles per coordinate block compete through shared mutual inhibition; the predicted coordinate is the LABEL of the ensemble that fired MOST — a raw neural read of `cp_firing_states`. **No host `feat @ Ws`, no argmax over host logits** (grep-verified: `content_path_clean=True`).
- **Two biological companion mechanisms** were built to attack the wall (below): a feedforward common-mode CANCELLER and homeostatic excitability EQUALIZATION of the read-out neurons.

## The WALL, quantified — the Dale common-mode swamps the decision margin
Realizing a SIGNED read-out weight vector as NON-NEGATIVE (Dale) synapses forces a uniform per-block offset `|Ws.min()|` onto every weight, injecting a COMMON-MODE drive `|Ws.min()| · (total reservoir spikes)` identical across a block's ensembles. Measured at seed 42 (n_pool=300):

<!--derived-->

| quantity | value | meaning |
|---|---|---|
| x-block linear top1−top2 margin (mean / min) | 0.436 / 0.121 | the discriminative signal (healthy at the LINEAR level) |
| Dale shift magnitude `|Ws.min()|` | 10.9 | uniform offset forced onto every read-out synapse |
| common-mode drive per ensemble ≈ `|shift|·Σfᵢ` | ≈ 60 | ~140× the margin — the margin rides as <1% modulation |
| ridge (host) held-out (joint) | **0.80** | the SAME reservoir feature decodes cleanly at the linear level |
| spiking synaptic read-out held-out (per block x / y) | **0.20 / 0.20** | = chance (1/G) — the spikes do not resolve the <1% margin |
| spiking synaptic read-out held-out (joint) | **0.04** | = chance (1/G²) |
| spiking read-out TRAIN (per block x / y) | 0.24 / 0.28 | barely above chance even where memorized |
| reservoir active (spikes/neuron/step, feature / read) | 0.0243 / 0.0244 | genuinely spiking (sparse fluctuation-driven LSM regime) |

A second, coupled limiter: with only P=16 neurons/ensemble, **heterogeneity gives each ensemble a fixed baseline-rate BIAS (~5–10 spikes) that itself exceeds the margin** — direct per-ensemble baseline subtraction lifts x-block TRAIN to 0.47, but a biologically-faithful homeostatic floor calibration only partially corrects it and the correction does NOT survive to held-out (per-block held-out stays at chance). Input REPLAY (more spike samples) did not recover held-out either.

## Anti-cheats (all present; the read-out is genuinely neural — the negative is real, not an instrument failure)
<!--derived-->

| anti-cheat | result | reading |
|---|---|---|
| NEURAL SOURCE | reservoir active (0.024 spk/neuron/step); winner off `cp_firing_states` | the drive is synaptic, the winner is a neural read |
| CONTENT PATH CLEAN | `content_path_clean=True` (grep of `_neural_predict` code) | no host `feat @ Ws`, no argmax over host logits |
| READ-OUT LESION | held-out 0.04 → 0.00 | zeroing the read-out synapses removes the (weak) signal |
| RESERVOIR-SILENCE LESION | held-out 0.04 → 0.00 | silencing the reservoir removes it |
| MATCHED SHAM | held-out 0.04 → 0.04 (|Δ|=0.00) | a count-matched lesion of an OFF-DECODE decoy pathway leaves the decode UNCHANGED — the collapse is specific, not perturbation-magnitude |
| SEEDED | two same-seed builds hash identically (`cfg.seed`) | byte-identical substrate |

The lesion "collapse" (0.04→0.00) has weak TEETH precisely because the intact read-out is already at chance — there is little held-out signal for a lesion to remove. That is itself part of the negative. The specificity control (matched sham unchanged) and the neural-source / content-path-clean checks confirm the run is well-formed: this is a genuine substrate-resolution limit, not a wiring bug (the same reservoir feature reaches ridge 0.80).

## The named SURPASS (what the substrate needs — do NOT defer; this is the next build)
The residual mechanism is the one `_rungB1c` explicitly named and this arc partially scaffolds:
1. **A SIGNED per-ensemble read-out via an inhibitory relay** — deliver the negative read-out weights `max(−Ws,0)` through a per-ensemble inhibitory relay pool so each ensemble's net drive is `Ws_r·s` EXACTLY (zero common-mode), instead of the single feedforward canceller here which only APPROXIMATELY subtracts `|shift|·S` and collapses the block at the operating point. This removes the ~140× common-mode at the source and is the highest-leverage next step.
2. **Larger ensembles (P≈80) with re-tuned WTA** (rungB1c's read-out-resolution fix: P=80 + a longer read window average out heterogeneity + resolve the sub-1% margin into a spike-count difference); the WTA E→I/E→E/I→E weights must be re-scaled ~1/(P/P₀) for the larger presynaptic counts.
3. **Direct intrinsic-plasticity equalization** of the read-out neurons (the floor-current homeostasis under-corrects because the floor→rate map is compressive; an intrinsic-excitability rule that regulates baseline SPIKE-RATE to a set-point would equalize exactly).

## Honest scope
Single-seed SMOKE (seed 42, n_pool=300, G=5, held-out 25/100 cells). `enable_stdp/hebbian/STP/structural = OFF` (fixed reservoir + fixed read-out synapses; the delta rule set the weights at train time off-substrate; the on-substrate homeostasis is the read-out floor calibration). The verdict is a VALID NO-GO (validity preconditions hold; the GO criterion "matches ridge" fails). Numbers trace to `research/findings/raw/_fm_spiking_synaptic_readout_smoke.json`.

**6-seed:**  `SIM_BACKEND=numpy python -m research.runners._fm_spiking_synaptic_readout_derisk --seeds 42 43 44 100 101 102 --n-pool 300`
