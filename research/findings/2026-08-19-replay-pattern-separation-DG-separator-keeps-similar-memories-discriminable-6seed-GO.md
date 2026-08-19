---
type: finding
status: positive
date: 2026-08-19
mechanism: replay-dg-pattern-separation
runner: research/runners/_replay_dg_pattern_separation_lif.py
artifacts:
  - research/findings/raw/replay_dg_sep/lif_6seed.json
  - research/findings/raw/replay_dg_sep/bridge_substrate_6seed.json
---

# A DG-style sparse-expansive separator on the replay stream keeps similar memories discriminable — 6-seed GO

**Board #43** ("keep similar memories from blurring during sleep-like replay").

## Verdict

**GO (6/6 seeds, all 7 checks).** Two SIMILAR memories (input Jaccard 0.60) are consolidated by offline
replay into a cortical answer store. With a dentate-gyrus-style **sparse-expansive pattern-separator** on the
replay stream, each memory stays retrievable without the reinstatement teacher and the correct answer wins over
the confusable one. **Lesion the separator and the two memories collapse into an identical code — the blur
returns.** The runner's own aggregate verdict is GO on 6 seeds (42/43/44/100/101/102), byte-identical on re-run.

The precise residual named up front: separation HALVES the false recall (correct:wrong 2:1), it does not
eliminate it — the ~27% of DG granule cells still shared by the two sparse engrams cross-write. Pushing DG
sparser to close that gap hits a reliability cliff (below). This is the 2026-05-31 separation-vs-reliability
tradeoff, now quantified on the replay-consolidation write rather than on VSA symbol grounding.

## Why this was the right mechanism, and where the prior wall actually sat

The replay-consolidation NO-GOs of 2026-08-03 (`_replay_cortical_consolidation_gate` v1/v2) established a
CAUSAL replay->cortex path whose write "alternates between diffuse false recall (broad coactivity strengthens
the wrong target) and near-inert learning." That false recall IS the blur board #43 names. The v2 doc's own
next-step was "make the reinstatement reliable without increasing false recall, using a local competitive /
inhibitory mechanism." A DG sparse-expansive recoding is exactly that mechanism: orthogonalize the two episodes
BEFORE the cortical write so replay of A cannot co-activate B's target.

The separate 2026-05-31 "DG FUNDAMENTAL BOUNDARY" is NOT a blocker here. That boundary was about turning DG
activity into a NEAR-ORTHOGONAL (cos ~ 0) VSA symbol; its own refinement records that the substrate is already
id-separable and that the unmet bar was VSA binding, not discriminability. This gate's bar is discriminability /
no-confusion — a much lower bar, which is why it is reachable.

## Mechanism (all spiking / synaptic)

A leaky-integrate-and-fire network with current-based exponential synapses:

    input (EC)  --fixed random EXPANSIVE (5x)-->  dg (granule)
    input,dg    --feedforward-->  dg_basket (PV)  --[separator gate]-->  dg   (winner-take-few)
    dg  --PLASTIC rate-Hebbian coactivity (replay/sleep only)-->  answer (cortex)
    answer  <--opponent inhibition-->  answer_inh

Consolidation is OFFLINE replay: each event reinstates a memory's input (-> its sparse dg engram) together with
its answer assembly (the hippocampal index), and the coincidence potentiates dg_engram -> answer. Retrieval
drives ONLY the input, with the index-teacher and plasticity OFF, and reads which answer assembly wins. The
single manipulated variable across the dissociation is the basket->granule (separator) gain; every drive,
pattern, seed and schedule is identical. Biology: DG sparse-expansive recoding via random perforant projection +
PV-basket feedforward inhibition (Marr 1971; O'Reilly & McClelland 1994; Leutgeb 2007; Bakker 2008); replay as
the transport of the hippocampal engram to cortex (`research/biology/swr-sequence-replay.md`,
`research/biology/systems-consolidation.md`).

## Results — 6 seeds, per-seed and pooled (`lif_6seed.json`)

<!--derived-->
_All numbers in this section are rounded / per-probe-averaged from the cited full-precision artifacts
`research/findings/raw/replay_dg_sep/lif_6seed.json` and `bridge_substrate_6seed.json`._

| seed | DG Jaccard ON | DG Jaccard OFF | ON correct | ON wrong | OFF correct | OFF wrong | scramble-teach sel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42  | 0.217 | 1.000 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |
| 43  | 0.310 | 0.995 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |
| 44  | 0.321 | 1.000 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |
| 100 | 0.225 | 0.998 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |
| 101 | 0.245 | 1.000 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |
| 102 | 0.284 | 1.000 | 0.033 | 0.017 | 0.267 | 0.267 | -0.333 |

Raw answer magnitudes (seed 42, memory m0): ON fires 48 answer spikes total — the correct 16-cell assembly
carries ~32, the confusable assembly ~16 (2:1, correct wins). OFF fires 512 answer spikes with the correct and
confusable assemblies EQUAL (0.267 = 0.267) — a total blur in which neither memory wins. DG is 15-19% active ON
(sparse) versus 100% active OFF (dense) on every seed.

**Anti-cheat 1 — similar-memory discriminability after consolidation, with a NULL that blurs.** ON: both
memories win, mean selectivity 0.333 (correct-vs-wrong from raw magnitudes above). NULL (separator OFF): mean
selectivity 0.000, the two answer assemblies fire identically. Dissociation +0.333 on all 6 seeds.

**Anti-cheat 2 — the separator is load-bearing.** The OFF condition IS the biological lesion: removing the
basket->granule inhibition removes DG sparsity, DG active fraction returns to 100% and the engram Jaccard
returns to ~1.0 (the two memories become the SAME code), and discriminability collapses to chance. The gate
stays lesioned throughout the OFF read (no plasticity re-grows it). Separator ON vs OFF is the only difference.

**Anti-cheat 3 — no catastrophic over-orthogonalization.** DISSIMILAR memories (input Jaccard 0.04) stay
perfectly discriminable ON (selectivity 1.000, 6/6), and a SINGLE consolidated memory recalls correctly (6/6).
So the separator does not orthogonalize everything into uselessness.

**Anti-cheat 4 — discriminability rides the LEARNED mapping.** Scramble-teach control: run the identical replay
on a fresh network with the memory->answer pairing SWAPPED, then probe the TRUE pairing. Selectivity INVERTS to
-0.333 on all 6 seeds — each memory now recalls the OTHER answer. The read is caused by the learned
engram->answer write, not by the separator or the readout geometry.

**Determinism.** The 6-seed artifact is byte-identical on re-run (`cfg.seed` seeds every draw; the LIF loop is
noise-free).

## Honest limits and the named next mechanisms

1. **Separation is partial, not clean.** ON leaves correct:wrong at 2:1 (false-recall fraction 0.33) because
   ~27% of the two sparse DG engrams still overlap. That residual maps directly to the next lever: a stronger
   separator (lower DG Jaccard) OR a completion/cleanup stage AFTER separation.

2. **The reliability cliff is real, even in clean LIF.** The stable operating window is basket-feedforward
   weight 0.45-0.60 (DG Jaccard ~0.27 throughout); at ~0.65 the sparse code COLLAPSES to zero active granule
   cells and recall fails on every seed. Sparser-therefore-cleaner is not freely available — this is the
   2026-05-31 separation-vs-reliability tradeoff, reproduced on the consolidation write.

3. **Production-substrate port is the load-bearing next step (NOT done here).** The sibling runner
   `_replay_dg_pattern_separation_gate.py` builds this exact circuit on the production `SimulationBridge`
   (Izhikevich + rate-window Hebbian). It reproduces the SEPARATION dissociation there — DG engram Jaccard
   drops from ~1.00 (dense, OFF) to 0.39-0.68 (competition ON, 6 seeds; `bridge_substrate_6seed.json`) — but the
   consolidation READ stays at chance (mean selectivity ~0, 6/6) because of two measured substrate properties:
   (a) Izhikevich RS granule cells POST-INHIBITORY REBOUND-burst under the strong phasic basket inhibition a
   sparse code needs (competition ON delivers g_i ~18x g_e yet RAISES DG spikes 851->2679), and (b) the spiking
   k-WTA has a razor-thin, seed-variable window. So on the production substrate the separation exists but does
   not yet translate into a discriminable write. The next mechanism is explicit: an adapting/non-rebounding
   granule phenotype (or graded tonic rather than phasic basket inhibition) plus a stabilized k-WTA, then
   re-run the consolidation read on the bridge.

4. **Tracked scaffolds (host, not brain).** Host-defined input (sensory) patterns and answer assemblies; host
   reinstatement of each memory's input AND answer during replay (the hippocampal index / SWR trigger); a
   rate-window Hebbian coactivity write (the same stand-in the consolidation gates use); an argmax over answer
   spike counts for MEASUREMENT only. The retrieval read is index-teacher-OFF, so it is a cortical-store read,
   not a full hippocampus-lesioned systems-consolidation gate (that end-to-end gate is the v2 bridge runner's
   target and remains open per limit 3).

## Reproduce

    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_lif \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/replay_dg_sep/lif_6seed.json

    # production-substrate attempt (separation dissociation; consolidation read at chance)
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_gate \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/replay_dg_sep/bridge_substrate_6seed.json

## Sources

EXTERNAL-SEARCH-RAN: dentate-gyrus sparse-expansive pattern separation and its role in reducing interference
during consolidation of similar memories (logged to the external-search record, 2026-08-19).

- Leutgeb, J.K., Leutgeb, S., Moser, M.-B., Moser, E.I. (2007). Pattern separation in the dentate gyrus and CA3
  of the hippocampus. Science 315:961-966. — DG orthogonalizes similar entorhinal inputs; CA3 completes.
- Bakker, A., Kirwan, C.B., Miller, M., Stark, C.E.L. (2008). Pattern separation in the human hippocampal CA3
  and dentate gyrus. Science 319:1640-1642. — DG/CA3 signal tracks separation of similar inputs.
- O'Reilly, R.C., McClelland, J.L. (1994). Hippocampal conjunctive encoding, storage, and recall: avoiding a
  trade-off. Hippocampus 4:661-682. — the sparse-conjunctive DG code and the separation/completion trade-off.
- Marr, D. (1971). Simple memory: a theory for archicortex. Phil. Trans. R. Soc. Lond. B 262:23-81. — expansive
  sparse recoding as the substrate of pattern separation.

Internal biology bindings: `research/biology/swr-sequence-replay.md`, `research/biology/systems-consolidation.md`.
Note: the "2026-05-31 FUNDAMENTAL BOUNDARY" wording above is a REFERENCE to a prior internal finding (VSA
near-orthogonality), not a boundary verdict of THIS finding — this finding's own verdict is GO.
