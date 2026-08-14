---
type: finding
status: live
date: 2026-08-13
mechanism: onebrain-composer-transducer
---

# One-brain transducer — the RF-phasor RECALL drives the cross-organ synapse on the shared substrate (GO)

**Date:** 2026-08-13 · **Status:** GO (6-seed 42/43/44/100/101/102). The composer's RF-phasor RECALL now
NATIVELY drives a load-bearing cross-organ synapse into the SURPRISE organ, on ONE shared `SimulationBridge`,
via a PHASE→SPIKE TRANSDUCER — the composer's own validated spiking-cleanup Izhikevich WTA read routed onto
the shared bridge as a first-class `cleanup` region. This closes the one nuance the composer↔surprise MERGE
left open (`2026-08-13-onebrain-composer-merge-GO.md`: the RF recall was a PHASE state that did not drive the
edge — RF-recall interaction 0/6). Recall-driven cross interaction goes **0/6 → 6/6**, with composer recall +
moat + surprise read all byte-identical (max delta 0.0), determinism intact, both cross edges load-bearing.

## What this de-risks (the mission)

The merge de-risk put the recall COMPOSER and the SURPRISE organ on one pool and proved a `composer→surprise`
synapse is load-bearing when its source neurons emit Izhikevich SPIKES (a current stand-in). The one nuance:
the composer's RF-phasor recall leaves neurons in a PHASE state (|Z|~1), and `rf_resonate_steps` never
traverses `cp_connections`, so the RECALL itself did NOT drive the edge (RF-phasor ↔ spike-rate CODE gap). The
named surpass was a PHASE→SPIKE TRANSDUCER. This lane BUILDS it (still a de-risk, not a production flip) and
measures: does the composer's own recall, transduced to a spike rate, drive the cross-organ synapse on the
shared substrate, byte-identity + moat intact?

## The transducer — the validated spiking-cleanup WTA, routed onto the shared bridge

```
RECALL (RF unbind on the shared composer slice)        -- phase state |Z|~1
   |  Stage-1 matched filter (RFPhasorComposer._spiking_cleanup Stage 1: Re(rec . conj(code_k)))
   v
cleanup REGION on the shared bridge (Izhikevich WTA, the `_izh_bank` Stage-2 read ROUTED onto the pool)
   |  the winner block SPIKES -- a genuine Izhikevich spike RATE on the shared cp_membrane_potential_v
   v  `cleanup->surprise` synapse in the shared cp_connections (traversed by the Izhikevich `_step`)
SURPRISE organ  -- the recall now DRIVES the cross-organ synapse.
```

- **`cleanup` region:** V=9 word-blocks of 24 Izhikevich neurons (default GENERIC_UNSTRUCTURED type — the SAME
  neuron config as the validated standalone `_izh_bank` WTA and the merge composer block). The transducer drives
  each word-block k with `(scores/peak)[k] * win_pA` — the SAME input-normalized WTA drive as `_spiking_cleanup`
  (winner ~win_pA, off-targets rectified to ~0). Only the winner block spikes.
- **`cleanup→surprise`:** TOPOGRAPHIC word→fact wiring — cleanup word-block(patient_i) → surprise fact-block i,
  for each composer fact (cat→fact0, mouse→fact1, deer→fact2). So recalling patient "cat" drives the surprise
  unit for the dog-chase-cat fact.
- **`surprise→cleanup`:** the reverse topographic edge (surprise fact-block i → cleanup word-block(patient_i)),
  so the surprise signal biases recall THROUGH the shared substrate.

The winner block is SELECTED by the RF recall's matched-filter argmax (== the recalled patient word, verified
6/6); the Izhikevich WTA on the shared bridge TRANSDUCES that phase-derived score to a spike RATE that drives
the cross synapse. This is NOT the merge runner's injected-current stand-in — the drive is the composer's OWN
recall readout.

## Result (`_onebrain_composer_transducer_derisk.py`, 6-seed; `--seeds 42,43,44,100,101,102`)

Facts `[dog→chase→cat, owl→eat→mouse, wolf→hunt→deer]`; unstored cue `lion roar` → the moat must abstain.

| Axis | Verdict | Detail |
|---|---|---|
| one shared neuron pool (surprise+composer+cleanup) | 6/6 | N=1720 = 1056 surprise + 448 composer + 216 cleanup, one `cp_membrane_potential_v`, both extra regions contiguous |
| determinism (`cfg.seed` incl. thresholds) | 6/6 | two fresh transducer builds at one seed → identical v / connections / thresholds |
| SURPRISE read byte-identical (WITH the transducer machinery) | 6/6 | max err 0.0 Hz over confirm/contradict/novel per-fact rates; cleanup region present but adds no footprint |
| surprise faculty alive (contradict ≫ confirm) | 6/6 | separation 5.4–61.3× (byte-identical of a LIVE organ) |
| COMPOSER recall byte-identical + CORRECT | 6/6 | `['cat','mouse','deer']` == standalone `RFPhasorComposer`, every seed; == stored patients |
| no-confab MOAT preserved (unstored → abstain) | 6/6 | shared `query_patient('lion','roar')` == None == isolated |
| transducer WTA winner block == recalled patient word | 6/6 | the shared-bridge WTA emits the CORRECT recall as spikes |
| shared-bridge `cleanup` region SPIKED under the recall | 6/6 | winner-block firing 180–199 Hz |
| transducer ABSTAINS on the unstored cue (no drive) | 6/6 | an abstain (peak score ~0) produces NO cleanup drive — the moat carried into the transducer |
| **RECALL DRIVES the cross-organ synapse (0/6 → 6/6)** | **6/6** | recall→surprise interaction **+37.5…+78.7 Hz**, lesion `cleanup→surprise` −0.93…+0.00 Hz, attribution frac 1.00–1.02 |
| SURPRISE→CLEANUP load-bearing (surprise biases recall) | 6/6 | contradiction drives cleanup **+31.7…+52.5 Hz**, lesion `surprise→cleanup` +0.00 Hz, frac 1.00 |
| **TRANSDUCER GO** | **6/6** | one pool + determinism + byte-identity + surprise alive + recall-drives-edge + abstain-clean |

## The two cross-organ interactions — both load-bearing on the shared pool

**Recall → surprise (interaction A, the headline).** A CONFIRM read of surprise fact i (surprise ~0 when the
prediction cancels the assertion), with the `cleanup` region driven by the RECALL of fact i, raises surprise
BLOCK i by +37.5…+78.7 Hz. LESIONING `cleanup→surprise` (weight→0, `plastic=False` so the lesion holds)
collapses it to −0.93…+0.00 Hz (attribution frac 1.00–1.02). The SAME run reproduces the merge runner's
boundary: the composer's RF phase state WITHOUT the transducer does not reach surprise. So the phase→spike
transducer is exactly what makes the recall drive the edge: **0/6 → 6/6**.

**Surprise → recall (interaction B).** A CONTRADICTION (the surprise faculty fires high) drives the `cleanup`
region — the recall's own WTA readout — by +31.7…+52.5 Hz through `surprise→cleanup`; lesion → +0.00 Hz (frac
1.00). The surprise signal biases recall through the shared substrate, and the reverse synapse is load-bearing.

## Anti-cheats

- **Load-bearing, not co-driven:** both interactions collapse to ~0 when their edge is lesioned (weight→0,
  `plastic=False`), attribution frac 1.00–1.02 via `tools.lab.attributable_to`. The recall-driven rise is on
  the block the recall's winner TOPOGRAPHICALLY targets (surprise block i), the like-for-like measurement.
- **Byte-identity in the DATA:** surprise read max err 0.0 Hz (exact per-fact compare), composer recall exact
  string compare vs a standalone `RFPhasorComposer`, determinism by hashing v / connections / thresholds.
- **Moat preserved AND carried into the transducer:** the unstored cue abstains (None), and the transducer
  produces NO drive on an abstain (peak matched-filter score ~0) — it only fires on a REAL recall.
- **Correct recall as spikes:** the shared-bridge WTA winner block == the recalled patient word (6/6), so it is
  the RIGHT recall driving the RIGHT surprise unit, not exact-of-garbage.
- Genuinely one pool (one `cp_membrane_potential_v`, contiguous composer + cleanup spans). Determinism via
  `cfg.seed` (not `actual_seed_used`). 6 seeds; smoke first.

## Honest scope

1. **This is a DE-RISK, not a production flip.** The composer stays its own bridge in `/api/brain-chat`; this
   proves the recall drives the cross-organ synapse byte-safely when merged. Per `docs/TERMS.md`, "closed"
   requires production integration (wired + on-by-default + scaffold-retired) — this is a 6-seed GO de-risk of
   the mechanism, the next rung toward the composer joining a production pool.
2. **The transducer inherits the validated spiking-cleanup's documented residual.** Stage-1 (the matched
   filter) is the composer's per-op RF readout, exactly as in `_spiking_cleanup`; the phase→spike TRANSDUCTION
   (Stage-2 Izhikevich WTA) and the cross-organ drive are on the SHARED substrate. The Stage1→Stage2 score
   read+normalize is host arithmetic — the residual of the spiking cleanup this reuses, not a new shortcut.
3. **The composer's fact-store here is the numpy-kb idealization** (its documented "principled idealization";
   the RF resonate/unbind ops themselves ARE on the shared bridge).
4. **The cross weight (8.0) and the WTA drive scale (win_pA=600) are synaptic-gain parameters** (both within
   the working range; `hebbian_max_weight=45`). The interaction is LINEAR in the weight and collapses to 0 on
   lesion; the winner is selected by the RF recall's matched filter regardless of the absolute scale.

## Read-out — the RF-phasor ↔ spike-rate code gap is bridged on the shared substrate

⇒ the composer's RF-phasor RECALL now emits a SPIKE RATE (via the shared-bridge Izhikevich WTA transducer) that
drives a load-bearing `cleanup→surprise` synapse, and the surprise faculty biases the recall back through
`surprise→cleanup` — both on ONE pool, with composer recall + moat + surprise read byte-identical and
determinism intact. The merge finding's one open nuance (recall-driven cross interaction 0/6) is now 6/6. The
composer↔organ interaction is DRIVEN by the recall itself on the shared spiking substrate — a real step toward
the composer joining a production Gate-B pool.

CI/repro: `SIM_BACKEND=numpy python -m research.runners._onebrain_composer_transducer_derisk --seeds
42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_transducer_6seed.json`. Runner:
`research/runners/_onebrain_composer_transducer_derisk.py` (`--seed`, `--seeds`, `--D-cmp`, `--cblk`,
`--cross-weight`, `--win-pa`). De-risk chain: composer+surprise MERGE
(`2026-08-13-onebrain-composer-merge-GO.md`) → THIS (the recall natively drives the cross synapse via the
phase→spike transducer). The transducer reuses the validated spiking cleanup
(`2026-06-05-phase1-tpam-cleanup-derisk-GO.md`).
