---
type: finding
status: no-go
claim_check: measured
date: 2026-09-24
lane: B — curiosity (novelty / epistemic-gap crave) x E1 metacognition
mechanism: curiosity-commitment-veto v5 (v4's metacog comparator -> frozen point-edge onto curiosity's ASK, plus phasic
  lc_ne feedback-withdrawal gain, PLUS a per-channel commitment-veto population whose rival-relay inhibition onto ASK
  is set by inhibitory STDP to a constant no-evidence rate in a one-time calibration epoch, then frozen; G11 scored on
  a reference-placed fine grid)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_curiosity_commitment_veto_v5_derisk.py
prereg: docs/plans/2026-09-24-curiosity-commitment-veto-v5-PREREG.md (v1.1, amendment 12b6d3e3e; governs runner
  revision 4cbedbaf8)
artifacts: research/findings/raw/_curiosity_commitment_veto_v5_6seed_combined.json (the pre-registered `--combine`
  over the six `_s<seed>.json` files beside it, all pinned to runner revision bf8d6a0c2, git_dirty: false)
verdict: NO-GO 1/6 (only seed 42 passes every required gate). The dominant residual is G13, the inhibitory-STDP
  no-evidence set-point on the veto: it misses on 4/6 seeds (43, 44, 100, 102), in both directions the prereg's own
  risk section named (a strongly-driven channel stuck above target at maximal inhibition; a weakly-driven channel
  that inhibition alone cannot raise). The 6th failing mode is different -- seed 101 passes G13 and every rho-based
  gate but fails G1 on the 1.0 Hz range floor, because the veto compresses its already-weak range (0.79 Hz, down
  from v4's 1.66 Hz on the same seed) below the floor. Every non-G13/G1 required gate (G3, G6, G7, G8, G10, G11, G12)
  passes on all 6 seeds, and every integrity precondition passes on all 6 seeds -- the run is COMPLETE and DEFINED,
  not UNDEFINED.
---

# Curiosity x metacog v5, commitment veto + inhibitory-STDP set-point: 6-seed NO-GO (1/6), G13 set-point is the residual

## 1. Result

Artifact: `research/findings/raw/_curiosity_commitment_veto_v5_6seed_combined.json`, produced by the pre-registered
`--combine` step over `research/findings/raw/_curiosity_commitment_veto_v5_s{42,43,44,100,101,102}.json`. All six
inputs are pinned to runner revision `bf8d6a0c2af380af4b49a9d6b695e849e45f9306` (`git_dirty: false`, identical
`source_manifest_sha256` across all six), a descendant of the governed revision `4cbedbaf8` (the v1.1 amendment
commit) with **no diff to the runner file between the two** -- confirmed with
`git diff 4cbedbaf8 bf8d6a0c2 -- research/runners/_curiosity_commitment_veto_v5_derisk.py` (empty). `--combine`
did not refuse: the seed union is exactly {42,43,44,100,101,102} with no duplicate, one mechanism string, one
operating point, and one runner blob across all six inputs' provenance SHAs -- the check the prereg names as **the
only pre-registered verdict**. `evaluation_set: true`, `GO: false`, `n_go: 1`, `n_seeds: 6`, `undefined_reasons: []`.

The operating point was chosen on dev seeds 7-12 only (prereg §3); all six evaluation seeds were held out before
this run. The table below is read from `per_seed` in the combined artifact, rounded to 2-3 decimals.

<!--derived-->
| seed | go | failing gate(s) | rho (intact) | ASK range (combined, Hz) | G13 setpoint ratio (ch0, ch1; target 1.0) |
|---|---|---|---|---|---|
| 42 | **GO** | none | -0.991 | 1.470 | 0.648, 0.509 |
| 43 | -- | G13 | -0.917 | 4.028 | 0.139, **1.806** |
| 44 | -- | G13 | -0.810 | 2.581 | **1.944**, 0.556 |
| 100 | -- | G13 | -0.936 | 4.907 | 0.463, 1.204 |
| 101 | -- | G1 (range floor) | -0.863 | **0.787** | 0.648, 1.019 |
| 102 | -- | G13 | -0.917 | 3.900 | 0.093, 0.185 |

Every other required gate (G3 gain-pathway load-bearing, G6 fresh-process determinism, G7 class-swap monotone, G8
relay-lesion abolishes coupling, G10 lc_ne evidence-graded, G11 multiplicative-not-additive on the reference-placed
fine grid, G12 lc_ne phasic) passes on all 6 seeds. Every integrity precondition (byte-off, GIRK routing exact,
metacog-vs-base-pool exact, G5 metacog unchanged across lesion arms, restore-exact, lesions held at measurement,
lc_ne acts only through the feedback loop, non-ASK regions silent, no host novelty/neuromodulator signal, inhibitory
STDP eligible ONLY on the two declared rival-relay -> veto rows, calibration changed only the plastic rows, veto
weights frozen after calibration) passes on all 6 seeds. The run is complete and every seed's verdict is DEFINED --
none is UNDEFINED, and none is scored as a pass by default.

## 2. Reading it

- **G13 (the inhibitory-STDP no-evidence set-point) is the dominant residual, exactly as the prereg's own risk
  section predicted (§6.1).** It misses on 4 of 6 seeds, in both of the directions the prereg named before this run:
  seed 44's channel 0 sits at 1.94x target (a strongly-driven channel inhibition-only plasticity cannot pull back
  down to target), while seeds 43, 100 and 102 each have a channel stuck below target (43 ch0 0.14x, 100 ch0
  0.46x, 102 both channels 0.09x/0.19x -- a channel inhibition alone cannot raise). The prereg named this as the
  bound of an inhibition-only set-point and flagged the missing companion (a multiplicative excitatory scaling arm)
  before any evaluation seed ran.
- **Seed 101 fails differently: G1's 1.0 Hz range floor, not its rho.** Seed 101's rho (-0.86) and every other
  rho-based gate pass; what fails is `ask_range_hz.combined = 0.79 Hz < G1_MIN_RANGE_HZ (1.0 Hz)`. This is the same
  seed the v4 finding flagged as the weakest evaluation range (1.66 Hz there); v5's veto compresses the most-
  uncertain-level ASK response further (prereg §6.2's named risk: "the veto removes drive at the most uncertain
  level too"), and on this seed that compression pushes an already-thin range under the floor.
- **The mechanism does what it was built to do.** Seed 42 -- the only GO -- shows the veto behaving as designed:
  `ask_range_hz.veto_lesion = 2.97 Hz` vs `combined = 1.47 Hz`, i.e. the veto measurably removes the confident-end
  rise the v4 comparator-summed read left in (the defect this rung targeted), while G11's reference-placed fine grid
  still reads a defined 8-point rising limb and G3's gain-pathway share stays load-bearing (0.35). The other five
  seeds show the same qualitative behavior (`attributable_frac.veto_confident_half` reads 0.89-1.0 "intact" and
  0.95-1.0 "class_swap" everywhere) -- the veto's OWN function is not what fails these seeds.
- **No instrument artifact.** G11 (the fine, reference-placed grid the v1.1 amendment fixed) is defined with 7-10
  limb points on every seed and passes on every seed; this rung's own instrument fix held up on the real
  evaluation-seed substrate, not just the 4-seed dev re-smoke.

## 3. Next (per the LAW: the capability stays open)

1. **The prereg's own named companion:** an excitatory arm on the set-point (multiplicative synaptic scaling of the
   veto-relay drive, not only inhibitory STDP), so a below-target channel can be raised as well as an above-target
   one suppressed -- named as "the next companion if G13 fails" in prereg §6.1, now confirmed as the failure mode on
   4/6 seeds.
2. **Seed 101's range floor** is a companion-process question, not a new lever on this circuit: the veto's
   confident-end suppression and the most-uncertain-level range floor are in tension on weak-range seeds, which
   argues for the same set-point fix (a better-regulated ASK operating point) rather than a veto-strength retune
   that would re-open the confident-end tail this rung closed on seed 42.
3. Bank this method's own claim (a downstream opponent read of the comparator, frozen by inhibitory STDP, does
   remove the confident-end artifact where the v4 comparator-summed read failed) and take the excitatory-arm
   companion as the next lever, per `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`'s standing rule: a wall is
   a verdict on a method, not a license to abandon the capability.
