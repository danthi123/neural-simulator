---
type: finding
status: contributing
date: 2026-08-26
mechanism: cls-interleaved-consolidation
lane: "#66 knowledge-scale"
seeds: [42, 43, 44, 100, 101, 102]
---

# Interleaved SWR-replay of a fast store into a slow store defeats catastrophic interference (CLS, 6-seed GO)

## Claim
The Complementary-Learning-Systems ARCHITECTURE protects old memories from new learning: sequentially teaching
set B does NOT overwrite set A when A lives in a fast hippocampal store that is REPLAYED INTERLEAVED with B into
a slow neocortical associator. The protection is load-bearing on the replay and on its CONTENT — remove the
replay, scramble the replayed targets, or spend the same extra steps on more B, and A is forgotten. This lands
the de-risk at 6 seeds (the prior commit's "robust across 42/43/44" prose had no committed artifact); the runner's
own verdict is GO.

## Result
`research/runners/_cls_interleaved_consolidation_derisk.py`, 6 seeds (42/43/44/100/101/102),
`SIM_BACKEND=numpy`, one process (`--seeds 42 43 44 100 101 102`). Artifact:
`research/findings/raw/cls_consol/cls_consol_6seed.json` -> `verdict: GO` (full precision);
`research/findings/raw/cls_consol/cls_consol_6seed_agg.json` is the same run rounded to the 3 decimals
displayed below.

Over-capacity slow cortex (60 items / dim=24 / 8 categories, `cortex_lr=0.02`), so the single-store baseline is
FORCED to reuse weights and forgets. Mean A-retention (fraction of post-phase-A baseline, baseline acc = 1.000)
and new-learning (B accuracy) by arm:

| arm | A-retention | B-acc (new learning) |
|-----|-------------|----------------------|
| interleaved (replay ON, true CLS) | **0.939** | 0.911 |
| seq (replay LESIONED, single store) | 0.317 | 0.994 |
| shuffled-replay (wrong A targets) | 0.133 | 0.922 |
| exposure-match (extra steps = more B) | 0.278 | 1.000 |

Per-seed A-retention (interleaved / seq / shuffled / exposure):
- s42: 0.967 / 0.333 / 0.167 / 0.267
- s43: 0.933 / 0.333 / 0.067 / 0.300
- s44: 0.967 / 0.200 / 0.200 / 0.133
- s100: 0.900 / 0.367 / 0.133 / 0.333
- s101: 0.900 / 0.300 / 0.100 / 0.300
- s102: 0.967 / 0.367 / 0.133 / 0.333

Every seed individually clears the pre-registered thresholds (interleaved >= 0.80, every lesion arm <= 0.60,
B-acc >= 0.80) — the mean-based GO is not carried by an outlier.

## Instrument + controls (all PASS — a GO whose controls fail is a NO-GO)
Reported verbatim from the runner's printed VERDICT / the JSON `anti_cheats` block:
- **AC1 replay-load-bearing** (seq forgets, ret <= 0.60): PASS — seq ret = 0.317.
- **AC2 content-specific** (shuffled replay does not protect): PASS — shuf ret = 0.133.
- **AC3 exposure-matched** (more-B, equal total updates, does not protect): PASS — exp ret = 0.278.
- **AC4 new-learning-works** (B acc >= 0.80 in ALL arms): PASS — min B = 0.911.
- **AC5 lesion-load-bearing margin** (interleaved beats every lesion by >= 0.20): PASS — margin = 0.622.
- **AC6 attribution** (replay owns >= 50% of retained-A; `tools.lab.attributable_to`): PASS — frac = 0.663
  (interleaved +0.939 vs seq control +0.317; 33.7% of retention is also present in the no-replay control).
- **PROTECT** (interleaved retention >= 0.80): PASS — ret = 0.939.

The replay branch actually EXECUTES (avg 3600 replay `train_step` calls / interleaved arm; not merely present in
source), and A-retention is read by a cortex-only WTA (`argmax(W @ x)`) that never consults the hippocampal
store — so the measured trace is already independent of the fast store at test. The verdict travels with a
`preconditions` block (`tools.verdict.Verdict`, all 8 checks ok:true): `acc_A_baseline` is a REQUIRED
precondition pinned at ceiling (1.0 every seed) BY DESIGN — A must be fully learned before B, else "forgetting"
is undefined — NOT the discriminating metric, which is the retention spread (interleaved 0.939 vs lesion arms
0.13–0.32); the advisory discriminating-power gate flags that ceiling and it is expected here. AC3 closes the
more-training / lower-effective-lr confound the biology binding demands; AC2 closes the generic-extra-activity
confound. The lesion arms hold by construction (seq never enters the replay branch; shuffled deranges the
replayed targets for the whole phase), so no plasticity re-instates the lesioned signal before measurement.

## Biology binding
`research/biology/cls-interleaved-consolidation.md` (status: established; anchors resolve 2026-08-26):
catalog N.14 hippocampal-neocortical dialogue ("gradually transfers memory from HC-dependent" -> neocortex),
Buzsáki two-stage model (waking encoding writes the fast HC store; sleep SWR-replay drives the SAME content into
a slow neocortex), Buzsáki *Rhythms of the Brain* ("could be replayed multiple times, assisting with the
consolidation"). `constraints_config cortex_lr=0.02` (the slow-store requirement of
McClelland/McNaughton/O'Reilly 1995) matches the runner's `cortex_lr` — biology_check PASS. The replay generator
this de-risks the FUNCTION of is the GO `swr-sequence-replay` organ (`_gap5_ecker_recurrent_replay.py`).

## Scope / honesty
- This de-risks the CLS **function** at the architecture level: a separate fast store whose content is replayed
  interleaved into a slow store defeats catastrophic interference, and the load-bearing variable is the replay
  (and its correct content), not a substrate regime. It does NOT claim a spiking neocortex — the slow store is a
  rate-coded three-factor Hebbian (delta-rule) associator, a documented idealization of the slow cortex. Making
  the cortical learner spiking, and sourcing the interleaved samples from the live spiking SWR organ rather than a
  buffered replay of the true mapping, are the next rungs.
- It does NOT contradict the 2026-05-21 arc
  (`2026-05-21-catastrophic-forgetting-FULL-3x3-matrix-COMPLETE-substrate-resistance-is-seed-dependent-not-regime-specific-CLS-regime-prediction-NOT-robust-multi-seed-at-any-intensity.md`):
  that tested whether a UNIFIED substrate's compositional-vs-direct REGIME predicts interference resistance (it
  does not, robustly). This isolates the architecture-level claim that arc never tested — separate store +
  interleaved replay vs a single store — and it IS robust at 6 seeds.

## Next (no-defer)
GO. Next rung toward one-brain: replace the rate-coded slow associator with the shared spiking substrate and feed
the interleaved samples from the live spiking SWR-replay organ (learn-through-use), so consolidation happens
through the brain's own replay rather than a host-buffered re-instatement of the true A mapping.
