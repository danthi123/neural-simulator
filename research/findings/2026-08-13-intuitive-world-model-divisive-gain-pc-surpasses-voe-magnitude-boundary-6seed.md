---
type: finding
status: live
date: 2026-08-13
mechanism: intuitive-world-model-divisive-gain-predictive-coding
verdict: GO (the strict 6-criteria gate passes 6/6) — DIVISIVE/GAIN (biased-competition) predictive
  coding SURPASSES the first rung's mapped VoE-magnitude boundary. The maintained prediction now
  DIVIDES (shunts) the sensory reveal instead of subtracting it: VoE magnitude >=2x on BOTH sets goes
  4/6 -> 6/6, and the FS-WTA one-of-K hold_correct goes to 1.0 on all six seeds (train AND held), while
  every first-rung control is KEPT and STRENGTHENED — the no-maintenance (recur=0) lesion still
  collapses the VoE, attributability 6/6, intact-lesion separation LARGER than the subtractive rung.
lane: T1-7 · Intuitive world model / core common-sense (the biggest faculty no domain owns; 2026-08-12 audit)
artifacts:
  - research/findings/raw/_intuitive_world_model_divisive_pc_6seed.json
  - research/findings/raw/_intuitive_world_model_divisive_pc_6seed.log
verification: >
  substrate seeded (cfg.seed/heterogeneity_seed/ou_seed set on all three RNGs, via the imported
  build_world_model). The object-file + occlusion + VoE machinery is IMPORTED UNCHANGED from the first
  rung (_intuitive_world_model_permanence_derisk.py); the ONLY change is the READ: the ipred_k->err_k
  prediction and the fs->wm competition are made SHUNTING (a per-region GABA_A reversal override, ~-56 mV
  on err / ~-60 mV on wm) so the engine's conductance-based I_syn = g_i*(E_i - v) (bridge.py:7744)
  DIVIDES the gain instead of subtracting a current. NO sim/ edit; the reversal override is an existing
  BrainRegion field (syn_reversal_potential_i_override). The surprise is a cp_firing_states err_*
  population rate (no host argmax over object codes); occlusion input asserted identically ZERO
  (permanence is a spiking memory); occlusion window kept >=110 ms (the instrument fix). The lesion is a
  NO-MAINTENANCE build (recur=0, nmda off) WITH the divisive machinery present -> ipred silent at reveal
  -> no shunt -> the VoE collapses; the control is unweakened. An ISOLATION control (same operating point,
  subtractive err reversal) rules out the load/WTA tuning as the cause of the magnitude lift.
---

# Divisive/gain predictive coding surpasses the world-model VoE-magnitude boundary — object permanence with a >=2x, WTA-clean violation-of-expectation on 6/6 seeds (2026-08-13)

## Result
The intuitive-world-model first rung (2026-08-13) built a spiking OBJECT FILE + a predictive-coding
surprise = object permanence + a persistence-caused, generalizing violation-of-expectation (VoE);
its load-bearing scientific claim was 6/6 but the STRICT gate returned 1/6, held back by TWO mapped,
NAMED-as-surpassable boundaries: (i) the VoE MAGNITUDE cleared >=2x on only 4/6 seeds, and (ii) the
FS-WTA occasionally seated the WRONG object (hold_correct 0.75 on ~half the seeds; seed 43 permanence
degraded to ratio 2.77). This rung closes both.

**Verdict: GO — the strict 6-criteria gate passes 6/6** (seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`):

| criterion | first rung (subtractive) | this rung (divisive/gain) |
|---|---|---|
| PERMANENCE ratio >= 5 (min over seeds) | 5/6 (seed 43 = 2.77) | **6/6** (min 132.4) |
| WTA-CLEAN: hold_correct = 1.0 train AND held | ~half the seeds miss (0.75) | **6/6** (1.0 all seeds, both sets) |
| VoE PRESENT + GENERALIZES (>=1.3 train AND held) | 6/6 | 6/6 |
| VoE MAGNITUDE >= 2x train AND held | 4/6 | **6/6** |
| PERSISTENCE-ATTRIBUTABLE (intact - lesion >= 0.3) | 6/6 | 6/6 (separation LARGER) |

- **VoE magnitude, per seed (train / held):** 3.67/2.51, 5.36/2.68, 2.30/8.11, 8.30/3.24, 5.91/4.25,
  3.10/2.56 — every seed clears 2x on BOTH sets (min 2.30). The first rung was 1.77-2.86 / 1.51-3.53.
- **The lesion still collapses the VoE (control KEPT):** the no-maintenance (recur=0, nmda off) build
  reads VoE 0.80-1.32 / 0.76-1.16 — attributability 6/6, intact-lesion separation 1.2-7+. Because the
  divisive read makes the INTACT VoE larger, the separation is LARGER than the subtractive rung: the
  control is strengthened, not weakened.

## The boundary, and why a subtractive single-relay caps the magnitude
In the first rung the maintained prediction cancels the sensory reveal by CURRENT SUBTRACTION (ipred_k
injects a hyperpolarizing current into err_k, E_i = -75 mV). A subtractive relay faces the classic
subtractive-PC trade-off (Rao-Ballard): to null a strong matched sensory transient it needs an
inhibitory current MATCHED to the sensory current, but the required match depends on the (seed-dependent)
sensory magnitude — so on some seeds the cancellation under-shoots (match residual leaks -> match_alarm
up -> ratio down), and it cannot simultaneously (a) fully null a strong match and (b) leave a large
violation response. That is the ~2x cap: an OPERATING-POINT-fragile balance, not a floor of the surprise.

## The fix — divisive / shunting (biased-competition) PC; brain-based; NO sim/ edit
Divisive normalization (Carandini & Heeger 2012) / biased-competition predictive coding (Spratling 2008
*J. Vis.* 8(7):1; Spratling 2010) computes the error as a RATIO — the input GAIN-DIVIDED by the
prediction — not a subtraction. Its biological substrate is SHUNTING inhibition: an inhibitory
conductance whose reversal sits near the operating point raises the membrane conductance and DIVIDES the
neuron's gain rather than subtracting a fixed current (Holt & Koch 1997; the regions.py note: ~-60 mV =
"shunting, depolarizing-near-rest"). The engine already delivers conductance-based inhibition
(`I_syn = g_i*(E_i - v)`) and a per-region reversal override, so the fix is RUNNER-SIDE config on the
SAME (imported) object-file machinery:
- **err_reversal_i = -56 mV** — the ipred_k -> err_k prediction SHUNTS (divides) the sensory reveal. On
  a MATCH the strong maintained prediction divides err_k down SCALE-ROBUSTLY (independent of the sensory
  magnitude — the seed-robustness the subtractive relay lacked); on a VIOLATION ipred_m is silent (wm_m
  not maintained) so err_m responds FULLY. That is the divisive/gain read AND the attentional
  amplification of the maintained prediction, in one shunting mechanism (adding a separate prediction-
  gain current, ipred_to_err up, HURT — over-shunting; the shunt itself is the amplification).
- **wm_reversal_i = -60 mV** — the fs -> wm competition becomes SHUNTING = a divisive BIASED-COMPETITION
  WTA (Reynolds-Desimone) -> a cleaner, better-separated one-of-K winner (the WTA-cleanliness rung).
- **a decisive LOAD** (load_w 36, fs_to_wm 16) so the PRESENTED object locks the attractor before
  occlusion — the wrong-winner was the loaded object being out-competed by a weight-jitter-favoured slot.

## The isolation control (rules out over-tuning — the divisive READ is what closes the boundary)
At a FIXED operating point (same biased-competition WTA, same load), flipping ONLY the err inhibition
from SUBTRACTIVE (E_i -75) to SHUNTING (E_i -56):

```
VoE magnitude >=2x (both sets):  SUBTRACTIVE err read  2/6   ->   DIVISIVE err read  6/6
  per-seed min VoE (subtractive): [2.17, 1.50, 1.41, 1.60, 2.16, 1.60]
  per-seed min VoE (divisive):    [2.51, 2.68, 2.30, 3.24, 4.25, 2.56]
```

The load/WTA changes alone (subtractive read) clear 2x on only 2/6; the divisive READ clears 6/6. The
magnitude lift is caused by the read, not by the tuning. (`--isolate` reproduces this.)

## Anti-cheats (all KEPT from the first rung; none weakened)
- **LESION (decisive)** — no-maintenance build (recur=0, nmda off) with the divisive machinery present:
  VoE collapses (0.76-1.32), attributability 6/6. Holds by construction (no NMDA recurrence to regrow).
- **GENERALIZATION (world-model-vs-memory)** — 4 HELD-OUT objects (never used to set the operating
  point) show the same permanence + WTA-clean + >=2x VoE + lesion-collapse as the 4 tune objects.
- **BRAIN-BASED / no host compare** — the surprise is a `cp_firing_states` err_* population rate;
  occlusion input asserted identically zero; the read is shunting inhibition, not a host `x/(1+pred)`.
- **INSTRUMENT preserved** — occlusion window kept >=110 ms (the first rung's presentation-history-
  residual fix); the surprise read at the err_* population, not a diluting downstream alarm pool.
- **DEVELOPMENTAL control (characterization, not gated)** — a naive substrate (wm->ipred un-potentiated)
  shows no VoE (~1.0-1.7); a teacher-scaffolded STDP+DA potentiation did NOT bootstrap it (naive ~=
  trained), exactly as the first rung: the simple Hebbian route does not self-organize the object-file
  binding. Reported honestly; self-organized binding remains the named next rung.

## The secondary characterization (reported, not gated) and the honest residuals
- **Absolute lesion floor <=1.15: 4/6** (seeds 101 train 1.32, 44 held 1.16 exceed it by noise) —
  IDENTICAL to the first rung. The load-bearing collapse read is ATTRIBUTABILITY (intact - lesion >= 0.3,
  6/6), not the absolute floor; the first rung established this, and the divisive read makes the
  attributable separation larger. So the floor is a noise property of the instrument, not a regression.
- **Self-organized object-file BINDING** — the comparator is still a topographic template (object-
  independent -> it generalizes, the anti-cheat), NOT learned per object; the Hebbian developmental
  control did not self-organize it. The named next rung.
- **Occlusion/reveal EVENT grounding** — the occlusion + reveal events + presented object remain sensory
  drive (the environment boundary, as E2's valence + T1-4's events were). Grounding them in the emergent
  relational/spatial code is the follow-on.

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._intuitive_world_model_divisive_pc_derisk \
    --seeds 42,43,44,100,101,102 --learn-control \
    --out research/findings/raw/_intuitive_world_model_divisive_pc_6seed.json
# fast: --smoke  (seed 43, the first rung's worst seed: divisive VoE + lesion + isolation)
# the boundary demo: --isolate  (subtractive 2/6 -> divisive 6/6 at a fixed operating point)
```

## External-literature check
`EXTERNAL-SEARCH-RAN: 2026-08-13` (logged to the external-search record, lane-tagged
T1-7). The boundary was a KNOWN property of subtractive predictive coding; the surpass is an ESTABLISHED
mechanism: **Spratling 2008 (J. Vis. 8(7):1) / Spratling 2010** (PC/BC divisive predictive coding),
**Carandini & Heeger 2012 (Nat. Rev. Neurosci. 13:51)** (normalization as a canonical divisive
computation), **Holt & Koch 1997** (shunting inhibition divides gain), grounded on the engine's existing
conductance-based inhibition. Not a novel wall; a named next rung, now closed.

## Path to production (the point of T1-7)
A production turn's world-model organ can gain a co-resident object-file with a DIVISIVE surprise read:
maintain referents mentioned in dialogue across intervening turns (occlusion = the referent leaving the
discourse focus) and raise a graded, robust surprise / honest self-report ("that doesn't match what I
was tracking") when a later statement violates a maintained one — moat-safe (it notices a mismatch,
never manufactures a fact). The divisive read gives the >=2x, seed-robust separation a graded read-out
needs. Wiring it default-on is the integration follow-on.

## Provenance
`research/runners/_intuitive_world_model_divisive_pc_derisk.py` (NEW; imports the first rung's object-
file/occlusion/VoE machinery unchanged; modes: default 6-arm, `--smoke`, `--isolate`). Additive change
to `_intuitive_world_model_permanence_derisk.py::build_world_model`: two optional reversal-override
kwargs (`err_reversal_i`, `wm_reversal_i`), default None = byte-identical to the subtractive build. Uses
`tools.lab.attributable_to` + `tools.verdict.Verdict`. NO `sim/` edit. CPU/numpy (~600-neuron bridge).
