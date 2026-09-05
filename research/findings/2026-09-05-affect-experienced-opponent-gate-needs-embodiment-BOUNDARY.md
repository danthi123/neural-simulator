---
type: finding
status: boundary
lane: affect-salience-gate-retirement
date: 2026-09-05
mechanism: experience-bound-spiking-opponent-gate
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_experienced_opponent_gate_derisk.py
artifacts:
  - research/findings/raw/_affect_experienced_opponent_gate_6seed.json
builds_on:
  - research/findings/2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md
  - research/findings/2026-08-13-affect-appraisal-origin-self-organizes-from-reinforcement-6seed-GO.md
  - research/findings/2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md
---

# Retiring the affect SALIENCE GATE, attempt 3 — the NAMED next rung (a fully-spiking, experience-bound opponent V+/V- population) ALSO fails at joint FP=0, and the sharpened diagnosis is that a TEXT-ONLY experience source cannot escape the register confound: the surpass needs a NON-TEXTUAL (embodied) US.

**Scaffold-retirement backlog rank 7** (`research/coordination/scaffold_retirement_backlog.md`). The production affect
appraisal has two halves: the per-word VALUE (RETIRED to a learned map, DR-2 6-seed GO; its ORIGIN self-organizes from
~10 innate primaries, DR-2b 6-seed GO), and the SALIENCE GATE — which words are allowed to move the mood at all — still
the host's fixed `abs(warriner_valence - 5.0) >= 2.0` threshold (`_STRONG_MARGIN`, `affect_production_organ.py:73`,
applied at `:118`). Two prior attempts to retire the GATE from a STATISTIC OF THE TEXT CO-OCCURRENCE GRAPH failed
(2026-08-12 D1: "unseparable by any gain or threshold"; 2026-09-05: four mechanistically-distinct statistics + a combo,
all BOUNDARY, best 29.4% recall at FP=0), with the sharpened diagnosis that **every statistic of that one graph inherits
the same REGISTER confound**, so the surpass is a DIFFERENT INFORMATION CHANNEL, not another graph statistic. This runner
builds the channel that BOUNDARY finding (and D1 before it) NAMED as the surpass — the fully-spiking, experience-bound
opponent V+/V- appraisal population — and de-risks it on the SAME 164-word benchmark. **Result: BOUNDARY, and worse than
the co-occurrence channel it was meant to beat.**

## The mechanism built (D1's own named next rung — the experience-bound spiking opponent)

Reuse-by-import only (no reimplementation, NO `sim/` edit, numpy-CPU — the established lane for the affect de-risks):

- **Concept CODE** = the self-organized PPMI stream-cortex code (`build_cooccurrence` -> `codes_from_cooccurrence`). [pre]
- **~10 INNATE primaries** (hug/hurt/cry/warm...) -> per-word evaluative-conditioning valence `s_c` (Rescorla-Wagner
  asymptote of co-occurrence with the primaries; `rescorla_wagner_valence` / `build_primary_cooccurrence`, Warriner-FREE). [CS<->US]
- **EXPERIENCE-CONDITIONED opponent weights** = `selforg_opponent_weights` (three-factor Hebbian outer-product over the
  learned code, rectified Namburi-Tye V+/V- split; Warriner is NOT an argument — asserted in the reused primitive). [SYNAPTIC]
- **SPIKING opponent bridge** = `build_bridge` (a `code_in` relay -> `appr_vplus`/`appr_vminus` pools that CROSS-INHIBIT
  via `xinh_vp`/`xinh_vm`; `OPP_TONIC_PA=0`, so the pools fire ONLY from the code FF). Read via `read_valence`. [the SPIKING population]

The KEY hypothesis that made this a genuinely DIFFERENT channel (not a re-badge of the refuted lever): the GATE / salience
is the TOTAL opponent DRIVE `rate(V+)+rate(V-)` (how much the affect system is activated at all), which is DISTINCT from
the VALENCE differential `rate(V+)-rate(V-)` the prior levers proxied (Namburi-Tye 2015: opponent populations code
valence by their balance, salience by total activation); and the read is a NONLINEAR, COMPETITIVE population response —
mutual inhibition should CANCEL the balanced/mixed drives of register-confounded neutral words while passing the clean
one-sided drive of genuine affect words. The gate was read HELD-OUT (a 2-fold cross-fit: each partition word read from a
W built on the OTHER fold's reinforced words, so no word reads its own conditioning), with a per-seed LABEL-FREE median
gain-control (a single threshold comparable across seeds).

## Benchmark + pre-registered GO bar (identical to the 2026-09-05 BOUNDARY finding's)

The SAME 164-word partition — Warriner words present in the full corpus AND all 6 bootstrap resamples: **102 raw-gated
"true affect" (`|v-5|>=2`) + 62 raw-excluded "neutral"** (reproduced exactly). Warriner is used ONLY to define the
partition and as the negative-control input — NEVER in the opponent weights. **G1** worst-case recall (min across 6
seeds) >= 0.5 at joint FP=0. **G2** that recall STRICTLY EXCEEDS the refuted co-occurrence-magnitude negative control on
the same words. **G3** (anti-hollow) the no-conditioning lesion collapses the separation. GO iff G1 AND G2 AND G3.

## Result (6-seed, `research/findings/raw/_affect_experienced_opponent_gate_6seed.json`) — BOUNDARY

<!--derived-->
Numbers below are read/rounded from the cited 6seed JSON (`gate_worst_case_recall`, `read_variant_worst_case_recall`,
`negctrl_worst_case_recall`, `conditioning_window_sweep`, `input_lesion_floor_max`, per-seed `corr_s_c_warriner`).

| read of the experience-bound spiking opponent | worst-case recall @ joint FP=0 (6-seed) |
|---|---|
| total opponent DRIVE `rate(V+)+rate(V-)` (the pre-registered salience read) | **0.000** |
| valence `|differential|` (the confounded-proxy alt read) | 0.000 |
| total drive with the arc's label-free relatedness (value⊥plausibility) control | 0.000 |
| **negative control: refuted co-occurrence learned-magnitude, naive threshold** | **0.000** |

- **G1 FAILS.** The best of three biologically-motivated held-out reads recovers **0.000** worst-case recall at joint
  FP=0 — to gate ZERO neutral words in every seed, the threshold must exclude every affect word too. (Below the 0.5 bar.)
- **G2 FAILS.** The naive co-occurrence-magnitude negative control also reads 0.000 here (a single magnitude threshold
  cannot achieve joint FP=0 across the full corpus + 6 resamples — matching the NAIVE row of the 2026-09-05 finding);
  0.000 does not STRICTLY exceed 0.000. And the experience-bound gate is far below the 2026-09-05 finding's BEST
  *elaborated* co-occurrence mechanism (D+B combined, **0.294** <!--derived--> quoted from that finding, not this run).
  So the experience-bound channel is **not a surpass — it is worse** than the co-occurrence channel it was to beat.
- **The read IS experience-driven (not hollow), it just doesn't separate.** The input-lesion floor is ~silent
  (max 0.0068) and the no-conditioning lesion (s_c:=0 -> weights collapse to 0) drives the opponent to ZERO for every
  word — so the intact drive (mean 0.089) exists ONLY because of the conditioning; the mechanism genuinely uses the
  experienced-affect channel. It is the channel's SEPARATING power that is absent, not the wiring.
- **The VALUE channel works while the GATE does not** — a clean dissociation. `corr(acquired s_c, Warriner)` is
  +0.42..+0.52 across the 6 seeds (the conditioning acquires honest signed valence, consistent with DR-2b), yet no read
  of the resulting opponent separates affect from neutral. VALUE (which sign) is learnable from this channel; SALIENCE
  (is it affect at all) is not.
- **The EMERGENT-ignition threshold gates everything.** At "drives the opponent above the silent floor", FP=62/62 at
  recall 1.0 in every seed — ALL 164 words (affect AND neutral) ignite the opponent. Neutral words DO drive it, because
  they live in the same register as the primaries.

## Diagnosis: a TEXT-ONLY experience source cannot escape the register confound — this is NOT a spiking operating-point artifact

The decisive control is the CONDITIONING-LEVEL window sweep (no bridge, numpy-only): across pairing windows 1-4 and the
label-free conditioning statistics {total pairing, `|s_c|`, one-sidedness `|n+ - n-|`, each per word-frequency}, the BEST
single-seed recall@FP0 is **0.088** (window 1-2, total-pairing-per-frequency) <!--derived-->. So the experience signal
itself — before any spiking read — does not separate the classes at ANY contiguity window. The spiking population,
mutual inhibition, and the total-drive-vs-differential distinction are all downstream of a signal that already lacks the
separation; they cannot manufacture it.

**Why the named channel inherits the SAME confound it was meant to escape.** The premise (the 2026-09-05 finding's, and
the task's) was that the experience-bound channel "does not read affect off text co-occurrence at all". But in a
TEXT-ONLY teaching stream, the brain's ONLY "experienced affective response" IS what the text co-occurrence drives —
there is no separate embodied US. The ~10 innate primaries (hug/hurt/cry/warm/cozy...) are themselves WORDS that live
INSIDE the same TinyStories narrative register as the neutral words: `cat` curls up `warm` and `cozy`; a `dragon` is in
the same dramatic (primary-rich) scenes as fear words. So conditioning-to-primaries is STILL a co-occurrence statistic —
just anchored to 10 words instead of 500 — and it inherits the register confound wholesale (neutral `cat`/`day`/`dragon`
have primary-pairing rates indistinguishable from affect `happy`/`joy`/`love`; several genuine affect words are so
promiscuous their per-encounter pairing rate is diluted below vivid neutral nouns). Anchoring narrower did not help; the
narrowing target is inside the register too.

## The genuine surpass (the boundary's next rung, not a stop): a NON-TEXTUAL experienced US

The residual, isolated: the failure is NOT the opponent population, the salience-vs-valence read, the contiguity window,
or the operating point — it is the EXPERIENCE SOURCE. Real amygdala/BLA valence conditioning (Namburi, Tye et al. 2015)
binds a cue to the animal's OWN affective response to a REAL unconditioned stimulus (a shock, a taste, a reward
delivery) — a NON-LINGUISTIC, embodied signal — not to which words a narrative places nearby. The genuinely different
channel the record keeps pointing to therefore needs the pairing to be driven by an EMBODIED / INTEROCEPTIVE US from the
world+body (the allowed host boundary), the way the production interoceptive-relay appraisal afferent already delivers a
bodily current to the ladder (board #84, `_appraisal_interoceptive_ladder_derisk`, production-default 2026-09-05) — a US
that is affective by construction and independent of the text register. Under a text-only teacher there is no such
signal, so the gate cannot be retired from the corpus stream by ANY mechanism; that is the sharpened, buildable
redirection this negative earns. (This proposal is fresh, not a refuted register mechanism.)

## What this means for the backlog item

Rank 7's gate-half remains **failed, now on a THIRD attempt** — and crucially the failure has moved OFF the
co-occurrence-graph family: the named "different channel" (experience-bound spiking opponent) was built and measured, and
it ALSO fails, for a reason that unifies all three attempts — in a text-only stream every affect signal (propagated
valence, any graph statistic, OR conditioning to primaries) is a read of the SAME narrative co-occurrence, whose register
frames ordinary words as consistently as emotion words. The production `_STRONG_MARGIN` threshold stays. The gate is a
genuine boundary of TEXT-ONLY affect learning; retiring it is now gated on an embodied experience source, which reframes
it as an EMBODIMENT rung rather than a lexicon/statistics rung.

## Honest residuals

1. **Not literally exhaustive.** Three held-out reads of the opponent (drive, differential, relatedness-residualized) +
   the conditioning signal at 4 windows were tested; a fourth read might squeeze marginally more. But all fail via the
   SAME diagnosed cause (the experience source is the register), and the conditioning-level control shows the signal is
   absent BEFORE any read choice — evidence against "a better read of the same channel" being the fix.
2. **numpy-CPU, rate-level conditioning map.** The opponent READ is spiking (off `cp_firing_states` through the
   cross-inhibiting population); the conditioning WRITE is the reused rate-level outer-product map. A fully-spiking
   three-factor WRITE is a further rung, but it would not change the diagnosis (the signal it would write is the same
   register-confounded conditioning signal).
3. **The embodiment claim is a REDIRECTION, not a measurement.** This finding proves the text-only channel fails and
   argues from biology (Namburi-Tye: conditioning binds to a real US) that the source must be embodied; it does not yet
   demonstrate an embodied-US gate works. That is the next build.
4. **The corpus is TinyStories.** A corpus with a less emotionally-resolved register might confound less; but the
   production teacher stream is this register, so the boundary is the operative one.

## Production wiring: NONE

`research/runners/affect_production_organ.py` and `webapp/wkv_mouth_generator.py` are byte-unchanged (this runner imports
from, and does not modify, the production organ). The fixed `_STRONG_MARGIN = 2.0` Warriner-norm threshold remains the
production salience gate. This file is additive-only; the controller decides any flip (there is nothing to flip — NO-GO).

## Anti-cheats

<!--derived-->
(numbers below restate values from the cited 6seed JSON, or quote the 2026-09-05 finding — see the Result table.)

- **Warriner-free weights (asserted, not commented):** the reused `selforg_opponent_weights` takes no Warriner argument
  and its assertion (corrupting `s_true` leaves the weights byte-identical) travels with it; Warriner enters ONLY the
  partition definition and the negative control.
- **Held-out reads (2-fold cross-fit):** no partition word reads its own conditioning (each read from the W built on the
  other fold), so recall is not inflated by a word re-reading its own `s_c`.
- **The refuted lever is reproduced as an explicit negative control** (the naive co-occurrence learned-magnitude, via
  `build_gate_features`), on the SAME words + seeds — it reads 0.000 here, and the 2026-09-05 finding's best elaborated
  co-occurrence mechanism (0.294) is quoted as the harder bar the experience-bound channel is below.
- **The conditioning-level window sweep** isolates the boundary to the experience SOURCE (0.088 best across windows 1-4,
  no bridge) — proving it is not a spiking operating-point artifact.
- **Load-bearing wiring proof for a negative:** input-lesion floor ~silent (0.0068) + no-conditioning lesion drives the
  opponent to 0 for all words -> the read is genuinely experience-driven; the separation, not the wiring, is absent.
- **`tools.lab.void_if`/`undefined_if_empty`** guard the partition size + the affect/neutral split so a degenerate build
  aborts loudly rather than reporting a hollow 0/0.

## Sources

<!--derived-->
(the 0.294 below is quoted from the cited 2026-09-05 co-occurrence BOUNDARY finding, not this run's artifact.)

- Namburi, P., Tye, K.M. et al. (2015, *Nature*), "A circuit mechanism for differentiating positive and negative
  associations" — opposing valence-coding BLA populations bound by conditioning to a REAL unconditioned stimulus
  (shock/reward), not to lexical company: the biological grounding for the opponent V+/V-, the salience(total)-vs-
  valence(differential) distinction, and the embodied-US redirection.
- Rescorla, R.A. & Wagner, A.R. (1972) — the associative-strength asymptote used for the conditioning `s_c`.
- `research/findings/2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md` — the co-occurrence-graph
  channel is register-confounded (best 0.294 at FP=0); named the experience-bound opponent as the surpass this tests.
- `research/findings/2026-08-13-affect-appraisal-origin-self-organizes-from-reinforcement-6seed-GO.md` (DR-2b) — the
  experience-bound conditioning map (concept->valence from ~10 innate primaries) reused here for the opponent weights.

## Reproduce

```
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_experienced_opponent_gate_derisk --smoke   # ~6s
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_experienced_opponent_gate_derisk \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/_affect_experienced_opponent_gate_6seed.json      # ~36s
```
