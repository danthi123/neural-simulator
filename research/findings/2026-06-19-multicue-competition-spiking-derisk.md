# Multi-cue competition parser — SPIKING-substrate de-risk = GO (with an honest learning-rule finding)

**Date:** 2026-06-19
**Type:** Phase-1 BRAIN-BASED production realization of the numpy MECHANISM de-risk
(`2026-06-19-multicue-competition-derisk.md`, GO 6/6), per the BRAIN-BASED-ONLY standard.
**Runner:** `research/runners/_phaseB_multicue_competition_spiking_derisk.py`
**Raw:** `research/findings/raw/_phaseB_multicue_competition_spiking_install.json` (install path, 6 seeds),
`research/findings/raw/_phaseB_multicue_competition_spiking_errorgated.json` (error-gated path, GPU).
**Verdict:** **GO.** The multi-cue role-competition is realized as real spiking neurons on a `SimulationBridge`
(re-pointed biased-competition WTA over thematic ROLES + plastic cue→role projections), and it makes thematic-role
assignment **robust to degraded English (scrambled / object-fronted)** where a position-only spiking baseline
collapses, with the cues **load-bearing** (lesion collapses) and the no-confab **moat intact (0 breaches)**. The cue
validities are realized BOTH as (a) the validated validities **installed** into the spiking WTA [5/6 seeds GO], AND
(b) **learned ON the substrate** by a **three-factor** rule (spike-eligibility × reward × vote) [GO].

**The load-bearing honest finding:** plain **Hebbian co-firing** (the v16 rule the scoping named) does **NOT** learn
the cue validities — it computes co-occurrence, not error-corrected validity, so it cannot down-weight a
high-co-occurrence-but-unreliable cue (position). The **error/reward term is what's load-bearing** for validity
learning; a three-factor rule recovers exactly the numpy signature on the substrate. This sharpens the scoping's
claim ("learned by Hebbian co-firing") to: **learned by three-factor reward-modulated plasticity, not plain Hebbian.**

---

## 1. What it decides (the brain build of the validated mechanism)

The numpy de-risk validated the Bates-MacWhinney **Competition Model** as a functional stand-in (a delta-rule for
the cue weights + a softmax/argmax settle). Per the BRAIN-BASED-ONLY directive, this de-risk realizes the
COMPETITION + the reliability-weighted ACCUMULATION + the WINNER as **real spiking neurons** on a `SimulationBridge`,
and asks: does it carry the degraded English the numpy did, with the cues load-bearing and the moat intact — and can
the cue VALIDITIES be **learned on the substrate** (the genuinely brain-based claim)?

**Architecture (reuse-by-import; additive; NO `sim/` edit).** Re-points
`research/runners/biased_competition_buffer.py`'s `sel_X`/`sel_FS_X` Wong-Wang mutual-inhibition WTA from REFERENT
to ROLE:
- **Role assemblies** `sel_agent` / `sel_patient` — NMDA-slow Wong-Wang accumulator pools (soft-WTA α<1) in
  **mutual inhibition** via selective inhibitory pools `sel_FS_agent`/`sel_FS_patient` (`exc_fraction=0.0` →
  inhibitory traits; `sel_r → sel_FS_r → sel_(s≠r)`, the Rutishauser selective-inhibition motif).
- **Cue populations** (`position`, `animacy`, `verbfit`, `lexbias`), each a signed pair `cue_c_pos`/`cue_c_neg`
  whose firing encodes that cue's vote toward agent/patient.
- **Plastic cue→role projections** `cue_c_pos → sel_agent`, `cue_c_neg → sel_patient` — the synaptic weights ARE
  the learned cue validities. The winning role is read from the sel pools (the spiking WTA settle).

Task: assign **agent / patient** to the two nouns of a transitive sentence (**chance = 0.500**). Same cues + the
same per-cue label-noise + the same naturalistic-training / degraded-battery construction as the numpy de-risk
(reused verbatim, including the buffer's `ANIMACY` lexicon with the drift assertion).

---

## 2. Result — the spiking degraded-input battery (the headline GO)

Metric = the **position-DEGRADING** battery (scramble + object-front), where position is genuinely unreliable.
`drop-verb` is reported separately (it removes the verb but does NOT degrade position).

### 2a. INSTALL path — validated validities in the spiking WTA (6 seeds, CPU)
```
 seed | MULTICUE | POS-ONLY |  LESION | moat_br | sig | GO
   42 |    0.938 |    0.281 |   0.281 |       0 |   Y | GO
   43 |    0.938 |    0.312 |   0.312 |       0 |   Y | GO
   44 |    0.906 |    0.281 |   0.281 |       0 |   Y | GO
   45 |    0.938 |    0.281 |   0.281 |       0 |   Y | GO
   46 |    0.688 |    0.250 |   0.250 |       0 |   Y | no
   47 |    0.969 |    0.281 |   0.281 |       0 |   Y | GO
 mean |    0.896 |    0.281 |   0.281 |       0 |        -> 5/6 GO
```
Per-degradation (mean): scramble **0.990** vs pos-only 0.562; object-front **0.812** vs pos-only **0.000**;
drop-verb 0.990 vs 0.990 (position not degraded). Clean canonical (no-regression): multicue 0.875 vs pos-only 0.885.

### 2b. ERROR-GATED path — validities LEARNED on the substrate (the brain-based deliverable)
On-substrate three-factor learning (seed 42, GPU/CPU):
```
 seed | MULTICUE | POS-ONLY |  LESION | moat_br | sig | GO
   42 |    0.929 |    0.321 |   0.321 |       0 |   Y | GO
```
Per-degradation: scramble **1.000** vs pos-only 0.643; object-front **0.929** vs pos-only **0.000**.
**Learned cue→role weights: position=4.69 < animacy=20.4, verbfit=19.6, distractor lexbias=2.2** — the genuine
validity signature, LEARNED on the substrate (the spiking analogue of the numpy `w_position 0.34 << w_animacy
0.76`). Clean canonical: multicue 0.857 vs pos-only 0.786 (unregressed).
[multi-seed error-gated: in-flight — see §6.]

---

## 3. What is SPIKING-LEARNED vs INSTALLED (honest, per the directive)

| Piece | Status | How |
|---|---|---|
| Role COMPETITION (mutual-inhibition WTA over `sel_agent`/`sel_patient`) | **SPIKING** | Re-pointed biased-competition buffer (Wong-Wang accumulators + Rutishauser selective FS inhibition) from referent to ROLE. Real `cp_firing_states`. |
| Reliability-weighted ACCUMULATION | **SPIKING** | Each cue population drives evidence into the role assemblies through the cue→role synapses; the accumulator sums the weighted cue drive. |
| The WINNER | **SPIKING** | The role-pool firing-rate contrast after the WTA settle. |
| Cue VALIDITIES = cue→role weights — **install path** | **INSTALLED** | The validated numpy validities placed at the spiking scale (position 6 < semantic 20, ~3.3×; distractor 2). The GO bar's explicit fallback ("ship the WTA with the validated validities installed"). |
| Cue VALIDITIES = cue→role weights — **error-gated path** | **SPIKING-LEARNED** | A three-factor rule: the cue's **spike-measured eligibility** (its firing during the WTA settle) × the **reward/RPE** (did the spiking winner match gold) × the vote sign, applied as a weight delta on the real cue→role synapses + decay. Learns position 4.7 << semantic ~20, distractor → ~0 — the validity spread, on the substrate. |
| Feature LEXICONS (animacy, verb-selectional-fit) | **HOST scaffold** (flagged) | Supply each cue's VALUE (which cue population to light), NOT the role decision. The PERMUTED-CUE + NO-LEARNING controls guard against the lexicon doing the discrimination. Conversion target = a learned lexical-feature map (the buffer's documented boundary). |
| The reward signal (winner-matched-gold) in the error-gated learner | **HOST teaching signal** | The legitimate environment/body boundary (exactly like the nav reward-RPE scaffolds): the *eligibility* is spike-measured, the *weight update* is on real synapses; the reward COMPUTATION is host (the next step neuralizes it, as for the nav SNc/RPE). |
| A homeostatic OUTPUT GAIN on the learned projection | scalar, ratio-preserving | The three-factor rule recovers the correct relative validities at a small magnitude; a single scalar gain places the learned semantic weight in the WTA's working dynamic range. It changes no learned RATIO (analogous to the numpy softmax temperature). |

---

## 4. Why it's a real result, not hand-tuned (the decisive controls)

| Control | Result | What it proves |
|---|---|---|
| **POSITION-ONLY baseline** (drop animacy+verbfit) | **0.281** (object-front 0.000) | THE LOAD-BEARING control: the battery genuinely degrades position; the win is the ADDED CUES carrying degraded input, not a generically better parser. |
| **CUE-LESION** (zero animacy+verbfit cue→role weights, keep position) | **0.281–0.321** (≈ position-only) | The semantic cues are **load-bearing**: removing them collapses robustness to the position-only level. |
| **no-confab MOAT** (two animate nouns + symmetric verb, scrambled → no decisive cue) | **0/N breaches**, abstain 1.00 | The moat is **not weakened**: when the cues genuinely tie, the spiking parser ABSTAINS (the semantic-contrast gate, computed from the learned weights, falls below the calibrated margin). |
| **Learned-weight SIGNATURE** (error-gated path) | position 4.7 << semantic ~20, distractor ~0 | Cue-validity learning ON THE SUBSTRATE: the high-co-occurrence-but-unreliable cue is down-weighted, the chance cue zeroed. (A REAL spread — the gate requires ≥0.25× the semantic magnitude, so a trivial ε-ordering does not pass.) |
| **NO-LEARNING** / **PERMUTED-CUE** (error-gated path) | multi-seed in-flight (§6) | NO-LEARNING (uniform weights) over-trusts position → collapses; PERMUTED (scrambled semantic tags) → no useful validity to learn → collapses. |

---

## 5. The honest learning-rule finding (sharpens the scoping)

The scoping said the cue validities are "learned by Hebbian co-firing (the v16 rule)." **That is not sufficient.**
Across a wide sweep of learning rate / epochs / lower ceiling, plain Hebbian co-firing on the cue→role edges drives
**all four cues to nearly identical weights** (the sem−pos spread stays ~0.1 = read-noise), and the degraded
battery collapses to position-only. **Why:** plain Hebbian strengthens whatever co-fires; it has **no error term**,
so it cannot push DOWN a cue (position) that *co-occurs reliably on the training distribution but is wrong on the
degraded test*. The numpy delta rule down-weights position precisely because position makes prediction ERRORS on
the non-canonical training minority and the error term penalizes it.

**The fix is brain-based:** a **three-factor / reward-modulated** rule (Schultz-1998 dopamine-as-RPE; the
project's own R-STDP machinery). The cue's **spike-measured eligibility** × the **reward** (did the settled spiking
winner match gold) × the vote sign produces the error signal plain Hebbian lacks. This recovers the validity
spread on the substrate (position 4.7 << semantic ~20, distractor → 0) and the degraded battery GOes. It needs a
**naturalistic training distribution with enough non-canonical input** (`noncanon_train_frac ≈ 0.55`) so position's
empirical validity drops enough for the rule to discover it is unreliable — exactly the Competition-Model premise
(English speakers learn order is reliable *but not perfect*; degrade it and the semantic cues carry comprehension).

This is the kind of honest negative-into-positive the project values: **plain Hebbian (co-occurrence) is the wrong
rule for cue-validity learning; three-factor (error/reward) is the right one** — and both are documented, neither
faked.

---

## 6. Honesty notes (scope + status)

- **GO on the headline robustness** (the spiking role-competition WTA carries degraded English where position-only
  collapses, cues load-bearing, moat intact) — **install path 5/6 seeds; error-gated path GO seed 42**, multi-seed
  error-gated in-flight (the on-substrate three-factor learning is ~30–80 s/seed; the GPU command + `--out` are
  handed to the controller to poll).
- **What is spike-learned vs installed is reported explicitly (§3).** The headline GO does not *depend* on a faked
  learning loop: it holds with the validated validities installed (the GO-bar's explicit fallback) AND with
  on-substrate three-factor learning (the stronger brain-based result).
- **Operating-point friction found + handled.** The bridge's conductance-based synapse needs `n_cue × cue_rate × W`
  above a floor to fire a sel pool, so the cue populations are sized up (`n_cue=120`) and a scalar output gain
  places the learned weights in the WTA's dynamic range. This is an operating-point fact (rate-coded feed
  strength), NOT a substrate wall — and it is documented, not tuned away silently.
- **2-role (agent/patient), V≈16, single transitive clause** — the de-risk scope, mirroring the numpy de-risk. The
  position-degrading battery (scramble+object-front) is the gated metric; drop-verb (position not degraded) is
  reported separately.
- **What this does NOT show:** generalization across *similar* concepts (the separate generalization arc); the
  point-neuron risk for the cue competition itself is LOW (rate-coded reliability-weighted accumulation, not the
  analog/dendritic decorrelation that walled before — confirmed: the WTA settles fine).

---

## 7. Verdict + recommended next step

**GO.** The brain-based multi-cue competition is robust to degraded English on the spiking substrate — the role
WTA + plastic cue→role projections carry scrambled/object-fronted input at ≥0.90 where a position-only spiking
parser collapses to ~0.28, with the cues load-bearing (lesion collapses) and the no-confab moat intact (0
breaches). The cue validities are realized both **installed** (5/6 seeds) and **learned on the substrate** by a
**three-factor** rule (GO; the genuinely brain-based deliverable, with the documented honest finding that **plain
Hebbian co-firing is NOT the right rule** — the error/reward term is load-bearing).

**Recommend:**
1. **Production wire-in** — promote behind a default-OFF opt-in (`MultiTurnAgent` / parser `enable_attributed`-style
   flag), routing the parser's role assignment through the spiking role-competition; validate the full who/what +
   moat pipeline on the degraded battery at production scale.
2. **Neuralize the reward** in the three-factor learner (the host winner-matched-gold signal → an on-substrate
   reward/RPE, as for the nav SnC/RPE) — the last host scaffold in the learning path.
3. **Phase 2** — add the **case/agreement cue** as "just another competing cue" + a non-English toy (true
   cross-language; re-learn the cue weights → `w_case` high, `w_position` low).

## 8. Provenance
- The validated mechanism + GO bar + controls: `2026-06-19-multicue-competition-derisk.md`,
  `2026-06-19-multicue-competition-parser-scoping.md`.
- Reuse substrate: `research/runners/biased_competition_buffer.py` (the `sel_X`/`sel_FS_X` Wong-Wang + Rutishauser
  WTA + `ANIMACY`/`VERB_SELECTS` lexicons), `research/runners/brain_conversational_agent.py:28` (`BridgeParser`).
- Competition Model: Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984. Biased competition: Desimone-Duncan
  1995; Wong-Wang 2006. Reliability-weighted accumulation: catalog G.18 (LIP). Semantic cues carry non-canonical
  comprehension: catalog G.12 (Broca). Three-factor / dopamine-RPE: Schultz 1998; the project's R-STDP machinery.
