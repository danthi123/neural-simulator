---
type: finding
status: contributing
date: 2026-08-26
mechanism: dual-route-morphology
lane: E-language
artifacts:
  - research/findings/raw/morph_recog_6seed/recognition_gated_6seed.json
---

# E·Language: recognition-THRESHOLD-on-the-blocking-interneuron does NOT rescue the seed-fragile dual-route rule — 6-seed NO-GO (REFUTED lever, banks the negative)

<!--derived-->
**One-line verdict (the runner's own, verbatim):** `NEGATIVE -- no op-point does BOTH (both 0/6; reg 0,
irr 0) -- sweep --recog-threshold-mV / --inh-drive`. The recognition-gate lever for the seed-fragile procedural
rule in the dual-route (Pinker–Ullman words-and-rules) past-tense de-risk resolves to **NO-GO on all 6 seeds**,
and its OWN gate-selectivity anti-cheat is what refutes it. Raising the di-synaptic blocking interneuron's spike
threshold (the "recognition/familiarity gate" the linear relay had replaced with a constant) does **not** make
blocking selective: the pooled interneuron fires ~equally for irregular vs novel cues on every seed, so no
threshold separates genuine whole-form retrieval from the tonic entrenched-attractor floor. This CONFIRMS at 6
seeds the 1-seed smoke (seed 43) recorded on the source branch. The residual is UPSTREAM (the tonic floor, to be
removed at source by spike-frequency adaptation — named, NOT built here).

Artifact: `research/findings/raw/morph_recog_6seed/recognition_gated_6seed.json` (backend numpy/CPU; base seed 42,
`--n-seeds 6`; the runner sweeps `base..base+n_seeds-1`, i.e. seeds 42–47, which includes the smoke seed 43).
`inhib_strength=6.0, inh_drive=6.0, recog_threshold_mV=8.0, n_lex=2000, n_proc=800`.

## Result — 6 seeds {42,43,44,45,46,47}

<!--derived-->
| read-out | 6-seed | gate | reading |
|---|---|---|---|
| **reg_acc** (novel/held-out rule generalization) | 0/6 ≥0.90 (0.875 every seed) | ≥0.90 | **FAILS** — the rule is not the robust default (one novel stem, `kick`, is captured by `went`) |
| **irr_acc** (irregular blocking) | 0/6 ≥0.85 (0.00 every seed) | ≥0.85 | **FAILS** — the raised threshold silences blocking entirely: every irregular over-regularizes to "-ed" |
| **both_gates** (rule AND blocking on the SAME brain) | 0/6 | ≥5/6 | **FAILS** |
| **gate_selective** (anti-cheat 4: inh fires for irr, silent for novel) | 0/6 | selective | **NOT selective on any seed** — the refutation |
| **full-GO** (runner verdict) | 0/6 | ≥5/6 | **NO-GO** |

Per-seed gate read-out — RAW interneuron firing magnitudes (irr vs novel; not a bare ratio, per the `selective`
term condition): seed42 17.56/17.51, seed43 16.33/15.68, seed44 16.31/16.29, seed45 16.99/16.73, seed46
16.64/16.47, seed47 16.96/16.93. Every pair is ~1.0× (max 1.04×), far under the 2.0× the gate requires to count
as selective. reg_acc, irr_acc, overreg_rate_lesion, permuted_binding_irr_acc are identical across all 6 seeds
(0.875 / 0.00 / 1.00 / 0.00) — the failure is structural, not stochastic.

## What refuted the lever (its own instrument caught it)

<!--derived-->
The lever adds a recognition THRESHOLD on the Dale-compliant di-synaptic blocking interneuron
(whole-form(exc)→interneuron(inh)→affix): a real GABAergic interneuron should fire — and block the default "-ed"
— only on supra-threshold (genuinely recognized) whole-form retrieval. Anti-cheat 4 measures whether the gate is
load-bearing: the interneuron pool must FIRE for irregular cues and be (near-)SILENT for novel-stem cues.

It is not. The interneuron POOLS over all 7 stored whole-forms, and the most-entrenched irregular attractors
(went/ran) leave a stem-INDEPENDENT ~0.20 cosine FLOOR in the LEX readout — present even for held-out regulars
never co-encoded with any whole-form. That floor keeps "some whole-form active" true for EVERY cue, so there is no
genuine-vs-spurious firing-rate difference to threshold on (inh_fire_irr ≈ inh_fire_novel on all 6 seeds). Raising
the threshold high enough to silence the spurious floor also silences genuine retrieval, so blocking collapses
(irr_acc 0/6) without robustly fixing the rule (reg_acc still 0/6 at ≥0.90). The gate-selectivity anti-cheat reads
NOT selective on every seed → the runner's verdict is NO-GO.

## Anti-cheats / controls (verbatim, and which are informative here)

<!--derived-->
The artifact carries a guarded `preconditions` block (`tools.verdict.Verdict` → `gates/verdict_preconditions`);
`verdict_status = NO-GO` is EARNED, not asserted. Both preconditions hold:
- **Anti-cheat 5 (substrate seeded by `cfg.seed`): PASSES** — `seed_check.seeds_substrate = True`
  (same_seed_identical True, cross_seed_differs True). This NO-GO is not the 2026-07-17 unseeded-substrate
  confound.
- **Gate read-out is LIVE (discriminating power for the refutation): PASSES** — mean interneuron firing
  `inh_fire_irr 16.80 / inh_fire_novel 16.60`, both far above silence and unsaturated (window 40), so
  `gate_selective = False` is a *measured equality*, not a dead read-out.
- **Anti-cheat 4 (recognition-gate load-bearing): the DECISIVE, UNCONFOUNDED control, and it collapses as
  designed** — `gate_selective = False` on 6/6 (raw firing above). This is the control that refutes the lever:
  a firing interneuron does not discriminate irregular from novel cues.

Two other built-in controls are DEGENERATE in this refuted regime (reported for completeness, not relied on):
- **Anti-cheat 3 (permuted stem→whole-form binding): DEGENERATE** — `permuted_binding_irr_acc = 0.00` on 6/6,
  but the INTACT `irr_acc` is already 0.00 (the raised threshold zeroed blocking pre-permute), so there is no
  intact→permuted contrast to read.
- **Lesion the LEX store → over-regularization: DEGENERATE** — `overreg_rate_lesion = 1.00` on 6/6, but the
  pre-lesion "-ed" rate is ALSO 1.00 (blocking already broken), so the lesion→over-reg gap is 0.

**Why `irr_acc` is NOT the discriminating-power anchor (honest caveat).** `irr_acc = 0.00` on all 6 lever seeds
is at floor, and a matched threshold=0 baseline does NOT lift it off floor at this operating point: a direct
probe (seed 42, `recog_threshold_mV=0`, the linear-relay two-pool baseline) reads `irr_acc = 0.00` at BOTH
`inh_drive` 6.0 and 3.0 — i.e. blocking is itself SEED-FRAGILE here (0.00 at seed 42; ~0.857 at seed 43 per the
source-branch smoke), consistent with the 2026-08-01 two-pool NO-GO (blocking seed-fragile). So `irr_acc = 0`
under the lever is a CONFOUNDED floor (part lever, part pre-existing baseline fragility), and the clean,
unconfounded refutation is the gate-selectivity read-out on the live interneuron firing — which is why the
verdict's discriminating-power precondition is anchored there, not on `irr_acc`.

## Honest scope + next (capability NOT abandoned — a verdict on the METHOD)

<!--derived-->
This is a NO-GO for the recognition-THRESHOLD-ON-THE-INTERNEURON method, not for the dual-route capability. It
advances the 2026-08-01 dual-route NO-GO (declarative route 6/6; procedural rule seed-fragile, reg_acc 0.25–1.0)
by localizing the seed-fragility to a SOURCE-side defect: the tonic entrenched-attractor floor in the LEX readout.
A downstream threshold cannot fix an upstream floor that is stem-independent and pooled — silencing the floor
silences genuine retrieval too.

The next mechanism is NAMED in the biology binding and is NOT built here: SOURCE-side floor removal via
**spike-frequency ADAPTATION (AHP / M-current)** to quench persistent attractors, so an entrenched whole-form
fires only transiently on its OWN cue rather than tonically for every cue — restoring the genuine-vs-spurious rate
difference a recognition gate could then act on. Open next step; do NOT re-attack the threshold lever (its
refutation is banked here at 6 seeds).

Biology binding: `research/biology/dual-route-past-tense-recognition-gated-blocking.md` (Kandel 6e Ch 55 p.1373,
"go becomes went rather than goed"; Pinker–Ullman 2002 words-and-rules; Marcus 1992). Runner:
`research/runners/_productive_morphology_recognition_gated_derisk.py`. NO `sim/` edit; numpy pool-portable;
substrate seeded by `cfg.seed`.
