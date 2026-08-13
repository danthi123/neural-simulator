---
type: finding
status: live
date: 2026-08-12
mechanism: affective-tagging
integration_faculty: affect-coloring
---

# D1: the production affect appraisal VALUE is now the DR-2 LEARNED distributional valence, not a hardcoded lexicon lookup (WIRED)

**Status:** WIRED (production-integration). Corrects the 2026-08-12 faculty-audit's #1 over-credit: the production AFFECT
organ (`research/runners/affect_production_organ.py`, wired into `webapp/server.py brain_chat`) read its per-word
appraisal VALUE from the raw Warriner-approximate seed lexicon while the code/docs mislabeled it "the DR-2 learned
per-word valence lexicon". The appraisal cognition between sensation and action was a Python dictionary lookup — the
host shortcut named by the audit. This wires the de-risked DR-2 learned valence as the VALUE source. Verified
SYNCHRONOUSLY numpy-CPU through the real `/api/brain-chat` handler.

Artifacts: `research/findings/raw/_affect_dr2_wiring_verify.json` (the wiring verify + reproduction numbers),
`research/findings/raw/_affect_distributional_tag_6seed.json` (6-seed DR-2), `research/findings/raw/_affect_learned_valence_map.json`
(the cached learned map).

## Verify-first assessment (honest): is DR-2 a genuine improvement, or a lateral host move?

DR-2 reproduces (`research/findings/raw/_affect_dr2_wiring_verify.json`): 6-seed held-out valence **r = +0.811**
(per-seed 0.765–0.831), smoke r = +0.806; permuted-graph collapses (−0.064) and shuffled-labels collapse (−0.144) —
the affect rides the LEARNED co-occurrence structure, not a lookup. On a LEAVE-ONE-OUT map (every word's valence inferred by label-propagation seeded from all OTHERS — no word
carries its own hand-assigned norm) the learned map agrees with the norm at **r = +0.907 / 97.4% SIGN** on the strong
words the production filter selects (only 2 near-zero edge flips, win/war — ambiguous in a children's corpus, both land
neutral). So the mechanism is a GENUINE, validated capability, and the per-word appraisal VALUE is now inferred from the
brain's own learned corpus structure rather than a hand-typed table — a real step in PROVENANCE toward brain-based.

**What remains HOST (declared, not faked):** (a) the affect-word SALIENCE GATE (which words move the mood) + the SEED
norms are still Warriner — DR-2 is SEEDED from them, so it does NOT retire the lexicon, it propagates it; (b) the
LEARNING is numpy PPMI + Zhu-Ghahramani label-propagation, NOT spiking; (c) the injection of the appraisal into the
ladder is still host. **The fully-spiking on-bridge opponent V+/V- appraisal population is the named NEXT RUNG** (the
DR-2 finding's own caveat a; the ledger's declared D1/E2 residual).

**Why a value-only swap, not a full drop-in (a load-bearing measured boundary):** distributional valence GENUINELY
bleeds moderate positive affect onto high-frequency action words — `sit`(learned +0.46), `run`, `jump`, `play`, and
`cat` acquire learned valence >= real affect words (`sad` learned +0.42) because in TinyStories they keep warm company
(sit-on-lap, play, cozy). This is real signal, not noise, and it is UNSEPARABLE from genuine affect by any gain or
threshold (sit > sad in learned magnitude). A full drop-in (learned value AND learned gate) would color a plain "what
does the cat eat" positive — breaking the neutral-fact invariant and the byte-identical smoke. So the norm-based
affect-salience GATE is kept (preserving neutral-default) and only the VALUE is sourced from the learned map. Honest
consequence: because learned ≈ norm on the gated affect words (97% sign), the observable coloring is ~unchanged — this
is a PROVENANCE step (values now learned-from-corpus), not a behavior change.

## What changed (additive, NO `sim/` edit; default-ON, `BRAIN_AFFECT_DR2=0` escape)

- `research/runners/_affect_distributional_tag_derisk.py` — added `build_learned_valence_map()` (composes the existing
  de-risked primitives: `build_cooccurrence` -> `codes_from_cooccurrence` -> `affinity_knn` -> leave-one-out
  `propagate`/`opponent_seed`; deterministic; a single global std-ratio gain restores the norm SPREAD, per-word
  sign/rank 100% learned) + a `--emit-map` CLI. NO reimplementation; `run_seed`/`main`/the de-risk path untouched.
- `research/findings/raw/_affect_learned_valence_map.json` — the cached 139-word learned valence/arousal map
  (gV=1.63, gA=2.26), emitted once, loaded by the organ (missing artifact -> raw-norm fallback).
- `research/runners/affect_production_organ.py` `appraise_text` — the affect-word SALIENCE GATE is unchanged (raw norm
  |v-5|>=2, so neutral queries stay neutral); the per-word VALUE is now `LEARNED_VALENCE[w]` (raw norm iff
  `BRAIN_AFFECT_DR2=0` or the word is not in the learned map). Downstream ladder + co-resident read UNCHANGED.

## Verify (SYNCHRONOUS, in-process real `/api/brain-chat`, numpy-CPU, stub renderer, 979s)

- **Affect coloring works, driven by DR-2:** POS induction ("I am so happy, full of joy and love") -> **level +2**,
  learned words happy/joy/love, ladder differential +0.0389, forthcomingness elaborations 2->3; recall under the held
  positive mood stays level +2. (NEG induction read level 0 here only because it followed the positive turn — the mood
  EMA blends down from +; from neutral, an isolated appraisal −0.60 reads level −2.)
- **Faculties UNREGRESSED (byte-identical, asserted by exact per-turn compare):** the (answer, abstained,
  recalled_svo, tone_level) tuple for teach/recall/abstain/anaphora is byte-identical (exact string compare, in the
  verify artifact) between DR-2 ON and OFF (recall "The dog chases cat.", abstain "I don't know…", anaphora "The cat
  eats fish.", all level 0, no affect words).
  The gate keeps neutral/factual turns at n_hits=0 under both, so recall/abstain/anaphora/D2/D4/E1/E2/curiosity/D5/D6
  are unaffected. (Unit proof: `appraise_text` with `BRAIN_AFFECT_DR2=0` == the pre-change raw Warriner lookup, byte
  for byte, over a word battery — since that is the only changed function, DR2-OFF == pristine HEAD through the handler.)
- **Flag-off reverts to the lexicon:** DR-2 OFF POS-induction valence = the raw norm mean (+0.485 EMA of +0.808) vs the
  learned +0.4697 (EMA of +0.783) — both level +2; the neutral/faculty turns byte-identical. `BRAIN_AFFECT_DR2=0` is the
  byte-identical oracle.
- **Lesion load-bearing:** `affect_out=0` (`BRAIN_AFFECT_LESION=1`) collapses the POS-induction from level +2 to level
  0, ladder differential -> 0.0, while the appraisal words are still detected — the neural READ-BACK is load-bearing;
  the matched fact ("The dog chases cat.") is byte-identical.

## Honest residuals (declared; ride existing burn-down items)

- The affect-salience GATE + the SEED norms remain Warriner (DR-2 is seeded from them — the lexicon is NOT retired, it
  is propagated). The learning is numpy, not spiking. The injection into the ladder is host.
- **Next rung:** a fully-spiking on-bridge opponent V+/V- appraisal population (reads the message, produces the valence
  via neurons) — that moves the LEARNING onto the substrate; this cheap slice moves only the VALUE's provenance.
- The observable coloring is ~unchanged (learned ≈ norm at 97% sign on the gated words) — a provenance win, measured.

## Repro

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._affect_distributional_tag_derisk \
    --emit-map research/findings/raw/_affect_learned_valence_map.json          # rebuild the learned map (deterministic)
# default-ON in /api/brain-chat; BRAIN_AFFECT_DR2=0 -> raw-norm value (byte-identical oracle);
# BRAIN_AFFECT_LESION=1 -> affect_out=0 collapses the coloring.
```
