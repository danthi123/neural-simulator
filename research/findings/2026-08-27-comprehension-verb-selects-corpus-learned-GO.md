---
type: finding
status: live
lane: comprehension-scaffold-conversion
date: 2026-08-27
mechanism: comprehension-verb-selects-cue-lexicon
---

# Comprehension cue-lexicon — the VERB_SELECTS selectional-fit residual is LEARNABLE for OPEN vocabulary from real-corpus co-occurrence (6-seed GO, numpy label-propagation AND spiking-realized)

**Vikunja #175.** The direct SIBLING de-risk to the already-GO ANIMACY cue
(`research/findings/2026-08-26-comprehension-cue-lexicon-open-vocab-animacy-learnable-GO.md`, 6-seed GO,
mean_learned=0.837 vs shuffled=0.504 vs frequency=0.511 <!--derived-->). That finding's own wire-in
(`research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md`) explicitly declares the
VERB_SELECTS half a RESIDUAL: *"VERB_SELECTS stays the pre-existing hand-coded closed set (8 verbs) — no GO
artifact validates an open-vocab verb-selects cue, so claiming that conversion would be an overclaim."* This
finding is that GO artifact.

## What VERB_SELECTS actually encodes

`_phaseB_multicue_competition_spiking_derisk.py:64`: a per-verb dict `{"agent": "animate",
"patient": "animate"|"inanimate"}` for 8 hand-typed verbs (chase, eat, push, carry, bite, kick, grab, watch).
The AGENT slot is "animate" for every one of the 8 — the table never varies it, so it carries no discriminating
signal. The bit the table actually encodes is the PATIENT slot: does this verb select an ANIMATE direct object
(chase, watch) or an INANIMATE one (eat, push, carry, bite, kick, grab)? That patient-slot preference —
classical selectional association (Resnik, 1996) — is the open-vocab target learned here.

## Mechanism — same graph, reused code, one honest addition over the sibling cue

`research/runners/_comprehension_learned_verbselects_cue_derisk.py` reuses (by import, not reinvention)
`load_tokens`/`build_vocab`/`cooccur_ppmi`/`label_spread`/`shuffle_graph` from
`_comprehension_learned_animacy_cue_derisk.py` — the identical PPMI word-word co-occurrence graph over the
top-1500 content words of the real TinyStories corpus (nouns and verbs share ONE graph), the identical Zhou
label-spread, the identical shuffled-graph anti-cheat.

**Ground truth (new, disjoint from the hand table by construction).** 20 animate-patient verbs (hug, kiss,
help, feed, pet, scare, meet, thank, warn, comfort, teach, protect, save, forgive, trust, invite, visit,
rescue, nurse, marry) and 24 inanimate-patient verbs (drink, open, close, break, build, buy, wear, read, write,
paint, wash, clean, cook, bake, pour, drop, pick, plant, dig, fix, sell, wrap, cut, fly) — obvious,
uncontroversial common TinyStories-register transitive verbs, each checked for real corpus frequency (>=25
occurrences) and an unambiguous majority reading. An `assert` at import time enforces
`(GT_ANIMATE_PATIENT | GT_INANIM_PATIENT) & VERB_SELECTS.keys() == {}` — held-out evaluation is to verbs
GENUINELY never in the hand table, not interpolation inside it.

**The one honest addition, MEASURED not assumed.** A pilot seeding the propagation with ONLY the verb ground
truth reached held-out accuracy ~0.64 at window=4 — real signal (beats its own shuffled control) but below the
0.75 GO bar. The PRIMARY configuration additionally seeds the SAME propagation with the already-established,
independently-GO'd noun ANIMACY ground truth (`GT_ANIMATE`/`GT_INANIM`, imported verbatim from the sibling
cue's own module — a different, already-validated fact, not the verb-selects answer). This is the literal SVO
mechanism: a verb's patient class is recoverable from the animacy of the nouns it keeps distributional company
with (Resnik's selectional association), executed as one-hop label propagation on the shared word-word graph
rather than a separate PMI-weighted-average computation. Measured lift: noun+verb joint seeding reaches
mean_learned=0.952 vs the verb-seed-alone ablation's 0.690 (both reported with their citing artifact below, neither hidden <!--derived-->).

## Result — 6-seed GO (numpy label-propagation)

`research/findings/raw/_comprehension_learned_verbselects_cue_6seed.json` (window=4, k_seed=8, seeds
42/43/44/100/101/102):

<!--derived-->
| metric | mean(6 seeds) | per-seed range |
|---|---|---|
| learned (noun+verb joint seed, PRIMARY) | **0.952** | 0.929 – 1.000 |
| shuffled-graph control | 0.446 | 0.321 – 0.536 |
| frequency-only control | 0.315 | 0.250 – 0.393 |
| learned − shuffled | **+0.506** | — |

GO-gate (learned>=0.75 AND shuffled<=0.60 AND gap>=0.15): **GO**, all 6 seeds individually above 0.75.

**Honest controls.** (1) SHUFFLED-GRAPH — permuting the off-diagonal PPMI edges collapses accuracy to
0.32–0.54 (chance), confirming the signal is CORPUS-DERIVED, not smuggled through the (real, but seeded) noun
ground truth alone. (2) FREQUENCY-ONLY — 0.315, well below the learned accuracy and, if anything, mildly
ANTI-correlated (not merely at chance); patient-animacy preference is not a frequency artifact. (3) Held-out
verbs are disjoint from this run's seed subsample (standard no-leakage split) AND from the entire original
8-verb VERB_SELECTS table (asserted at import time, not merely claimed).

**Attribution ablation (honesty, not gated).** Verb-seed ALONE (no noun scaffold): mean_learned=0.690,
mean_shuffled=0.506 — real signal, beats its own shuffled control, but below the 0.75 GO bar on its own. Most
of the primary configuration's accuracy is carried by the (already independently-GO'd) noun-animacy scaffold
propagating onto verbs through their object-noun co-occurrence context, not by the small verb-seed subsample
itself — stated plainly rather than left implicit, per this repo's one-term-one-meaning / attribution
discipline. This does not weaken the GO: the noun scaffold is real, corpus-derived, independently-validated
prior knowledge (not the verb-selects answer being tested), and the shuffled-graph control still collapses this
exact noun+verb configuration to chance, proving corpus STRUCTURE — not seed richness alone — does the work.

## Result — spiking realization matches the rate/numpy read (6-seed GO)

`research/runners/_comprehension_learned_verbselects_cue_derisk.py --output ...` (same file, same run),
output `research/findings/raw/_comprehension_learned_verbselects_spiking_verify.json`. Reuses gap#3-A1's
already-validated F_anim/F_inanim coincidence-detector pools VERBATIM (`_gap3_spiking_feature_compat_derisk.py`
`_build`, the SAME 2-pool bridge the animacy cue's own spiking realization reuses) — the learned score's SIGN
drives one pool with a fixed current, the pools compete for 25 ticks, the WINNER (by firing rate) is the
classification; a word off the learned graph drives neither pool -> tie at 0 -> ABSTAIN.

<!--derived-->
| metric | mean(6 seeds) |
|---|---|
| spiking `classify()` accuracy on held-out verbs | **0.952** |
| shuffled-graph control (spiking read) | 0.560 |
| gap (spiking − shuffled) | **+0.393** |
| abstain rate on held-out verbs | 0.0 |

GO-gate: **GO**. Per-seed spiking accuracy (0.929, 0.929, 0.964, 0.929, 0.964, 1.000) is numerically IDENTICAL,
seed for seed, to the numpy label-propagation read — the spiking WTA readout loses no signal relative to the
offline `score > 0` sign it replaces (delta=+0.000 exactly, mean 0.952 vs 0.952).

**LESION check** (`set_lesion(True)`, re-verified immediately after the call, not merely asserted): 6 held-out
words classified pre-lesion (mix of animate_patient/inanimate_patient, zero abstains) all revert to `None`
post-lesion — coverage reverts. **Attribution**: `attributable_to` reports 100% of the held-out coverage
effect attributable to the F_anim/F_inanim coupling (treatment=+1, control=+0) — not an artifact of an
already-abstaining baseline.

**Residual — the shuffled-graph control is noisier on the spiking path (per-seed 0.464–0.679) than on the
numpy path (0.321–0.536).** Root cause (verified, not merely observed): `run_seed`'s `shuffle_graph` call uses
an RNG that has already been advanced by two prior list-shuffles (the verb seed/held-out split), while
`LearnedVerbSelectsLexicon`'s shuffle_control path instantiates a FRESH RNG at the same nominal seed value for
the graph permutation — a different point in the PCG64 stream produces a different permutation. **This quirk is
inherited unmodified from the already-GO'd animacy pattern**, not introduced here: the identical
run_seed-vs-Lexicon discrepancy was checked directly against `_comprehension_learned_animacy_cue_derisk.py` /
`_comprehension_learned_animacy_spiking.py` and reproduces there too (only 2 of 6 seeds match exactly,
0.4318–0.5682 range) — it simply happened not to threaten that cue's larger (n=44) held-out set's mean. The
MEAN-based gate is robust to it here as well (spiking shuffled mean 0.560, comfortably <=0.60), but it is a
genuine, minor residual worth unifying (same RNG draw order in both call sites) if this cue's spiking control
becomes load-bearing on a tighter bar.

## Honest verdict

**GO — the VERB_SELECTS vocab-ceiling residual named by both sibling findings is now closed at the DE-RISK
level**, mirroring the ANIMACY cue's own status exactly: a corpus-learned, open-vocab, held-out-verb-validated
cue exists (numpy GO), and it is spiking-realized via the SAME already-validated F_anim/F_inanim mechanism with
no signal loss (spiking GO, matching the rate version bit-for-bit on accuracy). Per `docs/TERMS.md`, this is
correctly described as **de-risked / GO at runner level** — it is NOT `wired` (no call path from
`webapp/server.py`'s `/api/brain-chat` reaches this module), NOT `on-by-default`, and the hand VERB_SELECTS
table is untouched in `comprehension_production_organ.py`. Per the task scope (Vikunja #175, DE-RISK ONLY), no
production wiring was attempted this session — that is the deliberate next rung, symmetric to how the ANIMACY
cue's own wire-in (`BRAIN_LEARNED_ANIMACY_CUE`, default OFF) followed its de-risk in a separate session.

**Next rung (not done here, scoped out by the task).** Wire `_animacy_of`-style choke points for the PATIENT
slot of `v in VERB_SELECTS` (`comprehension_production_organ.py:485,542` and the multicue competition's
`VERB_SELECTS` lookups) behind a new default-OFF flag (e.g. `BRAIN_LEARNED_VERBSELECTS_CUE`), following the
exact wire-in pattern `research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md`
already validated for animacy: byte-identical flag-off, load-bearing flag-on, lesion-reverts, and a moat check
on genuinely novel verbs.

## Sources / external grounding

Verb selectional preference recoverable from corpus co-occurrence: Resnik (1996), *Selectional Constraints: An
Information-Theoretic Model and its Computational Realization*. The propagation step is the same standard
graph label-spreading already cited by the sibling animacy finding (Zhou et al., 2004, *Learning with local and
global consistency*). `NO-EXTERNAL-NEEDED` beyond this: the mechanism class (PPMI graph + label-spread) and its
anti-cheat protocol are unchanged from the already-externally-grounded sibling cue; only the seed/eval word set
and the noun-scaffold addition are new, and the noun-scaffold addition is itself grounded in the SAME Resnik
citation (selectional association is literally "infer a predicate's argument preference from its distributional
company").

## Files

* `research/runners/_comprehension_learned_verbselects_cue_derisk.py` — new: `GT_ANIMATE_PATIENT`/
  `GT_INANIM_PATIENT` ground truth (disjoint-asserted from the hand `VERB_SELECTS`), `build_vocab_with_verbs`,
  `run_seed` (numpy label-propagation GO gate, `seed_nouns` toggle for the ablation), `LearnedVerbSelectsLexicon`
  (spiking realization reusing gap#3-A1's F_anim/F_inanim pools), `eval_seed_spiking`, and a combined `main()`
  that runs both stages from one invocation.
* `research/findings/raw/_comprehension_learned_verbselects_cue_6seed.json` (+ `.prov.json`) — the numpy
  label-propagation 6-seed GO artifact, including the verb-seed-only ablation numbers.
* `research/findings/raw/_comprehension_learned_verbselects_spiking_verify.json` (+ `.prov.json`) — the
  spiking-realization 6-seed GO artifact, including the lesion check and attribution readout.

## Residuals (declared)

* No production wiring (task scope: de-risk only). The hand `VERB_SELECTS` table is unchanged and remains the
  live production path.
* The spiking path's shuffled-graph control is noisier than the numpy path's (RNG-stream-order quirk inherited
  from the animacy pattern, detailed above) — the mean-based gate is robust to it, a per-seed floor is not.
* The AGENT slot of `VERB_SELECTS` (always "animate" across all 8 hand verbs) is not modeled — there was no
  discriminating signal to learn from the existing table on that slot, and no corpus evidence was sought for an
  agent-side selectional cue in this session. If a future verb genuinely selects an inanimate/abstract agent
  (e.g. metaphorical or causative-inchoative usage), this cue would not represent it; out of scope for #175.
* The ground-truth verb lists, like the sibling cue's noun lists, encode a majority reading of each verb's
  typical patient (e.g. "protect"/"save"/"visit" have real inanimate-object usage too) — real corpus text is
  not perfectly clean, and this is the same class of annotation noise the sibling animacy finding accepted for
  its own noun ground truth.
