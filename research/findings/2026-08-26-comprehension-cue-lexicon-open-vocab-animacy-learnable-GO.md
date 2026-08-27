---
type: finding
status: live
lane: comprehension-scaffold-conversion
date: 2026-08-26
mechanism: comprehension-cue-lexicon
---

# Comprehension cue-lexicon — the ANIMACY vocab-ceiling is LEARNABLE for OPEN vocabulary from real-corpus co-occurrence (6-seed GO)

**The scaffold picked.** The PI-ledger declares the SAME host scaffold on five comprehension organs — D4
comprehension-monitor, D6 multiref-WM, D3 discourse-register, T1-6 other-repair, D2 surprise — each verbatim:
*"VOCAB CEILING: the cue lexicon (ANIMACY / VERB_SELECTS) is the toy 2-noun transitive scope"*
(`research/runners/comprehension_production_organ.py:41`). It is the single most-shared, highest-leverage
comprehension scaffold: `fully_covered = (v in VERB_SELECTS) and (n0 in ANIMACY) and (n1 in ANIMACY)`
(`comprehension_production_organ.py:287`) gates the monitor's COMPETENCE on membership in a **19-noun / 8-verb**
hand-coded table. Any real word outside it is OUT OF SCOPE, so the substrate cannot judge open-vocab sentences.
Each row names its own removal path as *"a LEARNED cue lexicon [is] the next rung"*.

**What was already known (and its gap).** gap#3-A1 (`2026-07-18-gap3-A1-learned-feature-compatibility-cheap-first-GO.md`)
proved animacy×selectional-preference is corpus-DERIVABLE and even spiking-realised it (two feature pools F_anim/F_inanim
+ coincidence, 6-seed GO, permuted collapse; `tests/test_gap3_spiking_feature_compat.py` 7/7) — **but only (a) for the
referent-BIAS, not comprehension, and (b) on a CLOSED synthetic corpus generated FROM the ground-truth table, evaluated
on the SAME vocabulary.** The genuinely-OPEN question for comprehension — can a learned cue assign animacy to HELD-OUT
words it was never labelled, from REAL text, well enough to lift the vocab ceiling — was never run.

## Result — 6-seed GO
`research/runners/_comprehension_learned_animacy_cue_derisk.py` (numpy). Same mechanism class as the production affect
DR-2 organ (label-propagation over the brain's learned co-occurrence graph): build a PPMI word-word graph over the
top-1500 content words of the REAL TinyStories corpus (3.9M tokens), seed **8 animate + 8 inanimate** obvious words,
Zhou label-spread, read each HELD-OUT word's propagated sign. Held-out words are never labelled (eval ground truth only).

Primary artifact: `research/findings/raw/_comprehension_learned_animacy_cue_6seed.json` — `mean_learned`
0.8371212121212123, `mean_shuffled` 0.5037878787878788, `mean_frequency` 0.5113636363636364 (window=4, all 6 seeds).
The rounded table below and the window=2 / smoke rows are computed off the same runner (separate window settings,
not re-saved as artifacts):

<!--derived-->
| config (6 seeds: 42,43,44,100,101,102) | learned | shuffled-graph | frequency-only | verdict |
|---|---|---|---|---|
| window=4 (primary, artifact) | **0.837** | 0.504 | 0.511 | GO |
| window=2 (more syntactic) | **0.913** | 0.504 | 0.511 | GO |
| 2M-char slice (smoke) | 0.716 | 0.489 | 0.534 | signal present, data-bound |

All 6 seeds ≥ 0.75 at window=4. **GO-gate:** learned≥0.75 AND shuffled≤0.60 AND (learned−shuffled)≥0.15 → GO
(gap = +0.333 <!--derived-->).

**Honest controls (all pass).** (1) SHUFFLED-GRAPH — permute the off-diagonal PPMI edges (destroy the real
co-occurrence structure), same seeds/held-out words: collapses to **0.504** <!--derived--> (chance) → the signal is
CORPUS-DERIVED, not smuggled through the 16-word seed set. (2) FREQUENCY-ONLY — predict from raw frequency: **0.511**
<!--derived--> (chance) → animacy is not a frequency artifact. (3) No label leakage — seed and held-out sets are disjoint. The corpus structure
(which word is animate) is NOT injected; it is real distributional English. This is **not a hand rule in a spiking
costume** — it is a learned distributional cue beating a no-learning control.

## Honest verdict — the scaffold conversion is PARALLELIZABLE-NOW; full open-ended comprehension is EMERGENCE-GATED

**Sources / external grounding.** The learnability half rests on established distributional semantics: verb
selectional preference is recoverable from corpus co-occurrence (Resnik, 1996, *Selectional constraints*); animacy is
a robustly distributional feature (Zaenen et al., 2004, animacy annotation; Bowman & Chopra, 2012); the propagation
step is standard graph label-spreading (Zhou et al., 2004, *Learning with local and global consistency*). In-repo,
gap#3-A1 (2026-07-18) already validated the same cue on a closed corpus and spiking-realised it.
`NO-EXTERNAL-NEEDED:` the boundary statement below (beyond-transitive comprehension needs a learned parse) is not a
newly-banked capability limit — it RESTATES the PI-ledger's already-declared "Fixed grammar frames" / "learned
temporal language sequencing" scaffold and the mouth's emergence class; this finding banks a POSITIVE result (a GO),
not a wall.

**GO (parallelizable-now, no new deep-engine dependency):**
- The animacy vocab-ceiling named on all five comprehension organs is **not intrinsic to the substrate** — the
  noun-animacy cue generalises to open vocabulary from co-occurrence (mechanism GO above).
- The SPIKING realisation is a **known, already-validated mechanism** (gap#3-A1's F_anim/F_inanim pools + coincidence,
  6-seed GO). Extending it from the closed referent-bias corpus to the open comprehension cue is the same class of
  build, not new emergence risk.
- Remaining bounded builds (all parallelizable-now): (i) the VERB-selects half (gap#3-A1 learned it jointly; extend to
  real corpus by the same EM/propagation over approximate subject/object co-occurrence); (ii) wire the learned map in
  place of the `ANIMACY`/`VERB_SELECTS` lookups (`comprehension_production_organ.py` lines 122/134/141/143/287 …) as
  the cue value into `SpikingRoleCompetition`, keeping the 6-seed D4 GO + moat + byte-identical `BRAIN_COMPREHENSION_GATE=0`
  escape; (iii) a graded/near-threshold open-vocab comprehension battery to recalibrate the competence gate off the
  learned confidence rather than table-membership.

**The genuine emergence bar this does NOT lift (deep-engine-gated).** The comprehension monitor is still a
**3-content-token TRANSITIVE role-binder**; the animacy+verbfit cue resolves who-did-what-to-whom for a single
transitive clause. Full open-ended comprehension — arbitrary syntax (embedding, relative clauses, coordination),
abstract/non-animacy-contrastive predicates, multi-clause discourse, novel argument structures — is **not a
cue-lexicon problem**. It needs a LEARNED grammar/parse that develops from the language stream, the SAME emergence
class as the mouth (a learned sequence/structure faculty on the substrate; the ledger's own "learned temporal language
sequencing" replacement for the grammar-frames scaffold). **Substrate + stream that makes it emerge:** a cortical
sequence/structure model trained through the continuous language stream (child-level corpus → real interaction),
credit-assigned on the substrate — i.e. it rides the SAME continuous-substrate + learned-parse arc as fluent speech,
and is NOT liftable by any lexicon. That is the deep-engine-gated half.

**One-line steer for the 3&4 parallelization decision:** the comprehension **cue-lexicon** conversion (animacy +
verb-selects → learned distributional cue → the already-validated spiking pools → wire into the five organs) is
**parallelizable-now**; **beyond-transitive open-ended comprehension** is gated on the learned-parse emergence bar
(same class as the mouth), which no cue-lexicon closes.
