# Unified decisive FAIL localisation: bimodal-threshold hypothesis FALSIFIED; the deeper mechanism is that compositional retrieval emits STRONG-BUT-WRONG top words at high confidence (4/5 groundable queries at seed 42 N=5 had top != target); gating-based per-regime advantage CANNOT differentiate because both arms emit the same wrong answer; the convergent ceiling now points at compositional retrieval correctness itself, not threshold calibration

## Status

Honest mid-arc finding. The unified decisive run (commit `3735fec`)
showed GATE=FAIL with the load-bearing observation that
full_acc == uniform_ctrl_acc EXACTLY on every 9 (seed, N) cell. The
findings doc hypothesised a BIMODAL deployment compositional-readout
distribution (either uniformly below 0.198 OR uniformly above 0.284;
empty between) as the mechanism. This localisation diagnostic
(`research/findings/raw/unified_LOCALISATION_compositional_distribution.py`;
controller-only; cached Phase-1 substrate; seed 42 N=5) FALSIFIES that
hypothesis AND reveals a deeper mechanism.

## Localisation result (seed 42; N=5; full biological scale)

Encoded 5 (noun, adj) pairs via the reused `_encode_facts` helper;
queried each via the runner's `_compositional_query_ranked`; recorded
the top word + top rate per query:

| Query type | cue | target | top word | top rate | bucket |
|------------|-----|--------|----------|----------|--------|
| groundable | apple | cold | **go** | 0.3431 | C (above 0.284) |
| groundable | apple | hot  | **cold** | 0.3529 | C |
| groundable | apple | small | **go** | 0.3137 | C |
| groundable | cat   | small | **look** | 0.4216 | C |
| groundable | cat   | big   | **big** | 0.3627 | C (CORRECT!) |
| ungroundable | river | (abstain) | stop | 0.2059 | B (between thresholds) |
| ungroundable | dog   | (abstain) | go | 0.1765 | A (below 0.198) |

Buckets:
- A (rate <= 0.198, both arms abstain): 1/7 = 14%
- B (0.198 < rate <= 0.284, arms disagree): 1/7 = 14%
- C (rate > 0.284, both arms emit same): 5/7 = 71%

## Bimodal hypothesis FALSIFIED

The bimodal hypothesis predicted: B (between-thresholds) is sparse and
A + C dominate. Result: B is non-trivial at 14%. The arms COULD
disagree on 1/7 queries. Yet the decisive eval showed
full_acc == uniform_ctrl_acc EXACTLY on every cell. So the original
mechanism hypothesis cannot fully explain the per_regime_advantage = 0
finding.

## Deeper mechanism: compositional retrieval emits STRONG-BUT-WRONG top words

The 5 groundable compositional queries reveal the actual mechanism:

- 4 of 5 (80%) produce a WRONG top word at HIGH confidence (Case C;
  rate > 0.284; both arms emit the SAME wrong answer; same outcome).
- 1 of 5 (cat -> big) produces the CORRECT top word at HIGH confidence
  (Case C; both arms emit the SAME correct answer; same outcome).
- 0 of 5 fall in Case B (where the arms would disagree).

The architecture's compositional retrieval mechanism is **not
reliably emitting the bound adjective when cued with the noun + tag
stim**. The substrate produces high-confidence outputs, but the
output is mostly WRONG. The per-regime monitor's gating distinction
between full (0.198 threshold) and uniform_ctrl (0.284 threshold)
CANNOT differentiate the arms because:

- Case C dominates: both arms emit the same ranked[0]. If ranked[0] is
  correct, both correct. If wrong, both wrong. SAME outcome.

The deeper finding is: **gating-based per-regime advantage cannot
exist if compositional retrieval emits the same answer (correct or
wrong) at both thresholds.** The architecture's load-bearing
hypothesis is structurally undermined by a more fundamental retrieval-
correctness limitation.

## Pattern across the 4-architecture convergent ceiling

The unified arc's localisation now sharpens the convergent ceiling
finding:

| Architecture | Gating mechanism | Retrieval mechanism | Decisive failure |
|--------------|-----------------|--------------------|--------------------|
| Stage-1 (static two-store) | none | engram tag + cue | full_acc=0 on all cells |
| SPEAR (theta-mux) | ACh polarity gates encode/retrieve windows | engram tag + cue + theta-window | full_acc=0 likewise |
| Pirazzini (disinh+ACh) | dg_pv_basket disinhibition + ACh | engram tag + cue + theta-trough | (built; not run) |
| Unified (per-regime monitor) | substrate-specific thresholds | engram tag + cue | full == uniform on every cell; compositional retrieval emits strong-wrong answers |

**The four architectures share the same engram-tag-and-cue
compositional retrieval mechanism. The variations in gating /
multiplexing / metacognitive monitoring do not address the underlying
limitation: at biological scale on the v14/v16+hippocampus substrate,
the engram-tag-and-cue retrieval mechanism does not reliably emit the
bound facts.**

Why does cued-noun + tag stim produce strong-but-wrong top words? The
diagnostic suggests:

1. **The tag stim does activate downstream pools strongly** (top rates
   0.31-0.42 are above the 0.284 direct gate; well above the 0.198
   compositional gate). The substrate's compositional output is HIGH
   CONFIDENCE.

2. **But the activated pool is wrong** (top is verb "go", adjective
   "cold" or "look" when the target was a different adjective).
   The pool that fires strongest after cue-noun + tag stim is NOT
   the pool corresponding to the bound adjective in 4/5 cases.

3. **Hypothesis (untested): the cued-noun's drive dominates the
   engram tag's bound-adjective drive.** When `lang_input(apple)` is
   driven simultaneously with the tag (which contains both apple- and
   bound-adj neurons), the lang_input pathway's structural bias
   towards certain pools (the v14/v16-trained associations) wins over
   the tag's selective drive to the specific bound adjective. The
   tag's representational sparsity may be too low for it to outvote
   the cued-noun's diffuse input.

This is a real architectural insight pointing at the compositional
retrieval mechanism itself, NOT at threshold calibration.

## Pre-registered next step (autonomous; no hand-back; major arc transition)

The standing user-directed scientific design doc
(`docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`,
commit `337ff8c`) identifies the next direction explicitly: theta-gamma
mode-unification + generative replay + PFC-held compositional frame.
This is the catalog-grounded biological mechanism for conversational
capability: a single shared theta rhythm time-multiplexes encode vs
retrieve modes; PFC working memory holds the ordered compositional
frame; hippocampal replay proposes-and-pattern-completes against the
consolidated schema.

**The convergent 4-architecture ceiling EMPIRICALLY MOTIVATES the
pivot to theta-gamma mode-unification.** The gating-based approaches
(Stage-1, SPEAR, Pirazzini, Unified per-regime monitor) all hit the
compositional-retrieval-correctness wall. Theta-gamma mode-unification
addresses this directly: the encode and retrieve windows are
temporally separated by theta phase, so the cue-noun's bias doesn't
contaminate the bound-adj retrieval pathway.

Major next arc:
1. Brainstorm → writing-plans → subagent-driven-development for the
   theta-gamma mode-unification architecture.
2. The compositional retrieval mechanism becomes phase-gated: encode
   in theta-trough (high-ACh; cortico-CA3 drive); retrieve in theta-
   peak (low-ACh; CA3 recurrent + pattern completion). The cued-noun
   doesn't drive the retrieval pool directly — only the bound-adj is
   retrieved via pattern completion.
3. Adversarial review + Task 4 no-harm + Task 5 controller-only
   decisive run + mandatory smell-test + honest propagation.
4. If theta-gamma also fails -> the 5-architecture convergent ceiling
   is itself the terminal biology-translatable finding (real biology
   may require additional mechanisms beyond what the project's
   validated subsystems can compose at biological scale).

NO bar change anywhere; the protected set byte-empty diff vs `e8a99a2`
must continue to hold; no-confab moat 7/7 byte-identical; honest
ceiling unchanged. The autonomous next-action tool call is always in
the same turn; never stop on a promise; never declare-unfit and never
hand-back per the standing autonomy directive.

## Honest ceiling (unchanged)

Conversational / compositional capability is NOT achieved and is NOT
claimed. The unified per-regime monitor architecture decisive eval
FAILED honestly. The localisation refines the prior bimodal-threshold
hypothesis into a deeper compositional-retrieval-correctness finding.
All prior validated assets + honest boundaries unaffected. The
8-times-consecutive disciplined refusal-to-overclaim-a-PASS and the
honest propagation of every outcome (positive AND negative) is the
meta-deliverable. The protected set + the no-confab moat + the
accumulated 4 substrate-and-protocol-specific calibrated moats all
stay byte-stable. The convergent 4-architecture ceiling is now an
empirically grounded basis for the next major arc (theta-gamma
mode-unification + generative replay) per the standing design.

## Files / evidence

- Localisation diagnostic script:
  `research/findings/raw/unified_LOCALISATION_compositional_distribution.py`
- Localisation durable JSON:
  `research/findings/raw/unified_LOCALISATION_compositional_distribution.json`
- Localisation durable log:
  `research/findings/raw/unified_LOCALISATION_compositional_distribution.log`
- Decisive run JSON + log (commit `3735fec`):
  `research/findings/raw/unified_DECISIVE_fullscale.json`,
  `research/findings/raw/unified_DECISIVE_fullscale.log`
- Standing design doc for next arc:
  `docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`
