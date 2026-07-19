# Reader/binder over the stream cortex's OWN codes (scout #1 de-risk) — the 787 codes are correlated via a UNIFORM COMMON MODE that a cheap feedforward op removes (→ decorrelated); the fixed HRR bind RECALLS facts on them; the LEARNED-binder SYSTEMATICITY/generalization on structured codes is the remaining open test.

**2026-07-19.** Following the strategic scout's #1 recommendation (a reader/binder over the stream cortex's OWN structured
codes = the convergence lever under gaps #2/#4/#5), ran the cheapest-first measurements on the cached 787-concept
stream-cortex codes (`bridges/developed/scale787/day_23/grounded_codes.npz`, 787×128 float32, corr(M,C)=+0.81 code-fidelity
per `2026-07-17-stream-cortex-787-concept-scale-test-RESULT`). NO `sim/` edit; CPU/numpy on cached artifacts.

## Measurements
1. **The 787 codes are highly correlated — via a UNIFORM COMMON MODE.** Between-code cosine mean **+0.751**, std only 0.029
   (ALL pairs >0.5, max 0.898), codes nonneg (0-1), dense (100% nonzero). The low-std uniform ~0.75 = a large shared
   common component (the "common mode" the 2026-06-11 cleanup arc identified; the earlier PPMI codes were ≈0.81 similar).
2. **A cheap FEEDFORWARD common-mode removal DECORRELATES them.** Per-dimension mean-subtraction (= the project's own
   CYCLE-88 PPMI LOCAL NORMALIZATION, biologically per-feature/divisive inhibition) takes between-code cosine
   **0.751 → −0.001** (std 0.095, max 0.538, 99.8% of pairs <0.3). ⇒ the codes are decorrelatable by a SIMPLE feedforward
   op — NOT the attractor-cleanup the 2026-06-11 arc found impossible (Hopfield/Storkey/DG all failed to clean them
   post-hoc). This confirms the CYCLE-88 insight ("off-diagonal decorrelation was a red herring; the fix is feedforward
   local normalization") ON the best-available 787 codes.
3. **The FIXED HRR bind (Plate circular convolution) RECALLS held-out SVO facts on the 787 codes** — who-recall 0.997 /
   what-recall 0.994 on RAW, 1.000/0.994 on common-mode-removed, 6-seed, chance 0.0013. The shuffled-facts anti-cheat
   collapses (who 0.000) — the binding is REAL. **BUT the permuted-codes anti-cheat ALSO recalls 1.000** → this who/what
   recall is a DISTINCTNESS test (any distinct codes bind + nearest-match; the uniform common mode doesn't block argmax),
   NOT a structure/generalization test. So recall being high is NOT evidence the code STRUCTURE is leveraged.

## Honest read + the remaining open test
- **What this de-risks:** the stream cortex's codes are (a) recall-usable by a fixed bind (distinct + nearest-matchable,
  robust to the common mode) and (b) DECORRELATABLE by a cheap feedforward normalization (→ bindable regime for a learned
  binder). The "correlated codes are unusable" fear is softened: a simple feedforward op puts them in the decorrelated
  regime the 2026-06-11 positive control showed a learned binder is SYSTEMATIC over.
- **What remains open (the genuine 2026-06-11 question):** does a LEARNED binder FIT + generalize SYSTEMATICALLY (bind
  held-out novel role-filler combinations; category-generalization — a similar concept inherits) over the (now
  decorrelated-by-normalization) 787 codes? That needs the GENERALIZATION test with category structure (which similar
  concepts should bind alike), not who/what recall. The measurement here (codes decorrelate cheaply) predicts this should
  now WORK (the 2026-06-11 negative was on un-normalized correlated codes; normalize first → the positive-control regime).
- **NEXT (the decisive de-risk):** train a learned bilinear binder on the COMMON-MODE-REMOVED 787 codes; test held-out
  systematicity (novel combos) + category-generalization vs (a) the same binder on RAW codes (expect the 2026-06-11
  ≈chance) and (b) a memorizer floor; category-derangement anti-cheat; 6-seed `cfg.seed` set. GO = systematicity/
  generalization >> RAW + memorizer on the normalized codes → the "learned binder over the brain's own codes" is unblocked
  by a feedforward normalization → wire the normalization into the composer front-end. This is the convergence lever
  (gaps #2/#4/#5): grounded reasoning over the brain's OWN learned semantics.
## ⚠️ TEMPERING FINDING (nearest-neighbor semantic-structure check) — the codes' generalizable structure is WEAK, and decorrelation DESTROYS it
Checked whether the 787 codes encode SEMANTIC SIMILARITY (the property that enables generalization — similar concepts →
similar codes):
- **RAW codes:** nearest neighbors are MOSTLY SEMANTICALLY UNRELATED (dog→time/wife/show/**cat**(0.83)/coat/girl;
  king→storm/is/church/bottle/elephant; water→airport/display/camera; fish→piano/butterfly/traffic). A WEAK semantic
  signal exists (cat IS in dog's top-6), swamped by the uniform common mode + noise.
- **COMMON-MODE-REMOVED codes:** the weak semantic signal is DESTROYED — dog→show/time/wife/history/cross/question (NO cat).
  ⇒ **the common mode CONTAINS the (weak) semantic similarity; global decorrelation REMOVES it.** This is exactly the
  CYCLE-88 tension (CLAUDE.md): *"a generalizing cortex needs feedforward LOCAL normalization, NOT cross-neuron
  decorrelation (which would DESTROY generalization)."* My global per-dim mean-subtraction is NOT the PPMI local
  normalization (log + per-hub + per-concept + threshold) — it conditions for BINDING but destroys generalization.
- **⇒ HONEST, TEMPERED read of the scout's #1:** the 787 codes at develop_D=128 (a) are recall-usable + decorrelatable
  (bindable), but (b) encode WEAK semantic-similarity structure (nearest neighbors mostly unrelated), and (c) the
  decorrelation that enables binding DESTROYS the little generalizable structure there is. So "a learned binder over the
  brain's own STRUCTURED codes for GENERALIZATION" is NOT a cheap win on these codes — it is limited by weak semantic
  structure (likely a develop_D=128 capacity limit at 787 concepts; the 787 finding itself flagged the D=128 FHRR-capacity
  curve) AND the binding-vs-generalization tension. The levers: (i) HIGHER develop_D (more capacity → stronger semantic
  structure), (ii) the PROPER PPMI local normalization (conditions without destroying generalization, per CYCLE-88),
  (iii) validate on codes with KNOWN category structure (the EMERGE taxonomy) rather than the raw 787. This is a
  genuine, precisely-characterized boundary for the convergence-lever direction — NOT the cheap unblock the first
  decorrelation measurement suggested. The recall (distinctness) works today; the GENERALIZATION needs stronger codes.
## FURTHER (develop-day check) — ⚠️ my "degrades over training" claim was ONE-CONCEPT NOISE (multi-concept REFUTES it); structure is weak but IMPROVES with training
The 788-concept vocab is PRE-ALLOCATED (all days have 788). I initially saw day-0 dog→toy/cat (semantic) vs day-23
dog→show/time/wife (unrelated) and claimed structure DEGRADES over training. **A proper MULTI-CONCEPT check (10
hand-labeled semantic pairs: dog-cat, king-queen, mother-father, run-walk, boy-girl, sun-moon, big-small, happy-sad,
eat-food, water-fire) REFUTES it:** day 23 is BETTER than day 0 (semantic-pair median rank **174 vs 222** of 788, top-20
frac **0.38 vs 0.25**) — structure IMPROVES with training. The dog case was one-concept noise (verify-don't-assert: my 4th
self-caught error this session). **But the load-bearing tempering STANDS: the structure is WEAK even at day 23** — semantic
pairs rank ~174th of 788 (top-20 frac 0.38), i.e. only ~38% of clearly-semantic pairs land in the top-20 nearest. ⇒ the
codes DO learn a real-but-WEAK semantic structure (improves with training), not strong enough for robust generalization
via a binder. **Net: the reader/binder-over-stream-codes direction is thoroughly characterized as NOT a cheap win — the
codes are bindable (recall) but their generalizable semantic structure is real-but-WEAK (the code-QUALITY limit is
upstream of the binder). Levers: higher develop_D, PPMI local-normalization (conditions without destroying structure),
or validate the binder on the EMERGE known-category codes (strong structure by construction) to isolate binder-vs-codes.**
## PPMI-normalization test — post-hoc conditioning does NOT fix the weak structure (the limit is the LEARNING)
Tested whether the PROPER PPMI local normalization (CYCLE-88: log + per-hub + per-concept center + positive threshold —
conditions WITHOUT global decorrelation) strengthens the semantic structure vs RAW, on the 10 semantic pairs:
| normalization | median rank / 788 | top-20 | top-50 |
|---|---|---|---|
| **RAW** | **88** | 0.25 | 0.38 |
| global mean-sub (my earlier) | 174 | 0.38 | 0.38 |
| PPMI (log+per-hub+per-concept+thresh) | 127 | 0.38 | **0.50** |
| PPMI no-threshold | 257 | 0.12 | 0.12 |
| per-dim z-score | 185 | 0.38 | 0.38 |
⇒ **NO normalization dramatically fixes it.** RAW is best for median (~88th — a CORRECTION: my earlier "~174th" was the
mean-subtracted codes, so RAW structure is a bit better than I'd stated); PPMI helps the TAIL (top-50 0.50) but hurts the
median; all remain WEAK (semantic pairs ~88-127th of 788, top-50 0.38-0.50). **The code-quality limit is in the LEARNING
(the develop_D=128 stream cortex does not strongly encode semantic similarity), NOT post-hoc-fixable by any normalization.**
⇒ **FINAL characterization of the scout's #1 convergence lever: it is blocked UPSTREAM by weak code semantic-structure,
which needs a code-QUALITY improvement in the LEARNING — a re-develop at higher develop_D (more capacity) and/or a better
co-occurrence learning rule — NOT a cheap post-hoc fix. The binder itself works when codes ARE strong (the EMERGE
inheritance/taxonomy arcs, EMERGE-26/27/28, generalize on is-a-structured codes). So: binder OK; stream-code QUALITY is
the gating sub-problem; the concrete next build is a higher-develop_D re-develop + re-measure the semantic-pair ranks (a
multi-hour GPU job).** This precisely bounds the highest-value next direction.
- Diagnostics: `scratchpad/bind_over_stream_codes.py`. NO `sim/` edit. Uses cached scale787 codes.
