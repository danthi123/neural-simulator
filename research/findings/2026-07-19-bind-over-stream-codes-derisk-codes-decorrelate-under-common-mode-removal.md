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
- Diagnostics: `scratchpad/bind_over_stream_codes.py`. NO `sim/` edit. Uses cached scale787 codes.
