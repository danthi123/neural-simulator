# Shortcut #12 codes-half (option 2): swap the production conversational composer to LEARNED PPMI codes — GO (2026-06-20)

**Owner decision (path B2, `2026-06-20-fhrr-frontier-decision-scoping.md`, commit `3c79651a`):** #12 (the FHRR composer)
has two halves — the **exact-inverse bind FORM** (KEEP, close as honest-negative; NOT touched here) and the **CODES**
(SWAP to learned). This deliverable is the codes-half: make the production conversational composer use the **LEARNED
PPMI stream-cortex codes** (the 2026-06-15/16 generalization arc — a cortex that learns each word's meaning word-by-word
from a conversation stream by population-Hebbian co-occurrence; cached as `_phaseB_stream_codes_320_*.npy`), retiring
the "curated codes" label, **without changing the answers and never weakening the no-confab moat**.

**VERDICT: GO — multi-seed (42/43/44), CPU/numpy.** On the production agent at the full 320-concept scale, the LEARNED
PPMI codes answer the **SAME** who/what/yes-no matrix as the curated (composer-self-generated random) codes (**==
curated who/what**, answer-identical, 0 mismatches every seed) AND hold the **no-confab moat at 0 false-accepts** on
both paths. The documented cost is a **lower cleanup margin** (the learned codes are less decisive) — the answers and
the moat are unchanged.

## The two artifacts (both CPU/numpy, NO `sim/` edit, reuse-by-import)

1. **The 320-scale A/B** — `research/runners/_codeswap_12codes_ppmi_ab.py`. Builds the production
   `BrainConversationalAgent` two ways over the SAME 320-word taxonomy — (A) curated codes (no `grounded_codes` -> the
   composer self-generates random phases) and (B) the LEARNED PPMI codes (the `grounded_codes` path the flagship
   already uses) — runs the identical who/what/yes-no/moat matrix, and reports: answers-identical, moat false-accepts
   (both paths), and the mean cleanup margin (a READ-ONLY recompute of the unbind+matched-filter sims; it does NOT
   touch the composer's cleanup code). `composer_kind="rf"` (CPU-fine; the onebrain path is the controller's GPU
   confirm).

   | seed | answers identical | moat FA (curated / ppmi) | mean margin curated -> ppmi | margin cost | verdict |
   |---|---|---|---|---|---|
   | 42 | True (0 mismatches) | 0 / **0** | 0.559 -> 0.524 | **+0.035** | GO |
   | 43 | True (0 mismatches) | 0 / **0** | 0.571 -> 0.513 | **+0.058** | GO |
   | 44 | True (0 mismatches) | 0 / **0** | 0.598 -> 0.477 | **+0.121** | GO |

   Mean margin cost **+0.071**. (Run: `SIM_BACKEND=numpy python -m research.runners._codeswap_12codes_ppmi_ab
   --seeds 42 43 44 --readout neural`.)

2. **The CI guard** — `tests/test_codeswap_ppmi_codes.py` (CPU-only, 4/4 pass). Pins the SAME invariant at a small
   vocab with synthetic learned codes (so it runs in CI without the 320 `.npy` cache or GPU): the production agent on
   learned codes answers == on curated codes (who/what/yes-no), the moat holds 0-FA on BOTH, and the curated-codes
   escape (no `grounded_codes`) still works. The swap can't silently bit-rot.

## The honest margin cost (the documented trade-off)

The decision-scoping noted the learned-codes path lands ~0.39 below the curated margin ceiling on the bundling-A/B
harness; on the **production who/what** matrix measured here the per-fact patient-unbind cleanup margin drops by
**+0.035 to +0.121** (mean +0.071) going from curated to PPMI codes. This is the cost of using
learned-from-experience codes — they are semantically structured (they carry category similarity, which is what lets
the cortex generalize), so they are correlated and therefore less decisive than orthogonal-ish random codes. **The
answers and the moat are unchanged** because (a) the role-binding decorrelates the cross-terms (the composer tolerates
code-similarity up to ~0.98, `_step3_correlated_percept_boundary.py`), so recall stays exact, and (b) the no-confab
moat is RELATIONAL (it abstains on whether the fact was stored, independent of code geometry).

**The moat was NOT weakened to absorb the lower margin.** If a lower margin had pushed a false-accept, that would be
the honest finding reported here — not a reason to relax the gate. It did not: 0 false-accepts on every seed.

## Production status (the "retire the curated codes label" point)

The flagship production conversation, `research/runners/consolidated_320_conversation_demo.py`, **already** defaults to
the learned codes — `--composer onebrain` + `--readout neural`, loading the PPMI `_phaseB_stream_codes_320_neural_*.npy`
via the `grounded` projection. Confirmed GREEN on the `neural` codes multi-seed this pass (rf composer, CPU): seeds
42/43/44 all recall 1.00, abstain 1.00, 0 false-accepts (`research/findings/raw/_codeswap_demo_ppmi_neural.json`). The
`neural`-readout PPMI codes for seeds 43/44 are now cached (closing the burndown #5 P3 "only seed 42 cached" gap for
the `neural` path). So the production conversation runs on **codes it LEARNED FROM CONVERSATION**; this deliverable
adds the A/B that PROVES the swap is answer-safe + moat-safe vs curated, and the CI guard that keeps it so.

**The escape:** the curated-codes path stays available (don't pass `grounded_codes` -> the composer self-generates the
random codes — the test-oracle / numpy-CPU default). The A/B exercises that escape side-by-side.

## Scope / what was NOT touched

- **The bind FORM is untouched** (the owner's KEEP half): the exact-inverse self-inverse FHRR bind/bundle/unbind
  algebra is unchanged. This is the codes-half only.
- **NO `sim/` edit.** Reuse-by-import of `BrainConversationalAgent` + the existing `grounded_codes` plumbing through
  `RFPhasorComposer`/`OneBrainComposer`. (A concurrent subagent's in-flight edit to `rf_phasor_composer.py`'s cleanup
  is theirs, not in this deliverable's PATHSPEC commit.)
- The nav files, the sequencer/`_scan`, and the rf cleanup were not touched (other subagents own those).

## Files
- `research/runners/_codeswap_12codes_ppmi_ab.py` — the 320-scale curated-vs-PPMI A/B harness
- `tests/test_codeswap_ppmi_codes.py` — the CPU CI guard (4/4)
- `research/findings/raw/_codeswap_12codes_ppmi_ab.json`, `_codeswap_demo_ppmi_neural.json` — results

## Sources
- `research/findings/2026-06-20-fhrr-frontier-decision-scoping.md` (path B2: fixed bind + learned codes; the
  ~0.39-below-ceiling cost; the strategic "retire the curated-codes label" point)
- `research/findings/2026-06-20-shortcut-burndown-inventory.md` (#12 = the FHRR algebra + given/curated codes; #5 the
  PPMI neural read-out norm)
- `research/runners/consolidated_320_conversation_demo.py` (the production flagship already on the PPMI `grounded` path)
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the PPMI stream cortex —
  learned codes ship + generalize)
- `research/findings/2026-06-17-within-category-error-signature-NEGATIVE.md` (the codes' category margin is thin —
  consistent with the lower cleanup margin measured here)
