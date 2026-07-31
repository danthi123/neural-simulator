---
type: finding
status: contributing
mechanism: btsp-place-field-formation
claim_check: synthesis
lane: gap#5
date: 2026-07-31
---

# gap#5 — the STEP C place-specificity control had NO POWER in the density sweep; its ⛔ verdicts are VOID, not negative

**Date:** 2026-07-31 · **Status:** instrument defect FOUND + FIXED + the fix VERIFIED both directions ·
**Scope correction included** (my own first statement of this overreached — see §5)

---

## 1. What was measured

The 18-seed read-density sweep (`research/findings/raw/gap5_density/*.json`, operating point
`--dwell 180 --lr 0.005 --w-max 150`, `GAP5_PLACE_READ_DENSITY=0.25`) printed
`⛔ NOT place-specific (generic potentiation)` on every run. Before recording that as a negative, the artifacts
were checked against their own controls:

| quantity | value |
|---|---|
| treatment `d_circ` (seed 200, btsp) | **-0.00759451** |
| its RANDOM-SET null `randset_d_circ` | **-0.00759449** |
| absolute difference | **1.9e-08** |

Across the whole sweep: **29 of 36 arm-runs had the treatment and its random-set null agreeing to <1e-6**,
frequently to 1e-9 — floating-point identical.

**A null that reproduces the effect to nine significant figures is not controlling anything.** The verdicts it
produced are VOID. The place-specificity question at this operating point is **UNANSWERED, not answered no.**

## 2. Root cause — read from the code, not inferred

Two separate defects in the STEP C block of `research/runners/_gap5_btsp_place_field_derisk.py`:

1. **The gate compared incomparable quantities.** `real` (line 361) was a *between-arm* contrast
   `circ(btsp) − circ(lr0_btsp)`; `randset_d_circ` (line 352) was a *within-run pre/post* delta
   `circ(M1r) − circ(M0r)`. Different quantities, compared directly.
2. **Both controls were computed on `circ`** — the circular resultant of the **FINAL weight matrix**, which is
   dominated by the random INITIAL structure. With increments small relative to that structure, `M1 ≈ M0`, so
   both the treatment's and the null's pre/post deltas collapse onto the same structural residue regardless of
   what the drive was doing.

The decisive contrast: the manipulation moved `circ` by ~1e-8 while the headline `circ_dW` (resultant of the
weight **CHANGE**) sat at **0.51–0.81**. The signal was large and the gate was reading a quantity blind to it.

**This is the same conflation already documented in this file's own STEP B comment** (`circ` vs `circ_dW`), where
it was fixed on the ARM side — and left unfixed on the CONTROL side. The earlier fix recorded `circ_dW` for the
arms and stopped there.

## 3. The fix

`permuted_increment_circ_dW_null()` evaluates the null on the **same quantity the headline reports** (`circ_dW`),
differs from the treatment in **exactly one property** (position — magnitude and concentration are held fixed by
construction), and is **threshold-free**: the p-value is the fraction of position-shuffles reaching the observed
value, so no `2x` rule has to be invented. It reshuffles increments that are already computed, so it adds **no
simulation cost**.

A `void_if` degeneracy guard now **asserts** the legacy control differs from its own treatment rather than
assuming it. "This control is doing something" is a hypothesis, so it gets a test. Legacy numbers are still
written to the artifact so the regression stays visible.

## 4. The instrument was verified BEFORE use — both directions

A single synthetic draw first read `p=0.0398` on the negative control and looked like a failure. That was one
unusually clustered draw, not a broken gate — so the **rate** was measured rather than judged from one sample
(60 independent draws, `n_perm=200`, α=0.05):

| check | result | wanted |
|---|---|---|
| POWER — contiguous increments flagged place-specific | **1.000** | ~1.0 |
| FPR — scattered increments of identical mass/count flagged | **0.000** | ~0.05 |
| median null p-value on the negative control | **0.679** | ≫0.05 |

The instrument has power and does not cry wolf.

## 5. ⚠️ SCOPE CORRECTION to my own first statement

The commit message for the fix (`70292a01`) says *"EVERY such verdict from this control is VOID."* **That
overreaches and is corrected here.** The banked gap#5 field-quality GO on the board does **not** come from this
code path: it uses a different artifact schema (`research/findings/raw/gap5_reader/fieldquality_gpu6.json`, keys
`circ / randset / width / peaks / sat`), where seed 42 reads `circ 0.664` against `randset 0.122` — a control with
ample power and a wide separation.

**Correct scope:** the defect is in `_gap5_btsp_place_field_derisk.py`'s STEP C control, and it degenerates when
the weight change is small relative to the initial weight structure — the regime this sweep ran in
(`lr 0.005`, `w_max 150`, both far below the runner's defaults of `0.02` / `2500`). **The banked GO is
unaffected.** Whether STEP C retained power at larger `dW` is now moot: the replacement gate always reports a
p-value, and the degeneracy guard fires whenever the legacy control collapses onto its treatment.

## 6. What this does and does not change

- **Does NOT change** any banked gap#5 claim, including the field-quality GO (different runner, working control).
- **Does void** every `⛔ NOT place-specific` line from the 18-seed density sweep.
- **Prevented** the conclusion I was about to draw — *"read density 0.25 loses place-specificity"* — which the
  data never supported.
- **Re-runs staged** on the fixed instrument: 6 seeds at density 0.25 and 6 at density 1.0
  (`g5fix_d025_*`, `g5fix_d100_*`), dispatched to the mini-PC pool.

## 7. The transferable lesson

This is the silent-failure shape the operating discipline names: **the machinery to check the claim already
existed and nothing invoked it.** `circ_dW` was already being recorded — by a comment explaining this precise
conflation — and the control block one screen below still gated on `circ`. Fixing a defect on one side of a
comparison and not the other leaves a gate that reports confidently and measures nothing.

**A control is not trustworthy because it exists. Assert that it differs from what it controls.**
