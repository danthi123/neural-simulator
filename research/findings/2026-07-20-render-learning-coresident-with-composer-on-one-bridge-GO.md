# Render read-out delta-update WIRE-UP runs co-resident with the composer on ONE bridge, and generalizes (GO)

**Date:** 2026-07-20 (revised post adversarial-audit) · **Status:** GO (3-seed 42/43/100) — the render read-out's
delta-update WIRE-UP runs on the shared bridge (on-bridge FORWARD `cp_ssm_readout_out = cp_ssm_readout_w @
cp_ssm_state` + on-bridge ELIGIBILITY `cp_ssm_state`; the delta weight-arithmetic is host numpy) WHILE the composer
binds/queries on the same bridge, and it GENERALIZES on a teacher-student task with a HELD-OUT set (genuine learning
of a MAP, not memorization of one point). NO `sim/` edit.

## Honest framing (this replaces the original "render-LEARNING learns / ≥7×10⁸ drop")

An adversarial audit correctly flagged the first version as overclaimed: it trained on a SINGLE fixed input→target,
where an over-parameterized linear read-out reaches machine-zero by construction (trivial LMS-on-one-point), and it
called that "learning." This revision fixes it:
- **Teacher-student, over-determined, held-out.** A fixed random teacher `T` maps the chan-region `cp_ssm_state` →
  target. `n_train=96 >> n_read=32` makes it OVER-determined, so the read-out must recover the TRUE `T` (it cannot
  memorize individual points); a low HELD-OUT loss (on 16 unseen inputs) is genuine generalization.
- **The credit arithmetic is host numpy** (`dw = -lr·err·state`), so this is a PURE DELTA RULE, not yet a spiking
  local rule. The on-bridge BDSP graded-clean-error (anticipated at `bridge.py:355-356`) is the follow-on. The claim
  is precisely: the delta-update wire-up runs co-resident, reading the ON-BRIDGE `cp_ssm_state`, and generalizes.

## Result (`_gap_onebridge_learning_coresident_derisk.py`, 3-seed 42/43/100)

- **HELD-OUT loss drops ~3000–6000× (generalizes=True) — all seeds** (e.g. seed 42: 0.0059 → 0.0000): genuine
  generalization to unseen inputs, not memorization.
- **INTERLEAVE non-interference: training WITH a composer op every 40 steps gives an IDENTICAL held-out loss to
  training WITHOUT (≤1e-6) — all seeds.** The composer op (RF ops on its region, touching `v`/`u`+`cp_rf_*`) does not
  perturb the WKV learning (which lives in `cp_ssm_state`+`cp_ssm_readout_w`).
- **composer recall `['cat','mouse']` + no-confab moat `None` intact — all seeds.**
- **ANTI-CHEAT (frozen read-out): held-out loss does NOT drop (0.0059 → 0.0059)** — the delta update is load-bearing
  (the earlier "flat" mis-spec is moot here: held-out loss with no update simply stays at its initial value).

CI: `tests/test_onebridge_learning_coresident.py` (2 tests, GPU-only).

## Read-out

- **⇒ the render read-out's delta-update wire-up runs on the shared bridge (on-bridge forward + on-bridge
  `cp_ssm_state` eligibility), GENERALIZES to held-out, and co-exists with the composer without perturbation.**
  Combined with the capstone (composer + WKV forward on one bridge), the grounded loop's read-out training reads
  on-bridge state on the same substrate the composer + WKV use.
- **Honest residual:** the weight arithmetic is off-bridge host numpy — a pure delta rule, NOT a spiking/synaptic
  local rule. Making the CREDIT ASSIGNMENT itself on-bridge (BDSP graded-clean-error) is the follow-on toward
  "learning is on the substrate" in the fullest sense.

Runner: `_gap_onebridge_learning_coresident_derisk.py` (`--seed`, `--epochs`, `--lr`, `--frozen`).
