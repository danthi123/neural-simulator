# Biology sweep — VERDICT B (no real learning)

**Date:** 2026-05-04 ~09:13 EDT
**Source:** `research/findings/raw/g11_bg/text_eval_biology_*_seed*.json` (24 runs)
**Pre-staged plan:** `research/findings/2026-05-04-biology-sweep-followup-plan.md`
**Auto-fired follow-up:** B1 (eval_sanity_check) — see VERDICT B section below

---

## TL;DR

The biology-grounded sweep tested whether topographic Wernicke→motor
priors (Pulvermüller 2001-2003) and motor PV-FSI lateral inhibition
(Vogels 2011 / Hofer 2011) — alone or combined — would break the 0/N
alignment streak from the v2 architecture investigation.

**They did not.** All four conditions show 0-1/6 aligned with TRUE
labels. The architecture exhibits 7-13pp "best-perm excess" structure
(seed-dependent random alignment with SOME mapping), but it's not aligned
with the task-defined N/E/S/W mapping.

| Condition | Mean TRUE | Mean best perm | Excess | Aligned/n |
|---|---|---|---|---|
| baseline (random+STDP, no FS) | 21.5% | 35.0% | +13.5pp | 1/6 |
| +FS only | 27.0% | 34.0% | +7.0pp | 0/6 |
| +Topo only | 21.7% | 35.0% | +13.3pp | 1/6 |
| +Topo +FS | 27.3% | 35.2% | +7.8pp | 0/6 |

The single aligned seed in baseline + topo_only is **the same seed (101)
in both cases** — i.e., random architecture-noise alignment that the
biology fixes don't disturb.

## What this rules out

1. **PV-FSI lateral inhibition is NOT the missing ingredient.** Adding
   it raises TRUE accuracy (+5-6pp toward chance) but doesn't increase
   alignment with TRUE labels. FS sharpens action selection; it doesn't
   make the network learn the correct mapping.

2. **Topographic Wernicke→motor prior is NOT the missing ingredient.**
   At 1.5/0.7 (mid-Pulvermüller range), the prior is too mild to
   override training dynamics. STDP under paired-stim training
   reinforces seed-dependent random patterns regardless of the prior.

3. **Combined biology fix is NOT enough.** topo_fs gives 0/6 aligned,
   identical to fs_only. The combination doesn't compose into a useful
   alignment signal.

## What it suggests

The 0/N alignment phenomenon is more fundamental than we thought. Possible
mechanisms (in the auto-launched B-branch investigation):

- **Eval methodology bug** (B1): if the eval can't even detect a
  hand-built perfect mapping, then we've been chasing a phantom signal.
- **Sparse code overlap** (B2): if vocab_to_drive_pattern produces
  too-overlapping codes at sparsity=0.10, no learning rule can
  differentiate them. Test sparsity 0.05/0.02.
- **Training-dose limitation** (B4): 1000 events/dir may be too few.
  Test 5000 and 10000.
- **Plasticity rule itself** (B3): if STDP+R-STDP fundamentally can't
  do this task, supervised gradient learning would succeed. Tests
  upper bound.

## The auto-launched B1 result (with bugs)

The first B1 run encountered two bugs:
1. `hand_build_perfect_weights` had a `'mode': mode` summary entry that
   the verbose-print loop tried to dict-access, crashing all perfect/
   wrong runs with TypeError.
2. `result_aggregator --out FILE` only printed "Wrote {path}" to stdout,
   so the waiter parsed verdict from "Wrote..." instead of the report
   and returned "unknown".

Result: 18/24 runs failed (only random-mode succeeded). Verdict was
inconclusive. **Both bugs fixed in commit `cfc9487`.** B1 re-launched
manually under `wait_b1_then_b2.ps1` (PID 2224 polling).

## Per-seed aligned details (12 conditions × 6 seeds = 72 evals)

The full per-seed breakdown is in
`research/findings/2026-05-04-biology-sweep-results.md` (auto-aggregated
by the waiter at 09:13).

Key per-seed pattern:
- Seed 101 aligns in baseline (NESW @ 32%) and topo_only (NESW @ 33%) —
  "lucky" random init.
- All other seeds show varied non-TRUE best perms across conditions.

## Implication for the cheat-5 ON HOLD reframe

The 2026-04-28 reframe described cheat-5 (cross-projections) as "ON HOLD
pending biology buildout." Tonight's results suggest the biology
buildout itself doesn't unlock W→A alignment in the minimal architecture.
This doesn't necessarily impact the cheat-5 navigation question (which
operates on a different eval), but it does cast doubt on the
"biology-grounded > engineering shortcuts" framing for the language task.

## Next step (auto-firing)

`wait_b1_then_b2.ps1` polls for the rerun completion. After ~30 min:
- If perfect mode aligns ≥ 4/6: eval is sound → B2 (sparse codes) auto-fires
- If perfect mode aligns 0-1/6: eval methodology is broken → manual review

Both outcomes are scientifically informative.
