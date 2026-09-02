---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-integration-r3v2-r3v3-read-isolation-refix
board: 108 / one-brain integration program (read-isolation-audit followup)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_integration_r3v2_noncorrupting_readfix_numpy6seed.json
  - research/findings/raw/_onebrain_integration_r3v3_functional_drive_readfix_numpy6seed.json
runner: research/runners/_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py
supersedes_diagnosis_of: research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md
---

# C2-class read-isolation fix on R3v2/R3v3: R3v2 stays a clean NO-GO 0/6 (confirmed, not flipped); R3v3's banked GO 6/6 FLIPS to NO-GO 3/6 — the shuffled-credit control now exceeds its threshold on half the seeds once the training-time leak is closed

**2026-09-02, numpy, 6 seeds (42/43/44/100/101/102).** Follow-up to
`research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`, which flagged
`R3v2Pool._hard_reset()` (`research/runners/_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py`) as
missing the same C2-class reset gap the metacog fix closed: `cp_refractory_timers`/`cp_prev_firing_states`
(live leak vectors here — core Izhikevich hard-refractory-gate bookkeeping, updated unconditionally every step)
never restored to true rest. `R3v3Pool` (`_onebrain_integration_r3v3_functional_drive.py`) subclasses
`R3v2Pool` and inherits `_hard_reset()`/`__init__` byte-identically, so ONE fix in the shared base covers both
generations.

**Predicted:** the audit flagged r3v2's NO-GO as a *possible* false wall (leak comparable-or-larger than the
F2 floor shortfall, sign unpinned) and asked this lane to additionally confirm r3v3's banked GO 6/6 **survives**
isolation. **Actual result: r3v2's NO-GO is a real wall, confirmed unchanged by the fix — but r3v3's GO does
NOT survive isolation.** Both are honest, load-bearing findings; neither is the predicted outcome.

## The fix (Port A, landed in the shared `R3v2Pool` base)

`R3v2Pool.__init__` now snapshots the framework's own `onebrain_merge_framework.MergedPool._PER_NEURON_STATE`
arrays not already covered by the runner's existing reset (`cp_prev_firing_states`, `cp_refractory_timers`,
`cp_neuron_firing_thresholds`, `cp_neuron_activity_ema`, plus a dead no-op `cp_refractory` entry — the
framework tuple's own inert placeholder) at the SAME true-rest point `rest_v`/`rest_u` are already captured
(after the 40-step no-drive settle). `_hard_reset()` now restores all of them on every call — a strict
superset of the prior reset, provably a no-op on every array the runner already handled correctly (same
values, restored twice). `R3v3Pool` overrides only `train()` (to freeze the candidate-edge gate), never
`__init__`/`_hard_reset`, so this one edit covers both runners.

**Selftest (fails-in-failing-direction, per the audit's own requirement):** `_selftest_read_isolation()`
(`--selftest`) builds a mechanism-zeroed pool (`mode="removed"` — the teacher drive withheld from every
episode, so nothing trains) and runs repeat `hard_reset -> drive -> hard_reset` cycles, asserting every extra
array is bitwise-identical (`np.array_equal`) to the true-rest snapshot after each reset. Verified to actually
fail when the restore loop is disabled: raises on `cp_prev_firing_states`, **33/1752 entries differing** — the
exact figure the 2026-09-02 audit's own dynamic probe reported independently.

## R3v2: NO-GO 0/6, confirmed unchanged — the false-wall prediction did NOT hold

<!--derived-->
| seed | BEFORE `delta_agent_intact` | AFTER | BEFORE `delta_patient_intact` | AFTER | BEFORE F2 | AFTER F2 |
|---|---|---|---|---|---|---|
| 42  | 0.005278 | 0.005556 | 0.004074 | 0.003981 | False | False |
| 43  | 0.002685 | 0.005000 | 0.000741 | -0.000926 | False | False |
| 44  | 0.005000 | 0.005000 | 0.001481 | 0.002593 | False | False |
| 100 | 0.006019 | 0.006296 | 0.005093 | 0.002870 | False | False |
| 101 | 0.005000 | 0.004630 | 0.005185 | 0.000648 | False | False |
| 102 | 0.004722 | 0.004537 | 0.004444 | 0.002870 | False | False |

BEFORE: `research/findings/raw/_onebrain_integration_r3v2_noncorrupting_6seed.json`. AFTER:
`research/findings/raw/_onebrain_integration_r3v2_noncorrupting_readfix_numpy6seed.json`. Best seed (100)
still misses `F2_INTACT_FLOOR=0.008` — now by 0.00170, vs 0.00198 before the fix (a small, non-monotonic
per-seed shift, sign genuinely unpinned as the audit warned, but never crosses the floor). Every other arm
(F1/F3/F4/lesion-recovers-migration/R3a-three-factor/dopamine-lesion) stays 6/6, unaffected — the fix's whole
measurable effect is a sub-0.001 wobble on F2's already-sub-floor numbers. `emergence.no_corruption_intact`
stays 6/6 (drift exactly 0.0 every seed): the C2 leak arrays are provably outside `_frozen_w0`'s tracked set,
so this fix cannot and does not touch that check.

**Honest diagnosis of why the leak did not flip this wall:** the audit's own DYNAMIC B evidence (untrained
probe: repeated reads moved the candidate weight from `W0=0.05` by >0.4/read) already pointed at the DOMINANT
confound being a *different*, larger bug — `enable_stdp`/`enable_reward_modulation` are global always-on flags
and no gate ever closes between `train()` and a read on `R3v2Pool` itself, so a "read" is mechanically
identical to a training step. That is precisely the defect R3-v3's own Fix #1 (freeze the candidate-edge gate
the instant `train()` returns) closes on ITS lineage — R3v2 never got that fix, by design (it is R3v2's F2
NO-GO, the pre-registered comparison point R3v3 was built to surpass). This C2 fix and R3v3's gate-freeze fix
are complementary, not duplicative: closing the smaller (missing-4-array) leak here is integrity/hardening
work that leaves R3v2 an honest, still-standing NO-GO — not the false wall the audit flagged as plausible.

## R3v3: GO 6/6 → NO-GO 3/6 — a REAL flip, not a false-wall confirmation

<!--derived-->
R3v3 inherited the identical C2 gap and had never been isolation-tested (its own `read_isolation_verified`
check only proves a READ doesn't move the candidate weights — it says nothing about repeat-read bitwise
identity of the refractory/prev-firing arrays this gap concerns). The lane's task was to confirm the banked
GO 6/6 SURVIVES isolation. **It does not.**

| seed | BEFORE `delta_agent_intact` | AFTER | BEFORE sel_intact | AFTER | BEFORE sel_shuffled | AFTER | BEFORE ratio | AFTER ratio | BEFORE R3a | AFTER R3a | AFTER overall PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.01222 | 0.01917 | 11.225 | 14.671 | 2.293 | 5.153 | 0.2043 | 0.3512 | True | **False** | **False** |
| 43  | 0.01417 | 0.02083 | 10.680 | 15.089 | 1.845 | 4.299 | 0.1728 | 0.2849 | True | True | True |
| 44  | 0.01241 | 0.01944 | 10.769 | 14.974 | 2.525 | 5.142 | 0.2345 | 0.3434 | True | True | True |
| 100 | 0.01361 | 0.02083 | 11.059 | 14.939 | 2.506 | 5.111 | 0.2266 | 0.3421 | True | True | True |
| 101 | 0.01296 | 0.01861 | 10.972 | 14.931 | 2.987 | 6.240 | 0.2723 | 0.4180 | True | **False** | **False** |
| 102 | 0.01306 | 0.01972 | 10.913 | 15.083 | 2.739 | 5.745 | 0.2510 | 0.3809 | True | **False** | **False** |

BEFORE: `research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md`'s own artifact
(`research/findings/raw/_onebrain_integration_r3v3_functional_drive_6seed.json`). AFTER:
`research/findings/raw/_onebrain_integration_r3v3_functional_drive_readfix_numpy6seed.json`. "ratio" =
`selectivity_shuffled / selectivity_intact`, must stay under `SEL_SHUFFLE_RATIO=0.35` for
`R3a_three_factor_PASS` (`_onebrain_integration_r3_spiking_dopamine_credit.py:441`) — the shuffled-credit
control must degrade relative to intact, or the mechanism is not shown to be selective. `n_go: 3/6` (payload
`GO: false`), per-arm `{'F1.PASS': '6/6', 'F2.PASS': '6/6', 'F3.PASS': '6/6', 'F4.PASS': '6/6',
'lesion_recovers_migration.PASS': '6/6', 'emergence.R3a_three_factor_PASS': '3/6',
'emergence.R3_dopamine_lesion_PASS': '6/6', 'emergence.no_corruption_intact': '6/6',
'emergence.read_isolation_verified': '6/6'}` — every arm stays 6/6 except R3a, which is the SOLE blocker on
all 3 failing seeds.

**F2 itself did not weaken — it got STRONGER** (`delta_agent_intact` 0.0122-0.0142 before, 0.0186-0.0208
after; `frac_attributable` stays 1.0/1.0 every seed). The flip is entirely in R3a's shuffled-control check.
`no_corruption_intact` and `read_isolation_verified` both stay 6/6 (the fix does not corrupt non-candidate
synapses, and the gate-freeze from R3v3's Fix #1 still holds).

**Root cause — this is a TRAINING-dynamics effect, not a read artifact.** Unlike R3v2's F2 read protocol
(which runs `_hard_reset` many times per read, so a leak concentrated there), R3v3's `_episode()` calls
`_hard_reset()` at the start of EVERY one of the 400 training episodes — intact, removed, AND shuffled arms
alike. Restoring the true-rest refractory/prev-firing state before each episode changes how much residual
hard-gating carries into that episode's drive, which changes exactly how much each episode's spikes coincide
with the dopamine coincidence-detector's window — for ALL four training arms, not just the read. Both
`selectivity_intact` and `selectivity_shuffled` grew substantially post-fix (intact +30-40%, shuffled
+80-120%) — the shuffled arm's growth outpaced intact's on 3 of 6 seeds, pushing the ratio past 0.35. This is
the C2 bug class biting a DIFFERENT arm of the same runner than the one the audit's own probe exercised (F2's
reads), a genuinely new failure mode this lane surfaced, not the one predicted.

## What this means (honest)

**R3v2: closed as re-confirmed.** The 2026-09-02 audit's "plausible flip, sign unpinned" framing for R3v2 is
resolved — the C2 fix is real and correctly applied, but it is not the dominant confound at R3v2's operating
point; R3v2's NO-GO stands, now hardened against this specific integrity question rather than merely
unexamined.

**R3v3: the banked GO 6/6 is RETRACTED (partial — the `n_go: 6/6` verdict only).** The mechanism itself is
unaffected and remains genuinely load-bearing: F2 (functional drive) is undiminished and actually stronger,
the dopamine-lesion crux stays 6/6, `no_corruption_intact` stays 6/6, `read_isolation_verified` (R3v3's own
Fix #1 proof) stays 6/6. What changed is R3a's shuffled-credit-degrades control, which now fails on 3/6 seeds
because training-time (not just read-time) leak-closure grew both the intact and shuffled arms' selectivity,
with the shuffled arm growing disproportionately more on half the seeds. This is NOT evidence the credit
assignment stopped being selective in the sense F2 measures (that check strengthened); it is evidence the
SPECIFIC three-factor R3a control (shuffle the credited/uncredited pairing, expect selectivity to collapse) is
now borderline at this substrate's operating point once the residual leak that was suppressing the shuffled
arm's growth is closed. `research/runners/_onebrain_integration_r3v3_functional_drive.py` is **not** wired
into `/api/brain-chat` or any production endpoint (grep-verified: no reference to
`onebrain_integration_r3v2`/`r3v3` outside `research/runners/`/`research/findings/`) — it is a research
de-risk runner, so this flip has no live production-default flag at stake (contrast the audit's IG-1
`_onebrain_crossedge_curiosity_to_d6wm`, which IS wired default-on and carries that separate, higher-priority
integrity question).

**Next rung (not attempted here, out of this lane's scope):** either recalibrate `SEL_SHUFFLE_RATIO` for the
DA-mediated pathway's now-larger operating scale (a legitimate pathway-specific floor recalibration, the same
class of fix R3v3's own `DA_SENSITIVITY` change was — never touching a floor without an explicit, cited
justification), or find/verify a training-time control (a per-episode reset that only-the-shuffled-arm's
episodes see, isolating whether the leak-closure specifically helped the shuffled arm's spurious coincidences)
that would show the R3a check itself, not the mechanism, needs the update. Until then the honest status is
**NO-GO 3/6**, not GO.

`docs/RETRACTED.md` carries a PARTIAL retraction of
`research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md`'s `n_go: 6/6` /
`R3a_three_factor_PASS: 6/6` claim; its F2/dopamine-lesion/no-corruption measurements are unaffected and still
stand, cited above.

Functional read-outs only; no phenomenal-experience claim.
