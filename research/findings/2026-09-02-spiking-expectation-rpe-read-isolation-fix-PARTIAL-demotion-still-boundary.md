---
type: finding
status: live
date: 2026-09-02
mechanism: spiking-expectation-rpe-read-isolation-fix
board: FW-1 / read-isolation audit re-verify (C2 bug class)
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.4_AFTER.json
---

# Spiking expectation-RPE low-prior "wall": the read-isolation fix PARTIALLY demotes it — 3/6 -> 4/6 (one seed of three), still BOUNDARY not GO

**2026-09-02, numpy, 6 seeds (42/43/44/100/101/102), same runner
(`research/runners/_spiking_expectation_rpe_derisk.py`), gain sweep unchanged.** Re-verifies the FW-1 prediction
from `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`
(`research/findings/raw/_read_isolation_audit/synthesis.md` lines 17-20), which forecast the low-prior
(`--cue-to-expected-weight 0.4`) 3/6 "precision/homeostatic-companion wall" narrated in
`research/findings/2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md` would
flip to **~5/6** (seeds 100 AND 102) once reads are isolated. **The actual result is 4/6, not 5/6: only seed 102
flips FAIL->PASS; seed 100 does not.** The wall is a MIXTURE, not a pure artifact — this finding reports the
honest measured split.

## The fix ported (Port B, verbatim template)

Ported `_EXTRA_RESET_ARRAYS` from `research/runners/_crossedge_surprise_metacog_derisk.py` (L195-196, snapshot at
L322-328) into `_spiking_expectation_rpe_derisk.py`. The runner's OWN `_hard_reset()` already zeroed
membrane/recovery/conductances/firing but reset a **nonexistent** attribute `cp_refractory` (typo for
`cp_refractory_timers`; `getattr(bridge, "cp_refractory", None)` returned `None` -> the reset line was a dead
no-op) and never touched `cp_prev_firing_states` / `cp_neuron_activity_ema` / `cp_neuron_firing_thresholds` at
all. This runner does not override `cfg.enable_homeostasis` (default `True`, `sim/config.py:856`), so all 4
arrays are live and leak here — the worst case in the audit's severity ranking for this bug class.
`cp_neuron_firing_thresholds` is heterogeneous and non-zero at true rest (drawn per-neuron at build), so the fix
snapshots the true-rest value once in `build_expectation_circuit` (`bridge._rest_extra`) and restores it on every
`_hard_reset()` call, exactly as `_rest_v`/`_rest_u` already were. No `sim/` edit.

## Selftest: fails in the failing direction, passes with the fix

Added `selftest_read_isolation()` (`--selftest`): on a pool that is BOTH untrained (no learned recall) and
lesioned (`patient_expected->surprise` zeroed) — nothing left to legitimately vary — two consecutive
`measure_conditions()` calls on the same bridge must be bitwise identical. Verified both directions directly
(temporarily reverted `_hard_reset()` to the pre-fix body, re-ran, restored):

<!--derived-->
| `_hard_reset()` version | `confirm_per` read 1 | `confirm_per` read 2 | selftest |
|---|---|---|---|
| pre-fix (`cp_refractory` typo, no extra-array restore) | `[10.317, 8.730, 8.829, 8.532, 8.234]` | `[10.317, 8.829, 8.829, 8.532, 8.234]` | **FAIL** (2nd trial's confirm rate shifts 8.730 -> 8.829 Hz; `recall_hz` 10.040 -> 10.159, `confirm_hz` 8.929 -> 8.948) |
| fixed (`_EXTRA_RESET_ARRAYS` restored) | (same) | (same, bitwise) | **PASS** |

## BEFORE vs AFTER, gain=0.4 (the "wall" config) — 3/6 -> 4/6

BEFORE: `research/findings/raw/_spiking_expectation_rpe_6seed_lowprior.json` (banked, pre-fix `_hard_reset`).
AFTER: `research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.4_AFTER.json` (this run, fixed
`_hard_reset`, identical seeds/args otherwise).

<!--derived-->
| seed | confirm BEFORE->AFTER (Hz) | contradict ratio BEFORE->AFTER | novel ratio BEFORE->AFTER | GO BEFORE->AFTER |
|---|---|---|---|---|
| 42  | 1.210 -> 1.190 | 7.475x -> 7.500x | 8.213x -> 8.317x | Y -> Y |
| 43  | 1.468 -> 1.448 | 6.257x -> 6.342x | 5.878x -> 5.945x | Y -> Y |
| 44  | 4.623 -> 4.544 | 1.923x -> 1.948x | 1.863x -> 1.869x | **N -> N (unchanged)** |
| 100 | 3.492 -> 3.433 | 2.807x -> 2.855x | 2.369x -> 2.399x | **N -> N (unchanged, predicted to flip)** |
| 101 | 2.044 -> 2.044 | 3.883x -> 4.505x | 4.563x -> 4.515x | Y -> Y |
| 102 | 2.698 -> 2.579 | 2.794x -> 3.369x | 3.257x -> 3.392x | **N -> Y (flips, as predicted)** |

`n_go`: **3 -> 4**. `verdict_label` stays **BOUNDARY** (`verdict` stays **UNDEFINED** — the `Verdict` gate
requires >=5/6, unmet both times); the two failing preconditions in the AFTER artifact are `intact GO on >=5/6
seeds` (measured=4) and `separation ratio >= 3x (min over seeds)` (measured=1.869, seed 44's novel ratio) — see
`preconditions` in the cited JSON, a non-empty `{name, ok}` list per `tools.verdict.Verdict`.

**Seed 102 (predicted, confirmed):** both ratios move from below-3x to above-3x (2.794x/3.257x -> 3.369x/3.392x)
purely from the reset fix — no config change. This is the demonstrated read-isolation artifact: the seed's true
separation was always >=3x; the leak from the prior trial suppressed it below threshold.

**Seed 100 (predicted, NOT confirmed):** the synthesis's per-fact isolation diagnostic forecast 2.807x -> 4.372x.
The actual runner-level fix moves it only 2.807x -> 2.855x (novel ratio 2.369x -> 2.399x) — nowhere near 3x, let
alone 4.372x. <!--derived--> The two diagnostics disagree by roughly 1.5x on this seed; this finding trusts the
actual production-code re-run over the audit's lighter-weight isolation probe and reports seed 100 as a **genuine
low-prior separation failure**, not a leak artifact.

**Seed 44 (not predicted to flip, and does not):** ratios move by <0.03x in either direction — this seed's
failure was never attributed to the leak (see the original 2026-08-12 finding: "the seeds whose recall is weak ...
do not fully cancel confirm") and the fix confirms that attribution: this is the honest precision-sensitivity
wall, unaffected by measurement isolation.

## BEFORE vs AFTER, gain=0.8 (the shipped GO operating point) — 6/6 -> 6/6, margins HARDEN

BEFORE: `research/findings/raw/_spiking_expectation_rpe_6seed.json`. AFTER:
`research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.8_AFTER.json`.

<!--derived-->
| seed | contradict ratio BEFORE->AFTER | novel ratio BEFORE->AFTER |
|---|---|---|
| 42  | 22.80x -> 21.43x | 25.05x -> 23.76x |
| 43  | 30.87x -> 27.24x | 29.00x -> 25.53x |
| 44  | **3.50x -> 3.78x** | **3.39x -> 3.63x** |
| 100 | 17.03x -> 17.64x | 14.38x -> 14.82x |
| 101 | 12.50x -> 24.42x | 14.69x -> 24.47x |
| 102 | 10.00x -> 11.53x | 11.66x -> 11.61x |

`n_go` stays **6/6, GO** both before and after — the primary result is unaffected in category. As predicted, the
one thin margin at this operating point (seed 44, the closest to the 3x floor) **hardens** (3.50x/3.39x ->
3.78x/3.63x): the leak was working AGAINST this seed at gain 0.8, opposite of its effect at gain 0.4 (the sign of
this order-dependent leak is not fixed — it depends on which trial/condition happened to run immediately before,
consistent with the audit's general finding that the leak is order-dependent, not a constant bias).

## Verdict: matches the prediction's DIRECTION, not its MAGNITUDE — report both honestly

The audit's synthesis forecast **~5/6** for gain=0.4 (seeds 100 and 102 both flipping). The measured result is
**4/6** (only seed 102 flips). This is a genuine, partial confirmation:

- **Confirmed:** the read-isolation leak was real here and DID inflate the apparent severity of the low-prior
  wall by exactly one seed (3/6 measured -> 4/6 true). The 2026-08-12 finding's headline claim ("6/6 GO at
  gain=0.8, lesion-decisive") is untouched and its margins are now demonstrably tighter without the leak's help
  at the one seed that mattered (44).
- **NOT confirmed:** the predicted magnitude (~5/6) overshot. Seed 100's failure is NOT a read-isolation artifact
  at the production-code level, contradicting the audit's own lighter-weight per-fact isolation probe on that
  seed by ~1.5x. The "precision / homeostatic-companion-process" framing in the 2026-08-12 finding's "wall /
  companion process" section **survives** — the low-prior regime genuinely needs the homeostatic precision
  mechanism it names, on 2 of 6 seeds (44, 100), not the 3 originally measured.

**Per the project law** ("a fix that does not flip a false wall is itself a finding — the wall was real"): the
wall here was **mostly** real. One measurement artifact does not license retiring the "needs a homeostatic
precision companion" call to action; if anything the corrected, tighter number (4/6, with seed 100's own leak
effect now shown to be small) makes that companion process look MORE necessary, not less, since the residual
failures are cleanly attributable to the mechanism rather than partly hidden inside measurement noise.

## ⛔ Correction to `2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md`

<!--derived--> That finding's "The wall / companion process" section states low-prior gain=0.4 "intact GO drops
to **3/6**" citing `research/findings/raw/_spiking_expectation_rpe_6seed_lowprior.json`. Under the isolated
(fixed) `_hard_reset()`, the correct figure is **4/6** (this doc's cited artifact). This is a **PARTIAL**
correction: the qualitative claim ("the separation robustness scales with the gain match between the
recalled-prediction inhibition and the asserted excitation ... the honest next mechanism is a HOMEOSTATIC
intrinsic-plasticity precision on the prediction pool") **survives fully** — it is, if anything, strengthened by
this finding's seed-by-seed attribution. Only the specific number **3/6** is superseded by **4/6**. The gain=0.8
GO 6/6 headline is unaffected (margins tighten, verdict unchanged). No `docs/RETRACTED.md` row: `verdict_label`
does not flip GO<->NO-GO (BOUNDARY both before and after this fix), so this is a correction, not a retraction,
per `docs/WRITING.md` W1's registry criterion.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk --selftest

SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.4 \
    --out research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.4_AFTER.json

SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.8 \
    --out research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.8_AFTER.json
```

## Cupy re-verify (queued, not run here per this arc's scope)

The numpy result above hardens the primary gain=0.8 GO but leaves gain=0.4 at BOUNDARY (4/6, unchanged category)
— per the read-isolation audit's own re-verify plan (`synthesis.md` §6 row 2), the decisive cupy 6-seed re-verify
is queued via `tools/gpu_queue.sh` (see the companion commit) so the production backend confirms the same split
before any default-on decision is made for the low-prior regime. **This finding does not change any production
default** — the runner has never been wired default-on (it is a de-risk, not a live faculty).

## Provenance
- Fix template: `research/findings/2026-09-02-c2-metacog-read-isolation-fix-GO.md` +
  `research/runners/_crossedge_surprise_metacog_derisk.py` (`_EXTRA_RESET_ARRAYS` L195-196, restore L322-328).
- Bug-class audit: `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md` (FW-1)
  + `research/findings/raw/_read_isolation_audit/synthesis.md`.
- Corrects (PARTIAL, see ⛔ above): `research/findings/2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md`.
