---
type: finding
status: live
lane: load-bearing
date: 2026-09-22
---

# source-provenance-honesty: the abstain-at-tie fix makes it load-bearing 6/6 seeds (2026-09-22)

Follow-on to the fragile-step LOCATION in
`research/findings/2026-09-22-borderline-separability-stabilizer-is-buildable.md`, which found source-provenance's
opponent read `d` separates intact-vs-lesion PERFECTLY on every seed (min-intact 1.000, max-lesion 0.000), yet the
production battery (`research.runners.load_bearing_fraction --only source-provenance-honesty`) read it
load-bearing only 4/6 seeds pre-fix (off@s44, s102). That finding pinned the fragile step: `_judge()`'s HOST
TIE-BREAK coin-flips `winner` from a per-monitor RNG at the lesion's genuine no-signal collapse
(`rate_perceived == rate_generated`, `d == 0.0`), and that per-seed coin lands the lesion's label on the SAME
side as the intact arm's confident "perceived" on 2 of 6 seeds — masking the lesion there.

## The fix

`BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE` (default OFF; `research/runners/source_provenance_honesty.py`): at a genuine
opponent no-signal collapse (`|d| < TIE_D_EPS = 1e-6`), `SourceProvenanceHonestyMonitor.judge_fact()` now reports a
DETERMINISTIC abstain (`label=None`) instead of forwarding `_laneC_source_provenance_opponent_derisk._judge()`'s
seed-dependent host coin-flip. `_judge()` itself is UNCHANGED (it is the validated, 6-seed-confirmed de-risk primitive, and
its own docstring is explicit the coin-flip is deliberate for a genuinely no-signal control). `label=None` is not
a new sentinel: it is the exact value `judge_fact()` already returns for a never-encoded key, and
`provenance_framed_text()`'s own docstring already names this branch ("the judgment ties/is undecided ->
UNCHANGED") — the mapping existed as documented, unwired intent. This is an HONEST functional read-out (a
collapsed opponent genuinely cannot report "I saw this" vs "I inferred this"; abstaining is the truthful state),
not tuning: `TIE_D_EPS` is derived from the SAME float-exact-tie floor `_judge()` itself already uses on the raw
margin (`1e-9`), and the intact arm's `d` sits nowhere near it BY CONSTRUCTION, not by luck: for a purely-perceived
fact `rate_generated` reads exactly 0 (the opposing pool never fired), so `d = rp/(rp+0+1e-9)` and
`1-d ~= 1e-9/rp` — six-plus orders of magnitude below `TIE_D_EPS=1e-6` for any firing rate above ~1e-3 Hz. Every
intact `d` this fix's own verify artifacts measured across all 6 seeds exceeds 0.9999998 (see the per-seed table
below for the exact values) — the fix can only ever engage on the already-degenerate no-signal state, never on a
confident intact read.

## Result: load-bearing 6/6 with the flag ON, byte-identical to the pre-fix baseline with it OFF

<!--derived-->
| seed | flag | `provenance.d` intact / lesion | `provenance.label` intact / lesion | treatment_diffs | load_bearing |
|---|---|---|---|---|---|
| 42  | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 43  | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 44  | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 100 | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 101 | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 102 | ON  | 1.000 / 0.000 | perceived / None       | 2 | **True** |
| 44  | OFF | 1.000 / 0.000 | perceived / perceived  | 0 | False (matches pre-fix `research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s44.json`) |
| 102 | OFF | 1.000 / 0.000 | perceived / perceived  | 0 | False (matches pre-fix `research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s102.json`) |

6/6 mandated seeds (42/43/44/100/101/102) read `load_bearing: true` with the flag ON — the pre-fix 4/6
(off@s44,s102) is now robust 6/6. With the flag OFF (its shipped default), s44 and s102 reproduce the pre-fix
`_adequate6` battery's own recorded `treatment_diffs=0`/`load_bearing=false` EXACTLY (an exact value compare
against that already-committed artifact, not an inferred-from-code claim) — the fix is inert unless explicitly
enabled.

## Byte-identical when OFF (asserted in the data, not inferred from the code)

- `tests/test_source_provenance_honesty_wirein.py` (6 tests, all pre-existing literal-value assertions on
  `judge_fact()["label"]` / `answer_text`) PASS UNCHANGED with the fix in place and the flag unset.
- `research.runners.load_bearing_fraction --selftest` PASSES unchanged (no brain build; instrument-logic only).
- Full re-confirmation at the two previously-fragile seeds (44, 102), flag OFF: `treatment_diffs=0`,
  `control_diffs=0`, `load_bearing=false`, `verdict="pass"` — an EXACT match, field by field, to the
  already-committed pre-fix artifacts `research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s44.json`
  and `..._s102.json` (produced by an earlier session, before this fix existed). The LESION arm (the one arm the
  fix actually touches) was rebuilt FRESH under an environment with `BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE` unset; the
  two INTACT arms were reused byte-identical from the flag-ON run rather than redundantly rebuilt, because the
  flag provably cannot change their measured field (source-provenance's intact `d` exceeds 0.9999998 on
  every seed this fix observed, nowhere near `TIE_D_EPS=1e-6`) — `load_bearing_fraction.py`'s own
  `LB_RESUME_SKIP_EXISTING` resume path (loads an existing arm file instead of rebuilding: no env check, just
  `os.path.exists` + `json.load`) makes this an explicit, code-verified reuse rather than an assumption. The
  per-run `source_prov_abstain_at_tie_env` stamp confirms the flag actually read back unset for the fresh lesion
  build, not merely unpassed.

## Artifacts (numpy CPU, `BRAIN_CHAT_SEED` threaded per seed, memcapped)

`research/findings/raw/_lbf_fix_source_prov_abstain/lbf_s{42,43,44,100,101,102}_flagON.json`,
`research/findings/raw/_lbf_fix_source_prov_abstain/off/lbf_s{44,102}_flagOFF.json` (a separate `off/`
subdirectory -- the per-arm filenames do not encode the flag, so an OFF re-run sharing the ON run's directory
for the same seed would silently overwrite that seed's ON lesion artifact),
`research/findings/raw/_lbf_fix_source_prov_abstain/VERDICT.json` (the `tools.verdict.Verdict` preconditions
block over all 8 per-seed reads, via `research/runners/_lbf_fix_source_prov_abstain_verify.py`).

## Mechanism note — what this is and is not

The fix touches only the INTEGRATION step (the `d`->`label` discretizer inside the production wrapper's
`judge_fact()`), exactly the step the prior finding located. It does not touch `_judge()`, the opponent
comparator, the Hebbian encode/recall pathway, or the lesion mechanism itself — the lesion still collapses the
SAME substrate quantity (`rate_perceived`/`rate_generated` both silent) it always did; what changed is that the
readout of a collapsed, no-signal state is now reported honestly (abstain) rather than resolved by a coin flip
that happened to sometimes land on the confident answer. No phenomenal claim is made: this is a functional
correlate of source monitoring (Johnson-Hashtroudi-Lindsay 1993), read out from a spiking opponent comparator.
