---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-crossedge-provenance-to-selfschema-read-isolation-fix
board: one-brain integration / measurement-integrity (C2 bug class, a 15th instance found outside the audited 14)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_crossedge_provenance_to_selfschema_readfix_6seed.json
runner: research/runners/_onebrain_crossedge_provenance_to_selfschema.py
builds_on:
  - research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
  - research/findings/2026-09-02-onebrain-r4-selfschema-provenance-read-isolation-fix-flips-GO-to-NOGO.md
  - research/findings/2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md
---

# The provenance→self-schema reciprocal edge has its OWN, previously-unaudited instance of the C2 read-isolation bug (not inherited from R4) — fixed + selftest-verified; the GO 6/6 SURVIVES, but with a substantially narrower margin than banked (headroom over floor drops from ~2.1-2.4x to ~1.18-1.37x)

**One-line:** the record-follow-through task asked whether
`2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md`'s verdict depends on R4's now-corrected
result (`2026-09-02-onebrain-r4-selfschema-provenance-read-isolation-fix-flips-GO-to-NOGO.md`). It does not — the
reciprocal edge's own `ProvToAuthorPool` trains an entirely separate, independently-initialized cross-edge
(`provgen_to_author`, `W0=0.05`) and never reads R4's `author_to_provgen` weight. But inspecting
`ProvToAuthorPool._hard_reset()` directly (this file predates, and was not part of, the 14-runner audit) found it
is a byte-for-byte copy of the SAME vulnerable hand-rolled `_hard_reset` shape the audit's H-1/IG-1 items named —
missing `cp_refractory_timers`/`cp_prev_firing_states`/`cp_neuron_activity_ema`/`cp_neuron_firing_thresholds` — a
**15th, previously-undiscovered instance of the C2 bug class**, unrelated to R4's own leak but the identical code
pattern. Fixed with the same Port A recipe R4's own fix used (`fa4e10271`); a new fails-in-failing-direction
`--selftest` confirms the fix has teeth (diverges when disabled, bitwise-identical when enabled). The honestly
re-measured 6-seed verdict is **GO 6/6, unchanged in DIRECTION**, but the grown weight and the load-bearing
`delta_intact` both shrank substantially — this rung's own "2.1-2.4x headroom over the registered floor" claim is
now false; the true headroom is 1.18-1.37x.

## 1. The independently-discovered bug (not delegated from R4)

`research/runners/_onebrain_crossedge_provenance_to_selfschema.py`'s `ProvToAuthorPool.__init__`/`_hard_reset`
(lines 150-186 pre-fix) restore `cp_membrane_potential_v`/`cp_recovery_variable_u`/the 9-array `_CONDUCT`
tuple/`cp_firing_states`/`cp_hebb_coactivity_trace` — the SAME set every pre-fix `_hard_reset` in this codebase
restored — but never touched the audited 4-array C2 class. `train()` calls `_hard_reset()` every one of 60
episodes (compounding the leak across the whole training trajectory, exactly the R4/curiosity-d6wm shape); `read_
author()` also calls it once per read (4 reads averaged per condition). This file is dated 2026-09-01 and was
never included in the 14-runner audit (`2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`), so
neither the audit's own prediction table nor either landed fix commit (`ffa229876`, `fa4e10271`) touched it — it
was found only because this record-follow-through task's own dependency check inspected the file's source
directly rather than assuming "doesn't cite R4's weight" meant "unaffected."

## 2. The fix — Port A, identical recipe to R4's own fix

Added `_ALREADY_RESET` (module-level, mirrors R4's own `fa4e10271` constant) naming the arrays `_hard_reset`
already restores explicitly; snapshotted every OTHER name in `MergedPool._PER_NEURON_STATE`
(`onebrain_merge_framework.py:246-250`) at true rest in `ProvToAuthorPool.__init__` (immediately after the
post-build 40-step settle); restored the full snapshot on every `_hard_reset()` call. Added
`_selftest_read_isolation()` / `--selftest` (mirrors R4's own `_selftest_repeat_read_identity`): on a fresh pool,
after driving `author`+`ctx_generated` to induce asymmetric residue, two back-to-back `read_author("perceived")`
calls must be bitwise identical.

**Selftest run, both directions (fails-in-failing-direction guard, not assumed to have teeth).** The table below
is ad hoc `--selftest` stdout captured this session, not a saved JSON artifact — marked accordingly:

| | fix DISABLED (simulates pre-fix `_hard_reset`) | fix ENABLED (this commit) |
|---|---|---|
| repeat-read `{author, gen, perc}` <!--derived--> | `{'author': 0.0, 'gen': 0.0139..., 'perc': 0.0955...}` then `{'author': 0.0, 'gen': 0.0122..., 'perc': 0.0963...}` — **DIVERGE** | `{'author': 0.0, 'gen': 0.0125, 'perc': 0.0947}` then identical — **bitwise IDENTICAL** |

Restoring `MergedPool._PER_NEURON_STATE` alone was sufficient here (unlike the curiosity-d6wm pair, this read is
single-phase — one `_drive()` call per read, not a multi-turn load-then-scored-read — so the `_SEQ_EXTRA_STATE`
extension that pair additionally needed was not required; confirmed empirically by the selftest passing without
it, not assumed by analogy).

**Also fixed the same latent `Verdict` outcome-as-precondition bug** the curiosity-d6wm fix found:
`main()` wrapped `all_seeds_go` (the OUTCOME) as a `Vd.require()` precondition; per
`tools/gates/verdict_preconditions.py`'s rule this is latent-harmless only while the outcome never fails. Fixed
to pass the outcome directly to `Vd.decide()`; preconditions now carry only the 2 genuine validity checks
(`lesion_removes_bias`, `byte_identical_off`).

## 3. The result — GO 6/6 survives, margin narrows substantially

BEFORE (banked, the original 2026-09-01 finding's own table, quoted verbatim — marked `<!--derived-->` since it
is not from an artifact this finding cites) vs AFTER (`research/findings/raw/
_onebrain_crossedge_provenance_to_selfschema_readfix_6seed.json`, this finding's own cited artifact), full
precision:

| seed | grown BEFORE <!--derived--> | grown AFTER | Δintact BEFORE <!--derived--> | Δintact AFTER | Δlesion (both) | frac_attrib (both) | byte-off (both) | GO (both) |
|---|---|---|---|---|---|---|---|---|
| 42  | 4.204289 <!--derived--> | 2.741609 | +0.022083 <!--derived--> | +0.013000 | +0.000000 | 1.0000 | PASS | GO |
| 43  | 4.670042 <!--derived--> | 2.777717 | +0.023750 <!--derived--> | +0.011833 | +0.000000 | 1.0000 | PASS | GO |
| 44  | 4.439690 <!--derived--> | 3.082381 | +0.022083 <!--derived--> | +0.013667 | +0.000000 | 1.0000 | PASS | GO |
| 100 | 4.021028 <!--derived--> | 2.905727 | +0.021167 <!--derived--> | +0.013500 | +0.000000 | 1.0000 | PASS | GO |
| 101 | 4.434182 <!--derived--> | 2.698699 | +0.022042 <!--derived--> | +0.012667 | +0.000000 | 1.0000 | PASS | GO |
| 102 | 4.340328 <!--derived--> | 2.851291 | +0.023208 <!--derived--> | +0.013167 | +0.000000 | 1.0000 | PASS | GO |

`INTACT_FLOOR=0.010` (unchanged). Every seed still clears the floor — `n_go` stays **6/6**; `payload.GO` stays
**true**. `delta_lesion` was already exactly `0.0` before the fix (this rung's `author` population has no other
drive source in the reduced read protocol, so the lesion control was never leak-corrupted the way the parent
r4/curiosity-d6wm reads were) and remains exactly `0.0` after — the lesion-attributability claim is genuinely
unaffected. `no_corruption` (max\|Δ\| over every non-edge synapse) and `byte_off` (the no-edge baseline's
connectivity) both PASS on all 6 seeds, both before and after — the fix touches only the harness reset, never
the pool's wiring.

### 3a. What changed, numerically (a mix of cited-artifact values and quotes from the original finding)

<!--derived-->

**What changed, and by how much:** the grown weight fell ~27-39% per seed (e.g. seed 43: 4.670 → 2.778, a 40.5%
drop); `delta_intact` fell ~39-46% per seed (e.g. seed 43: 0.02375 → 0.01183, a 50.2% drop — the single largest
proportional shrink of the six). **Headroom over the registered floor** (`delta_intact / INTACT_FLOOR`) — the
number the original finding characterized as "2.1-2.4x headroom" (§3 of that finding) — is now **1.18x-1.37x**
(min at seed 43: 0.011833/0.010 = 1.1833; max at seed 44: 0.013667/0.010 = 1.3667). The verdict category (GO)
does not change, but a reader relying on the original headroom characterization to judge this rung's robustness
margin would be materially misled — this is why a PARTIAL retraction row is added for that specific claim (§5),
even though the GO itself is not retracted.

**Why this shrink, unlike the curiosity-d6wm pair, did not flip any seed:** that pair's pre-fix headroom
(computed the same way against its own `INTACT_FLOOR=0.008`) ranged roughly 1.25x-1.78x across its 6 seeds —
already close enough to the floor that a leak-driven ~40-75% swing (documented in that finding, both directions
depending on seed) pushed half the seeds under 1.0x. This rung's pre-fix headroom (2.1-2.4x) had roughly double
the margin, so an isolated-read correction of comparable proportional size (~40-50% shrink) left every seed still
above 1.0x, if narrowly on the weakest seed (43, at 1.18x).

## 4. Answering the task's own question: does this depend on R4's corrected result?

**No, not numerically.** `ProvToAuthorPool` never reads R4's `author_to_provgen` cross-edge weight or R4's
trained pool state — the two edges are declared, trained, and read entirely independently (this file's own
`CROSS_EDGES` list contains only `provgen_to_author`; R4's edge is absent from this pool, per the module
docstring's own "nothing is double-grown" claim, confirmed unaffected by this fix — `no_corruption` stayed exact
zero). **Yes, structurally** — both files share the SAME author, the SAME hand-rolled `_hard_reset` house style
(explicitly documented as reused: "R1/R4 house style" in both files' own comments), and evidently the SAME
omission was copied when this file was written from R4's own pre-fix template. The correct framing per the
task's own distinction: this finding's GO does not depend on R4's WEIGHT, but it independently INHERITED R4's
BUG, because the bug lived in a code pattern that was copied, not in a value that was read.

## 5. Retraction scope — PARTIAL, the GO itself is not retracted

<!--derived-->

`docs/RETRACTED.md` gets a PARTIAL row for `2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-
GO.md` (quoting that finding's own superseded figures, not an artifact this finding cites): the specific
grown-weight range ("4.0-4.7 across seeds"), the specific `delta_intact` range ("+0.021 to +0.024"), and the
"2.1-2.4x headroom" characterization are superseded by the values in §3 above. The `GO 6/6` verdict, the
lesion-attributability (`frac_attributable=1.0`), the byte-identical-off property, and the mechanism itself (a
genuinely Hebbian-grown reciprocal edge) all survive unretracted.

## 6. A broader residual, flagged not chased

A quick grep for the SAME hand-rolled `_hard_reset` shape (present, but lacking `_rest_extra`/`_PER_NEURON_STATE`
restoration) across `research/runners/*.py` returns ~29 files beyond the 14 the original audit covered and the
one this finding fixes. Most are very likely already covered by the audit's OWN clean-verdict reasoning (warmup
washout, or a lesion numerator structurally pinned to zero) rather than genuinely vulnerable — the audit's own
§"9 ROBUST" section explicitly named several files with this exact shape as clean for those reasons. This
finding does NOT re-audit all 29; that is a distinct, larger task outside this lane's scope (read-isolation
record-correction for two specifically-named findings), flagged here so it is not lost, not silently expanded
into.

## 7. Files

`research/runners/_onebrain_crossedge_provenance_to_selfschema.py` (MODIFIED — `_ALREADY_RESET`, `__init__`/
`_hard_reset` read-isolation fix, `_selftest_read_isolation`, `--selftest` CLI, the `Verdict` outcome-as-
precondition fix) · `research/findings/raw/_onebrain_crossedge_provenance_to_selfschema_readfix_6seed.json`.
Reused, unmodified: `research/runners/onebrain_merge_framework.py` (`MergedPool._PER_NEURON_STATE`) ·
`research/runners/_onebrain_integration_r4_selfschema_provenance.py` (`AUTHOR_PA`, `CTX_DRIVE_PA`, `TRAIN_STEPS`,
`_CONDUCT` — constants/primitives only, no logic reimplemented, per the original finding's own Files section).
No `sim/` file touched; no `webapp/server.py` edit (this rung was never wired into production — the original
finding's own §6 declares "Not yet wired into production").

Functional read-outs only; no phenomenal-experience claim.
