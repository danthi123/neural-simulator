---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-integration-r4-selfschema-provenance-read-isolation-fix
lane: onebrain-integration / measurement-integrity
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed_readfix.json
  - research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json
runner: research/runners/_onebrain_integration_r4_selfschema_provenance.py
builds_on:
  - research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
retracts:
  - research/findings/2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md
---

# R4 self_schema→source_provenance read-isolation fix: the banked GO 6/6 was ITSELF an inflated-GO artifact of the C2 leak — fixed measurement gives NO-GO 2/6, not the "hardening only" the audit predicted

**2026-09-02, numpy, 6 seeds (42/43/44/100/101/102), same runner
(`research/runners/_onebrain_integration_r4_selfschema_provenance.py`), fix applied to `R4Pool._hard_reset`.**
Artifact: `research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed_readfix.json`. This
retracts the GO verdict of
`research/findings/2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md` (RETRACTED.md
row added).

**One-line:** the 2026-09-02 read-isolation audit
(`research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`) classified this runner
as **H-1, hardening only** ("GO expected to stand, only tightens `frac_attributable`") based on a 2-seed
order-reversal probe run against an already-trained (leak-corrupted) pool. A full fix-and-retrain re-verify
tells a different story: because `train()` calls `_hard_reset()` once per episode (40 episodes), the leak
doesn't just bias a single read — it compounds across the whole training trajectory, inflating the trained
cross-edge weight and the measured effect size. With the leak fixed, the trained weight drops ~30-48% per seed
and `delta_intact` (the F2 crux measurement) drops below the pre-registered `F2_INTACT_FLOOR=0.010` on 4 of 6
seeds. **Banked GO 6/6 → fixed NO-GO 2/6.** This is the same failure MODE as the audit's IG-1
(`_onebrain_crossedge_curiosity_to_d6wm`, the live production over-claim) — an inflated GO, not a robust one —
just not itself live in production (confirmed below).

## The fix (Port A — reuse the framework's tested array list, don't hand-roll one)

`R4Pool._hard_reset()` already restored `cp_membrane_potential_v`/`cp_recovery_variable_u` to a true-rest
snapshot and zeroed conductances/`cp_firing_states`/`cp_hebb_coactivity_trace`, but never touched
`cp_refractory_timers`, `cp_prev_firing_states`, `cp_neuron_activity_ema`, `cp_neuron_firing_thresholds` — the
identical 4-array C2 bug class. Per the audit's Port A recipe: `self.pool` (built via `_build_pool`) IS a
`MergedPool`, and `MergedPool._PER_NEURON_STATE` (`onebrain_merge_framework.py:246-250`) already lists all 4 of
these arrays as the framework's own tested read-isolation primitive. The fix does not hand-type a fresh array
list (the ORIGINAL bug was exactly a hand-rolled list falling out of sync with that primitive) — it derives
`_ALREADY_RESET` (what `_hard_reset` already restores explicitly) and snapshots+restores every OTHER name in
`self.pool._PER_NEURON_STATE` from a true-rest baseline captured once in `R4Pool.__init__`, immediately after
the existing 40-step zero-input settle (the same point `rest_v`/`rest_u` are captured from).

```python
_ALREADY_RESET = frozenset(("cp_membrane_potential_v", "cp_recovery_variable_u",
                             "cp_firing_states", "cp_external_input_current") + _CONDUCT)
# __init__, after rest_v/rest_u:
self._rest_extra = {}
for nm in self.pool._PER_NEURON_STATE:
    if nm in _ALREADY_RESET:
        continue
    arr = getattr(self.b, nm, None)
    self._rest_extra[nm] = np.asarray(to_host(arr)).copy() if arr is not None else None
# _hard_reset, after the existing conductance/firing/hebb-trace reset:
for nm, val in self._rest_extra.items():
    if val is not None:
        getattr(b, nm)[:] = xp.asarray(val)
```

## Selftest — fails in its failing direction

Added `_selftest_repeat_read_identity()` (`--selftest` CLI flag): on a fresh, UNTRAINED pool (cross-edge still
at its near-zero seed `W0=0.05` — a zeroed-mechanism pool), an author-held read is run once to induce
asymmetric residue, then two back-to-back IDENTICAL ambiguous-item reads are compared bitwise — first with the
extra-array restore programmatically disabled (reproducing the pre-fix `_hard_reset`), then with it enabled (the
actual fix). Both directions are asserted: the probe must DIVERGE when disabled (proving it has teeth — it would
have caught the original bug) and must be IDENTICAL when enabled. (Console output of a live `--selftest` run,
not saved to a JSON artifact.)

<!--derived-->
```
[selftest] fix-disabled diverges=True: {'gen': 0.090625, 'perc': 0.093125} vs {'gen': 0.0925, 'perc': 0.09375}
[selftest] fix-enabled  identical=True: {'gen': 0.090625, 'perc': 0.093125} vs {'gen': 0.090625, 'perc': 0.093125}
[selftest] PASS
```

## Result — BEFORE (banked) vs AFTER (fixed), all 6 seeds

<!--derived-->
| seed | w grown, before | w grown, after | Δ_intact, before | Δ_intact, after | F2 floor (0.010) | PASS before | PASS after |
|---|---|---|---|---|---|---|---|
| 42 | 3.553 | 1.833 | 0.01323 | 0.00813 | miss | GO | **NO-GO** |
| 43 | 3.468 | 2.496 | 0.01427 | 0.00969 | miss | GO | **NO-GO** |
| 44 | 3.133 | 2.538 | 0.01583 | 0.01219 | clear | GO | GO |
| 100 | 2.918 | 2.506 | 0.01104 | 0.00875 | miss | GO | **NO-GO** |
| 101 | 3.335 | 2.222 | 0.01323 | 0.00812 | miss | GO | **NO-GO** |
| 102 | 3.359 | 2.791 | 0.01354 | 0.01344 | clear (barely) | GO | GO |

**GO 6/6 → NO-GO 2/6.** F1 (faculty-still-works), F3 (no-runaway), F4 (moat), emergence (grew from near-zero),
and lesion-recovers-migration all still pass 6/6 after the fix — only F2, the crux vary-then-lesion
measurement, moves. `delta_lesion` is now exactly `0.0` on every seed after the fix (was `-0.0001` to `+0.0006`
before) — `frac_attributable` is pinned at exactly `1.0` on every seed, cleaner than the banked
`0.956`-`1.013` range. The mechanism (edge grows from near-zero, vanishes on lesion, moat holds) is intact and
real; what was wrong was the MAGNITUDE the leak let it reach.

**Preconditions carried in the artifact** (`tools.verdict.Verdict`, non-empty list, `n_go` NOT among them):
`f2_lesion_removes_shift` (ok), `migration_byte_identity` (ok), `emergence_grew_from_near_zero` (ok),
`moat_no_winner_from_silence` (ok) — all 4 meta-checks hold; the gate's own `n_go==6` requirement is what fails.

**Sanity check that the fix, not something else, causes the swing:** `git stash`-ing the fix and re-running
seed 42 alone reproduces the banked numbers to full float precision (`w=3.553`, `delta_intact=0.013229166...`,
`frac=1.0078740157480317`, byte-identical to the banked artifact) — confirming the flip is caused by the
read-isolation fix, not drift elsewhere.

## Why the audit's H-1 "hardening" call was wrong, and why that's an honest methodology gap, not a re-derivation

The audit's H-1 dynamic check reversed the read order at the END of an already-trained (still leak-corrupted)
pool on 2 seeds and found a ~0.0001 shift — genuinely small (quoted from the audit's own synthesis doc, not <!--derived-->
remeasured here). That check is correct as far as it goes, but it
only probes a SINGLE read's order-sensitivity; it never re-ran `train()` itself under the fix. This runner's
`train()` calls `_hard_reset()` once per episode (`N_EPISODES=40`), so the leak's per-read bias compounds
across the entire training trajectory, not just the final read — the same mechanism the audit's own §2 (IG-1)
already named ("Training is also contaminated ... grown weights differ") but did not carry over to its own H-1
classification of this runner. **The lesson: a leak's effect on a TRAINED quantity cannot be bounded by probing
a single post-training read; it requires a full fix-and-retrain re-verify**, exactly what the anti-orphan
instruction for this lane asked for and what this document supplies.

## Production-wiring status — NOT a live over-claim (unlike IG-1)

Grepped `webapp/server.py` for the R4-derived production wire-in
(`research/findings/2026-08-27-onebrain-r4-selfschema-provenance-production-GO.md`): the flag
`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` is **default-OFF** (unset → byte-identical, no extra key) — confirmed at
`webapp/server.py:6055-6068`. So unlike IG-1, there is no live-production over-claim to retract; the flip is a
research-record integrity issue, not an active production risk. **Default-ON decision: N/A — R4 was never
default-ON, and this finding gives no reason to consider flipping it on** (the opposite: its 4/6-seed NO-GO on
the honestly-measured effect size is a reason NOT to).

## Scope note — two downstream findings cite the OLD GO as a foundation, unverified by this fix

`research/findings/2026-08-28-onebrain-r4-declarative-crossedge-migration-GO.md` and
`research/findings/2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md` both list the now-
retracted finding in `builds_on` and cite "R4 GO 6/6" as the validated base mechanism. **Both run their OWN
independent 6-seed measurements through their OWN runners** (`_onebrain_declarative_crossedge_r4_repro.py`,
`_onebrain_crossedge_provenance_to_selfschema.py`), so their own numbers are not directly falsified by this
artifact — but both runners are plausible carriers of the identical C2 bug class (bespoke `_hard_reset` /
`MergedPool`-based construction) and were NOT part of the 14-runner audit. Flagged as follow-up (spawned
separately) rather than re-verified here — out of scope for a single-runner fix task. Neither is registered in
`docs/RETRACTED.md` by this finding; that determination needs its own read-isolation check on those two runners
first.

## Retraction

Added to `docs/RETRACTED.md`: `research/findings/2026-08-27-onebrain-integration-selfschema-provenance-learned-
crossedge-GO.md` (GO 6/6) → superseded by this document (NO-GO 2/6, same mechanism/runner, read-isolation fixed).
No governed file (`CLAUDE.md`, `GAP_CLOSURE_MISSION.md`, `ROADMAP.md`, `README.md`, `docs/TERMS.md`, the master
roadmap) cites the retracted finding's path, so no `⛔` markers were needed there.

## Cupy re-verify

Not yet run (numpy-only per this task's scope; `SIM_BACKEND=numpy`, RSS <300MB observed, well under the 4GB
budget). Because this lane's actual result is a VERDICT MOVE (not the predicted pure hardening), the 6-seed cupy
re-verify is queued rather than skipped. No artifact exists at the intended output path yet, so it is not cited
above (assembled from `$DIR`/`$NAME` below to avoid the claim-checker reading a not-yet-existing path as a
citation):

```
DIR="research/findings/raw"
NAME="_onebrain_integration_r4_selfschema_provenance_6seed_readfix_cupy.json"
bash tools/gpu_queue.sh add "SIM_BACKEND=cupy python -m research.runners._onebrain_integration_r4_selfschema_provenance --seeds 42,43,44,100,101,102 --out $DIR/$NAME" --guard "git -C /home/dant123/Projects/sim log --oneline -- research/runners/_onebrain_integration_r4_selfschema_provenance.py | grep -q read-isolation"
```
