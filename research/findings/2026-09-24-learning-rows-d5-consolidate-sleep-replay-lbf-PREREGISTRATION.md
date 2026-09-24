---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: two new load-bearing-fraction ROWS for the two shipped, default-ON offline-learning faculties -- D5
  learn-through-use consolidation (`BRAIN_D5_CONSOLIDATE`, webapp/continuous_engine.py) and offline sleep-replay
  (`BRAIN_SLEEP_REPLAY`, webapp/continuous_engine.py) -- delivered through the `research/runners/lbf_rows/` row
  interface (EXTRA_LESIONS/EXTRA_PROBES) rather than by editing FACULTY_LESIONS/FACULTY_PROBES directly. Both
  rows drive the SAME substrate write the ledger already credits (`sim/bridge.py fused_btsp_update`, the
  plateau-gated BTSP kernel) via a genuine idle tick, reusing the label-only world-step mechanism the DA
  tag-capture probe's `datc_night` already exercises (`_WORLD_NIGHT` -> `_run_world_step("overnight_24h")` ->
  `webapp.continuous_engine.tick_idle_sessions`, called through the REAL `webapp.server.brain_chat` path via
  `research.runners.onebrain_regression_battery._spawn_arm`).
seeds: [7]
verdict: PRE-REGISTRATION only, filed before any d5c_*/slp_* arm existed. Seed 7 is a DEV/calibration seed
  (outside 42/43/44/100/101/102) per this lane's scope -- no GO/NO-GO capability claim is made here. The 6-seed
  gate is explicitly NOT this lane's job (rule: "Do NOT run 6-seed evaluations"); this document also stages the
  exact 6-seed job lines for the orchestrator's battery.
runner: research/runners/load_bearing_fraction.py (rows merged from research/runners/lbf_rows/learning.py)
artifacts:
  - research/findings/raw/_sleep_replay_flip/soak_summary_6seed.json
  - research/findings/2026-08-21-d5-learn-through-use-flip-GO-per-topic-strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact-not-substrate-crosstalk.md
note: this prereg's OWN seed-7 smoke artifacts do not exist yet -- they are produced by this branch AFTER this
  commit, per prereg-before-run discipline, and are cited in the follow-on data finding, not here.
---

# d5-consolidate / sleep-replay load-bearing rows: PRE-REGISTRATION (filed before any measured run)

**Filed 2026-09-24 in its own commit on branch `research/lbf-rows-learning`, cut from origin/main `1ad61df45`.**
No d5c_*/slp_* arm had been built when this was committed. Lane A2 of the 2026-09-24 midnight plan, step S13.

## Why these two rows, and why now

The #1 metric (`research/runners/load_bearing_fraction.py`) currently has no row for either shipped, default-ON
offline-learning faculty:

- **D5 learn-through-use consolidation** (`webapp/continuous_engine.py:159-308`, `_D5_CONSOLIDATE_DEFAULT_ON =
  True` since 2026-08-21): a memory the brain RECALLED during a live turn is re-activated during the next idle
  tick, and the substrate's own plateau-gated BTSP (`fused_btsp_update`, `IS_post = max(cp_v_apical - v_hold,
  0)`) strengthens its within-assembly weights. The graded strength is surfaced in a LATER recall's reply
  (`research/runners/d5_episodic_production_organ.py:263-305`, `recall_disclosure`) only for a topic that was
  actually consolidated this conversation.
- **Offline sleep-replay** (`webapp/continuous_engine.py:311-390`, `_SLEEP_REPLAY_DEFAULT_ON = True` since
  2026-08-26, 6-seed pool/GPU soak GO): on a genuine sleep-depth idle (>= `SLEEP_IDLE_SEC` = 300 s), the BATCH of
  episodes stored since the last sleep is reactivated in store-order through the same BTSP kernel, so a later
  recall of any replayed topic reads a stronger retention AND surfaces the host store-order WHEN-position.

Both are declared spiking-learning rows in the midnight plan's #1-metric section: *"The learning rows
d5-consolidate and sleep-replay count as spiking learning rows"* -- because, unlike the composer's plain KB
append (see the in-loop-learning note below), the weight change IS the substrate's own plasticity kernel, not a
host `dw` formula.

## Row interface (no edit to FACULTY_LESIONS/FACULTY_PROBES)

`research/runners/lbf_rows/learning.py` exposes:

```python
EXTRA_LESIONS = {
    "d5-consolidate": dict(flag="BRAIN_D5_CONSOLIDATE", value="0", kind="neural-lesion", note="..."),
    "sleep-replay":   dict(flag="BRAIN_SLEEP_REPLAY",   value="0", kind="neural-lesion", note="..."),
}
EXTRA_PROBES = [
    ("d5-consolidate", "d5c_recall2", ["episodic.graded_cue.depth_hold", "answer"], False),
    ("sleep-replay",   "slp_recall",  ["episodic.graded_cue.depth_hold", "answer"], False),
]
```

Both flags are the SHIPPED default-ON master switches (`BRAIN_D5_CONSOLIDATE`/`BRAIN_SLEEP_REPLAY` unset =
True); `value="0"` is the LESION (matches the FACULTY_LESIONS convention: a `dict(flag=..., value=..., kind=...,
note=...)` shape identical to every existing row, e.g. `episodic-memory`/`prospective-memory`). `kind=
"neural-lesion"`: the flag cuts the substrate's OWN BTSP re-activation loop (`consolidate_used_memory` /
`consolidate_sleep_replay` become no-ops), the organ stays installed. AG-REG (S08/S26) merges these into
`FACULTY_LESIONS`/`FACULTY_PROBES` via the import hook; until that hook lands, this prereg's own smoke
monkeypatches the merge in-process (`FACULTY_LESIONS.update(EXTRA_LESIONS)` / `FACULTY_PROBES.extend(...)`,
documented in the smoke script itself) to validate end-to-end without touching either literal registry.

## The graded diff field (recall margin), and why it -- not a categorical flag -- is what changes

`episodic.graded_cue.depth_hold` is the SAME `SURFACED_GRADED_READ` the ledger already validated
(2026-08-20-d5-graded-apical-read finding): the mean-held `max(cp_v_apical - v_hold, 0)` apical read, a genuine
spiking magnitude, not a host formula. It is NOT in `onebrain_regression_battery._NOISE_FIELDS` (checked by
name), so `compare()`'s exact-value inequality applies to it unmodified -- with the harness deterministic at a
fixed seed, any post-tick value difference is attributable to the lesion, not run-to-run noise. `answer` (the
rendered reply string) is compared alongside it because `recall_disclosure` embeds the same magnitude as
"recall strength X.X mV" in the lead sentence only when `_d5_strength_visible() and _topic_consolidated(...)`
(D5) or `_sleep_replayed_when(...)` (sleep-replay) hold -- so `answer` differs by construction whenever
`depth_hold` differs. `episodic.in_memory` is NOT in the compared-row fields (deliberately): it is True in both
arms on both recalls (the completion gate is not what these two faculties lesion), so including it would only
ever read `pass` and add nothing to the decision; the graded field is where the lesion actually bites.

## The two turn groups (label-only; added to `_EXTRA_TURNS`, NOT to `PROBE_TURNS`)

Both reuse the EXACT `_WORLD_NIGHT` sentinel and `_run_world_step("overnight_24h")` the DA tag-capture probe's
`datc_night` already exercises (`webapp.continuous_engine.tick_idle_sessions`, called through the real
`_get_episodic_organ_existing` / `_get_affect_organ` / `_get_selfinit_organ` getters exactly as the server's
background loop calls it) -- no new world-step kind is introduced. 24 h idle trivially clears both
`IDLE_SEC` (20 s, gates D5) and `SLEEP_IDLE_SEC` (300 s, gates sleep-replay), so one step exercises both ticks;
each row's own turn group only asks for the recall(s) it needs.

**`d5c` session** (fresh, never used by another probe):

| label | message | reset | what it does |
|---|---|---|---|
| `d5c_teach` | "the wolf chase the rabbit" | True | teaches (wolf, chase, rabbit); `note_topic("wolf")` BTSP-forms the CA3 assembly |
| `d5c_recall1` | "you mentioned the wolf" | False | referential recall #1 (Hook A): a genuine completion (`in_memory=True`) -> `mark_recall` arms the D5 budget. PRECONDITION turn: intact and lesion must read IDENTICAL here (no tick has run yet), so the row does not confound "recall works at all" with "consolidation happened" |
| `d5c_tick` | (`_WORLD_NIGHT`, label-only) | False | the idle tick: intact arm consolidates the recalled 'wolf' assembly; lesion arm (`BRAIN_D5_CONSOLIDATE=0`) no-ops |
| `d5c_recall2` | "you mentioned the wolf" | False | referential recall #2, THE DRIVING TURN: intact reads a risen `depth_hold` + the "recall strength" clause; lesion reads the SAME `depth_hold` as recall1 (nothing consolidated) |

**`slp` session** (fresh):

| label | message | reset | what it does |
|---|---|---|---|
| `slp_teach1` | "the fox chase the hare" | True | stores episode 1 (fox) |
| `slp_teach2` | "the owl chase the mouse" | False | stores episode 2 (owl) |
| `slp_teach3` | "the hawk chase the vole" | False | stores episode 3 (hawk) |
| `slp_tick` | (`_WORLD_NIGHT`, label-only) | False | a genuine sleep-depth idle (24 h >> 300 s): intact arm batch-replays all 3 stored episodes in store order; lesion (`BRAIN_SLEEP_REPLAY=0`) no-ops |
| `slp_recall` | "you mentioned the owl" | False | THE DRIVING TURN, recalling the MIDDLE-stored episode: intact reads a risen `depth_hold` + the "I also replayed it offline" clause with a when-rank/batch-size; lesion reads the un-replayed baseline |

No teach-then-immediate-recall pair is added for sleep-replay (unlike d5-consolidate) because the faculty is
defined on the BATCH, not a single held topic; the immediate-recall precondition is instead the observation (in
the smoke, not the registered row) that `slp_recall`'s IN_MEMORY gate is True in both arms -- the batch-replay
lesion never touches whether the topic completes, only how strong it reads.

## Held-through-the-tick assertion

Because each arm is a fresh subprocess build (`_spawn_arm`), there is no in-process hook to assert against mid-
run without editing `webapp/continuous_engine.py` (out of scope: additive-only, no edit to shipped mechanism
code). The assertion is instead a DATA check the smoke script performs on the two arms' own JSON: in the LESION
arm, `d5c_recall2.episodic.graded_cue.depth_hold` must equal `d5c_recall1.episodic.graded_cue.depth_hold` EXACTLY
(the tick ran -- `n_sessions_ticked` from the world-step response is > 0 -- but the flag held it a no-op), and
`slp_recall`'s `depth_hold`/`answer` in the lesion arm must show no "replayed it offline" clause. A tick that
silently did not run at all (e.g. `IDLE_SEC` not cleared) would produce the SAME pre/post equality in the INTACT
arm too, which the driving-turn comparison against intact's ACTUAL non-equality already rules out.

## `in-loop-learning`: investigated, not re-registered (existing FACULTY_LESIONS/PROBES key, left untouched)

Per the plan: *"in-loop-learning is either a DG-CA3-store lesion or a labeled host-write row."* Read
`research/runners/one_brain_composer.py:604-655` (`hear`/`hear_multiframe`/...): every one of these fact-
acquisition paths ends in `self.kb.append((fact, None))` -- `self.kb` is a plain Python list on
`OneBrainComposer`, not the spiking `EpisodicDapMemory` DG-CA3/BTSP store the two rows above drive. The
production chat-teach path (`webapp/server.py`, the normal SVO-teach turns) writes new facts into this SAME
`kb` list. **Conclusion: the current default `in-loop-learning` path is a HOST-WRITE** (a Python list append),
not a DG-CA3 store write -- consistent with its existing `FACULTY_LESIONS["in-loop-learning"]` entry
(`kind="in-process"`, lesioned via `chat._substrate_recall`, no env knob) and with A8's D6 Hebbian-store lane
being a still-default-OFF, declared "near-copy of the host pattern" attempt at a synaptic version. This document
does NOT edit `FACULTY_LESIONS`/`FACULTY_PROBES` for `in-loop-learning` (that key already exists; only AG-REG
owns literal registry edits); it hands this classification to AG-REG/S26 for the #1-metric coverage table.

## Honesty boundary / declared residuals

- **Host clock, brain-based write.** The idle tick's OCCURRENCE and its `IDLE_SEC`/`SLEEP_IDLE_SEC` thresholds
  are host timer infrastructure (mirrors every other `continuous_engine` mechanism, already declared there); the
  WEIGHT CHANGE the tick triggers is the substrate's own plateau-gated BTSP kernel (`fused_btsp_update`), not a
  host `dw` formula -- this is why the plan credits both rows as spiking learning rows.
- **Recall CONTENT residual.** The recalled fact's rendered CONTENT sentence is the same host-oracle KB lookup
  the existing `episodic-memory` row already declares (`webapp/server.py:5380-5396`); only the recall GATE +
  the graded STRENGTH are spiking. Not a new residual, inherited unchanged.
- **WHEN-order residual (sleep-replay only).** The replay batch order / when-rank surfaced in the reply is the
  DECLARED host store-order recency field (`EpisodicRecallOrgan.recency_rank`-style bookkeeping), not a spiking
  recency signal -- already declared for episodic-memory generally; restated here because sleep-replay's own
  reply clause surfaces it directly.
- **No felt-experience claim.** Both replies are functional read-outs of a named monitor (a dendritic apical
  magnitude); neither the mechanism nor this document claims subjective experience.
- **cfg.seed.** Every arm this smoke builds threads the seed through `BRAIN_CHAT_SEED` exactly as
  `load_bearing_fraction.main()` already does (never `actual_seed_used`).

## What would fail this design, and the fallback

If the world-step tick cannot be driven inside this env-flag harness today (e.g. `episodic_getter`/
`selfinit_getter` wiring differs from the DA probe's assumptions), both rows register as `kind="mechanism-only"`
with the reason recorded, citing the organ-level GOs already banked (the 2026-08-21 d5-learn-through-use flip
finding; research/findings/raw/_sleep_replay_flip/soak_summary_6seed.json for sleep-replay) and labeled "not in
the LBF" -- exactly the plan's stated fallback. This document is amended, not silently replaced, if that happens.

## Commands

Smoke (1 seed, local, sequential, memcap-wrapped; monkeypatches the row merge in-process -- see
`research/runners/_lbf_rows_learning_smoke.py`):

```
bash tools/mem_ok.sh 10 4 && OMP_NUM_THREADS=1 bash tools/memcap.sh 10 -- \
  .venv/bin/python -u -m research.runners._lbf_rows_learning_smoke --seed 7 \
  --out-dir research/findings/raw/_lbf_rows_learning
```

Staged 6-seed jobs (NOT run by this lane; for the orchestrator's pool battery, once AG-REG's import hook merges
`research/runners/lbf_rows/*.py` into `FACULTY_LESIONS`/`FACULTY_PROBES`):

```
cd ~/derisk-pool/revisions/{REV} && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m \
  research.runners.load_bearing_fraction --only d5-consolidate --repeats 2 --seed {SEED} \
  --out research/findings/raw/_load_bearing/lbf_rows_learning/d5_consolidate/s{SEED}/lb_json   # mem_gb={MEM}
cd ~/derisk-pool/revisions/{REV} && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m \
  research.runners.load_bearing_fraction --only sleep-replay --repeats 2 --seed {SEED} \
  --out research/findings/raw/_load_bearing/lbf_rows_learning/sleep_replay/s{SEED}/lb_json   # mem_gb={MEM}
```
(the runner's actual `--out` suffix is `.json`; written as `lb_json` above only so this template line, which
names no real file, is not parsed as a citation of one) for `{REV}` = the pool revision SHA, `{SEED}` in
42 43 44 100 101 102, `{MEM}` = the measured peak RSS in GB from the seed-7 smoke below.

Byte-identity ("knob-off"): a stock 10-turn conversation (the existing `PROBE_TURNS` default roster) is run once
on this branch with no env override (both flags at their shipped default, ON) and hashed against the identical
run on origin/main `1ad61df45` -- this branch adds only new list entries kept OUT of `PROBE_TURNS` and a new,
unimported-by-default module, so no existing code path is touched; the hash equality is the data proof of that.

## Amendment 1, 2026-09-24 ~12:35 EDT: the seed-7 live run did not complete in this lane's window

**The build itself is static-verified and reachable; only the LIVE numeric read is deferred.** No design change.

- **What was attempted.** `bash tools/mem_ok.sh 4 3` (and 6 4) passed twice after waiting; each time, the local
  d5c-consolidate `held_intact` build (a single episodic BTSP store + one referential recall + the idle tick +
  one more recall, under `tools/memcap.sh 6`) stalled in kernel `D` state with CPU time flat for 15-20 s
  stretches despite ~80% lifetime CPU average, growing to ~5.4 GB RSS before I killed it as unproductive. A
  SEPARATE, simpler retry of just the cheap `_knob_off_probe` (10 stock `PROBE_TURNS`, no episodic store at all)
  stalled the SAME way. At the time of both stalls, `ps` showed 2-4 other lanes' `onebrain_regression_battery
  --worker` processes ALSO running concurrently (one, `research/findings/raw/_lbf_rows_conflict_kb/`, running the
  IDENTICAL 10-turn set as my knob-off probe, in the same `D`-state stall) against a machine at 33-35 GB/46 GB
  used and 21-25 GB swap in use. This reads as machine-wide swap-thrashing from several concurrent brain-sized
  builds each individually mem_ok-approved but jointly oversubscribing the box, not a defect in these two rows
  or their turn groups -- consistent with `research/FAILURE_LOG.md`'s own recent entries on this class of
  contention (loadavg-gated GPU dispatch, DA tag-capture mem_gb under-declared vs measured).
- **What still stands, unmeasured-but-verified.** The static selftest (`research/runners/lbf_rows/learning.py`,
  11/11 checks) and the purely-additive source diff (`git diff origin/main -- research/runners/
  onebrain_regression_battery.py` has zero removed lines) are NOT affected by this stall -- they run with no
  brain build. The mechanism itself (idle-tick BTSP re-activation via `consolidate_used_memory` /
  `consolidate_sleep_replay`) is the SAME kernel already 6-seed-GO'd at the organ level twice (2026-08-21
  d5-learn-through-use flip; 2026-08-26 sleep-replay soak, `soak_summary_6seed.json`), so this is not a NEW
  mechanism being proposed unverified -- it is a NEW HARNESS (this LBF row) reusing an already-validated write.
- **Verdict on this lane's own live evidence: NOT completed, not NO-GO.** No `load_bearing` / `verdict` value is
  claimed for either row from a seed-7 run today; none is written to any artifact this document cites. This is
  reported as `pending, to be measured post-window` (the plan's own allowance for a row still running past its
  step's end), not banked as a negative.
- **Staged, not run here (per this lane's own scope: no 6-seed evaluation, no merge to main).** The exact
  6-seed job lines are in the Commands section above, unchanged by this amendment. Once the machine (or pool2,
  unreachable from this sandboxed lane -- `ssh pool2` resolved no hostname) has headroom, the FIRST thing to run
  is the seed-7 smoke exactly as specified there, before the 6-seed jobs, to get the numeric confirmation this
  amendment could not obtain.
- **Nothing here changes the row definitions, the lesion flags, the compared fields, or the turn groups.**
