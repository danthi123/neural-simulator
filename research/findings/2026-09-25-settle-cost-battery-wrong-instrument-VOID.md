---
type: finding
status: void
date: 2026-09-25
mechanism: BRAIN_AFFECT_MARKER_SETTLE warm-turn cost on the production GPU chat path (criterion L,
  research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md)
lane: latency-and-cost (production GPU chat path)
artifacts:
  - research/findings/raw/_settle_cost/cupy_on.json
  - research/findings/raw/_settle_cost/cupy_on.json.prov.json
  - research/findings/raw/_settle_cost/cupy_off.json
  - research/findings/raw/_settle_cost/cupy_off.json.prov.json
  - research/findings/raw/_settle_cost/numpy_on.json
  - research/findings/raw/_settle_cost/numpy_on.json.prov.json
  - research/findings/raw/_settle_cost/numpy_off.json
  - research/findings/raw/_settle_cost/numpy_off.json.prov.json
  - research/findings/raw/_settle_cost/gpu_prod_chat_smoke_settle_on.txt
  - research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md
  - research/runners/_prod_chat_phase_timing.py
  - research/runners/_settle_turn_cost_probe.py
---

# `_settle_cost` battery scored against the 2026-09-24 PREREGISTRATION: VOID — every artifact was made by a different, prior instrument; criterion L was never run

**One-line verdict.** The four `research/findings/raw/_settle_cost/{cupy,numpy}_{on,off}.json` artifacts were all
produced by `research/runners/_settle_turn_cost_probe.py` — the OLD, isolated-affect-marker-circuit probe the
2026-09-24 PREREGISTRATION explicitly names as insufficient and distinct from what it registers
(`research/runners/_prod_chat_phase_timing.py`, timing the real `webapp.server.brain_chat()` entrypoint). None of
the four carries the `warm_turn_total_s` field criterion L reads; the registered comparison tool
(`_prod_chat_phase_timing.py --compare-on/--compare-off`) would raise `KeyError` on any of them. No amendment to
the PREREGISTRATION authorizes this substitution — its own Amendment log reads `(none yet)`. **Criterion L is
UNDEFINED, not scored, not a NO-GO on SETTLE's production-path cost** — this is an instrument mismatch, not a
result.

## Liveness (checked before scoring, per the task's own requirement)

Nothing is running anywhere. Confirmed 2026-09-25:

- **Pool nodes** (`ssh -n -F research/queue/.pool_ssh_config <node> 'ps -eo etimes,args | grep ...'`, read-only):
  pool1 (`ip-172-31-47-37`), pool2 (`ip-172-31-47-57`), pool41 (`node-dl4g243`), pool42 (`node-dl5d243`) all
  answered `alive`; `ps -eo etimes,args | grep -E '_settle_turn_cost_probe|_prod_chat_phase_timing|test_production_chat_gpu_smoke'`
  matched **zero processes** on all four.
- **Local GPU queue** (`research/queue/gpu.queue`, `gpu.queue.running`, `gpu_queue.log`): no entry for either
  runner or the smoke test; the most recent log lines are unrelated jobs (slotbinder production gate, a gap4
  transport-ceiling readout, an `_affect_marker_settle_gpu_timing` a3x smoke, a `_da_tag_capture_chat_probe` pair
  run).
- **Local `ps`**: `ps -eo etimes,args | grep -E '_settle_turn_cost_probe|_prod_chat_phase_timing|test_production_chat_gpu_smoke'`
  matched zero processes on this box.

The battery is complete in the sense that nothing is still producing it — the data that exists is final, static,
and (per the next section) simply not the preregistered data.

## Census: registered vs. actual

| | **Registered** (the PREREGISTRATION's Commands section) | **Actual** (what is on disk) |
|---|---|---|
| Runner | `research/runners/_prod_chat_phase_timing.py` | `research/runners/_settle_turn_cost_probe.py` (per every `.prov.json`'s own `runner` field) |
| What it measures | A full `webapp.server.brain_chat()` turn (build + Qwen load + turn 1 + a warm turn 2), through the real production entrypoint | A tight loop of 8 direct calls to `reader.select_valence()` / `select_arousal()` on the affect-marker circuit alone — no `brain_chat()` call, no Qwen, no gate/compose/render |
| Output filenames | `research/findings/raw/_settle_cost/prod_chat_phase_settle_{on,off}.json` (2 files, cupy/GPU only) | `cupy_{on,off}.json` + `numpy_{on,off}.json` (4 files, cupy AND numpy) |
| Backend | cupy/GPU only (criterion L is explicitly a GPU criterion) | cupy AND numpy (numpy was never in scope for criterion L; the PREREGISTRATION's only numpy path is a seed-7 dev smoke that is explicitly told to write to a scratch dir, never to `research/findings/raw/_settle_cost/`) |
| Revision | one pinned revision (a single `--compare-on`/`--compare-off` pair must be "at the same commit" per the PREREGISTRATION's Criterion L text) | **two** revisions: `5e571d6c6` (cupy pair) vs `9d83a242a` (numpy pair) |
| Tree state | (implicitly clean/pinned — queued via `tools/gpu_queue.sh`) | `git_dirty: true` on all 4 `.prov.json` sidecars |
| Key field | `warm_turn_total_s` (top level) | absent from all 4 files (see mechanical check below) |

`research/findings/raw/_settle_cost/gpu_prod_chat_smoke_settle_on.txt` is closer in spirit — it runs
`tests/test_production_chat_gpu_smoke.py`, which does call the real `brain_chat()` — but it is still not the
registered runner, and it recorded a **`TimeoutExpired` after 600s**: turn 1 of `brain_chat()` on the cupy/Qwen
path never returned inside the pytest subprocess's timeout, so this attempt produced no timing number at all, and
has no `settle_off` counterpart.

A full-tree search (`find research/findings/raw -iname '*prod_chat_phase*'`) turns up **zero** files matching the
registered `prod_chat_phase_settle_{on,off}.json` naming anywhere in the repository — not just under
`_settle_cost/`. The registered artifacts do not exist.

## Mechanical check: criterion L cannot be computed from this data

<!--derived-->
None of the four JSON files contains `warm_turn_total_s` (checked directly — `'warm_turn_total_s' in d` reads
`False` for `cupy_on.json`, `cupy_off.json`, `numpy_on.json`, `numpy_off.json`). `_prod_chat_phase_timing.py`'s
own `_compare()` function reads `on["warm_turn_total_s"] - off["warm_turn_total_s"]` unconditionally — running
the registered comparison command against these four files raises `KeyError: 'warm_turn_total_s'`, it does not
silently return a wrong number. Their actual top-level keys are `mode`, `backend`, `seed`, `settle_enabled`,
`deliberation_ms`, `interturn_rest_ms`, `build_s`, `first_turn_s`, `turn_s`, `steady_median_turn_s`,
`steady_max_turn_s` — the schema `_settle_turn_cost_probe.py` writes, not the schema
`_prod_chat_phase_timing.py` writes.

## Per-arm detail (the isolated-circuit probe's own numbers, for the record only — NOT criterion L)

| arm | file | `git_sha` | `git_dirty` | `source_kind` | `steady_median_turn_s` | `steady_max_turn_s` |
|---|---|---|---|---|---|---|
| settle ON / cupy | `research/findings/raw/_settle_cost/cupy_on.json` | `5e571d6c6` | true | null | 0.1618419729929883 | 0.16769706099876203 |
| settle OFF / cupy | `research/findings/raw/_settle_cost/cupy_off.json` | `5e571d6c6` | true | null | 0.020233646995620802 | 0.020806179993087426 |
| settle ON / numpy | `research/findings/raw/_settle_cost/numpy_on.json` | `9d83a242a` | true | null | 0.25486755699967034 | 0.29523003500071354 |
| settle OFF / numpy | `research/findings/raw/_settle_cost/numpy_off.json` | `9d83a242a` | true | null | 0.02594343898817897 | 0.02616187800595071 |

<!--derived-->
If one naively substituted this probe's `steady_median_turn_s` for criterion L's `warm_turn_total_s` (which the
next section says NOT to do), the deltas would read cupy +0.1416083259973675 s and numpy +0.22892411801149137 s
— both "under" the +0.3 s bar, but this is not a licensed reading: see below.

## Why the isolated-probe data cannot stand in for criterion L, even informally

1. **The PREREGISTRATION itself draws this line on purpose.** Its "Why" section states the open question as
   whether SETTLE slows a warm turn "on the real production GPU chat path, **not just the isolated
   affect-marker circuit** (`_settle_turn_cost_probe.py` already measures that in isolation — this measures the
   same flag through `webapp.server.brain_chat` itself)". The `_settle_cost` battery is exactly the measurement
   the PREREGISTRATION says is insufficient, re-run on 2026-09-24 (the same day the probe and the
   PREREGISTRATION both landed) rather than superseded by it.
2. **Magnitude.** The isolated probe's full "build + 8 calls" wall time tops out at 0.29523003500071354 s (numpy
   on, `steady_max_turn_s`). The one attempt at the real path (`gpu_prod_chat_smoke_settle_on.txt`) did not complete
   turn 1 of `brain_chat()` inside 600 s. Whatever the real production warm-turn cost is, it is not something
   this isolated loop — which never calls `brain_chat()`, loads no Qwen weights, and runs no gate/compose/render
   — can bound from above or below.
3. **No amendment.** The PREREGISTRATION's own "Amendment log" section reads `(none yet)` (verified against the
   file at its current HEAD; `git log --oneline --all -- research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md`
   shows a single commit, `228ba16f0`, no follow-up). A repo-wide grep of `research/findings/*.md`,
   `GAP_CLOSURE_MISSION.md` and `docs/*.md` for `_settle_turn_cost_probe`, `_settle_cost`, and
   `production-chat-phase-timing` returns the PREREGISTRATION file itself and nothing else — no amendment
   document registers the naming change, the mixed revisions, or the numpy arms.

## Provenance

All four `.prov.json` sidecars record `source_kind: null` and `git_dirty: true` — not `git_archive`, and not a
clean pinned tree. `source_kind: git_archive` has been available in `research/runners/__init__.py` since
2026-08-03/04 (commits `59eded5d6`, `b976f8945`, `49958c6a7`, `49958c6a780`), roughly seven weeks before this
2026-09-24 run — **this battery does not predate the provenance rule**; it was simply run ad hoc against a dirty
working tree rather than dispatched through the git-archive-pinned queue (`tools/gpu_queue.sh` /
`tools/sweep_pool.sh`), which is also why the cupy pair (`5e571d6c6`) and the numpy pair (`9d83a242a`) sit at two
different, unpinned revisions rather than one.

## Non-claims

- Does **not** say `BRAIN_AFFECT_MARKER_SETTLE` is or is not affordable on the production GPU chat path — that
  question remains exactly as open as before this battery ran.
- Does **not** retract or reweigh the separate, already-decided full-brain functional NO-GO on SETTLE
  (`research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json`, cited by the PREREGISTRATION as
  prior evidence). That is a functional verdict; this battery is cost-only de-risk and never touches it.
- The naive delta figures above are reported only to show they are far smaller than whatever the real
  production-path cost is (the one real-path attempt did not finish in 600 s) — they are explicitly **not** a
  criterion-L score.

## Next action (THE LAW: a wall/negative is a verdict on a METHOD, not a license to abandon the CAPABILITY)

1. Run the PREREGISTRATION's own Commands section, unmodified: queue `research.runners._prod_chat_phase_timing`
   with `--mode settle_on` and `--mode settle_off`, each writing into the directory
   `research/findings/raw/_settle_cost/` under the two filenames the PREREGISTRATION names
   (`prod_chat_phase_settle_on.json` / `prod_chat_phase_settle_off.json`), via `tools/gpu_queue.sh` at one pinned
   (git-archive) revision, then score with `--compare-on`/`--compare-off` — this is the only path that produces a
   scoreable Criterion L.
2. Before requeueing, separately look into why `tests/test_production_chat_gpu_smoke.py` timed out at 600 s on
   2026-09-24 (turn 1 of `brain_chat()` never returned under `tools/memcap.sh` 14G/12G). If the default
   production turn genuinely takes minutes end-to-end, that is a higher-priority finding on its own, independent
   of SETTLE, and it also means `_prod_chat_phase_timing.py`'s 1800 s watchdog should be re-checked against the
   observed cost rather than assumed generous.
3. If the isolated `_settle_turn_cost_probe.py` measurement is ever intended to stand in for criterion L, that
   requires a written amendment entry in the PREREGISTRATION's own Amendment log — not a silent substitution —
   before it can be scored as such.

## Sources

- `research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md` — Criterion L, the Commands
  section, the Amendment log.
- `research/runners/_prod_chat_phase_timing.py` — the registered instrument (`warm_turn_total_s`,
  `--compare-on`/`--compare-off`).
- `research/runners/_settle_turn_cost_probe.py` — the actual, different instrument that produced every JSON
  artifact under `research/findings/raw/_settle_cost/` (its own docstring names it as measuring the isolated
  circuit only).
- `tests/test_production_chat_gpu_smoke.py` — the class guard `gpu_prod_chat_smoke_settle_on.txt` ran; distinct
  from both runners above.
