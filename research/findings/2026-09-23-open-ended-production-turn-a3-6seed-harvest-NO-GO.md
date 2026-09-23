---
type: finding
status: no-go
lane: load-bearing
date: 2026-09-23
---

# Open-ended production-turn amendment-3 6-seed harvest: NO-GO — one seed reads UNDEFINED (degenerate null) and the mean effect over the remaining five sits just under the registered floor (2026-09-23)

Lane `research/open-ended-production-turn-lb` (charter D1). Pre-registration:
[`docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md`](../../docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md),
amendment 3 committed `eefdd666a` (before any run it governs; the design, GO rule, M=4/K=8/floor=0.10, is in that
commit's "The design (fixed now)"). Round-6 disclosure of an undisclosed post-launch scorer change: PREREG
"Amendment-log correction 3". Supersedes the "Staged" section of
[`2026-09-23-open-ended-generation-production-turn-draw-lesion-seed42-and-open-ended-mode-bypass.md`](2026-09-23-open-ended-generation-production-turn-draw-lesion-seed42-and-open-ended-mode-bypass.md)
("Correction 4"), which staged these sessions on the mini-PC pool; they finished elsewhere (see "Provenance").
Instrument: `research/runners/_lbf_open_ended_production_turn_probe.py --a3-score`.

## Verdict: **NO-GO**

Scored with the pre-registered command (`--a3-score --mode default --seeds 42,43,44,100,101,102`) against all 54
governed session files, exactly as amendment 3 registers. Aggregate artifact:
`research/findings/raw/_load_bearing/_oe_production_turn/a3/default_a3_aggregate.json`. Per-seed verdicts (with
the `attributable_to_host_weight_drive` diagnostic used in the table below):
`research/findings/raw/_load_bearing/_oe_production_turn/a3/default/default_s*_a3_verdict.json`.

- **5/6 seeds DEFINED, not 6/6** — the registered GO rule's first precondition ("6 seeds, all DEFINED") fails
  outright. Seed 100 reads **UNDEFINED**: `noise streams never changed a reply within an arm -> the null is
  DEGENERATE` (`noise_live: false`). Both arms answered "deer" on all 32 asks (4 sessions × 8 asks, each on its own
  noise stream) — the per-session noise reseeding that amendment 3 exists to guarantee a non-degenerate null did
  not, for this seed, ever change a reply in either arm, so the null this seed would contribute is exactly zero by
  construction, not a real draw from H0. This is the guard working as designed (a wall the scorer catches rather
  than silently passing), not a data-quality bug in the harvest — every session's own `noise_stream_competes` count
  is > 0 and no reply errored (see "Session validity" below).
- **The five DEFINED seeds' mean Delta (0.09375) sits under the registered 0.10 floor.** Individual seed deltas:
  s42 = 0.03125 (perm p = 0.2), s43 = 0.020833 (perm p = 0.414286), s44 = 0.010417 (perm p = 0.5),
  s101 = 0.145833 (perm p = 0.028571), s102 = 0.260417 (perm p = 0.014286). All five are positive
  (`n_delta_positive: 5`), but with only 5 of the registered 6 seeds DEFINED the exact sign-test cannot run over
  the full registered set (`p_sign_test: null`; `all_defined: false` blocks it by the scorer's own construction),
  and the held-out sign test (excluding seed 42) is likewise blocked (`p_sign_test_heldout_excl_seed42: null`,
  `GO_heldout_only: false`) for the same reason.
- **`"GO": false`** in the aggregate artifact — both preconditions fail independently (not-all-defined, and the
  descriptive mean under the floor among the defined seeds), so this is a definite NO-GO on the rule as written,
  not an ambiguous or partial read.

| seed | verdict | delta | perm p (descriptive) | attributable_to_host_weight_drive | reason if UNDEFINED |
|---|---|---|---|---|---|
| 42 | DEFINED | 0.03125 | 0.2 | 0.036585 | — |
| 43 | DEFINED | 0.020833 | 0.414286 | 0.022222 | — |
| 44 | DEFINED | 0.010417 | 0.5 | 0.010753 | — |
| 100 | **UNDEFINED** | — | — | — | noise streams never changed a reply within an arm (degenerate null) |
| 101 | DEFINED | 0.145833 | 0.028571 | 0.186667 | — |
| 102 | DEFINED | 0.260417 | 0.014286 | 0.342466 | — |

`attributable_to_host_weight_drive` (`tools.lab.attributable_to`, round-5 fix) reports what fraction of the
intact-minus-lesion difference in mean session value is not also present when the intact arm is compared to
itself — i.e. how much of the raw delta is attributable to the manipulation versus shared with a null comparison;
it is descriptive, not a second gate, and is smallest on the three seeds (42/43/44) whose deltas are also smallest.

## The disclosed scorer amendment made NO difference to this harvest

Per PREREG "Amendment-log correction 3": commit `a9eda3d0a` (2026-09-23 17:47:20 -0400) added one more per-seed
UNDEFINED condition to `score_seed_a3` — `stored facts differ across sessions` — after the PREREG's own amendment 3
was committed (`eefdd666a`) but without its own amendment-log entry at the time. That correction is disclosed there
in full (what changed, its strictly-conservative direction, and the mtime check against "no a3 result had been
read"). Here, empirically, on the actual 54-session harvest: `stored_facts_equal_across_sessions` reads `True` for
all 6 seeds (verified in each seed's own `default_s<seed>_a3_verdict.json`), so the added check never fires and
changes no verdict. Re-scoring this same harvest with the scorer exactly as it stood at `eefdd666a` (extracted
standalone and run against a scratch copy of the same 54 session files, output discarded rather than committed
over the amended verdicts) gives a **byte-identical** `summary` block to the one above. **Both the as-registered
and the amended scorer read this harvest as NO-GO, for the identical reasons.**

## Session validity (all 54 governed sessions + the seed-7 smoke)

Every governed session file (6 seeds × (4 intact + 4 lesion + 1 rebuild) = 54) and the non-governed seed-7 smoke
session were checked individually: non-empty `replies`, no `error` key in any reply, and `noise_stream_competes >
0` (the per-session noise stream was actually exercised at least once). All 54 governed files and the smoke file
pass; 0 failures. The `intact_rebuild` determinism check also passed for every seed that reached a verdict (a seed
that failed it would read NONDETERMINISTIC, and none did).

## Provenance (stated honestly)

The task that produced this harvest describes the 54 governed sessions and the seed-7 smoke as run on an AWS
instance, from a `git archive` of `eefdd666a`, with `SIM_POOL_HOST=aws2` — **not** the mini-PC-pool staging the
PREREG and the prior finding's "Staged" section describe (that pool staging did not finish; see "Correction 4" in
the linked finding). This is stated as reported, not independently re-derived by me: every pulled session's own
`.prov.json` sidecar records `"env": {"SIM_POOL_HOST": "aws2", "SIM_BACKEND": "numpy"}`, matching the claim, but
each sidecar's `git_sha` field reads **`"unknown"`** rather than `eefdd666a` — consistent with running from an
extracted `git archive` tarball (no `.git` directory for `git rev-parse` to read on the AWS box), but it means the
`eefdd666a` code-identity claim rests on the operator's own statement of what was archived and run, not on a
self-attesting field inside the artifacts themselves. I did not independently verify byte-for-byte that the code
running on `aws2` was an unmodified archive of `eefdd666a` (I have no access to that instance); I verified only
that the artifacts' own shape (fields, session count, per-arm/per-seed structure, `A3_SESSIONS`/`A3_K` implied by
4 sessions and 8 replies each) matches what `eefdd666a`'s registered protocol specifies.

The 54 governed session files (+ sidecars) and the seed-7 smoke were copied, unmodified, from the PRIMARY
checkout's untracked staging directory
`research/findings/raw/_load_bearing/_oe_production_turn/aws2_a3/{a3,a3_smoke}/default/` into this lane's
registered a3 output location (`.../a3/default/` and `.../a3_smoke/default/`) so that the pre-registered
`--a3-score` command's default `--out-dir` finds them at the path amendment 3 names. No session content was
altered; only the directory location changed. Provenance sidecars for the scoring run itself (this lane's own
`--a3-score` invocation, in this worktree) are stamped by `research/runners/__init__.py` as usual and carry this
worktree's own `git_sha` (not `eefdd666a` — that field correctly describes the SCORING code's commit, distinct
from the SESSION-GENERATING code's commit named above).

## What this does and does not show

- **Not a claim that the spiking generative draw is load-bearing on the production open-ended turn**, under this
  registered design. The registered GO rule is not met.
- **Not a claim that it is refuted either.** Five of six seeds show a positive, same-direction effect
  (`n_delta_positive: 5`), two of them nominally significant on the (descriptive, non-gating) per-seed exact
  permutation test (s101, s102), and the shortfall against the floor is on the mean of five DEFINED seeds, not a
  reversal. The sixth seed is UNDEFINED, not negative. Per this lane's standing rule, a NO-GO on this method banks
  the method and does not close the capability: the next lever is a design that does not let one seed's frozen
  draw (host affine map saturating, or the WTA landing on the same winner regardless of noise realization at that
  seed's particular host weight vector) zero out a sixth of the registered evidence.
- **The lesioned edge is a HOST vector** (`BRAIN_SPIKING_DRAW_LESION=1` replaces the host co-occurrence-derived
  weight vector w with `np.ones` before the host affine map into the spiking bank's drive), declared in the PREREG
  and repeated in the aggregate artifact's own `lesioned_edge` field. This design cannot show the spiking part
  itself is load-bearing; the `host_oracle` arm was not part of this amendment-3 governed run.

## Declared host shortcuts

Unchanged from the PREREG and the amendment-3 aggregate's own `host_shortcuts` field: teach KB + ask prompt
(world); histogram/TV/sign-test statistic and the per-session noise-stream assignment (instrument); the
co-occurrence matrix P, weight vector w, and the affine drive map (host, the lesioned input); argmax over the
bank's firing counts (host read-out); hypothesis role induction, the SVO template, the RF-composer moat verify,
and the prompt routers (host); the reply renderer and warm Qwen faculty stub (FORM not measured).
