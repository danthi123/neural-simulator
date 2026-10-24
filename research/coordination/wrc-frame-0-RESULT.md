# [wrc-frame-0] RESULT — step 0 deadlock read: GO, role-layer entry (14/15)

**PROVISIONAL** (agent-filed, 2026-10-21; Claude rules on the entry layer
before step 1 — RULINGS wrc-frame-go-1020, parallel-builds-1021).

## preconditions

- probe: `research/runners/wrc_frame_step0.py` on `research/wrc-frame` @
  `9d6f6edb6` (GO rule committed in-module BEFORE the run)
- run: dev seeds 1-15, local (103 s < 10-min compute-placement cap), numpy,
  `SIM_BACKEND=numpy OMP_NUM_THREADS=1`, python `.venv`, wall 103.0 s,
  peak VmHWM 85.3 MiB, rc=0
- inputs: iw1 turn-5 fact set (five built-ins + wolf-hunt-deer),
  `commit_organ_params(0)`, bias grid {0,50,100,200,400,800} pA
- outputs: `research/findings/raw/_wrc_frame/step0/` (15 seed JSON +
  `summary.json`); artifacts untracked by policy (raw/**)

## verdict (GO rule as written in the probe)

**GO: role-layer entry — 14/15 seeds pass** (bar: >= 14).
n_role_go=14, n_gate_go=14, n_singles_ok=14 of 15.

| seed | B_min role | B_false role | B_min gate | B_false gate | singles_ok | right i/l | attr |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 100 | 50 | 200 | yes | 3/2 | -0.50 |
| 2 | 50 | 200 | 50 | 200 | yes | 3/3 | 0 |
| 3 | 50 | 100 | 50 | 200 | yes | 3/3 | 0 |
| 4 | 50 | 100 | 100 | 200 | yes | 3/3 | 0 |
| 5 | 50 | 200 | 50 | 200 | yes | 2/2 | 0 |
| 6 | 50 | 100 | 50 | 200 | yes | 3/3 | 0 |
| 7 | 0 | 100 | 0 | 200 | yes | 2/2 | 0 |
| 8 | 50 | 100 | 100 | 200 | yes | 3/3 | 0 |
| 9 | 50 | 100 | 50 | 200 | yes | 3/3 | 0 |
| 10 | 50 | 100 | 50 | 200 | yes | 2/2 | 0 |
| 11 | 50 | 200 | 50 | 200 | yes | 3/3 | 0 |
| 12 | 50 | 100 | 50 | 200 | yes | 3/3 | 0 |
| 13 | 50 | 100 | 50 | 200 | yes | 2/3 | +0.33 |
| 14 | 100 | 200 | 200 | 200 | NO (eat->None) | 2/2 | 0 |
| 15 | 50 | 100 | 100 | 200 | yes | 3/3 | 0 |

- Margin: role B_min 50 pA with B_false >= 100 on all seeds (>= 2x rule);
  seed 7 latches agent alone with zero bias (B_min=0, window open by
  construction); gate layer also has a window (14 seeds) but the rule picks
  role-layer.
- (e) eat keeps action under the agent bias through 100 pA; stolen at
  >= 200 pA (consistent with B_false_role hitting at 100-200).
- (f) weights (seed 1): cat -> agent 11.88 ~ patient 10.91 (true dual role,
  the deadlock's weight signature); fish -> patient 12.48 only; word ->
  action ~0.05 (actions ride position, not word weight).
- (b) HONEST DEVIATION: the exclusion lesion does NOT confirm the gate-quench
  finding on this fact set. Right reads intact 2-3/4 vs lesioned 2-3/4,
  attributable mean -0.011 (n=15 defined, range -0.50..+0.33, mostly
  in-control). The battery-fact dual_role effect (17/24 -> 21/24) does not
  carry to the iw1 6-fact production set; per-seed attribution in the JSONs.
- only failure: seed 14 `eat` -> None at pos=None (1 seed; STOP branch
  needs >= 2, not triggered).

Await: Claude's entry-layer ruling, then step 1 (byte-identical read of
unchanged evidence).
