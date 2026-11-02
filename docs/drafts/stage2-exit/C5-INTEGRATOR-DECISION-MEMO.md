# Stage-2 retake, C5 (private -> inner speech): which integrator the counted arms run on (DRAFT memo for Claude)

status: DRAFT (build agent, 2026-11-01; branch research/stage2-retake-amendments). A decision memo, not a ruling: Claude
decides. Nothing was queued.
host-decision: none a memo choosing a run configuration; it builds nothing that decides a reply

## The question

The 10-26 gate scored C5 PASS 6/6 at the default integrator (explicit Euler, dt 1 ms). RULING substrate-1027 (2) puts a
caveat on every citation of that PASS. On dev seeds 7 and 8, C5 and recall verification are decided by the GNW
ignition-bus workspace, and that workspace ignites at the drive where all 80 cells of its driven slot leave the valid
membrane range (research/findings/2026-10-26-stable-integration-dev7.md, research/findings/2026-10-27-stable-recal-dev78.md).
For the retake, the counted `inner` (and every other) arm can run on:

- **Option A, default integrator** (production defaults, as the prereg registers). The caveat travels with any C5 PASS.
- **Option B, `SIM_STABLE_INTEGRATION=1` + `SIM_STABLE_RECAL=1`** (valid membranes on every bridge, with the bus, the
  thought-swap and the global-stop workspaces re-calibrated by their own rules). Both are default OFF on main.

## Evidence

### Option A (default integrator)

| data | revision | C5 correct | lesion_workspace | lesion_selfinit | permuted tracks |
|---|---|---|---|---|---|
| 10-26 gate, seeds 42-102 | 9bdb92ee8 | 6/6 | 6/6, reproduced 6/6 | 6/6, reproduced 6/6 | 6/6 |
| 11-01 dev preview, seeds 1-3 (105/105 cells) | 41d76bdbe | 3/3 | 3/3, reproduced 3/3 | 3/3, reproduced 3/3 | 3/3 |
| stable-recal battery `off` arm, dev seeds 1-12 (intact_a only) | 34a4e5f75 | 12/12 | not run | not run | not run |

(Preview read: `python -m research.runners.stage2_exit_gate --aggregate <preview> --seeds 1,2,3`; seed 3's two inner
intact cells and emo intact_b were read from the home server's landed copies, see AMENDMENT-A6.md.)
Against: the deciding workspace's ignition on dev 7/8 is the runaway of non-physical membranes (by intervention: the bus
alone on valid membranes inside a production brain loses recall verification and fails C5). The membrane census finds 12
production organs out of range on dev seed 7, replicated on dev 1-12. So a C5 PASS here shows the integrated system
passes, but not that the biology it emulates would.

### Option B (stable + recal)

The stable-recal held-out battery (RULING canary-recal-1028; 180 lines at `34a4e5f75`, which is BEFORE the emotion and C3
flips) ran intact_a of all five groups on dev seeds 1-12 under off / stable / stable_recal. Re-scored here with the
REGISTERED Stage-2 scorers by `c5_integrator_heldout_read.py` (output `c5_integrator_heldout_read.json`, committed data
only). Seeds 7 and 8 are in-sample (the swap workspace was fitted on them, the bus on 7).

| setting | C5 `all` (held-out / in-sample) | C2 `all` | C4 `all` | answered turns lost vs off (conv / infer / ask / emo / inner) | primary-field turns differing vs off (all groups) |
|---|---|---|---|---|---|
| off | 10/10 / 2/2 | 12/12 | 12/12 | - | - |
| stable (plain) | 0/10 / 0/2 | 0/12 | 12/12 | 12 / 12 / 36 / 0 / 12 | 423 |
| stable_recal | 9/9 / 2/2 | 11/11 | 11/11 | 0 / 0 / 0 / 0 / 0 | 186 (inner 18) |

The stable_recal cells of dev seed 5 are on the home server but not committed. Read from that copy, seed 5 also scores C5
`all` True and loses no answered turn, which makes stable_recal C5 12/12 dev seeds. That read is not in the JSON. C1 and
C3 are `all` False in every setting at that revision (no C1 fix; C3 flags were off then). The registered dev check
(`research/runners/stable_recal_dev_check.py`, Claude's 11-01 night read in `_scratch/night1101/recal/dev_check.json`)
reports the same answer rates.

What B has NOT measured:
- **Lesion and permuted arms.** The battery ran intact_a only. Under re-calibrated workspaces it is open whether
  `lesion_workspace` still changes the chain reply, whether `lesion_selfinit` still changes the idle reply, and whether
  the permuted chain is tracked. These are three of C5's four clauses. C4's lesion and permuted clauses are open too.
- **The current defaults.** Nothing ran stable+recal at a revision with the emotion and C3 flips ON (41d76bdbe).
- **Gate seeds.** None ran under stable or stable+recal.
- **Recal's own limits** (stable-recal-dev78 finding): the global-stop organ's gate over 15 dev seeds reads 5 GO,
  1 NO-GO and 9 UNDEFINED. The conflict sensor and boost it calibrates are host Python (a scaffold). The stop is
  saturated on dev 7/8.

### The pool lines that measure the gap

`research/coordination/stage2-stable-preview-pool-lines.tsv` holds 36 lines at 41d76bdbe (provisioned): dev seeds 1-3,
groups `inner` (intact_a, intact_b, lesion_workspace + .rep0, lesion_selfinit + .rep0, permuted) and `emo` (intact_a,
intact_b, lesion_affect + .rep0, permuted). Each line carries `SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1`, has no
gate-seed flags, writes under `research/findings/raw/_stage2_preview_1101/stable/`, and has the reason "RULING
stage2-retake-1101: stable+recal dev preview incl. lesion arms". They were emitted by:

    .venv/bin/python -m research.runners.stage2_exit_gate --emit-pool-jobs 41d76bdbe9b58d4362b9d480f301f1db13550570 \
        --seeds 1,2,3 --groups inner,emo --env SIM_STABLE_INTEGRATION=1 --env SIM_STABLE_RECAL=1 \
        --out-root research/findings/raw/_stage2_preview_1101/stable \
        --reason "RULING stage2-retake-1101: stable+recal dev preview incl. lesion arms"

They are byte-checked against the emitter by `tests/test_stage2_exit_gate_a6.py`. Cost: about 1000-1250 s and about 6-8 GB
per line (prereg A2), so about 12 job-hours, about 3 h on four nodes. Score: `--aggregate
research/findings/raw/_stage2_preview_1101/stable --seeds 1,2,3`. Its conv/infer/ask groups are absent, so C1-C3 read
INCOMPLETE; read C4 and C5 only. Not queued (the RULING key in the reason is a placeholder until Claude posts it).

## Constraints either option must respect

- The prereg measures the **integrated production brain at production defaults**. Setting B's env vars only in the gate's
  lines would measure a configuration no user meets. That breaks the prereg's own frame and integration-first
  (owner 2026-09-29: a faculty counts when it is ON BY DEFAULT).
- B as production therefore means a **default-ON flip** of both switches, with its own flip prereg, before the retake's
  pin. The usual pipeline is smoke-first, flag-off battery arms, answer rate on every group at the gate seeds. The C3
  flip took 10-29 (bars) to 11-01 (merged). RULING substrate-1027 (3) forbids plain `SIM_STABLE_INTEGRATION=1` in any
  production or gate run; stable_recal is the configuration that passed answer rate on dev 1-12.
- The calendar (owner 2026-09-29): the stage-exit attempt runs on its date even if expected to fail. The window is
  11-03..11-10 and the attempt is due 11-21.

## Recommendation

1. **Counted arms on the default integrator (A), as registered**, at the A7 pin. C5's verdict is cited only with the
   substrate-1027 caveat. Do not set B's env vars in counted gate lines.
2. **Queue the 36-line stable+recal dev preview now** (idle compute, dev seeds), so the lesion and permuted clauses
   under B are known before the gate data land.
3. **If that preview reads C5 and C4 correct, lesion-effect, reproduced and permuted-tracked on 3/3**, file the B flip
   prereg (both switches, all five groups, answer rate and scorer identity bars, flag-off arms). If B flips before
   11-21, re-pin and re-run the retake on it: then the production brain has valid membranes and the caveat retires
   with evidence instead of a citation rule. If the preview FAILs C5 under B, that is the measured residual of the
   caveat (C5 rests on the runaway), and the next method is the workspace's own, under the no-defer law.
4. **Optional, needs an amendment (A8, not drafted):** informational shadow arms in group `inner` at the gate seeds
   under B (intact_a, lesion_workspace, lesion_selfinit, permuted; 24 lines). They would put gate-seed evidence beside
   the caveat in the same attempt and never be counted. Worth it only if step 3 cannot land before the window.

## What this memo does NOT establish

- B's C5 PASS is intact_a only, and at a revision before both flips. Nothing here says B passes C5 as the gate scores it.
- A's PASS says nothing about whether the workspace's ignition is physiological. The 10-26 attribution is on dev 7 only.
- No gate seed was run or read for selection. The held-out battery and the preview are dev seeds.
