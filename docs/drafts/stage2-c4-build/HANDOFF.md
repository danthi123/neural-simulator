# HANDOFF: s2-c4-build dev checks (Hermes)

Written 2026-10-06 by Claude. Plan and reasons: `docs/drafts/stage2-c4-build/PLAN.md`. Execute top to bottom. At most
ONE fix round per issue; anything not covered here is a NEEDS line (`[s2-c4-build] NEEDS-CLAUDE: <what>`) and you stop
that line of work. Never change the organ, its GRID, the Stage-2 runner, its bars or turns. Never run a gate seed
(42/43/44/100/101/102): every command here refuses them, and so must you. Never flip the default yourself.

- `P=/home/dant123/Projects/neural-simulator`, `PY=$P/.venv/bin/python`. Work in the worktree of
  `research/stage2-exit-gate` (`$P/.claude/worktrees/stage2-exit-gate`, below `W=`).
- Routes: CPU checks under ~10 min may run on S; everything else goes to the pool through RULING `compute-placement`
  (provision by `provision.req`, then `pool_queue.sh add ... --checked`). Lines here are dev seeds only.

## 0. State at handoff

- Branch `claude-review/s2-c4-build` (Claude merges it to main). It adds, default OFF: the organ
  (`research/runners/emotion_category_organ.py`), its dev battery and line emitter
  (`research/runners/emotion_category_dev.py`), the banked-method probe (`research/runners/_emotion_category_corpus_probe.py`),
  the switch and two hooks in `webapp/server.py` (`_EMOTION_CATEGORY_DEFAULT_ON = False`), tests
  (`tests/test_emotion_category_organ.py`), the dev bar's bar-sanity spec and OK record, and the dev records under
  `research/findings/raw/_s2_c4_build/`.
- Organ dev battery: row 0 selected, 8/8 dev seeds pass all ten cases (`dev_select.json`).
- Handler smoke (Claude, workstation, dev seed 7, flag on, through the Stage-2 runner's own worker): intact_a, lesion_affect and permuted at dev seed 7 (shards in `research/findings/raw/_s2_c4_build/handler_smoke/on/s7/`, runner bd5c95bb4's file over this branch, workstation numpy, memcap 10, 846 s per arm). Read with the Stage-2 scorer (`score_emo`, `named_category_changed`): intact names joy, fear, anger, sadness (`all` true); the lesion arm names none on all four (`category_changed` true on every turn); the permuted arm names sadness, anger, fear, joy (`all` true); 4/4 feel reads parse in every arm. Every non-feel reply is identical to the committed flag-off dev shard (`_stage2_exit/dev/s7/emo__intact_a.json`), every feel reply is that reply plus the one clause. One dev seed, one process each: no reproduction (intact_b) and no other seed; that is H1's job.

## 1. Sync and reproduce (S, CPU, ~5 min) — by 2026-10-09

After the orchestrator has merged `claude-review/s2-c4-build` into main:

    cd $P && bash tools/agent_sync_branch.sh research/stage2-exit-gate
    cd $W && $PY -m pytest -q tests/test_emotion_category_organ.py tests/test_stage2_exit_gate.py
    cd $W && $PY -m research.runners.stage2_exit_gate --selftest
    cd $W && SIM_BACKEND=numpy $PY -m research.runners.emotion_category_dev --select --seeds 1-8 \
        --out research/findings/raw/_s2_c4_build/dev_select_repro.json

STOP (NEEDS) if the sync conflicts, a test fails, or the battery does not print `selected=0` with 8 PASS lines. The
battery's `--seeds 1-8` run is ~2.5 min on a workstation core; on S it is within the 10-minute rule. Commit
`dev_select_repro.json` (+ its `.prov.json`) on `research/stage2-exit-gate`: Claude's `dev_select.json` was produced
from the branch's files before they were committed (its sidecar reads `git_dirty: true`), so this run at a committed
tip is the clean-provenance record of the same battery.

## 2. Provision and queue the handler dev lines (pool) — queue by 2026-10-10

    cd $W && SHA=$(git rev-parse HEAD) && mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$SHA

Wait for `/opt/host-status/sim-provision/$SHA.json` to read `ok` or `already` (RULING `compute-placement`). Then emit
and queue, from `$W`:

    # H1: C4 at dev, flag ON, every registered emo arm, dev seeds 1-8 (32 lines)
    $PY -m research.runners.emotion_category_dev --emit-handler-lines $SHA --seeds 1-8 --groups emo > /tmp/h1.tsv
    # H2: flip specificity. Flag ON: the other four groups' intact_a at dev seeds 3 and 7 (8 lines).
    $PY -m research.runners.emotion_category_dev --emit-handler-lines $SHA --seeds 3,7 --groups conv,infer,ask,inner --arms intact_a > /tmp/h2on.tsv
    # Flag OFF at the SAME revision: those 8, plus emo intact_a at dev seeds 1-8 (16 lines)
    $PY -m research.runners.emotion_category_dev --emit-handler-lines $SHA --seeds 3,7 --groups conv,infer,ask,inner --arms intact_a --flag-off > /tmp/h2off.tsv
    $PY -m research.runners.emotion_category_dev --emit-handler-lines $SHA --seeds 1-8 --groups emo --arms intact_a --flag-off >> /tmp/h2off.tsv

Each TSV line is `<command>\t<why>`. Queue every line with
`bash tools/pool_queue.sh add '<command>' --checked 's2-c4-build H1|H2: <why>'` (the why already carries
`mem_gb=9 max_age_h=168`). 56 lines, ~10-15 min each, mem_gb 9 under memcap 10 (the Stage-2 lane's measured
emo peak is 8.1 GB). Canary one line first with `bash tools/pool_canary.sh run auto '<first H1 line>'` if the pool
asks for one (these are dev seeds; the gate-seed canary rule does not apply). Pull results with the usual pull into
`$P/research/findings/raw/_s2_c4_build/handler/{on,off}/s<seed>/`.

## 3. Score (S, CPU, < 1 min) — by 2026-10-15

    cd $W && $PY -m research.runners.stage2_exit_gate --aggregate research/findings/raw/_s2_c4_build/handler/on \
        --seeds 1,2,3,4,5,6,7,8 --out research/findings/raw/_s2_c4_build/handler/aggregate_on.json > /tmp/h1_report.md
    cd $W && $PY -c "import json; g=json.load(open('research/findings/raw/_s2_c4_build/handler/aggregate_on.json'))['groups']['emo']; print({k: g[k] for k in g if k != 'seeds'})"
    cd $W && $PY -m research.runners.emotion_category_dev --flip-check research/findings/raw/_s2_c4_build/handler/on \
        research/findings/raw/_s2_c4_build/handler/off --seeds 1,2,3,4,5,6,7,8 \
        --out research/findings/raw/_s2_c4_build/handler/flip_check.json

Read only the `emo` group of the aggregate: the other groups have no H1 shards there and read UNDEFINED, as expected.

Expected (the dev prediction, not a bar): `emo` PASS at dev, i.e. on 8/8 dev seeds the four feel replies name joy,
fear, anger, sadness; `lesion_affect` names no category on all four (`category_changed` true on every turn); intact_b
reproduces intact_a; the permuted arm names sadness, anger, fear, joy. `flip_check`: `flags` empty, `unmeasured` empty.

Commit the shards, `flip_check.json` and a WORKLOG entry in `docs/drafts/stage2-c4-build/WORKLOG.md` on
`research/stage2-exit-gate` (pathspec commit, then `bash tools/push_both.sh research/stage2-exit-gate`). Then write
`[s2-c4-build] INFO: H1 emo <verdict> at dev (n/8), flip_check flags=<...>` in the runbook NEEDS list.

## 4. STOP conditions (each is a NEEDS line; do not retry beyond one fix round)

- the sync conflicts; a test or the selftest fails; the organ battery is not 8/8 at row 0 (do NOT try other rows);
- provisioning reads `blocked`/`refused` over pool provisioning, or no result in 15 minutes;
- any H1/H2 shard has `emotion_category.error` non-null on a feel turn, a worker rc != 0 twice, or a memcap kill;
- `emo` is not PASS at dev on 8/8: write which clause failed on which seeds (named category, lesion
  `category_changed`, reproduced, permuted), from the aggregate, verbatim;
- `flip_check.flags` non-empty (FLIP-HURTS-ANSWERS, FLIP-CHANGES-NONFEEL, FEEL-NOT-APPEND-ONLY) or `unmeasured`
  non-empty after one requeue of the missing lines;
- anything would need a gate seed, a code edit, or a default change.

## 5. After a clean section 3 (Claude, not Hermes)

Claude runs verify-go on the H1/H2 data, then a flip commit: `_EMOTION_CATEGORY_DEFAULT_ON = True` in `webapp/server.py`
(explicit `BRAIN_EMOTION_CATEGORY=0` stays the byte-identical escape), the ledger row, test updates. Hermes then
re-runs the five dev-seed-7 `intact_a` arms of the Stage-2 gate at the flip tip (the gate's own dev smoke), and Claude
pins GATE_SHA at a revision that contains the flip, AMENDMENT-A3, the runner and its tests, before 2026-11-01.
