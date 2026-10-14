# HANDOFF: BG live seam, spiking commit read (amendment 2026-10-14), six gate seeds, for Hermes

*Written 2026-10-14 by Claude (build agent). Execute top to bottom; use no judgment beyond the tables. Do not change
any number, bar, arm, flag, row, job line or file named here; never flip a default. Verdicts you write are
PROVISIONAL until Claude confirms (runbook verdict rule).*

- Prereg (G1-G5, arms, preconditions): `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md`.
- Amendment (G4 on the commit read, engagement precondition, operating point, outcome table):
  `research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md`.
- Build: `research/runners/bg_action_selection_production_organ.py` region `BEGIN bg-commit` .. `END bg-commit`
  (flag `BRAIN_BG_COMMIT`, default OFF; selected row `BG_COMMIT_OP`); shared stage `research/runners/spiking_commit.py`.
- Runner and scorer: `research/runners/host_share_bg_seam.py` with `--commit` (sets `BRAIN_BG_COMMIT=1` on the three
  seam arms only). Worklog with the DEV table: `docs/drafts/bg-commit/WORKLOG.md`.

Every block is ONE Bash call. Set first in each: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees; RAW=research/findings/raw/_bg_live_seam/commit_gate`.

## 0. Start condition, worktree, pin, tests, provision

Start when `git -C $P log --oneline -1 main -- research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md` prints
a commit (the amendment is merged; no gate-seed data of this build exists before it).

```bash
cd $P && git worktree add -b research/bg-commit-gate $WT/bg-commit-gate main && cd $WT/bg-commit-gate && ln -s $P/.venv .venv && ln -s $P/data data
PIN=$(git rev-parse HEAD); mkdir -p docs/drafts/bg-commit && printf 'P\t%s\t%s\n' "$PIN" "$(date -u +%FT%TZ)" > docs/drafts/bg-commit/PINS.tsv
SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_bg_commit.py tests/test_bg_neural_seam.py tests/test_word_role_commit.py tests/test_idle_replay_organ.py
git add docs/drafts/bg-commit/PINS.tsv && git commit -m "ops(bg-commit): pin P=${PIN:0:9}" && echo "$PIN"
```

| result | do |
|---|---|
| a test fails | STOP. NEEDS-CLAUDE "[bg-commit] test <name> fails at <sha>" |
| all pass, commit made | provision P: `mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$PIN`; wait for `ok`/`already` in `/opt/host-status/sim-provision/$PIN.json`; `failed` with no `next_retry_utc`: STOP, NEEDS-CLAUDE |

## 1. The six gate lines (one per seed; one line = seven fresh-brain arms, ~3.5 h on one pool core)

```bash
cd $WT/bg-commit-gate && PIN=$(awk '$1=="P"{print $2}' docs/drafts/bg-commit/PINS.tsv) && mkdir -p docs/drafts/bg-commit/jobs
for s in 42 43 44 100 101 102; do
  echo "cd ~/derisk-pool/revisions/$PIN && bash tools/mem_ok.sh 8 2 && mkdir -p $RAW/${PIN:0:7} && bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.host_share_bg_seam --run-all $RAW/${PIN:0:7} --seed $s --allow-gate-seeds --commit"
done > docs/drafts/bg-commit/jobs/gate_${PIN:0:7}.txt
wc -l docs/drafts/bg-commit/jobs/gate_${PIN:0:7}.txt
git add docs/drafts/bg-commit/jobs && git commit -m "ops(bg-commit): six gate lines at ${PIN:0:9}"
```

Expect 6 lines. Any other count: STOP, NEEDS-CLAUDE.

## 2. Canary first (the seed-42 line on one pool node), then the engagement check

```bash
cd $P && J=$WT/bg-commit-gate/docs/drafts/bg-commit/jobs && PIN=$(awk '$1=="P"{print $2}' $WT/bg-commit-gate/docs/drafts/bg-commit/PINS.tsv)
bash tools/pool_canary.sh run pool41 "$(head -1 $J/gate_${PIN:0:7}.txt)" --wait-s 18000; echo canary rc=$?
```

Canary rc != 0: STOP, NEEDS-CLAUDE with the canary log path. Then open the canary's `seam_b.json` (under
`_pool_canary/`, as `pool_canary.sh` prints) and check, without reading any outcome value:

```bash
F=<the canary's seam_b.json>; $PY - "$F" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ev = [e for evs in d["events"].values() for e in evs if e.get("kind") == "bg_neural"]
ok = (d["env"].get("BRAIN_BG_COMMIT") == "1" and len(ev) == 4 and d.get("amendment") == "2026-10-14-bg-live-seam-commit"
      and all(e.get("commit") and e["commit"]["winner"] == e["winner"] and e["commit"]["organ_seed"] == 42
              and set(e["commit"]["read_variants"]) == {"x0.5", "x2"} and "old_latch_winner" in e["commit"] for e in ev))
print("ENGAGED" if ok else "NOT-ENGAGED")
EOF
```

`NOT-ENGAGED`: STOP, NEEDS-CLAUDE (quote the first event). `ENGAGED`: step 3.

## 3. Queue the six lines

```bash
cd $P && J=$WT/bg-commit-gate/docs/drafts/bg-commit/jobs && PIN=$(awk '$1=="P"{print $2}' $WT/bg-commit-gate/docs/drafts/bg-commit/PINS.tsv)
bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "bg-commit gate: prereg research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md + amendment research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md ; mem_gb=8 ; max_age_h=168" </dev/null || echo "ADD FAILED: $l"; done < '"$J/gate_${PIN:0:7}.txt"
bash tools/pool_queue.sh list | tail -8
```

## 4. Score (when the 42 arm files are in)

Results land in `$P/$RAW/<P7>/s<seed>/<arm>.json` (7 arms x 6 seeds).

```bash
cd $WT/bg-commit-gate && PIN=$(awk '$1=="P"{print $2}' docs/drafts/bg-commit/PINS.tsv) && P7=${PIN:0:7} && mkdir -p $RAW && cp -a $P/$RAW/$P7 $RAW/
ls $RAW/$P7/s*/*.json | grep -vc prov                                             # must be 42
$PY -m research.runners.host_share_bg_seam --score $RAW/$P7 --seeds 42 43 44 100 101 102 --commit --out $RAW/$P7/score.json
git add $RAW/$P7 && git commit -m "data(bg-commit): six-seed gate shards + score (amendment 2026-10-14)"
```

Fewer than 42 files after 24 h: re-queue only the missing seeds with their step-1 line (the runner keeps existing
arm files). Still missing after one re-queue: STOP, NEEDS-CLAUDE.

## 5. Outcome -> what to write (read `verdict` and `gates` from score.json; never recompute)

| verdict | gates | write | then |
|---|---|---|---|
| GO | all six true | PROVISIONAL finding `research/findings/<date>-bg-live-seam-commit-RESULT.md`: amendment section "Outcome -> verdict" sentence verbatim, the pooled and n tables copied from score.json, the reported numbers (old_read_agree, co_commit, first12_change, latch_thr_change of the old latch, S, D), the fixed caveats | NEEDS-CLAUDE "[bg-commit] GO PROVISIONAL, verify-go due" |
| NO-GO | G4_commit_read_threshold_invariant false | PROVISIONAL NO-GO finding: the amendment's G4 sentence verbatim, the per-seed seam_b events whose commit read changed (from `per_seed` and the shards' `commit.read_variants`) | NEEDS-CLAUDE |
| NO-GO | any other gate false | PROVISIONAL NO-GO finding naming each false gate with the amendment's sentence for it | NEEDS-CLAUDE |
| UNDEFINED | - | if a precondition names `commit stage engaged`: STOP, NEEDS-CLAUDE (do not re-run). Otherwise re-queue the seeds with missing arm files once; still UNDEFINED: STOP, NEEDS-CLAUDE |

Do not flip `BRAIN_BG_COMMIT`, `BRAIN_BG_NEURAL_SEAM` or `BRAIN_GNW_UNIQUE_IGNITION_READ`; a flip is a separate packet.
