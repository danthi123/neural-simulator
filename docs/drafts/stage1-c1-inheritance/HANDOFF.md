# HANDOFF (Hermes): C1 flip dev check, dev seeds 1-8 (Claude review 2026-10-06)

Plan: `docs/drafts/stage1-c1-inheritance/PLAN.md` steps A-B. Runner: `research/runners/c1_flip_dev_check.py`. Dev
seeds only (1-8); the runner refuses gate seeds. You run and record; Claude reads the score and decides the flip.
Never change `_CATEGORY_INHERIT_DEFAULT_ON`, never edit `webapp/`, `sim/` or `research/runners/`, never queue a gate
seed. Append one line per step to `docs/drafts/stage1-c1-inheritance/WORKLOG.md` (create it on first use).

```bash
P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python; cd $P
```

## 0. Precondition

`claude-review/c1-inheritance` is merged into main: `git merge-base --is-ancestor claude-review/c1-inheritance main
&& test -f research/runners/c1_flip_dev_check.py && echo READY`. Not READY: stop, nothing to do yet.

## 1. Commit the C1 wiring gate shards (record only)

The 52 shards under `research/findings/raw/_stage1_c1_category/gate/s*/` are untracked in the primary checkout
(their aggregate is on main from the merge). Commit exactly them:

```bash
git add research/findings/raw/_stage1_c1_category/gate/s42 research/findings/raw/_stage1_c1_category/gate/s43 \
  research/findings/raw/_stage1_c1_category/gate/s44 research/findings/raw/_stage1_c1_category/gate/s100 \
  research/findings/raw/_stage1_c1_category/gate/s101 research/findings/raw/_stage1_c1_category/gate/s102
git diff --cached --name-only | grep -c _stage1_c1_category/gate/    # expect 104 (52 json + 52 .prov.json)
git commit -m "data(stage1-c1): the 52 C1 wiring gate shards at 6c2cb1362 (scored by Claude 2026-10-06)" -- research/findings/raw/_stage1_c1_category/gate
```

If the count is not 104, or s42 now also holds the two lateral arms from RULING pool-hold-c1 (then 108), commit what
is there and write the count in the WORKLOG; the verdict does not change (those arms are informational).

## 2. Pin and check

```bash
DEV_SHA=$(git rev-parse main); echo $DEV_SHA
$PY -m pytest -q tests/test_c1_flip_dev_check.py tests/test_category_inherit_organ.py tests/test_stage1_exit_gate.py
$PY -m research.runners.c1_flip_dev_check --selftest 2>/dev/null | tail -1        # SELFTEST OK
$PY -m research.runners.stage1_c1_category --selftest 2>/dev/null | tail -1       # SELFTEST OK
```

Record DEV_SHA in the WORKLOG. Every later line uses this one revision.

## 3. Provision and canary

```bash
bash tools/pool_provision.sh --revision $DEV_SHA --isolated pool41 pool42      # LTM OK, CORPUS OK, RUNNER OK on each
$PY -m research.runners.c1_flip_dev_check --emit-pool-jobs $DEV_SHA > research/queue/c1_flipdev_${DEV_SHA:0:9}.tsv
wc -l < research/queue/c1_flipdev_${DEV_SHA:0:9}.tsv                              # 96
JOBS=research/queue/c1_flipdev_${DEV_SHA:0:9}.tsv
grep -- '--seed 1 --side on --group cat_disc ' $JOBS > research/queue/c1_flipdev_canary.tsv; wc -l < research/queue/c1_flipdev_canary.tsv   # 1
bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why"; done < research/queue/c1_flipdev_canary.tsv'
```

When the canary shard lands (`research/findings/raw/_stage1_c1_flipdev/on/s1/cat_disc__intact_a.json`), check it:

```bash
$PY - <<'EOF'
import json
d = json.load(open("research/findings/raw/_stage1_c1_flipdev/on/s1/cat_disc__intact_a.json"))
r = d["responses"]; pe = r["p_canary_eat"]
print("turns", len(r), "| flag", d.get("flip_dev"), "| sources", d.get("sources"), "| corpus", d.get("corpus_present"))
print("trace on every turn", all("category_inherit" in x for x in r.values()))
print("canary:", pe.get("abstained"), pe.get("recalled_svo"), pe.get("hypothesis_svo"))
EOF
```

Expect 16 turns, `flag_value` "1", sources `['tiny-demo +LTM']`, corpus True, trace on every turn True, and the canary
`False None ['canary', 'eat', 'worm']`. Then add the other 95 lines (skip the canary's):

```bash
bash -c 'n=0; while IFS=$'"'"'\t'"'"' read -r cmd why; do out=${cmd##*--out }; [ -f "$out" ] && continue; grep -qF -- "$out" research/queue/pool.queue && continue; n=$((n+1)); bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < '"$JOBS"'; echo "added $n"'
```

## 4. Score (when all 96 cells have landed, or 120 h after queueing)

```bash
$PY -m research.runners.c1_flip_dev_check --score research/findings/raw/_stage1_c1_flipdev \
  --out research/findings/raw/_stage1_c1_flipdev/flipdev_score.json
$PY tools/where_ran.py research/findings/raw/_stage1_c1_flipdev | tail -6
git add research/findings/raw/_stage1_c1_flipdev && git commit -m "data(c1-flipdev): dev seeds 1-8 at ${DEV_SHA:0:9}, scored" -- research/findings/raw/_stage1_c1_flipdev
```

UNDEFINED names its seeds: re-add only the missing or invalid cells' lines once (the step-3 loop skips landed
files; delete an invalid cell's file first and say so in the WORKLOG), then score again.

## 5. Report

Write `[c1-flipdev] RESULT <PASS|FAIL|UNDEFINED> n_pass=<k>/8, bridge moat guesses on <m> of <d> seeds, DEV_SHA
<sha9>` to NEEDS with the score's `why` lines and the where_ran host summary. Claude reviews (verify-go) and decides
the flip (PLAN section 4). Do not flip, re-tune or re-run anything after a scored verdict.

## STOP (write NEEDS and stop; never work around)

- Step 0 not READY; a step-2 test or selftest fails; a node lacks LTM, CORPUS or RUNNER after one re-provision.
- The canary: rc != 0, a missing trace, an errored turn, or the canary not a flagged worm guess (the merged stack does
  not reproduce the 09-30 gate: that is the result, report it; do not queue the other 95).
- A line fails twice (`worker FAILED`), or more than 4 lines fail once.
- The score is UNDEFINED after one re-run of its cells, or FAIL (report the `why` lines; no fixes).
- Anything would need a gate seed, a default change, or an edit outside research/findings/raw/_stage1_c1_flipdev/,
  research/findings/raw/_stage1_c1_category/gate/, research/queue/ and this folder's WORKLOG.md.
