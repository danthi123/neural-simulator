# HANDOFF (Hermes): C1 with the hop fix: gate re-measure + flip dev check (Claude review 2026-10-06, rev. c1-hop)

Plan: `docs/drafts/stage1-c1-inheritance/PLAN.md` steps A2, A3 and B (RULING c1-hop-first: the hop fix lands BEFORE
the flip). Runners: `research/runners/stage1_c1_category.py` (gate re-measure, `--hop`) and
`research/runners/c1_flip_dev_check.py` (flip dev check; every cell now runs with `BRAIN_CATEGORY_INHERIT_HOP=1` and
the bridge group is in the bar). You run and record; Claude reads both scores and decides the flip. Never change
`_CATEGORY_INHERIT_DEFAULT_ON` or the hop default, never edit `webapp/`, `sim/` or `research/runners/`, never queue a
gate seed except the step-3b lines emitted by `stage1_c1_category --emit-pool-jobs ... --hop`. Append one line per
step to `docs/drafts/stage1-c1-inheritance/WORKLOG.md` (create it on first use).

```bash
P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python; cd $P
```

## 0. Precondition

`claude-review/c1-hop` is merged into main:
`git merge-base --is-ancestor claude-review/c1-hop main && grep -q HOP_FLAG research/runners/c1_flip_dev_check.py
&& echo READY`. Not READY: stop, nothing to do yet.

## 1. Commit the C1 wiring gate shards (record only; skip if already committed)

The 52 shards under `research/findings/raw/_stage1_c1_category/gate/s*/` (the 09-30 gate, WITHOUT the hop fix):

```bash
git add research/findings/raw/_stage1_c1_category/gate/s42 research/findings/raw/_stage1_c1_category/gate/s43 \
  research/findings/raw/_stage1_c1_category/gate/s44 research/findings/raw/_stage1_c1_category/gate/s100 \
  research/findings/raw/_stage1_c1_category/gate/s101 research/findings/raw/_stage1_c1_category/gate/s102
git diff --cached --name-only | grep -c _stage1_c1_category/gate/    # expect 104 (52 json + 52 .prov.json)
git commit -m "data(stage1-c1): the 52 C1 wiring gate shards at 6c2cb1362 (scored by Claude 2026-10-06)" -- research/findings/raw/_stage1_c1_category/gate
```

If the count is not 104 (108 if seed 42's two lateral arms landed), commit what is there and write the count in the
WORKLOG; the verdict does not change (those arms are informational).

## 2. Pin and check (CPU, no brain)

```bash
DEV_SHA=$(git rev-parse main); echo $DEV_SHA
$PY -m pytest -q tests/test_c1_flip_dev_check.py tests/test_stage1_exit_gate.py \
  tests/test_category_inherit_hop.py -k "verdict or gate_ or env_parse"
$PY -m research.runners.c1_flip_dev_check --selftest 2>/dev/null | tail -1        # SELFTEST OK
$PY -m research.runners.stage1_c1_category --selftest 2>/dev/null | tail -1       # SELFTEST OK
```

Record DEV_SHA in the WORKLOG. Every later line uses this one revision. (The organ-building tests ran on the
workstation at the merge; do not run them on the server.)

## 3. Provision and the flip-dev canary

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
r = d["responses"]; pe = r["p_canary_eat"]; rd = (pe.get("category_inherit") or {}).get("read") or {}
print("turns", len(r), "| flag", d.get("flip_dev"), "| sources", d.get("sources"), "| corpus", d.get("corpus_present"))
print("trace on every turn", all("category_inherit" in x for x in r.values()))
print("canary:", pe.get("abstained"), pe.get("recalled_svo"), pe.get("hypothesis_svo"), "| read hop", rd.get("hop"),
      rd.get("query_steps"))
EOF
```

Expect 16 turns, `flag_value` "1" and `hop` "1" in flip_dev, sources `['tiny-demo +LTM']`, corpus True, trace on
every turn True, and the canary `False None ['canary', 'eat', 'worm'] | read hop True 35`. Then step 3b, then add the
other 95 flip-dev lines (skip the canary's):

```bash
bash -c 'n=0; while IFS=$'"'"'\t'"'"' read -r cmd why; do out=${cmd##*--out }; [ -f "$out" ] && continue; grep -qF -- "$out" research/queue/pool.queue && continue; n=$((n+1)); bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < '"$JOBS"'; echo "added $n"'
```

## 3b. The C1 gate re-measure WITH the fix (gate seeds; prereg 2026-09-30 section 13, amendment 2026-10-06)

The 09-30 C1 PASS 6/6 was measured without the hop fix. To keep a C1 PASS valid for the configuration the flip
ships, the same 54 gate lines run again with `BRAIN_CATEGORY_INHERIT_HOP=1`: seeds 42, 43, 44, 100, 101, 102 x arms
`intact_a`, `intact_b`, `lesion_inherit`, `lesion_inherit.rep0`, `permuted` (counted) and `off`,
`lesion_lateral_learn`, `lesion_lateral_read`, `lesion_draw` (informational), into
`research/findings/raw/_stage1_c1_category/gate_hop/`. Queue them after the flip-dev canary passed:

```bash
$PY -m research.runners.stage1_c1_category --emit-pool-jobs $DEV_SHA --hop > research/queue/c1_gatehop_${DEV_SHA:0:9}.tsv
wc -l < research/queue/c1_gatehop_${DEV_SHA:0:9}.tsv                              # 54
grep -c 'BRAIN_CATEGORY_INHERIT_HOP=1 ' research/queue/c1_gatehop_${DEV_SHA:0:9}.tsv  # 54
GJOBS=research/queue/c1_gatehop_${DEV_SHA:0:9}.tsv
bash -c 'n=0; while IFS=$'"'"'\t'"'"' read -r cmd why; do out=${cmd##*--out }; [ -f "$out" ] && continue; grep -qF -- "$out" research/queue/pool.queue && continue; n=$((n+1)); bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < '"$GJOBS"'; echo "added $n"'
```

## 4. Score (when all cells of a set have landed, or 120 h after queueing)

```bash
$PY -m research.runners.stage1_c1_category --aggregate research/findings/raw/_stage1_c1_category/gate_hop --hop \
  --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage1_c1_category/gate_hop/aggregate.json
$PY -m research.runners.c1_flip_dev_check --score research/findings/raw/_stage1_c1_flipdev \
  --out research/findings/raw/_stage1_c1_flipdev/flipdev_score.json
$PY tools/where_ran.py research/findings/raw/_stage1_c1_category/gate_hop | tail -6
$PY tools/where_ran.py research/findings/raw/_stage1_c1_flipdev | tail -6
git add research/findings/raw/_stage1_c1_category/gate_hop && git commit -m "data(stage1-c1): C1 gate re-measure with the hop fix at ${DEV_SHA:0:9}, scored" -- research/findings/raw/_stage1_c1_category/gate_hop
git add research/findings/raw/_stage1_c1_flipdev && git commit -m "data(c1-flipdev): dev seeds 1-8 with the hop fix at ${DEV_SHA:0:9}, scored" -- research/findings/raw/_stage1_c1_flipdev
```

Never aggregate `gate_hop` without `--hop`. UNDEFINED names its seeds or cells: re-add only the missing or invalid
cells' lines once (the loops skip landed files; delete an invalid cell's file first and say so in the WORKLOG), then
score again.

## 5. Report

Write two NEEDS lines with the `why` lines and the where_ran host summaries:
`[c1-gatehop] RESULT <PASS|FAIL|UNDEFINED> correct <k>/6, lesion effect <k>/6, reproduced <k>/6, permuted <k>/6, DEV_SHA <sha9>`
and `[c1-flipdev] RESULT <PASS|FAIL|UNDEFINED> n_pass=<k>/8, bridge: bat-swim guesses on <m> of <d> seeds, DEV_SHA <sha9>`.
Claude reviews (verify-go) and decides the flip (PLAN section 4). Do not flip, re-tune or re-run anything after a
scored verdict.

## STOP (write NEEDS and stop; never work around)

- Step 0 not READY; a step-2 test or selftest fails; a node lacks LTM, CORPUS or RUNNER after one re-provision.
- The flip-dev canary: rc != 0, a missing trace, an errored turn, `hop` not "1" in flip_dev, the read without
  `hop: True` / `query_steps: 35`, or the canary not a flagged worm guess (the merged stack with the fix does not
  reproduce the organ: that is the result, report it; queue neither the 95 flip-dev lines nor the 54 gate lines).
- The gate re-measure reads anything but PASS after one re-run of its missing cells (FAIL, or UNDEFINED with a
  `hop attenuation not applied` reason, or a lesion / permuted clause short of 6/6): report; no fixes, no flip.
- A line fails twice (`worker FAILED`), or more than 4 lines of a set fail once.
- The flip-dev score is UNDEFINED after one re-run of its cells, or FAIL (report the `why` lines; no fixes).
- Anything would need a gate seed outside step 3b, a default change, or an edit outside
  research/findings/raw/_stage1_c1_flipdev/, research/findings/raw/_stage1_c1_category/gate/ and gate_hop/,
  research/queue/ and this folder's WORKLOG.md.
