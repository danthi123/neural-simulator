# HANDOFF: s2p-flip (session-2 packet T21; execution packet for Hermes)

Written 2026-10-01 by the cloud Claude session 2. Prereg: `research/findings/2026-10-01-s2p-flip-PREREGISTRATION.md`.
Tool: `tools/s2p_flip.py` (tests: `tests/test_s2p_score.py::test_flip_tool_apply_then_revert_round_trips`). Use no
judgment beyond the tables. The flip commit below is a scripted edit on YOUR branch, the same exception the owner
granted for the iw1/iw2 flip commits; the merge to main is Claude's.

Every block is ONE Bash call. Set first: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees; K=docs/drafts/s2p-flip`.

## 0. Start condition

All of: (a) `cd $P && $PY tools/integration_wave2_flip.py status` shows the iw2 packets "on" on main; (b) a RULING
`s2p-confirmed: <packet>=<op>, ...` exists in runbook section 6 (Claude writes it when confirming T18-T20 PASSes);
(c) `research/integration-wave2/PINS.tsv`'s last `F` row is the iw2 flip pin F whose step B passed (it is the base).
Write FLIP = the ruling's list, typed literally (e.g. `fb-spiking-witness=4 transitive-spiking-reach=0`).

## 1. The flip commit G, provision G

```bash
cd $P && git worktree add -b research/s2p-flip $WT/s2p-flip main && cd $WT/s2p-flip && ln -s $P/.venv .venv && ln -s $P/data data
$PY tools/s2p_flip.py apply <FLIP>
SIM_NO_PROVENANCE=1 nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_s2p_score.py tests/test_fb_spiking_witness.py tests/test_transitive_spiking_reach.py tests/test_word_role_parse.py tests/test_integration_wave2_anchors.py
git add -u && git commit -m "flip(s2p, PROVISIONAL, branch only): <FLIP> via tools/s2p_flip.py; validation pending (B, F1)"
G=$(git rev-parse HEAD); mkdir -p $K && printf 'G\t%s\t%s\t%s\n' "$G" "<FLIP>" "$(date -u +%FT%TZ)" >> $K/PINS.tsv && echo $G
mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$G
```

Tests that fail only because a flipped anchor now reads True (an anchor test asserting False) are EXPECTED. List
them in the WORKLOG and continue only if every failure is of that kind. Any other failure: `git checkout -- .` is
denied to you, so run `$PY tools/s2p_flip.py revert <packets>`, then STOP with NEEDS-CLAUDE and the test names. A gate
block on the commit (other than the compute gates): revert as above, then STOP with NEEDS-CLAUDE and its message.
Commit the PINS.tsv line after the flip commit (`git add $K/PINS.tsv && git commit -m "ops(s2p-flip): pin G"`).

## 2. Step B lines at G (needs G provisioned: `ok`/`already`)

```bash
cd $WT/s2p-flip && G=$(awk '$1=="G"{g=$2} END{print g}' $K/PINS.tsv)
IW2_JOBS_OUTD=$K/jobs bash research/packets/integration-wave2/make_jobs.sh flip $G false-belief-chat,transitive-chat,neural-selfid
git add $K/jobs && git commit -m "ops(s2p-flip): step B lines at ${G:0:9}"
```

Queue in waves of 43 (each only when fewer than 10 `iw2-flip-${G:0:7}` lines are queued), exactly as the iw2 HANDOFF
section 7. Canary first, per RULING `s2p-canary`: `bash tools/pool_canary.sh run pool41 "$(head -1 $K/jobs/flip_${G:0:7}.txt)" --wait-s 7200`.

```bash
cd $P && G=$(awk '$1=="G"{g=$2} END{print g}' $WT/s2p-flip/$K/PINS.tsv)
bash -c 'R="s2p-flip step B: prereg research/findings/2026-10-01-s2p-flip-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216"; sed -n 1,43p '"$WT/s2p-flip/$K/jobs/flip_${G:0:7}.txt"' | while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "$R" </dev/null || echo "ADD FAILED: $l"; done'
# later waves: sed -n 44,86p, 87,129p, ... until the file ends; log each wave in $K/waves.tsv
```

## 3. Score (when every line has landed)

```bash
cd $WT/s2p-flip && G=$(awk '$1=="G"{g=$2} END{print g}' $K/PINS.tsv) && G7=${G:0:7}
F=<the iw2 flip pin F, full sha>; F7=${F:0:7}; V=research/findings/raw/_s2p/s2p-flip/$G7
mkdir -p research/findings/raw/_load_bearing/_shards $V && cp -a $P/research/findings/raw/_load_bearing/_shards/iw2-flip-$G7 research/findings/raw/_load_bearing/_shards/
$PY tools/iw2_score.py B --base-tag iw2-flip-$F7 --base-pin $F --flip-tag iw2-flip-$G7 --flip-pin $G --flipped false-belief-chat,transitive-chat,neural-selfid --out $V/B.json; echo B rc=$?
# for each packet p in FLIP, with its own selection.json (on its research/<p> branch, copied into $V/<p>.selection.json):
$PY tools/s2p_score.py R --packet <p> --tag iw2-flip-$G7 --pin $G --selection $V/<p>.selection.json --at-default --out $V/F1_<p>.json; echo F1 rc=$?
git add $V research/findings/raw/_load_bearing/_shards/iw2-flip-$G7 && git commit -m "data(s2p-flip): B and F1 at ${G:0:9} (PROVISIONAL)"
```

| B | every F1 | do |
|---|---|---|
| PASS | PASS | finding `research/findings/<date>-s2p-flip.md` (PROVISIONAL; numbers from B.json and F1_*.json; each packet's prereg section-6 sentence + "now the default path of <faculty>"; never the words "on by default" or "wired into production" without `integration_faculty:` frontmatter, which the production_integration gate checks), verify-go per RULING `verify-go`, NEEDS-CLAUDE "[s2p-flip] B + F1 PASS at research/s2p-flip @ <sha>; confirm, merge, record full Stage-1 credit" |
| any | an F1 FAIL for p | `$PY tools/s2p_flip.py revert p`, commit ("revert(s2p): p -- F1 FAIL"), append a new G row, provision, redo sections 2-3 |
| FAIL | PASS | the iw2 HANDOFF section 7 attribution (alloff/bisect with the s2p packets as the candidates: `$PY tools/s2p_flip.py revert <q>` in a scratch commit replaces `integration_wave2_flip.py revert`): one packet to blame -> revert it and redo; none or several -> revert all, NEEDS-CLAUDE "[s2p-flip] interaction at <faculty>" |
| UNDEFINED | - | one re-run per not-DEFINED cell (iw2 HANDOFF section 8), score again; still UNDEFINED: NEEDS-CLAUDE |

## Stop conditions

As the iw2 HANDOFF: any gate other than the compute gates; a needed edit to code, a prereg, a scorer or a job line;
a pool job failing twice outside the re-run rule.
