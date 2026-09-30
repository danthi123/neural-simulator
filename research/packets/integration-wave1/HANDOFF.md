# HANDOFF: integration wave 1 -- three default-flip packets (for the local model)

Written 2026-09-29 by Claude. Branch `research/integration-wave1`. Execute top to bottom. Use no judgment beyond the
tables: a row marked NEEDS-CLAUDE means STOP that thread and write one line under "NEEDS CLAUDE" in
`research/coordination/LOCAL_LLM_RUNBOOK.md` (on main), then continue with any other thread that is not blocked.

The three packets (each has its own HANDOFF with its outcome table):
- `d6-hebbian-write/HANDOFF.md` -- the in-conversation fact write by the local Hebbian rule (row `in-loop-learning`).
- `wm-focus-bind/HANDOFF.md` -- an anaphor resolves to the referent a spiking WTA retrieves (row `wm-binding-advanced`).
- `xedge-surprise-worldmodel/HANDOFF.md` -- cross-edge C1, the world model learns from its own surprise
  (row `worldmodel-forward`).

Governing preregs (committed before any gate-seed job): `research/findings/2026-09-29-iw1-wave-battery-
PREREGISTRATION.md` (shared steps D, B, F, the decision table, the UNDEFINED rule) and one per packet,
`research/findings/2026-09-29-iw1-<packet>-flip-PREREGISTRATION.md`.

**What you may and may not do here.** You make every commit below on `research/integration-wave1` only, never on
main, and you never merge. The flip commit in step 6 is a scripted edit on this branch that the decision table
authorizes; it changes no production default until Claude merges the branch. If your runbook's "never flip a
default" line stops you at step 6, STOP there and write NEEDS-CLAUDE "iw1: runbook blocks the scripted flip commit on
the packet branch". Every verdict you write is PROVISIONAL until Claude inspects it.

**Each fenced block below is ONE Bash call** and starts with `source .../env.sh` (the Bash tool keeps no state
between calls). `env.sh` sets `P` (primary checkout, where pool results land), `W` (this worktree), `K` (this packet
directory), `SH` (the shard root), and the pins recorded so far in `$K/PINS.tsv`: `PIN_P`, `P7`, `PIN_F`, `F7`,
`FLIP`. Loops are already wrapped in `bash -c`.

`$W/.venv` and `$W/data` are symlinks to the primary's (gitignored). If missing: `ln -s $P/.venv $W/.venv; ln -s
$P/data $W/data`.

## 0. Already done (do not redo)

- Anchors (all read False here): `_D6_LEARN_DEFAULT_ON` (research/runners/d6_hebbian_store.py; every reader of the
  three D6 flags now routes through it), `_MULTIREF_FOCUS_BIND_DEFAULT_ON` (research/runners/
  d6_multiref_wm_production_organ.py), and the existing `_XEDGE_SWM_DEFAULT_ON`.
- Opt-in LBF rows: `research/runners/lbf_rows/d6_hebbian_write.py`, `research/runners/lbf_rows/
  xedge_surprise_worldmodel.py` (REQUIRED_ENV keeps them out of every production-default battery until the flip).
- Tools: `tools/integration_wave1_flip.py` (apply / revert / status), `tools/assert_iw1_defaults.py` (job guard),
  `tools/iw1_score.py` (the registered scorer: steps R, B, A, X), `research/runners/iw1_default_identity.py`
  (real-handler identity arms), `$K/make_jobs.sh` (job files). Tests: `tests/test_integration_wave1_anchors.py`.
- Dev seed 7 results: section 9 below.
- Dropped from the wave, with the reason: the learned-referent lexicon (section 10).

## 1. Pin P (step 0 of the shared prereg)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $W && git status --short --untracked-files=no          # must print nothing
git fetch -q origin
git merge-base --is-ancestor origin/main HEAD || git merge --no-edit origin/main
.venv/bin/python -m pytest -q tests/test_integration_wave1_anchors.py
.venv/bin/python tools/integration_wave1_flip.py status    # all three must read "off"
bash tools/push_both.sh research/integration-wave1
PIN_P=$(git rev-parse HEAD); echo "$PIN_P"
```

| result | do |
|---|---|
| status printed tracked changes | STOP. NEEDS-CLAUDE "iw1: uncommitted tracked changes in the worktree" |
| the merge reports a conflict | `git merge --abort`; STOP. NEEDS-CLAUDE "iw1: merge conflict with origin/main: <files>" |
| a test fails, or a packet is not "off" | STOP. NEEDS-CLAUDE with the failing test name or the status line |
| the pre-commit/merge hook blocks | compute gates (`compute_idle_persistent`, `lane_starvation`): wait and retry. Any other gate: STOP, NEEDS-CLAUDE with its message. Never `--no-verify`, never a waiver file |
| push verified on both remotes | continue; write `P` and `$PIN_P` in your WORKLOG |

## 2. Provision P on the pool

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
PIN_P=$(git -C $W rev-parse HEAD); cd $P && bash tools/pool_provision.sh --revision $PIN_P --isolated pool41 pool42
```

Add AWS aliases (e.g. `pool1 pool2`) only if they are already up; you do not start AWS. If the provisioner refuses
(stale source or anything else): STOP, NEEDS-CLAUDE with its message. Never set `POOL_PROVISION_ALLOW_STALE`.

## 3. Job files for steps D and R at P, then commit them

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $W && PIN_P=$(git rev-parse HEAD) && git merge-base --is-ancestor origin/main HEAD
bash $K/make_jobs.sh identity $PIN_P default off on:d6-hebbian-write on:wm-focus-bind on:xedge-surprise-worldmodel
bash $K/make_jobs.sh R $PIN_P
printf 'P\t%s\t%s\n' "$PIN_P" "$(date -u +%FT%TZ)" >> $K/PINS.tsv
git add $K/PINS.tsv $K/jobs/ $SH/iw1-R-*-${PIN_P:0:7}/PIN.txt
git commit -m "ops(iw1): pin P=${PIN_P:0:7}; step D + R job files (nothing queued yet)"
bash tools/push_both.sh research/integration-wave1
```

(`PIN_P` here is the HEAD you provisioned in section 2; if HEAD moved since, STOP, NEEDS-CLAUDE.) Expect
`identity_<P7>.txt` with 5 lines and `R_d6_<P7>.txt`, `R_swm_<P7>.txt`, `R_wmf_<P7>.txt` with 6 lines each.
Any other count: STOP, NEEDS-CLAUDE. (This commit is a child of P and changes no code; jobs run at P.)

## 4. Queue steps D and R (from the primary checkout; each add goes through the queue's own gates)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $P
bash -c 'R="iw1 step D dev seed 7, prereg research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md; mem_gb=12"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/identity_$P7.txt"
bash -c 'R="iw1 step R, prereg research/findings/2026-09-29-iw1-d6-hebbian-write-flip-PREREGISTRATION.md; mem_gb=8"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_d6_$P7.txt"
bash -c 'R="iw1 step R, prereg research/findings/2026-09-29-iw1-xedge-surprise-worldmodel-flip-PREREGISTRATION.md; mem_gb=8"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_swm_$P7.txt"
bash -c 'R="iw1 step R, prereg research/findings/2026-09-29-iw1-wm-focus-bind-flip-PREREGISTRATION.md; mem_gb=10"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_wmf_$P7.txt"
```

| result | do |
|---|---|
| every add succeeds | continue; note the time in WORKLOG |
| an add fails (`ADD FAILED`, a shape-check or prereg refusal) | STOP that file. NEEDS-CLAUDE with the refusal text. Do not edit the line |

Expected wall time: D about 30-60 min per arm (each identity arm builds two chat sessions, ~12 GB peak: one per
node), R d6/swm about 30 min per seed, R wmf 2-3 h per seed (32 builds).

## 5. Score steps D and R (after `bash tools/status.sh` shows the lines done; results land under `$P`)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $P && I=research/findings/raw/_iw1/identity/$P7; V=$W/research/findings/raw/_iw1/verdicts/$P7; ID=$W/research/runners/iw1_default_identity.py
.venv/bin/python $ID --compare $I/default.json $I/off.json --expect same --out $V/D0.json
.venv/bin/python $ID --compare $I/default.json $I/on_d6-hebbian-write.json --expect differs --out $V/D_d6-hebbian-write.json
.venv/bin/python $ID --compare $I/default.json $I/on_wm-focus-bind.json --expect differs --out $V/D_wm-focus-bind.json
.venv/bin/python $ID --compare $I/default.json $I/on_xedge-surprise-worldmodel.json --expect differs --out $V/D_xedge-surprise-worldmodel.json
.venv/bin/python $W/tools/iw1_score.py R --packet d6-hebbian-write --tag iw1-R-d6-$P7 --pin $PIN_P --out $V/R_d6.json
.venv/bin/python $W/tools/iw1_score.py R --packet xedge-surprise-worldmodel --tag iw1-R-swm-$P7 --pin $PIN_P --out $V/R_swm.json
.venv/bin/python $W/tools/iw1_score.py R --packet wm-focus-bind --tag iw1-R-wmf-$P7 --pin $PIN_P --out $V/R_wmf.json
```

Scorers run from `$P` (they read the landed shards there) and write verdicts into `$W`. Exit codes: 0 PASS, 1 FAIL, 3 UNDEFINED. Read each packet's HANDOFF
table for R. The shared rows:

| D0 | D_p | R_p | do for packet p |
|---|---|---|---|
| PASS | PASS | PASS | p goes into the flip list |
| PASS | any | FAIL | p is not flipped. Draft the NO-GO finding named in p's HANDOFF (provisional) |
| PASS | FAIL ("vacuous") or UNDEFINED | any | p is not flipped. NEEDS-CLAUDE "iw1 p: D_p <verdict>" |
| PASS | PASS | UNDEFINED | apply the re-run rule (section 8) to the UNDEFINED seeds once; still UNDEFINED: p is not flipped, NEEDS-CLAUDE |
| FAIL or UNDEFINED | any | any | NO packet is flipped. STOP. NEEDS-CLAUDE "iw1 D0 <verdict>: an anchor edit may have changed the default path" |

Commit the verdict files and the landed step-D/R results (raw files + `.prov.json` sidecars) on the branch:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $W && bash $K/copy_results.sh $P7 iw1-R-d6-$P7 iw1-R-swm-$P7 iw1-R-wmf-$P7
git add research/findings/raw/_iw1 $SH/iw1-R-*-$P7 && git commit -m "data(iw1): steps D and R at P=$P7 -- D0 <verdict>; R d6 <v>, swm <v>, wmf <v> (provisional)" && bash tools/push_both.sh research/integration-wave1
```

If no packet passed: there is no flip. Write the NO-GO drafts and NEEDS-CLAUDE lines the tables named, and stop the
wave here.

## 6. The flip commit F, then provision F

`FLIP` = the comma-separated packets that passed section 5, in this order when present:
`d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel`.

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $W && FLIP=<the list, typed literally, e.g. d6-hebbian-write,wm-focus-bind>
.venv/bin/python tools/integration_wave1_flip.py apply ${FLIP//,/ } --date $(date +%F)
.venv/bin/python -m pytest -q tests/test_integration_wave1_anchors.py
.venv/bin/python tools/assert_iw1_defaults.py --expect-on=$FLIP && echo guard-ok
git add -u && git commit -m "flip(iw1, PROVISIONAL, branch only): $FLIP default-on via tools/integration_wave1_flip.py; ledger anchors + subflag notes; LBF rows lifted into the headline registry. Validation pending (shared prereg steps B, F)"
bash tools/push_both.sh research/integration-wave1
PIN_F=$(git rev-parse HEAD); printf 'F\t%s\t%s\t%s\n' "$PIN_F" "$FLIP" "$(date -u +%FT%TZ)" >> $K/PINS.tsv
cd $P && bash tools/pool_provision.sh --revision $PIN_F --isolated pool41 pool42
```

| result | do |
|---|---|
| apply refuses, a test fails, or the guard does not print guard-ok | `git checkout -- .` in $W; STOP. NEEDS-CLAUDE with the message |
| the pre-commit hook blocks with the production_integration gate | STOP. NEEDS-CLAUDE with the message (the ledger edit is scripted; a block means the script and the gate disagree) |
| the provisioner refuses | STOP. NEEDS-CLAUDE with its message |
| all succeed | continue |

## 7. Steps B and F at F

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $W
bash $K/make_jobs.sh flip $PIN_F $FLIP
bash $K/make_jobs.sh identity $PIN_F default off on:${FLIP//,/+}
git add $K/PINS.tsv $K/jobs/ $SH/iw1-flip-${PIN_F:0:7}/PIN.txt && git commit -m "ops(iw1): pin F=${PIN_F:0:7} ($FLIP); step B + F job files" && bash tools/push_both.sh research/integration-wave1
```

The flip battery has about 43 + 6 x (rows lifted) lines per seed (~270). Queue it in waves of 43 lines, each wave
only when fewer than 10 `iw1-flip` lines remain in `research/queue/pool.queue`
(`grep -c iw1-flip research/queue/pool.queue`), exactly like B2d's fallback:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $P
bash -c 'R="iw1 step B, prereg research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md; mem_gb=8"; sed -n 1,43p '"$W/$K/jobs/flip_$F7.txt"' | while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done'
# later waves: sed -n 44,86p, 87,129p, 130,172p, 173,215p, 216,258p, 259,301p (until the file ends)
bash -c 'R="iw1 step F dev seed 7, prereg research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md; mem_gb=12"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/identity_$F7.txt"
```

Log each wave (epoch, first-last line) in `$K/waves.tsv`. Every B line starts with
`cd ~/derisk-pool/revisions/<F> && .venv/bin/python tools/assert_iw1_defaults.py --expect-on=<FLIP> &&`; a line that
does not: STOP, NEEDS-CLAUDE. AWS: two r7i.4xlarge cut ~50 h pool-only to ~15 h; starting them is not yours
(NEEDS-CLAUDE if the pool alone would take more than 3 days).

When every line has landed:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave1/research/packets/integration-wave1/env.sh
cd $P && V=$W/research/findings/raw/_iw1/verdicts/$F7; IF=research/findings/raw/_iw1/identity/$F7; IP=research/findings/raw/_iw1/identity/$P7; ID=$W/research/runners/iw1_default_identity.py
.venv/bin/python $W/tools/iw1_score.py B --base-tag b2b0924-base --base-pin a308f1e09babcc9ed096c3c8046d00040391368c --flip-tag iw1-flip-$F7 --flip-pin $PIN_F --flipped $FLIP --out $V/B.json
.venv/bin/python $ID --compare $IF/default.json $IF/on_${FLIP//,/_}.json --expect same --out $V/F2.json
.venv/bin/python $ID --compare $IF/off.json $IP/default.json --expect same --out $V/F3.json
```

(The `on:` arm file name has `:` and `+` replaced by `_`.) Then apply the shared decision table:

| B.json | F2 | F3 | do |
|---|---|---|---|
| PASS | PASS | PASS | the flips stand on the branch. Section 8 re-run log, then section 11 (finding + NEEDS-CLAUDE review/merge) |
| FAIL, and every regression is B1 (level s0/s2) | PASS | PASS | attribution: for each regressed faculty X, `bash $K/make_jobs.sh alloff $PIN_F <X> $FLIP` (run in $W at F), queue it (reason as step B), then `iw1_score.py A --b-result $V/B.json --tag iw1-alloff-$F7 --pin $PIN_F --out $V/A.json`. A PASS (every drop excused) = the flips stand. A FAIL: next row |
| A.json FAIL (a flip-caused drop at X) | - | - | bisect: for each packet q in FLIP, `make_jobs.sh bisect $PIN_F <X> <q>`, queue, `iw1_score.py X --tag iw1-bisect-<q>-$F7 --pin $PIN_F --faculty <X>`. The ONE q whose run is higher than the F count at either level: `integration_wave1_flip.py revert <q>`, commit ("revert(iw1): <q> caused <X> drop"), push, provision, and redo section 7 at the new F. No q, or more than one: revert every packet in FLIP, commit, push; NEEDS-CLAUDE "iw1: interaction at <X>" |
| FAIL with an F1 regression for packet p | - | - | revert p (flip script), commit, push, provision, redo section 7 at the new F |

After ANY revert: `cd $W && .venv/bin/python tools/integration_wave1_flip.py revert <packets>`, run the tests, commit
("revert(iw1): <packets> -- <reason from the table>"), push, then append the new F row
`printf 'F\t%s\t%s\t%s\n' "$(git rev-parse HEAD)" "<the packets still on>" "$(date -u +%FT%TZ)" >> $K/PINS.tsv`,
provision that sha (section 6's last line), and redo section 7 from its first block. If no packet is still on, stop
the wave (NEEDS-CLAUDE with the verdict files).
| any | FAIL | any | revert every packet in FLIP, commit, push. NEEDS-CLAUDE "iw1 F2 FAIL: the default is not the capability" |
| any | any | FAIL | revert every packet in FLIP, commit, push. NEEDS-CLAUDE "iw1 F3 FAIL: the explicit-off escape changed" |
| UNDEFINED | - | - | section 8 re-run rule for each not-DEFINED cell, then score again; still UNDEFINED: NEEDS-CLAUDE, nothing merges |

Commit the landed shards, identity files and verdicts on the branch as in section 5:
`bash $K/copy_results.sh <F7> iw1-flip-<F7> [iw1-alloff-<F7>] [iw1-bisect-<q>-<F7> ...]`, then `git add
research/findings/raw/_iw1 $SH/iw1-*-<F7>`, commit ("data(iw1): steps B and F at F=<F7> -- <verdicts>
(provisional)"), push.

## 8. The re-run rule (UNDEFINED cells)

A cell that is not DEFINED is re-run ONCE with its job line unchanged. Before queueing it, append one line to
`$K/reruns.tsv`: `utc<TAB>tag<TAB>seed<TAB>faculty<TAB>reason (the scorer's fails text)`, and commit that line. Move
the old shard directory aside first (`mv $SH/<tag>/s<seed>/<fac> $SH/<tag>/_attempt1_s<seed>_<fac>`), then queue the
line with the same `--checked` reason. A second failure stays UNDEFINED.

## 9. Dev seed 7 results (Claude, 2026-09-29, worktree at the packet commit, local numpy)

Step D's shape, run locally at commit 6f4c02368 (dev seed 7, 13-turn script, real `/api/brain-chat` handler, one fresh
process per arm): `research/findings/raw/_iw1/dev_s7/identity/*.json`, verdicts in `.../dev_s7/verdicts/`.

- **D0 byte-identical-off: IDENTICAL** -- `default` equals `off` on all 13 turns.
- **D_wm-focus-bind: DIFFERS**, and the reply changes: in the swapped-order session "what does it chase" answers "The
  dog chases cat." (positional) vs "I don't know about that." (flag on; the WTA retrieved "cat").
- **D_xedge-surprise-worldmodel: DIFFERS** on turns 7-12; the reply changes on "what do you expect?": margin +430 Hz
  (default) vs +370 Hz (edge on). Four violation turns were not enough here for the notice to stop.
- **D_d6-hebbian-write: DIFFERS** on 12 of 13 turns by response body (the `d6_hebbian` block and engram-derived
  sets), with NO reply-text difference on this script: an intact recall reads the same whether the fact was copied
  or written. The write's reply-level effect shows only under the freeze lesion (step R; the dev LB run above).
- Latency (report only): arm wall time default 965.5, off 881.5, on_d6-hebbian-write 826.5, on_wm-focus-bind 951.3, on_xedge-surprise-worldmodel 892.2 s; the first turn of each session includes a brain build (up to ~290 s
  locally with other jobs running); later turns 5-25 s. Each identity arm peaks near 12 GB (two chat sessions): the
  first local attempt under a 10 GB cap stalled in memory reclaim and was killed; rerun under 16 GB.

Dev LB rows (local, `--repeats 2`, commit 106a20da5): `d6-hebbian-write` regressed at trace and reply level (the
freeze lesion abstains); `xedge-surprise-worldmodel` regressed at trace and reply level (the margin text). Details
in each packet's HANDOFF. These are dev runs: their sidecars are not pool git_archive, so they carry no gate weight.

## 10. Dropped from the wave: the learned-referent lexicon

`BRAIN_LEARNED_REFERENT_LEXICON` (GO 6/6 on its route, 2026-09-24) was the third candidate. Screening found that
with the flag on, the production `extract_referents` admits wh-words as referents ("what does the dog chase" ->
`['what', 'dog']`, "who are we talking about?" -> `['who']`), so a default flip would put ordinary wh-questions into
the multi-referent organ's two-referent scope. Logged in research/FAILURE_LOG.md (2026-09-29), pinned by a strict
xfail test. Its reply-level probe worked on dev seed 7 (intact: "I'm holding 2 referents in working memory at once:
air and alarm."; lesion: the ordinary abstain), so it is the natural wave-2 packet once the closed-class admission is
fixed. Nothing to do here.

Other default-off candidates screened and not chosen for this wave (for Claude's wave-2 list): the transitive
yes/no chat route (`BRAIN_TRANSITIVE_CHAT`; its keystone chase is 6/6 GO, but the chat route has only a
preregistration, no 6-seed chat-level gate); the false-belief chat wire (`BRAIN_FALSE_BELIEF_CHAT`; seed-7 smoke
only at chat level); DA tag-and-capture (`BRAIN_DA_TAG_CAPTURE`; runner-level 6/6, but the pair review found an
ordinary fact told 4 h before sleep is not kept on 6/6 seeds, so a flip would regress today's default); the
SlotBinder composer (GO scoped to N=32 of the 404-fact corpus); the affect-marker SETTLE route (its own flip arc is
in flight).

## 11. Finding and hand-back (after section 7 reads PASS)

Write ONE finding per flipped packet from the template in its HANDOFF, `status: provisional`, on this branch; sync
`GAP_CLOSURE_MISSION.md` CURRENT STATE with one dated line per packet (numbers only, from the verdict JSONs). Commit,
push. Then NEEDS-CLAUDE "iw1: flips <FLIP> validated at F=<F7> (provisional); adversarial review + merge".

## Stop conditions (any one stops the thread)

- A gate blocks for any reason other than the compute gates: STOP, NEEDS-CLAUDE. Never `--no-verify`, never a waiver.
- You would need to edit `sim/`, `webapp/`, `tools/gates/`, a prereg, a scorer or a job line: STOP, NEEDS-CLAUDE.
- A pool job fails twice (rc != 0) outside the re-run rule, or memory kills (rc 137/143) repeat: STOP, NEEDS-CLAUDE.
- AWS spend near the cap (`bash tools/aws_budget.sh status`): STOP queueing to AWS, NEEDS-CLAUDE.

## NEEDS CLAUDE (open; do not act on these)

1. Adversarial review (verify-go) of each flip that passes, then the merge of `research/integration-wave1` to main.
   The flip bar's review leg is Claude's, not the local model's.
2. The C1 world model is process-shared: once on by default it learns across every conversation a server process
   serves. Rule on whether that is acceptable for production, or whether it needs a per-session organ first
   (xedge-surprise-worldmodel HANDOFF).
3. After a wm-focus-bind flip: whether its paired-session probe should count in the headline fraction (kind
   `neural-lesion-opt-in` is excluded by design today) -- a harness decision.
4. The local-model runbook forbids flipping a default; this HANDOFF authorizes a scripted flip commit on the packet
   branch only. Confirm, or amend the runbook, before the local model reaches section 6.
5. The reference battery for step B is B2b (a308f1e09). If B2d lands first and reads COMPLETE, decide whether it
   replaces B2b as the reference (that is an amendment to the shared prereg, committed before step B is scored).
