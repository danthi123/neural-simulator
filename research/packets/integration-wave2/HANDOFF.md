# HANDOFF: integration wave 2 -- three default-flip packets (for the local model)

Written 2026-09-30 by Claude. Branch `research/integration-wave2`. Execute top to bottom. Use no judgment beyond the
tables: a row marked NEEDS-CLAUDE means STOP that thread and write one line under "NEEDS CLAUDE" in
`research/coordination/LOCAL_LLM_RUNBOOK.md` (on main), then continue with any other thread that is not blocked.

The three packets (each has its own HANDOFF with its outcome table and finding template):
- `false-belief-chat/HANDOFF.md` -- "where will Sally look?" answered off the W3 false-belief register (ToM).
- `transitive-chat/HANDOFF.md` -- "does cat precede river?" answered by the keystone re-entrant chase (held-out inference).
- `neural-selfid/HANDOFF.md` -- "what do you use?" comprehended by the on-brain parser, not the host router.

Governing preregs (committed before any gate-seed job): `research/findings/2026-09-30-iw2-wave-battery-
PREREGISTRATION.md` (the candidate RANKING, shared steps D/S, R's control rule, B, F, the decision table, the UNDEFINED
rule) and one per packet, `research/findings/2026-09-30-iw2-<packet>-flip-PREREGISTRATION.md`.

**What you may and may not do here.** You make every commit below on `research/integration-wave2` only, never on
main, and you never merge. The flip commit in section 6 is a scripted edit on this branch that the decision table
authorizes (the owner ruled on the same step for wave 1, 2026-09-29: one scripted flip commit on the packet branch is
allowed; the merge stays Claude's). If your runbook's "never flip a default" line stops you at section 6, STOP there
and write NEEDS-CLAUDE "iw2: runbook blocks the scripted flip commit on the packet branch". Every verdict you write is
PROVISIONAL until Claude inspects it.

**Each fenced block below is ONE Bash call** and starts with `source .../env.sh` (the Bash tool keeps no state
between calls). `env.sh` sets `P` (primary checkout, where pool results land), `W` (this worktree), `K` (this packet
directory), `SH` (the shard root), and the pins recorded so far in `$K/PINS.tsv`: `PIN_P`, `P7`, `PIN_F`, `F7`,
`FLIP`. Loops are already wrapped in `bash -c`.

`$W/.venv` and `$W/data` are symlinks to the primary's (gitignored). If missing: `ln -s $P/.venv $W/.venv; ln -s
$P/data $W/data`.

## Built in from wave 1's adversarial review (each pinned by `tests/test_integration_wave2_anchors.py`)

- Every `--checked` reason reads `prereg <path> ; mem_gb=N ; max_age_h=216` (a space before each `;`): the pool's
  GATE-SEED GUARD reads the prereg path up to whitespace and checks it with `git cat-file` at the pinned revision.
- Declared `mem_gb` <= 11 (the largest job the dispatcher hands a 15 GB node). LB shards `mem_gb=8` under
  `mem_ok.sh 8 2` + `memcap.sh 12`; identity arms keep ONE brain alive at a time (each script segment starts with
  `reset`) and declare the peak measured on dev seed 7 (section 9).
- Every LB line starts with `tools/assert_iw2_node.py` (exit 3 without data/corpus/ or a resolvable LTM bundle); the
  identity runner refuses the same way itself and `--compare` is UNDEFINED unless both arms had the corpus and the
  same LTM bundle.
- `tools/iw2_score.py` never passes an empty, absent, abstaining or echoed intact reply (UNDEFINED) or a WRONG one
  (`intact-wrong`, FAIL), and every step R includes a permuted control that must hold 6/6.
- UNDEFINED never counts; step A never turns a B that carried undefined criteria into a pass.
- Every queued line carries `max_age_h=216` (at packet time the pool held 441 lines, three to four days of work, with
  wave-1 and Stage-1 lines ahead of these); without it the dispatcher skips a line as debris after 12 h.
- New: step S (scope). An `on:<p>` arm must leave every turn it was not built for byte-identical (the wave-1 lexicon
  screening found a flag that silently widened its scope).

## 0. Already done (do not redo)

- Anchors (all read False here): `_FALSE_BELIEF_CHAT_DEFAULT_ON` (webapp/false_belief_chat.py, new),
  `_TRANSITIVE_CHAT_DEFAULT_ON` (webapp/reasoning_transitive_chat.py, new), and the existing `_NEURAL_SELFID_DEFAULT_ON`
  (research/runners/brain_chat_tui.py; `_NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON` stays False and is NOT part of this wave).
- Opt-in LBF rows: `tom-false-belief` (reply fields now), `reasoning-transitive-chat` (no longer thin or parked: its
  turns are a literal EXTRA_TURNS), `neural-selfid` (new), and the three permuted controls in
  `research/runners/lbf_rows/iw2_controls.py` (opt-in by `LB_IW2_CONTROLS=1`; never lifted by the flip).
- Tools: `tools/integration_wave2_flip.py` (apply / revert / status), `tools/assert_iw2_defaults.py` (job guard),
  `tools/assert_iw2_node.py` (node preflight), `tools/iw2_score.py` (the registered scorer: steps R, B, A, X),
  `research/runners/iw2_default_identity.py` (real-handler identity + scope arms), `$K/make_jobs.sh` (job files).
  Tests: `tests/test_integration_wave2_anchors.py`.
- Dev seed 7 results: section 9. The candidate ranking: the shared prereg's "Selection and ranking".

## 1. Pin P (step 0 of the shared prereg)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $W && git status --short --untracked-files=no -- . ":!research/findings/raw/_provenance/runs.jsonl"   # must print nothing
git fetch -q origin
git merge-base --is-ancestor origin/main HEAD || git merge --no-edit origin/main
SIM_NO_PROVENANCE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_integration_wave2_anchors.py
.venv/bin/python tools/integration_wave2_flip.py status    # all three must read "off"
bash tools/push_both.sh research/integration-wave2
PIN_P=$(git rev-parse HEAD); echo "$PIN_P"
```

| result | do |
|---|---|
| status printed tracked changes | STOP. NEEDS-CLAUDE "iw2: uncommitted tracked changes in the worktree" (the shared run log `research/findings/raw/_provenance/runs.jsonl` is excluded from the check and never committed) |
| the merge reports a conflict | `git merge --abort`; STOP. NEEDS-CLAUDE "iw2: merge conflict with origin/main: <files>" |
| a test fails, or a packet is not "off" | STOP. NEEDS-CLAUDE with the failing test name or the status line |
| the pre-commit/merge hook blocks | compute gates (`compute_idle_persistent`, `lane_starvation`): wait and retry. Any other gate: STOP, NEEDS-CLAUDE with its message. Never `--no-verify`, never a waiver file |
| push verified on both remotes | continue; write `P` and `$PIN_P` in your WORKLOG |

If wave 1 (`research/integration-wave1`) has been merged to main by now, the merge above brings its flips in; that
is expected (shared prereg, "Interaction with wave 1"). If the merge brings in a change to any file wave 2 edits
(`git diff --name-only HEAD@{1} HEAD` lists one of `tools/integration_wave2_flip.py files_touched()`), rerun the tests;
they must pass.

## 2. Provision P on the pool

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
PIN_P=$(git -C $W rev-parse HEAD); cd $P && bash tools/pool_provision.sh --revision $PIN_P --isolated pool41 pool42
for n in pool41 pool42; do ssh -n -o BatchMode=yes -F research/queue/.pool_ssh_config $n "cd ~/derisk-pool/revisions/$PIN_P && .venv/bin/python tools/assert_iw2_node.py" && echo "$n NODE OK" || echo "$n NODE DEGRADED"; done
```

Add AWS aliases only if they are already up; you do not start AWS. If the provisioner refuses (stale source or
anything else): STOP, NEEDS-CLAUDE with its message. Never set `POOL_PROVISION_ALLOW_STALE`. `NODE DEGRADED`: run
`bash tools/pool_sync_assets.sh <node>` once and re-check; still degraded: STOP, NEEDS-CLAUDE "iw2: <node> has no
corpus or LTM bundle at P".

## 3. Job files for steps D and R at P, then commit them

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $W && PIN_P=$(git rev-parse HEAD) && git merge-base --is-ancestor origin/main HEAD
bash $K/make_jobs.sh identity $PIN_P default off on:false-belief-chat on:transitive-chat on:neural-selfid
bash $K/make_jobs.sh R $PIN_P
printf 'P\t%s\t%s\n' "$PIN_P" "$(date -u +%FT%TZ)" >> $K/PINS.tsv
git add $K/PINS.tsv $K/jobs/ $SH/iw2-R-*-${PIN_P:0:7}/PIN.txt
git commit -m "ops(iw2): pin P=${PIN_P:0:7}; step D + R job files (nothing queued yet)"
bash tools/push_both.sh research/integration-wave2
```

(`PIN_P` here is the HEAD you provisioned in section 2; if HEAD moved since, STOP, NEEDS-CLAUDE.) Expect
`identity_<P7>.txt` with 5 lines and `R_fb_<P7>.txt`, `R_tr_<P7>.txt`, `R_si_<P7>.txt` with 12 lines each (6 capability
+ 6 control). Any other count: STOP, NEEDS-CLAUDE. (This commit is a child of P and changes no code; jobs run at P.)

## 4. Queue steps D and R (from the primary checkout; each add goes through the queue's own gates)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $P
bash -c 'R="iw2 step D dev seed 7: prereg research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md ; mem_gb=11 ; max_age_h=216"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/identity_$P7.txt"
bash -c 'R="iw2 step R: prereg research/findings/2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_fb_$P7.txt"
bash -c 'R="iw2 step R: prereg research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_tr_$P7.txt"
bash -c 'R="iw2 step R: prereg research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/R_si_$P7.txt"
```

| result | do |
|---|---|
| every add succeeds | continue; note the time in WORKLOG |
| an add fails (`ADD FAILED`, a shape-check or prereg refusal) | STOP that file. NEEDS-CLAUDE with the refusal text. Do not edit the line |

Expected wall time once dispatched (section 9 has the dev-seed numbers): an identity arm about 20-40 min (three brain
builds in sequence, one node each: `mem_gb=11`); an LB shard 10-20 min locally, so about 20-40 min on a pool node.
41 lines in all, about 20-30 node-hours.

## 5. Score steps D, S and R (after `bash tools/status.sh` shows the lines done; results land under `$P`)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $P && I=research/findings/raw/_iw2/identity/$P7; V=$W/research/findings/raw/_iw2/verdicts/$P7; ID=$W/research/runners/iw2_default_identity.py
.venv/bin/python $ID --compare $I/default.json $I/off.json --expect same --out $V/D0.json
for p in false-belief-chat transitive-chat neural-selfid; do .venv/bin/python $ID --compare $I/default.json $I/on_$p.json --expect differs --packet $p --out $V/D_$p.json; done
.venv/bin/python $W/tools/iw2_score.py R --packet false-belief-chat --tag iw2-R-fb-$P7 --pin $PIN_P --out $V/R_fb.json
.venv/bin/python $W/tools/iw2_score.py R --packet transitive-chat --tag iw2-R-tr-$P7 --pin $PIN_P --out $V/R_tr.json
.venv/bin/python $W/tools/iw2_score.py R --packet neural-selfid --tag iw2-R-si-$P7 --pin $PIN_P --out $V/R_si.json
```

Scorers run from `$P` (they read the landed shards there) and write verdicts into `$W`. Exit codes: 0 PASS, 1 FAIL,
3 UNDEFINED. `D_<p>.json` holds both D_p and S_p: `DIFFERS-IN-SCOPE` = both pass; `OUT-OF-SCOPE` = S_p FAIL;
`UNDEFINED (vacuous ...)` = D_p UNDEFINED. Read each packet's HANDOFF table for R. The shared rows:

| D0 | D_p verdict | R_p | do for packet p |
|---|---|---|---|
| PASS | DIFFERS-IN-SCOPE | PASS | p goes into the flip list |
| PASS | any | FAIL | p is not flipped. Draft the NO-GO finding named in p's HANDOFF (provisional) |
| PASS | OUT-OF-SCOPE | any | p is not flipped. NEEDS-CLAUDE "iw2 p: S_p OUT-OF-SCOPE on turns <list from D_p.json>" |
| PASS | UNDEFINED (vacuous or a failed precondition) | any | p is not flipped. NEEDS-CLAUDE "iw2 p: D_p <verdict>" |
| PASS | DIFFERS-IN-SCOPE | UNDEFINED | apply the re-run rule (section 8) to the not-DEFINED cells once; still UNDEFINED: p is not flipped, NEEDS-CLAUDE |
| FAIL or UNDEFINED | any | any | NO packet is flipped. STOP. NEEDS-CLAUDE "iw2 D0 <verdict>: an anchor edit may have changed the default path" |

Commit the verdict files and the landed step-D/R results (raw files + `.prov.json` sidecars) on the branch:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $W && bash $K/copy_results.sh $P7 iw2-R-fb-$P7 iw2-R-tr-$P7 iw2-R-si-$P7
git add research/findings/raw/_iw2 $SH/iw2-R-*-$P7 && git commit -m "data(iw2): steps D, S and R at P=$P7 -- D0 <verdict>; R fb <v>, tr <v>, si <v> (provisional)" && bash tools/push_both.sh research/integration-wave2
```

If no packet passed: there is no flip. Write the NO-GO drafts and NEEDS-CLAUDE lines the tables named, and stop the
wave here.

## 6. The flip commit F, then provision F

`FLIP` = the comma-separated packets that passed section 5, in this order when present:
`false-belief-chat,transitive-chat,neural-selfid`.

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $W && FLIP=<the list, typed literally, e.g. false-belief-chat,neural-selfid>
.venv/bin/python tools/integration_wave2_flip.py apply ${FLIP//,/ } --date $(date +%F)
SIM_NO_PROVENANCE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_integration_wave2_anchors.py
.venv/bin/python tools/assert_iw2_defaults.py --expect-on=$FLIP && echo guard-ok
git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null   # the shared run log; never part of a flip commit
git add -u && git commit -m "flip(iw2, PROVISIONAL, branch only): $FLIP default-on via tools/integration_wave2_flip.py; ledger anchors + subflag notes; LBF rows lifted into the headline registry. Validation pending (shared prereg steps B, F)"
bash tools/push_both.sh research/integration-wave2
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
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $W
bash $K/make_jobs.sh flip $PIN_F $FLIP
bash $K/make_jobs.sh identity $PIN_F default off on:${FLIP//,/+}
git add $K/PINS.tsv $K/jobs/ $SH/iw2-flip-${PIN_F:0:7}/PIN.txt && git commit -m "ops(iw2): pin F=${PIN_F:0:7} ($FLIP); step B + F job files" && bash tools/push_both.sh research/integration-wave2
```

The flip battery has one line per registry faculty and seed (about 50 faculties x 6 = ~300 lines). Queue it in waves
of 43 lines, each wave only when fewer than 10 `iw2-flip` lines remain in `research/queue/pool.queue`
(`grep -c iw2-flip research/queue/pool.queue`):

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $P
bash -c 'R="iw2 step B: prereg research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=216"; sed -n 1,43p '"$W/$K/jobs/flip_$F7.txt"' | while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done'
# later waves: sed -n 44,86p, 87,129p, 130,172p, 173,215p, 216,258p, 259,301p, 302,344p (until the file ends)
bash -c 'R="iw2 step F dev seed 7: prereg research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md ; mem_gb=11 ; max_age_h=216"; while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done < '"$W/$K/jobs/identity_$F7.txt"
```

Log each wave (epoch, first-last line) in `$K/waves.tsv`. Every B line starts with
`cd ~/derisk-pool/revisions/<F> && .venv/bin/python tools/assert_iw2_node.py && .venv/bin/python
tools/assert_iw2_defaults.py <FLIP> &&` (the packet list is POSITIONAL: `pool_queue.sh add` refuses a pinned line
carrying any `--flag` its `-m` module does not accept, and `--expect-on` is not a load_bearing_fraction flag); a line
that does not: STOP, NEEDS-CLAUDE. AWS: starting it is not
yours (NEEDS-CLAUDE if the pool alone would take more than 3 days).

When every line has landed:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/integration-wave2/research/packets/integration-wave2/env.sh
cd $P && V=$W/research/findings/raw/_iw2/verdicts/$F7; IF=research/findings/raw/_iw2/identity/$F7; IP=research/findings/raw/_iw2/identity/$P7; ID=$W/research/runners/iw2_default_identity.py
.venv/bin/python $W/tools/iw2_score.py B --base-tag b2b0924-base --base-pin a308f1e09babcc9ed096c3c8046d00040391368c --flip-tag iw2-flip-$F7 --flip-pin $PIN_F --flipped $FLIP --out $V/B.json
.venv/bin/python $ID --compare $IF/default.json $IF/on_${FLIP//,/_}.json --expect same --out $V/F2.json
.venv/bin/python $ID --compare $IF/off.json $IP/default.json --expect same --out $V/F3.json
```

(The `on:` arm file name has `:` and `+` replaced by `_`.) Then apply the shared decision table:

| B.json | F2 | F3 | do |
|---|---|---|---|
| PASS | PASS | PASS | the flips stand on the branch. Section 8 re-run log, then section 11 (finding + NEEDS-CLAUDE review/merge) |
| FAIL, and every regression is B1 (level s0/s2) | PASS | PASS | attribution: for each regressed faculty X, `bash $K/make_jobs.sh alloff $PIN_F <X> $FLIP` (run in $W at F), queue it (reason as step B), then `iw2_score.py A --b-result $V/B.json --tag iw2-alloff-$F7 --pin $PIN_F --out $V/A.json`. A PASS (every drop excused) = the flips stand. A FAIL: next row |
| A.json UNDEFINED | - | - | section 8 re-run rule for the not-DEFINED cells (B's and A's), then score B and A again; still UNDEFINED: nothing merges, NEEDS-CLAUDE with both verdict files |
| A.json FAIL (a flip-caused drop at X) | - | - | bisect: for each packet q in FLIP, `make_jobs.sh bisect $PIN_F <X> <q>`, queue, `iw2_score.py X --tag iw2-bisect-<q>-$F7 --pin $PIN_F --faculty <X>`. The ONE q whose run is higher than the F count at either level: `integration_wave2_flip.py revert <q>`, then the revert procedure below. No q, or more than one: revert every packet in FLIP, commit, push; NEEDS-CLAUDE "iw2: interaction at <X>" |
| FAIL with an F1 regression for packet p | - | - | revert p (the revert procedure below) and redo section 7 at the new F |
| any | FAIL | any | revert every packet in FLIP, commit, push. NEEDS-CLAUDE "iw2 F2 FAIL: the default is not the capability" |
| any | any | FAIL | revert every packet in FLIP, commit, push. NEEDS-CLAUDE "iw2 F3 FAIL: the explicit-off escape changed" |
| UNDEFINED | - | - | section 8 re-run rule for each not-DEFINED cell, then score again; still UNDEFINED: NEEDS-CLAUDE, nothing merges |

**Revert procedure.** `cd $W && .venv/bin/python tools/integration_wave2_flip.py revert <packets>`, run the tests,
commit ("revert(iw2): <packets> -- <reason from the table>"), push, append the new F row
`printf 'F\t%s\t%s\t%s\n' "$(git rev-parse HEAD)" "<the packets still on>" "$(date -u +%FT%TZ)" >> $K/PINS.tsv`,
provision that sha (section 6's last line), and redo section 7 from its first block. If no packet is still on, stop
the wave (NEEDS-CLAUDE with the verdict files).

Commit the landed shards, identity files and verdicts on the branch as in section 5:
`bash $K/copy_results.sh <F7> iw2-flip-<F7> [iw2-alloff-<F7>] [iw2-bisect-<q>-<F7> ...]`, then `git add
research/findings/raw/_iw2 $SH/iw2-*-<F7>`, commit ("data(iw2): steps B and F at F=<F7> -- <verdicts>
(provisional)"), push.

## 8. The re-run rule (UNDEFINED cells)

A cell that is not DEFINED is re-run ONCE with its job line unchanged. Before queueing it, append one line to
`$K/reruns.tsv`: `utc<TAB>tag<TAB>seed<TAB>faculty<TAB>reason (the scorer's fails text)`, and commit that line. Move
the old shard directory aside first (`mv $SH/<tag>/s<seed>/<fac> $SH/<tag>/_attempt1_s<seed>_<fac>`), then queue the
line with the same `--checked` reason. A second failure stays UNDEFINED. An identity arm that fails (rc != 0) is
re-queued once the same way (its output file moved aside to `<arm>.attempt1.json`).

## 9. Dev seed 7 results (Claude, 2026-09-30, local numpy)

**Identity and scope (step D's shape).** Commit 1718c83d0 (the packet code as it will be pinned, determinism guard
in), `PYTHONHASHSEED=0`, one BLAS thread, numpy, the real `/api/brain-chat` handler, one fresh process per arm. Arms:
`research/findings/raw/_iw2/dev_s7/identity/*.json`; verdicts (the registered `--compare`):
`research/findings/raw/_iw2/dev_s7/verdicts/`.

- **D0: IDENTICAL** -- `default` equals `off` on all 16 turns.
- **false-belief-chat: DIFFERS-IN-SCOPE**, turn 4 only: "I don't know about that. My curiosity is piqued -- I haven't
  learned about sally yet ..." (flag off) -> "Sally will look in the basket for the marble." The three narration turns
  are byte-identical (the fold is write-only).
- **transitive-chat: DIFFERS-IN-SCOPE**, turn 9 only: the flag-off path answers a different question ("... the cat
  precedes the ball", i.e. "what does cat precede?") -> "... I believe the cat precedes the river, but I reasoned that
  myself rather than being told it directly." The scope turn "does the cat sleep?" is identical (the closed-class
  guard) and so is the direct pair "does cat precede ball?".
- **neural-selfid: DIFFERS-IN-SCOPE**, turns 14-15 ("what are you?", "who are you?"): "the brain uses the spikes" ->
  "I don't know about that. My curiosity is piqued -- I haven't learned that yet: ..." (the prereg's "known behaviour
  change"; packet NEEDS-CLAUDE 2). Turn 13 "what do you use?" is byte-identical (the substrate route gives the same
  reply the router gave). The ordinary anaphora scope turns (11-12) are identical.
- **Memory.** Peak RSS 11.08-11.37 GB per arm (getrusage). Each segment reset leaves most of the previous brain
  resident (RSS about 7.1 -> 10.5 -> 11.5 GB across the three segments). Hence `mem_gb=11` (the largest job the
  dispatcher hands a 15 GB node; wave 1 declared 11 for a measured 11.75), `mem_ok.sh 11 1`, `memcap.sh 15`.
- **Time.** About 1050-1090 s per arm locally, with other jobs running: first build about 500 s, each segment rebuild
  130-160 s, turns 4-48 s (the derived chase turn 8.9 s with the flag on).
- **Two earlier attempts were set aside** (`dev_s7/_attempt1_2threads/`, `dev_s7/_attempt2_1thread_nohashseed/`): each
  read OUT-OF-SCOPE on turn 5 from a +/-1.1e-16 residue in `curiosity.novelty` that also differs between two fresh
  processes with the SAME flags (a scratch body diff, A vs C). It follows Python's hash-randomized iteration order;
  PYTHONHASHSEED=0 removes it. The runner now refuses to run without it (research/FAILURE_LOG.md 2026-09-30).

**Reply-level rows (step R's shape).** Local, `--repeats 2`, commit badabc703 (the row code has not changed since),
`research/findings/raw/_iw2/dev_s7/{fb,tr,si}/<row>/`:

| row | registered verdict (null clean, lesion reproduced) | intact reply | lesion reply |
|---|---|---|---|
| tom-false-belief | regressed (yes, yes) | Sally will look in the basket for the marble. | Sally will look in the box for the marble. |
| tom-true-belief-ctl | pass (yes, yes) | Sally will look in the box for the marble. | the same |
| reasoning-transitive-chat | regressed (yes, yes) | ... I believe the cat precedes the river, but I reasoned that myself rather than being told it directly. | ... I don't know about that. |
| transitive-reversed-ctl | pass (yes, yes) | ... I don't know about that. | the same |
| neural-selfid | regressed (yes, yes) | the brain uses the spikes | I don't know about that. |
| selfid-nofact-ctl | pass (yes, yes) | I don't know about that. My curiosity is piqued -- I haven't learned about drink yet ... | the same |

All three teach turns of the transitive chain were stored. `tools/iw2_score.py`'s own cell logic, run on these files
with its provenance rules bypassed (they are local sidecars, not git_archive: the real step R reads them as E) and the
dev tree placed in all six seed slots, reads R PASS for each packet (capability 6 of 6, control 6 of 6): the scorer
parses real arm files as intended. These are dev runs and carry no gate weight.

Scratch feasibility runs before the preregs were committed found the two probe defects the transitive prereg records
as deviations (e0..e4 are not words to the acquisition path; a chain of animals is held back by other-repair) and the
selfid class-(b) behaviour change.

## 10. Ranking (recorded in the shared prereg)

Chosen, in rank order: false-belief-chat (16), transitive-chat (16), neural-selfid (11), scored 3a + 2b + c on
(a) likelihood the lesion changes the reply, (b) readiness, (c) relevance to Stage 1/2 exit criteria. Next in line for
wave 3: biased competition's shared inhibitory pool (10; needs a production flag and a non-hollow lesion probe), the
metacog graded 3-band marker (7; needs a production wire). The table with every screened candidate and each
exclusion's reason is the shared prereg's "Selection and ranking".

## 11. Finding and hand-back (after section 7 reads PASS)

Write ONE finding per flipped packet from the template in its HANDOFF, `status: provisional`, on this branch; sync
`GAP_CLOSURE_MISSION.md` CURRENT STATE with one dated line per packet (numbers only, from the verdict JSONs). Commit,
push. Then NEEDS-CLAUDE "iw2: flips <FLIP> validated at F=<F7> (provisional); adversarial review + merge".

## Stop conditions (any one stops the thread)

- A gate blocks for any reason other than the compute gates: STOP, NEEDS-CLAUDE. Never `--no-verify`, never a waiver.
- You would need to edit `sim/`, `webapp/`, `tools/gates/`, a prereg, a scorer or a job line: STOP, NEEDS-CLAUDE.
- A pool job fails twice (rc != 0) outside the re-run rule, or memory kills (rc 137/143) repeat: STOP, NEEDS-CLAUDE.
- A node preflight refuses (rc 3 in `dispatch.log` with `assert_iw2_node`): STOP queueing to that node, NEEDS-CLAUDE.
- AWS spend near the cap (`bash tools/aws_budget.sh status`): STOP queueing to AWS, NEEDS-CLAUDE.

## NEEDS CLAUDE (open; do not act on these)

1. Adversarial review (verify-go) of each flip that passes, then the merge of `research/integration-wave2` to main.
   The flip bar's review leg is Claude's, not the local model's.
2. Ledger placement: false-belief chat is recorded as a sub-rung of `w5-affective-tom` (the only ToM row; that row is
   affective ToM). Decide at merge whether cognitive ToM gets its own row (a ratchet-count change, Check C).
3. neural-selfid default-on changes who answers every "what do you <verb>?" / "what/who are you?" question (the host
   router retires for these). Its step S covers one ordinary anaphora turn ("the cat eats fish" / "what does it eat?");
   the Stage-1 exit gate's reported sub-probe "what does it eat" after a NEW word (C2) is the same shape with an
   unadmitted referent, where 'it' may resolve to the brain itself. Confirm at merge against the Stage-1 C2 anaphora
   sub-probe's arm files (reported, never counted, in that gate).
4. If wave 1 and wave 2 both reach step B within a few days of each other, one combined battery at a branch holding
   both flip sets would halve the ~300-line cost; that is a prereg amendment (Claude's), not the local model's.
5. Wave 1's identity runner (`research/runners/iw1_default_identity.py`, branch research/integration-wave1) does not
   pin PYTHONHASHSEED, and its step-D lines are already queued. Wave 2 measured a hash-order-dependent rounding residue
   (`curiosity.novelty`, +/-1.1e-16) that made two fresh processes with the SAME flags differ on one turn (section 9;
   FAILURE_LOG 2026-09-30). Wave 1's 13-turn script read D0 IDENTICAL on its dev run, so it may not reach that field,
   but if its D0/F2/F3 read DIFFERS on a turn no flag touches, this is the first suspect: rule whether its lines get
   `PYTHONHASHSEED=0` (a re-queue) before they are scored.
6. This build registered itself for CPU lanes C and E in `research/coordination/lane_builds.jsonl` (entries expire
   after 6 h) so its commits could land while those lanes read unserved; the registrations name this branch and say
   the packet's step-R jobs are the lanes' next work. Nothing was queued by Claude.
