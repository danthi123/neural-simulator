# WORKLOG — Stage-2 exit gate (branch `research/stage2-exit-gate`)

Append-only. One line per step, newest last. The local model appends here; Claude reads this at inspection.

- 2026-09-30 09:55 Claude: worktree created from main (f5e3732fe); record checked (`before_you_build.sh` + five RAG
  queries, one per criterion; the Stage-1 packet's prereg, runner, HANDOFF and dev shards read in full); the five
  criteria operationalized against the live handler's code (`webapp/server.py`, `brain_chat_tui.py`,
  `curiosity_production_organ.py`, `compositional_chain_route.py`, `gnw_multistep_deliberation.py`,
  `spiking_mouth_recall_prod.py`, `affect_production_organ.py`, `self_initiated_production_organ.py`).
- 2026-09-30 10:00 Claude: runner + tests drafted; selftest OK (18 checks), 12 unit tests pass (no brain build).
- 2026-09-30 10:03 Claude: dev smoke (seed 7) started as a sequential chain under `mem_ok.sh 7 4` + `memcap.sh 9`:
  the workstation is shared with the B2d mouth-arm GPU job (10 GB RSS, `mem_ok.sh 10 4` is that job's own margin)
  and a sibling agent's feasibility probe (6 GB), so `mem_ok.sh 7` at the default 8 GB margin refuses; one brain at
  a time.
- 2026-09-30 10:35 Claude: host-parser pre-check of every protocol turn WITHOUT a brain build (the cheap part of
  the instrument): topic extraction reads eel/yak/ibex/toad/zorblat/deer/owl as intended (`heron's` on the
  possessive probe, irrelevant); `how do you feel` matches the feel query; `detect_chase` strips the marker on
  the three chase probes; the possessive parser reads (heron, prey, eat/chase); the transitive regex matches
  `does crow chase moth`; `say something` is the self-initiation trigger; `tell me more` is a follow-up; no
  situation sentence names a third party (W5 silent) and none is acquired as a fact (`extract_polar_assertion`
  -> None for all eight, intact and swapped); the acquisition extractor reads the passive as (mole, hunt, owl)
  and the prepositional as (deer, sleep, meadow), i.e. exactly the prereg's predicted C1 mis-parse and
  preposition loss. Wart noted, not changed: the swapped anger frame reads "scared and afraid and scared" (three
  anger words mapped onto two fear words); the affect words are what the appraisal reads.
- 2026-09-30 10:35 Claude: memory still refused (two ~10 GB brain jobs from the GPU queue's B2d row and the
  sibling probe); the chain waits on `mem_ok.sh 7 4` in a loop; the prereg's section 11 declares that the dev arms
  run after its commit and their lines land here.
- 2026-09-30 10:50 Claude: SESSION CUT SHORT by the orchestrator before any dev arm could get memory (the chain
  never started a brain: `mem_ok.sh 7 4` refused throughout). Committed in this order: the prereg alone (its
  `artifacts:` also cites the two existing record artifacts its C1/C3 predictions rest on, so doc-type's
  existing-path rule holds before the dev shards exist), then runner + tests + this packet. The dev-seed-7 smoke
  is NOT done: it is the first thing to run (HANDOFF section 2 lists the arms; run them one at a time under
  `mem_ok.sh 7` + `memcap.sh 9` once the B2d mouth-arm GPU job has finished) and its lines go here. The pin files
  are written after that smoke. NEEDS-CLAUDE: read the first dev shards against prereg section 4 before any gate
  seed (an instrument defect is a section-12 amendment in its own commit).
- 2026-09-30 11:00 Claude (adversarial review of 8c934312a): (1) CRITICAL, fixed `ed956d08b`: `--shard` spawned
  `--worker` without `--allow-gate-seed` and the gate-seed guard applies to `--worker`, so all 186 gate jobs would
  have died rc=2 (reproduced: `--shard --seed 42 --allow-gate-seed ...` -> "worker FAILED rc=2"). The Stage-1 runner
  (branch research/stage1-exit-gate, 204 lines already in the pool queue) has the identical code: raised to the
  orchestrator, not fixed here. (2) fixed: validity checked only on the intact pair; an errored scored turn in a lesion
  arm read "no effect" -> FAIL "not load-bearing"; organ errors unguarded (B2b); now UNDEFINED over every counted arm
  (prereg amendment A1, `7e3b2972c`). (3) fixed: the chase trace is never attached to the response (it lives on
  `chat._last_gnw_multistep`); the worker captures it. (4) fixed: the shipped LTM default is `wikidata_100k`, not
  `wikidata_core_15k` (HANDOFF pool check + test; the novel words are absent from both bundles, checked). (5) the
  HANDOFF section 0 claimed a done dev smoke and a GATE_SHA that did not exist: corrected. Checks run: selftest OK,
  20 tests pass (7 new ones fail against 8c934312a), 186 emitted lines pass `queue_job_shape_check.sh` and the
  pool_queue gate-seed guard (prereg present at the pinned revision), no sim/ or webapp/ file touched by the branch.
  Local RAM refused the dev smoke all morning (`mem_ok.sh 7 4`: avail -2 G), so the dev arms go to the pool FRONT=1.
- 2026-09-30 10:40 Claude: provisioned `6b9cb47f7` on pool41 + pool42 (--isolated; LTM wikidata_100k + corpus OK on
  both) and queued the 15 dev-seed-7 arms with `FRONT=1` (`mem_gb=7 max_age_h=24`, dev seed only, no gate flag).
  They did not dispatch: each node carries six `mem_gb=2` long-delay-credit jobs (committed 12 GB of 15, so the
  dispatcher's lifetime budget is 1 GB) and first-fit hands every freed slot to the next of the 97 queued `mem_gb=2`
  lines, so a 7 GB line (these, and the 204 Stage-1 gate lines) waits until the small lines drain (hours). Local RAM
  kept refusing (`mem_ok.sh 7`: avail 8 G after 19 G committed-unused). The shards land in
  `$P/research/findings/raw/_stage2_exit/dev/s7/` when they run; HANDOFF section 2 has the reader. GATE_SHA stays
  unpinned until they are read. The runner at `8b49dc1c1` is byte-identical to `6b9cb47f7` (tests/HANDOFF only).
- 2026-09-30 11:07 Claude: local RAM freed; dev arms run on the workstation at the final runner (one brain at a
  time, `mem_ok.sh 7 4` + `memcap.sh 9`). `inner__intact_a.json`: 10 turns, no errored turn, no organ error, source
  `tiny-demo +LTM`, hash ok, corpus, builtin_facts 6, peak RSS 6295 MB, ltm `wikidata_100k`, wall 793 s. The chase
  works: lynx -> mite (resolved_hops 3, cycles 4, the ignition trace n_ignited 1,1,1,0 -> COMMIT, captured by the
  worker's `internal`), vole -> mite (2 hops), mite abstains, single hop vole. PRIVATE FAILS the grounding clause:
  "Something's been on my mind — bird see seed." -- the self-initiation organ speaks from its OWN fixed 4-fact
  demo lexicon (`_self_initiated_utterance_derisk.AGENTS/VERBS/PATIENTS`: dog chase ball, cat eat worm, bird see
  seed, fish hear bug), never from the brain's facts or anything taught, so `private_voices_stored_fact` is False
  by the prereg's own definition (build-time list or taught). Prereg 4.5 predicted PASS from the 2026-08-18
  finding's example; per section 11 a dev capability read changes no prediction. NEEDS-CLAUDE #15. Also noted:
  the chase terminal is spoken as a plain assertion ("the lynx chases the mite", `derived` false) although it was
  never told -- a derived conclusion not framed as derived (production honesty, not this packet's instrument).
- 2026-09-30 11:22 Claude: `ask__intact_a.json`: 19 turns, valid (no errors, peak RSS 6524 MB, wall 913 s). Taught
  topics read LOW graded novelty (yak 2e-16, moose 0.0) and do not ask: the builder's question is answered yes.
  BUT the production DA/engagement crave-threshold (`BRAIN_CURIOSITY_DA`, default-ON since 2026-08-21) rises with
  every low-engagement abstain turn (DA 0.55 -> 0.26 -> 0.06 -> 0.046; eff threshold 17.7 -> 29.7 -> 57.4 -> 59.8 Hz
  against a want of ~38 Hz), so after two asks the brain stops asking everywhere: eel asked, crab asked, newt NOT
  (novelty 0.96), ibex_1 NOT, toad_1 NOT. Scores: ask_rate_gap 2/3, moat 0 (discriminates True at the bar edge),
  lp_learnable False, lp_unlearnable False (toad_1 never asked). Also `the goat eats the hay` was NOT acquired
  ("my role-binding didn't resolve the PATIENT"), so `m_goat` is a de facto novel gap (novelty 0.976) scored as a
  taught moat. Prereg 4.3's premise "the 4th probe reads exactly as the 1st" is false at HEAD: C3's lp clauses and
  its permuted arm (gaps at positions 2/4/6 instead of 1/3/5) are order-confounded by DA depletion. That is a
  protocol-design question (turns are frozen), NEEDS-CLAUDE #16; GATE_SHA must not be pinned before it is ruled on.
- 2026-09-30 11:43 Claude: `conv__intact_a.json`: 11 turns, valid (peak RSS 6303 MB, wall 876 s). fox_chase PASS ("As
  for it — the fox chases the hare"); the passive is stored as (mole, hunt, owl) exactly as prereg 4.1 predicted
  (owl_hunt abstains, the mole "hunts the owl"); the teach echo "the deer sleeps the meadow" shows the dropped
  preposition. Two reads differ from the prereg's expectation: `tell me more` does NOT elaborate ("I don't have
  anything more to add on that", abstained), and `where does the deer sleep` does NOT route to (deer, sleep): it
  abstains, so `deer_sleep` fails at the question parse before any frame is rendered (the prepositional sub-bar then
  measures the wh-parse, not the frame inventory; the teach echo is the frame evidence). Both still FAIL C1 as
  predicted; which gap the FAIL names is NEEDS-CLAUDE #17 (probe wording is a frozen turn).
- 2026-09-30 12:04 Claude: `infer__intact_a.json`: 10 turns, valid (peak RSS 6291 MB, wall 1246 s: longer than the
  prereg's ~20 min planning figure; `p_yesno` 295 s, `c_heron_hunt` 203 s). C2 reads PASS on every clause at dev 7:
  "I derived this from: heron hunt frog; frog eat gnat. the frog eats the gnat" (derived true), both premises, the
  moat abstains. Production finding (not this instrument): the bare yes/no `does crow chase moth` is ACQUIRED as an
  assertion and reconsolidates memory ("my mismatch monitor fired: I'd learned that crow chase wren. Updated --
  I'd stored ..."), and the next turn volunteers "the crow chases the moth" -- a question rewrote a stored fact.
  It comes after the scored C2 turns and the dog control still reads cat, so no C2 score moves. Open item for the
  owner / next inspection (moat: a question must never be stored).
- 2026-09-30 12:16 Claude: `emo__intact_a.json`: 13 turns, valid, wall 557 s, PEAK RSS 8101 MB (the highest group;
  the gate lines' `mem_gb=7` / memcap 9 left ~1 GB): fixed in `8691e84c1` (`mem_gb=9`, memcap 10) + prereg
  amendment A2 (`24dac499e`, lane only). C4 reads capability absent as predicted (no category word in any feel
  read); signs joy +0.038, fear -0.023 (within the 0.03 neutral tolerance, so "neutral"), anger -0.052, sadness
  -0.052: `sign_correct_all` False (prereg predicted 4/4 signs), `n_distinct_2d` 3. The exposure turns abstain and
  ask about "win"/"wolf"/"someone"/"alone" (curiosity on the situation's topic word); no category is echoed.
- 2026-09-30 12:30 Claude: committed the five `intact_a` dev shards (workstation, final worker). GATE_SHA NOT pinned:
  NEEDS-CLAUDE #15 (C5 private store), #16 (C3 DA-order confound + failed acquisition), #17 (C1 probes) need a ruling
  first. Remaining dev arms: `ask/intact_b`, `inner/intact_b`, `emo/lesion_affect` running locally (memcap 10); the
  rest queued on the pool (FRONT=1, `max_age_h=24`; they may starve behind `mem_gb=2` lines and expire).
- 2026-09-30 13:01 Claude: the nulls are clean at dev 7: `ask__intact_b` and `inner__intact_b` are byte-identical to
  their `intact_a` in EVERY response field (not only the primary ones) and in the captured chase trace, across two
  independent processes (the gate's null precondition should hold). `emo__lesion_affect`: valid, peak RSS 8104 MB
  (under memcap 10), every feel read clamped to "+0.000 neutral", so all four scored turns differ from intact
  (lesion effect present, no organ error). Committed the three shards. Still on the pool (FRONT=1, may starve):
  the other lesion arms, `infer/transitive_chat`, and duplicate intact_a arms (a workstation-vs-pool check).
- 2026-09-30 13:05 Claude: removed from `research/queue/pool.queue` (under its lock) the seven pool dev lines that the
  local runs made redundant; kept `inner/intact_a` as a workstation-vs-pool determinism check plus the seven
  lesion/informational arms no dev run has covered (`ask/lesion_graded`, `ask/lesion_curiosity`,
  `inner/lesion_workspace`, `inner/lesion_selfinit`, `conv/lesion_mouth`, `infer/lesion_chain_route`,
  `infer/transitive_chat`). They carry `max_age_h=24`; if the `mem_gb=2` backlog holds them past that they expire.

## 2026-10-06 (Hermes) RULING stage2-dev-smoke: fresh dev-smoke lines at the synced tip

- Local branch research/stage2-exit-gate created from gitea/research/stage2-exit-gate (eddb1a071; no fetch,
  RULING private-repos); agent_sync_branch merged main 519f14e24 in -> f2aaabc8a, no conflict, gates passed.
- before_you_build ran rc=0 (corpus check logged in .git/corpus_checks_shared.jsonl; receipt cites
  2026-10-06).
- Provisioned f2aaabc8a9272e6e2e094b9e0f5e1c15c4b7d526 on pool41/pool42/pool40/wspool: state ok, rc=0,
  markers after=true, venv deps include httpx.
- Withdrew the 8 expired 6b9cb47f7 lines -> research/queue/withdrawn.log (per the ruling: re-queue, but
  not the expired lines), then queued 11 fresh seed-7 lines at the tip, each shape-checked before
  add --checked: the 7 lesion/informational arms (ask/lesion_graded, inner/lesion_workspace,
  conv/lesion_mouth, infer/lesion_chain_route, inner/lesion_selfinit, ask/lesion_curiosity,
  infer/transitive_chat) plus intact_a of ask, inner, conv and infer. mem_ok 9 1 / memcap 10, out
  research/findings/raw/_stage2_exit/dev/s7/, max_age_h=168. The dispatcher consumed 3 during the add
  (ask/lesion_graded, conv/lesion_mouth, inner/lesion_workspace running); 8 remained queued.
- Canary: pool_canary.sh run auto on ask/intact_a launched stage2_exit_gate_f2aaabc8a_wspool_20261006T151306
  (alive: bridge initialized 144 neurons); the local `run` process was cut by the harness timeout while it
  polled, so the verdict + pool_canaries.tsv record are taken manually next wake from the node's
  ~/derisk-pool/canary/<id>.status (known: launch ssh rc=3 false negatives, NEEDS pool-canary).
- T14 starts when 8 files land in _stage2_exit/dev/s7/; INFO NEEDS on landing (ruling).
- 2026-10-06 16:35: RULING stage2-bar-sanity executed: `git merge --ff-only
  claude-review/stage-bar-sanity-s2` e125827f2 -> b7f8c1b41 (clean tree, was a true FF; adds
  C1-C5 + stage bar-sanity specs/records, all OK n=1000), then `agent_sync_branch.sh
  research/stage2-exit-gate` merged main @ f4b23f93f as b913e0c78 (rc=0, gates passed).
  GATE_SHA, PREREG_SHA and the 11-03..11-10 window unchanged. All 11 seed-7 dev shards have
  landed (11 results in _stage2_exit/dev/s7/).

## 2026-10-06 (Claude) AMENDMENT-A3: C4/C5 shortcut hardening (branch claude-review/stage2-c4c5)

- Outcome-blind (no gate data). C4 lesion clause now needs the named category to change on all four feel turns; C5
  permuted-tracks clause now needs the idle fact to differ across intact/permuted. Scorer + tests + bar-sanity specs and
  `_A3` records (all six OK) committed; prereg byte-identical (PREREG_SHA unchanged); GATE_SHA still unpinned and must
  contain AMENDMENT-A3.md. No dev line is re-run; NEEDS-CLAUDE #15-#17 are untouched by this amendment.
- 2026-10-06 16:52: RULING stage2-c4c5-ff executed: FF claude-review/stage2-c4c5 -> bd5c95bb4+
  then agent_sync_branch merged main (tip `43763e637`); pytest test_stage2_exit_gate + runner
  selftest run above; GATE_SHA pinned at `43763e637` (revision contains AMENDMENT-A3.md;
  checked). PREREG_SHA stays 24dac499e. Only _A3 bar-sanity records are cited for Stage 2.

- 2026-10-07 (Hermes, driver) — RULING stage2-c4c5-dev re-score DONE at the synced tip
  (f87b8c1ab = branch + main @ c0d675504; A3 code; GATE_SHA 43763e637 untouched): the 15
  existing seed-7 dev shards (8 committed + the 7 pool-only cells copied in from main's
  untracked f2aaabc8a landing, 14 files cp -a) re-scored under the amended aggregate ->
  research/findings/raw/_stage2_exit/dev/s7/aggregate.json. Verdict: UNDEFINED (INCOMPLETE) —
  29 primary shards missing (dev battery = subset), intact_b absent for conv/infer/emo, so
  C1-C5 all INCOMPLETE with n_defined=0 and the stage-level reads None; preconditions
  unmet = honest scorer output on a partial battery, no dev line re-run (per ruling).
  Integrity check before scoring: the 4 cells existing in both workstation (committed) and
  pool f2aaabc8a (main untracked) runs — ask/conv/infer/inner intact_a — are LOGIC-IDENTICAL
  after excluding timing keys; only <=1-ULP curiosity-novelty float noise (0.0 vs +-1.1e-16,
  0.9532050931328421 vs ...422) and elapsed/RSS differences. Committed workstation copies
  govern those 4 names; main's 11 untracked files stay put for the future merge.

- 2026-10-09 09:05 EDT — T14 Start met (11 files in dev/s7, landed Oct 6); [stage2-dev] INFO filed on main.
