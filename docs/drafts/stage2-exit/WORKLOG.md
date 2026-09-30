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
