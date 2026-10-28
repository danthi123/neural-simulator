# [lbf-gate] RESULT — §2 items 1-4 (prereg section 2), 2026-10-28

Command (unmodified; full 40-char pin per RULING lbf-gate-go-1020c + AMENDMENT 1 — a 9-char
pin reads every cell MISSING):

    tools/b2d_score.py --tag b2d1018 --pin f2885df70cfb99a8715c376bca12ac3c2134ef46 --seeds 42 43 44 100 101 102

Verdict: **INCOMPLETE; STOP-GATE BLOCKED at S0: reasoning-transitive-chat**
Counts: n_defined 72 / n_scored_cells 252 (coverable 40, host-input 1 defined, stop-gate 1 defined); preconditions_failed: None
All 294 gate files are on disk and committed; the UNDEFINED cells are recorded, never counted.

## Item 1 — the S0 core fraction per seed, mean, sd, and the S2 reading

| seed | defined | exercised | load-bearing | S0 fraction | S2 fraction |
|---|---|---|---|---|---|
| 42 | 42 | 40 | 33 | 0.8250 | 0.5366 |
| 43 | 31 | 29 | 25 | 0.8621 | 0.5667 |
| 44 | 0 | 0 | 0 | undefined | undefined |
| 100 | 0 | 0 | 0 | undefined | undefined |
| 101 | 0 | 0 | 0 | undefined | undefined |
| 102 | 0 | 0 | 0 | undefined | undefined |

S0 mean = 0.8435, sd = 0.0185 over the 2 defined seeds (2 of 6); union_n = 33; robust_core_n = 0 (needs all six seeds, so no robust core yet).
S2 mean = 0.5516, sd = 0.0150 (the reading printed beside S0).
Host-input row tom-false-belief is excluded from both (RULING lbf-host-input-rows; STORE-NOT-LOAD-BEARING, lb [1,6]).
Stop-gate reasoning-transitive-chat: seed 42 'not blocked'; seeds 43/44/100/101/102 'blocked:MISSING' -> gate BLOCKED (5 of 6 stop-gate cells UNDEFINED).

## Item 2 — the per-faculty table + every failed/UNDEFINED cell by name

S0 load-bearing [n, denominator] per faculty (n_load_bearing out of defined seeds):

| faculty | LB [n, denom] |
|---|---|
| affect-appraisal-interoceptive | [1, 6] |
| affect-coloring | [2, 6] |
| affect-drives-response | [2, 6] |
| affect-marker-spiking-wta | [1, 5] |
| affective-tom | [1, 6] |
| bg-action-selection | [2, 6] |
| category-inherit | [2, 6] |
| causal-whatif | [1, 6] |
| common-ground-drives | [2, 6] |
| comprehension-learned-animacy-cue | [2, 6] |
| comprehension-learned-verb-selects | [2, 6] |
| comprehension-monitor | [2, 6] |
| confidence-forthcomingness | [2, 6] |
| curiosity-followup | [2, 6] |
| d5-consolidate | [0, 4] |
| d6-hebbian-write | [1, 6] |
| da-gated-encoding | [0, 4] |
| da-mode-drives-response | [2, 6] |
| discourse-register | [2, 6] |
| episodic-memory | [2, 6] |
| gnw-bus | [0, 5] |
| gnw-multistep-deliberation | [2, 6] |
| metacog-monitor | [2, 6] |
| multiref-competition | [1, 6] |
| neural-selfid | [1, 6] |
| noncontradiction-gate | [2, 6] |
| open-ended-generation | [2, 6] |
| pragmatic-implicature | [2, 6] |
| prospective-memory | [2, 6] |
| reconsolidation | [2, 6] |
| self-initiated-utterance | [2, 6] |
| sleep-replay | [0, 4] |
| source-provenance-honesty | [2, 6] |
| spiking-anaphor | [0, 5] |
| surprise-monitor | [2, 6] |
| swap-drives-response | [0, 6] |
| vision-identity-spiking-hmax | [2, 6] |
| wm-binding-advanced | [0, 6] |
| worldmodel-forward | [2, 6] |
| xedge-surprise-worldmodel | [1, 6] |

UNDEFINED cells (n=179), reported as UNDEFINED — never counted load-bearing or hollow. Uniform fail class: 'BRAIN keys' — rule1 lb.json prov env and rule2 arm env carry BRAIN_CHAT_SEED/BRAIN_EPISODIC_STORE_VERIFY/BRAIN_PMEM_FACILITATION/BRAIN_PMEM_OP_STABILIZER/BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE (the old arm env), i.e. these are the PRE-clean-rerun files.

Missing by seed: s43 = 11 cells; s44 = all 42; s100 = all 42; s101 = all 42; s102 = all 42 (the full scored roster: affect-appraisal-interoceptive, affect-coloring, affect-drives-response, affect-marker-spiking-wta, affective-tom, bg-action-selection, category-inherit, causal-whatif, common-ground-drives, comprehension-learned-animacy-cue, comprehension-learned-verb-selects, comprehension-monitor, confidence-forthcomingness, curiosity-followup, d5-consolidate, d6-hebbian-write, da-gated-encoding, da-mode-drives-response, discourse-register, episodic-memory, gnw-bus, gnw-multistep-deliberation, metacog-monitor, multiref-competition, neural-selfid, noncontradiction-gate, open-ended-generation, pragmatic-implicature, prospective-memory, reasoning-transitive-chat, reconsolidation, self-initiated-utterance, sleep-replay, source-provenance-honesty, spiking-anaphor, surprise-monitor, swap-drives-response, tom-false-belief, vision-identity-spiking-hmax, wm-binding-advanced, worldmodel-forward, xedge-surprise-worldmodel).

s43's 11 UNDEFINED by name: affect-appraisal-interoceptive, affective-tom, causal-whatif, d6-hebbian-write, gnw-bus, multiref-competition, neural-selfid, reasoning-transitive-chat, spiking-anaphor, tom-false-belief, xedge-surprise-worldmodel.

No cell is marked failed (fail_cells = []). The clean re-runs that replace these files are still queued: 142 f2885df70 lines in research/queue/pool.queue at 12:56 (batch-1024(4) requeue, env byte-symmetric to the dev list).

## Item 3 — the host of every result (tools/where_ran.py, 294 files)

| host | class | files |
|---|---|---|
| dant123-wk | workstation | 95 |
| node-dl5d243 | pool | 69 |
| node-mxl1462drl | pool | 67 |
| node-dl4g243 | pool | 63 |

Every file records git_sha f2885df70cfb99a8715c376bca12ac3c2134ef46, git_dirty false; validity does not rest on one node.

## Item 4 — the dev read beside it, labelled as dev

Seeds 7+8 (dev): 33 of 40 = 0.825, sd 0, quoted from NEEDS [lbf-head] RESULT 2026-10-19 04:22 <!--derived-->, labelled **dev**. Gate seed 42 (S0 = 33/40 = 0.825) reproduces it exactly. The dev number is never cited as the result; the gate result above is INCOMPLETE at 2 of 6 seeds.
