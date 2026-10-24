# FIX_NOTES: #271 stage 1, round 2, fix round (Claude, 2026-10-22)

Scope. The final verification returned NEEDS-CHANGES from both readers: the refuter (`workflow_result.json` final[0], 14
findings: 2 BLOCKING, 7 MAJOR, 5 minor) and the completeness critic (final[1], 16 findings: 1 BLOCKING, 8 MAJOR, 7
minor). This round edited `round2/DESIGN.md` and `research/coordination/deixis-271-s1-PACKET.md` in place. Everything it ran lives in
`docs/drafts/deixis/s1-scratch/fix/`. Nothing in any repo was written, and the code was read at the workstation checkout. IDs below:
R = refuter, C = critic, numbered as in `workflow_result.json`.

## Evidence produced this round (all run; `docs/drafts/deixis/s1-scratch/fix/`)

| File | What it shows |
|---|---|
| `wscale_curve.py` / `.out` / `.json`, `fI_probe.py` | The engine's own transfer curve: numpy backend, heterogeneity on, 10 seeds, a 12-neuron target, a 40-step token. A route fires its pool at w ~ 6-8, and at w <= 1 it never fires. The f-I curve: about 150 pA to fire inside a token. |
| `cov_decay_replica2.py`, `replica2_baseline.out` | The judged replica, copied unchanged. It reproduces the refuter's numbers: COV passes every weight bar on 6/15 at lam x tau = 4, 6 and 9. |
| `replica3.py` | The revised replica: the GIVEN relation and its five forms, the S0 route bar, within-token bars, six learners (cov, covpre, hebb, hebbwta, bcm), and four nulls. |
| `final_v3.out` | Every arm on the same 8-cell grid, seeds 1-15, 4800 utterances. |
| `centres.out` | The graded centres: COV-PRE 6, COV 6, RW-HEBB 3, +WTA 3, BCM 1. |
| `final_v3b.out` / `.json` | At those centres: rates on 60 seeds, nulls, dose curves and convergence. |
| `s6_check_lt6.out` | The revised S6 at the COV-PRE centre: 15/15. |
| `rate60.out` | The exposure-length check. |
| `project_s1hold_v2.py` | The census projection under the R1-R3 exposure: 59 of 126 items held, m1 28 -> 3 (`moi`). |
| `b3_item_check.out` | Which honesty items flag-off B3 actually stores. |
| `hd_check.py` / `.out` | A read-only call of `tools/gates/host_decision_declared.py::judge` on both files at their planned repo paths. Problems: NONE for each. Exactly one `host-decision:` line per file, before the first `##`, within 60 lines. `decides` / `allowed-by` / `replaced-by` / `cap` are identical in both files, so they form one scaffold group. |

## BLOCKING

**R0, the step-0 GO rule was unreachable: FIXED (DESIGN 3.2-3.5, 13; PACKET step 0a/0b, Measures, Appendix A).**

- **Cause: two defects, separated by running the replica.**
  - *The bar's defect.* S2b compared every name with a class-maximum null set by the most-exposed name. Against each name's own dose-matched null, the same COV weights pass S2b on 15/15 (`final_v3.out`, columns S2b vs S2b_classmax).
  - *The learner's defect.* Under a uniform leak, `w* = (lam/delta) P(x) (E[y|x] - ybar)`. Every weight therefore scales with its form's frequency. The weakest must-fire routes are the rarest forms (POSS-G, ORD1, V3's names: 0.05-0.14). The strongest wrong routes are frequent forms' chance covariances (0.09-0.12). That ratio is <= 1 at all eight cells, but the engine needs 2 (theta_fire / theta_silent = 8 / 4). No lam/delta and no w_max can fix a ratio.
- **What changed: the bars.** All bars are now physical or within-token, and no threshold comes from any arm:
  - S0 is absolute, taken from the engine's transfer curve.
  - S2b is within-token, with no dose floor.
  - S3 and S4 use max(theta_silent, round 1's fraction).
  - S6 now checks that routes do not decrease, and that heavily dosed routes stay within 20%.
  - S8 and T-carry moved to step 1a.
- **What changed: a sixth learner arm, COV-PRE.** Its leak is presynaptically gated and weight-proportional: `dw = lam x (w_max (y - ybar) - kap w)`, with the outstar fixed point `w* = w_max (E[y|x] - ybar) / kap`, which does not depend on the dose. The biology: Grossberg 1974; Dudek & Bear 1992; van Rossum, Bi & Turrigiano 2000.
- **What changed: a fair procedure for every arm.**
  - One grid and one centre rule for all arms.
  - Each arm runs at its centre and the next value up.
  - One preregistered audit applies to every pass.
- **What changed: GO at 13/15, justified binomially.**
  - At per-seed rate q = 0.9: P(>= 13/15) = 0.82, but P(>= 14/15) = 0.55.
  - At q = 0.5: P(>= 13/15) = 0.004.
  - Nulls must pass on <= 1/15.
- **What changed: a written STOP rule.** If the 0a record fails, 0b is not queued.
- **Replica result.**
  - COV-PRE: 15/15 at its centre, 59/60 on seeds 1-60. The one-sided 95% lower bound is 0.92, where P(>= 13/15) = 0.90. Its window is lam x tau 4-9 at >= 14/15.
  - Every rival arm: 0/15 at all eight cells, and 0/60 at its centre.
  - All four nulls on COV-PRE: 0/15.
- **Deviation from the literal request, stated.** The refuter asked to show "COV passes >= 14/15". Under the per-name null it passes every round-2 weight bar. Under the S0 bar that BLOCKING R1 requires, it passes 0/15 at every cell. Lowering S0 would mean tuning a bar to one arm, so the learner was fixed instead. COV stays in the census as the fifth arm: if the spiking census shows it passing, it wins first.

**R1, plastic weights in [0, 1] could never fire a pool: FIXED (DESIGN 2.2, 2.6, 3.3, 3.5; PACKET Build).**

- **The engine's arithmetic, with file and line:**
  - A spike adds `g_e += w x 0.05` (`sim/bridge.py:9258-9261`; `sim/config.py:871`).
  - The conductance decays by exp(-1/5) per step (`bridge.py:2910`; `config.py:394`), a gain of 5.52.
  - The current is `I = g_e (0 - v)` (`sim/kernels.py:1115`; `config.py:392`).
  - Cells are Izhikevich-2007 RS (`config.py:20-23`; `kernels.py:76`).
  - A 1500-pA token fires 0.19 spikes per ms per neuron, so g_ss = 0.052 x NT x w nS. The cell needs about 150 pA, so w_fire ~ 52 / NT.
- **Measured curve, NT = 12:** w = 4 gives at most one target spike; w = 8 fires every target neuron on every seed.
- **Chosen setting:**
  - NT = 12; W_MAX = 20 (the word-role organ's, `word_role_parser_organ.py:127`); w_init = 0.
  - theta_fire = 8 and theta_silent = 4, both taken from the curve, never from a learning arm.
  - The replica maps to the engine as w_engine = 20 x w.
  - The fixed downstream weights stay at the engine's working values (W_RG 40-60, `:174-183`).
  - The [0, 20] clip touches only gain > 0 plastic synapses (`bridge.py:10390-10411`).
- **Checks added:** the saturated-route check (routes set by hand to 20, 4 and 0) and a re-measurement on the organ's own bridge. Both run at the 0b smoke, before any census line. S7 now reads [0, 20].

**C0, anaphors and pointing words outside the lexicon were stored as names: FIXED (DESIGN 1, 2.1, 2.2, 4.1-4.3, 7, 8.1-8.3, 11, 16; PACKET Why, Build, Measures, Appendix A/B).**

- **A third learned relation, GIVEN, added in stage 1:**
  - The toy has nonce forms ANA-1, DEM-1, POSS-G, ORD-1 and ORD-2. Its afferent is pointing (a world percept) plus the previous mention. The previous mention is a declared scaffold, replaced by the stage-2 mention trace.
  - Production renders no givenness afferent. A GIVEN form therefore fires its learned route into RF_GIV, and the turn is HELD.
- **English exposure built only from rules:**
  - R1: the reference-grammar paradigm of all persons, plus the demonstratives and the former / the latter.
  - R2: the reduced and regional forms listed by Wales 1996 ('em, ya, ye, yer, yous, youse, thee, thou, thy, thine).
  - R3: uniform chat-spelling rules (apostrophe deletion, u, ur, urself).
  - ~~A CI test proves that no form was added by hand. No form comes from a list, and no census file is read.~~
    **Corrected in round 3:** R1-R3 are a declared teacher scaffold chosen AFTER the census probe set was seen (R2 as
    written here matched the census slang residue). The CI rule test only checks that the generator's output equals what
    its rules produce; it cannot show that no form was chosen with the census in view. The anti-list evidence is the
    nonce arms, the K = 0 fixture and the synapse transplant on both readings. R2 is now the pinned source's complete
    list (Round 3, minor 1).
- **Bars:**
  - T14: one given entity is bound.
  - T15: two-candidate items are HELD 36/36.
  - T16: a GIVEN form with no afferent is HELD 36/36.
  - The `dx_giv` lesion.
  - A single- and two-turn former/latter/demonstrative arm ('the former likes jazz', 'this one likes jazz', 'kofi and dalia study at dunmere' then 'the latter married fenna') is HELD 100% on 6/6, with a made-up-marker twin at >= 29/32.
- **Oracle v2:** filed beside v1, relabelling p009 'latter' and p010 'that' as ANTECEDENT.
- **Projection:** 'the latter', 'the former', 'that guy', 'hes', 'em' and every other listed form are held. The flag-on default path stores no exposed pointing word as a name.
- **What is not covered:** `moi`, which no rule produces (3 census items). It is a counted residue with a method (teacher correction through `ai_teacher.py`, then learning from use under the stage-3 plasticity gate), a stage (3) and a date (method packet by 12-05).
- **Honesty about scope:** resolving these forms (moved ≠ answered) is #271 stage 2: packet 11-28, gate 12-15..12-18. Required-changes table 16 now says ANSWERED or MOVED for each item.

## MAJOR

| ID | Finding | Status | What changed (section) |
|---|---|---|---|
| R2 | Req. change 12 and the interim of 2 not answered; "100% unstored" by absence | FIXED | The GIVEN exposure plus T15/T16 and the 100% held former/latter arm (C0 above). 16.2 now reads "Interim ANSWERED, route MOVED"; 16.12 is ANSWERED with real bars; the absence claims are deleted (1, 16) |
| R3 | Rule census unfair: tuned on COV only; audit only for existing rules | FIXED | Every arm runs on the replica with one tuning procedure and one centre rule; the audit criteria are preregistered and applied to every pass, COV and COV-PRE included; no threshold comes from an arm, and S0 is re-measured only before any learned weight is seen (3.4, 3.5) |
| R4 | Step 0 under-costed; S8/T-carry need step 1a's build | FIXED | S8 and T-carry moved to step 1a. Recount: about 235 runs, 4800 utterances each, 20-105 CPU-h. Trim rule (the rivals drop to their centre cell) and a slip rule for the organ-gate window (3.4, 13; PACKET step 0b) |
| R5 | sim/ spec misdescribes the engine | FIXED | ybar is gated on the plasticity-open state (`hebbian_cov_ybar_gate`). The runner arm applies the same arithmetic. The co-firing condition (`bridge.py:10236`), the steps per occurrence and the trace value y_on are measured at the smoke, which sets every arm's per-occurrence rate and per-utterance decay. Time constants are counted in occurrences. S6 is redefined (2/3 vs end). Parity is checked on a replayed raster (3.3, 3.5). The requested "set the exposure length from the measurement" is replaced: 4800 utterances is fixed by the rarest form's dose (>= 83 occurrences of POSS-G, one route time constant 67 at the centre), and the measurement sets the per-step rates instead |
| R6 | Lesion and null engagement unprovable | FIXED | Gate values are read back; per-token P/RF/E spike counts; K, the permutation map, the shuffle seed and the stream sha; an arm missing any of these is UNDEFINED; the both-route bar counts only with that record (8; PACKET Measures) |
| R7 | Answer-rate battery cannot run | FIXED | One `intact_<probe>.json` per probe, the probe as the last answered turn, in its own faculty dir `deixis-hold-ar` (`paired` reads only `^intact`, and the row's driving pairs abstain by design). One tag pair per request shape, with the exact command. FLIP-HURTS by name needs a ruling (8.5; PACKET) |
| R8 | No where-ran record | FIXED | `tools/where_ran.py` summary in every RESULT; source, `corpus_present`, LTM sha, cache key and hit, and host class recorded; a cross-host-class comparison is UNDEFINED (8; PACKET) |
| C1 | Residues with no stage (no-defer) | FIXED | A residue table with method, stage and date for each; the premise-2 projection (every residue name except `moi` held); honest limits stated: no existing store is cleaned, and holding is not voicing (11) |
| C2 | W5 false-belief and W6 affective-ToM are unlisted host English writers | PARTLY | Both are added as census v2 writers, rows in the routing table (a held turn neither starts nor advances a scenario; W6 untouched and reported), and section 11 entries with their retiring stages (W6: S1-bind; W5: Stage-3 C3) (5.1, 5.2, 8.2, 11). NOT done: their stage-0 COUNTS. Measuring them needs the census-v2 extension run at the merge-base, which is code and a run in the repo, and this round may not write there. The counts are taken at the merge-base before any flag-on arm, as step 3's first census line |
| C3 | Reusable organs ignored without reason | FIXED | Each has a stated disposition. EMERGE-62 is #260's input: as the candidate gate it would strip content words, per its own real-corpus precision drop. The extended anaphor detector is an installed vocabulary, which wrc-frame-go-1020b forbids as decider here. Teacher correction is the route for spellings no rule produces (11) |
| C4 | Homographs ('brain', 'mine', 'us') | FIXED (cost route) | A toy homograph arm (N_SELF's form as an object token, 20%) and measure 6b, a frozen set of homographs flag-off stores ('a dog has a brain', 'kofi owns a mine', 'kofi and fenna were friends'). The cost is preregistered and a ruling requested before the gate; role context, the real fix, is #260 (4.1, 8.1, 8.2, 11) |
| C5 | Quoted / reported speech and identity claims | FIXED | H6 items are built from statements flag-off stores and HELD 100% on 6/6. S1-bind gets a HELD-REPORT class and is not flipped before stage 2's stacked speaker context passes its own gate. S1-bind bars ensure identity claims never rebind P* or reach a self-fact (6.8, 8.4, 10). Token sharing (two people typing in one conversation under one minted token) is declared in 10 as S1-bind's residue, with a counting arm and a method (a per-person token from an authenticated client login, a world afferent) |
| C6 | Production fail-open paths | FIXED | With the flag on, a missing organ refuses the flag at startup; a raise or text mismatch HOLDs and is counted. The TUI local fallback calls `read_turn`. The organ reads every turn `_discourse_turn` would echo, so the W3 bar is reachable. `test_deixis_fail_closed.py` (2.4, 5.1, 18) |
| C7 | S1-bind voicing has no production efference | FIXED (by scope) | No voicing in S1-bind until #272's reply-side spiking word-commit drives cd. That packet carries the production X1 bar (0/N first-person renderings on 6/6) and its cd lesion (10) |
| C8 | Calendar collisions | FIXED | Step 3 is gated on the step-2 verdict (no EN fixture if the organ gate fails). S1-bind development starts after the step-4 ruling. Queue placement is a NEEDS line instead of citing parallel-builds-1021b. Step 0b is scoped (no commit stage) and costed. Stage 2 is dated (packet 11-28, gate 12-15..12-18), and the S1-hold flip target is 11-24..11-28 (13) |

## Minor (all fixed)

- R9: the header sits before the first `##` and `decides:` now names the per-token latch reset. `judge()` returns no problems for either file.
- R10 / C9: the held turn on the rich default path short-circuits `_direct_fact` before the DIRECT-LTM-TOPIC fallback, which is a declared skip. The curiosity follow-up runs as on any abstention and sits inside the H bars. H5 and E3 run on the rich default shape (5.2, 8.4).
- R11 / C10: honesty items B3 never stores are dropped ('brain is conscious', the 'a wolf attacks you' short form). They are replaced by stored items ('the brain feels joy', 'a wolf bites your arm', 'the storm scares your cat'), checked in `b3_item_check.out` (6, 8.4).
- R12: the B3 contradiction reject is KEPT, because the hold applies only to a turn the flag-off path would store (2.4, 5.2).
- R13: the made-up-marker controls (K = 0, transplant, both-route lesion) run on both readings. "Held out" is dropped as evidence. S8 and GO are both at 13/15 (8.1, 8.3).
- C11: H1-H6 run under `BRAIN_ONE_BRAIN=1` with two sessions. H5-legacy counts the legacy store, and its quarantine is dated with S1-bind (5.3, 8.4).
- C12: the withheld VOICE, ADDR and GIV afferents are declared in `decides:`, with their replacements. Every production pointer hold is stated to be referential failure under a silenced afferent (2.1, 12).
- C13: co-commit risk cited (cs-race-f1d). Its rate is reported at S8 (step 1a) and at the organ gate, and counted as a miss in T1-T5 and T14 (2.4, 8.1, 14).
- C14: m5 is renamed for what it counts. The cross-speaker B3 measure gets a merge-base baseline, the bar "no increase" and its stage, S1-bind (8.2).
- C15: the organ cache key adds the blob shas of `sim/bridge.py`, `sim/config.py`, `sim/kernels.py` and the organ module. Fresh exposure is re-proven equal to the cache at each gate (5.1).

## Not fixed, or fixed differently from the request, and why

1. **C2's stage-0 counts for W5 and W6** are not given (see the table above): they need a census run in the repo. They are step 3's first census line, at the merge-base.
2. **R0's literal "COV >= 14/15"** is not met, by evidence. The physical S0 bar (required by R1) fails the uniform-leak rule at every operating point. COV-PRE is added rather than lowering S0, and the GO count is 13/15 for the binomial reason above. If the owner or judge prefers 14/15, the replica says COV-PRE still meets it (P(>= 14/15) = 0.97 at q = 0.983), so the choice of 13/15 is about the spiking arm's extra noise, not about rescuing an arm.
3. **R5's "set exposure length from the measured decay steps"** is replaced by setting the per-step rates from that measurement, with exposure fixed by dose (above). The intent, time constants that mean the same on every arm, is kept.
4. **Round-1 required changes that MOVED, not answered in stage 1** (DESIGN 16):
   - 2: the resolving route, to stage 2, 12-15..12-18.
   - 5, second half: told-by-X storage, to stage 2's M3b.
   - 6, production half: to #272's word-commit.
   - 7 and 9: to S1-bind, gate 12-01..12-10.
5. **Residues that remain, each counted with method, stage and date:**
   - `moi` (3 items): stage 3, method packet by 12-05.
   - Homograph false holds (6b): #260; the cost is preregistered and a ruling requested.
   - The declared recall cost (measure 7: 5 correct partial coordinations, 16 over-general possessives, 1 demonstrative): a ruling before the gate.
   - B3's own strip deciding which turns reach the organ (60 silent drops, 25 declines): S1-bind and stage 2.
   - Token sharing under one minted token: S1-bind.
6. **The replica is rate-level, not spiking.** Every step-0 number above is a prediction the spiking census (0b) can falsify, and 0b decides the GO. The transfer curve has not yet been measured on the organ's own bridge with its FS pools and mode gates. The 0b smoke does that before any census line, and re-sets the thresholds there if they differ.
7. **COV-PRE at overheard 0.75 passes 12/15.** The failing bar is not yet isolated; it is reported by the step-1 dose arm. The schedule's 0.6 passes 15/15.
8. **Rulings this design requests (NEEDS lines, not assumed):** queue placement for the lane; the FLIP-HURTS named list; the homograph cost (6b); the recall cost (measure 7).

## The judge's 15 unresolved items: what changed

| Item | Status |
|---|---|
| 7: slang residue, 19 of 28 | Superseded: R2/R3 expose the reduced and chat spellings, and the residue is `moi` x3 |
| 8: recall cost of 5 + 9 | Now 5 + 16 + 1, with 59 items held; a ruling is still required |
| 9: compute of 465 runs | Recounted: about 235 runs, 20-105 CPU-h |
| 12: S2b dose floor | Removed: S2b is within-token, and S0 requires every name to fire |
| 6: forced-fit replica numbers | Updated at the COV-PRE centre: 0.11, 0.17, 0.30, 0.41 |
| 13: replica is not spiking | Still true (item 6 above) |
| 11: cache byte-identity | Still a step-3 record claim, with the cache key now including the code revisions |
| 0-5, 10, 14 | Unchanged, and carried in DESIGN 10-13 |

## Round 3 (Claude, 2026-10-22, second fix round)

Input: `the fix-round workflow record (not committed)`, `result.ref.findings` (verdict NEEDS-CHANGES: 1 BLOCKING, 4 MAJOR, 3 minor).
DESIGN.md and PACKET.md were edited in place; the packet was restructured. Scratch: `docs/drafts/deixis/s1-scratch/fix2/` (the
`scratch/fix/` scripts copied there unchanged, never edited, plus new scripts that import them). Everything ran on the
workstation (dant123-wk, where_ran class `workstation`) and is not provenance-stamped: each number below is a forecast,
and the restructured step 0a regenerates it as a provenance-stamped module run on the pool. Nothing in any repo was
written; code was read at the workstation checkout `da12c3aac` (and `research/stage2-exit-gate` at `eddb1a071`).

### Evidence produced this round (`docs/drafts/deixis/s1-scratch/fix2/`)

| File | What it shows |
|---|---|
| `replica3.py`, `wscale_curve.py`, `fI_probe.py`, `final_v3b.py`, `hd_check.py` | Copied unchanged from `scratch/fix/`. |
| `stream_audit.py`, `.out`, `.json` | The toy world writes each seed's stream to disk (canonical JSONL, sha256); the replica and a spiking-noise stand-in both read the file. The audit on a shared stream passes the correct learner (COV-PRE: moderate A1 15/15, A2 15/15; harsh 13/15, 14/15). It fails every mislabelled pair (A1 <= 1/15, A2 0/15) and every rival. Independent streams are UNDEFINED by A0, with Spearman 0.45. |
| `calib_proto.py`, `calib_proto_*.out`, `.json` | A per-target threshold prototype on the engine (seeds 61-70, plasticity off, both heterogeneity settings). TOK -> P is 8/4. TOK -> E is 9/4 at the frozen FS constants (8/30) and at 20/20. TOK -> E is 16/4 with a strong real-FS pool (60/80, preset-centred on), and 9/4 with RS hybrids. The FS preset is real only with the flag on: a 0.206 and C 19.0 with it, a 0.031 and C 94.9 without. |
| `replica4_markers.py`, `.out`, `.json` | The toy with ART, ASP, IND, AGR and DEF (the original stream is unchanged, checked), scored with per-target thresholds. With P 8/4 and E 9/4, COV-PRE scores 15/15 and 59/60; with E 16/4 it scores 0/15. On COV-PRE, ART, ASP and IND stay <= 0.12 of W_MAX on every pool; AGR reaches 0.30-0.37 and DEF 0.66. COV-PRE's name routes measure 11.1-15.0 engine units. |
| `hd_check_round3.out` | `judge()` from `tools/gates/host_decision_declared.py`, run on both files at their planned paths: problems NONE for each. The decides / allowed-by / replaced-by / cap fields are identical across the two files. |
| `PACKET.before_round3.md` | The fix-round packet before the restructure. |

### Finding -> change

| # | Finding | Status | Change (DESIGN section; PACKET part) |
|---|---|---|---|
| B1 | BLOCKING: the audit was unreachable. Spearman >= 0.9 against a replica run on an independent stream gives about 0.46 for a correct learner | FIXED | DESIGN 3.4 ("The audit, defined on one recorded stream"): the toy world writes the stream; the spiking run and the replica both read that file. A0 (sha256 equality, else UNDEFINED); A1 (per-seed GO-bar pattern equal, >= 13/15); A2 (per-route same side of its own target's bar, 1-engine-unit band, >= 13/15). Spearman is reported only. The spiking-noise model is stated at two levels. The scratch result is in the table above. "Reproduces the census" at step 1 is now defined: the same per-seed pass/fail and route sides on 15/15 seeds, plus replayed-raster parity <= 1e-6 (3.4 GO rule; PACKET step 1). New outcome: PASS-UNEXPLAINED (instrument items hold but a replica item fails). It waits for a ruling and blocks later arms, so a rival is never rejected only because a rate replica predicted failure. PACKET: `_deixis_toy_world.py` writes streams, `_deixis_audit_check.py` runs the audit in 0a(ii), and STOP-AUD applies |
| M1 | Thresholds: one FS-free curve served both targets; re-set thresholds did not re-run 0a; the heterogeneity setting was not stated | FIXED | DESIGN 2.6 measures thresholds per target in 0a(i), on calibration seeds 61-70 (outside 1-15), plasticity off (weight sha256 checked), before any learning arm. TOK -> P is measured on its real hearing wiring (plus the WTA variant). TOK -> E is measured with the FS pool, the release projection open, and the toy-hearing H lines; presence drive is reported as an increment. A threshold rule is written down, and the values are committed with their sha256. The replica always reads the committed thresholds, so a moved threshold re-runs 0a's STOP rule by construction. DESIGN 2.2 states the heterogeneity setting: `enable_parameter_heterogeneity=True`, `preset_centred_parameter_heterogeneity=True` (without it, FS presets are RS hybrids). Both flags and the cell-type dump go into the bar-sanity record and the organ-cache key (5.1). The FS constants are frozen at the agency motif's 8/30 (PACKET Appendix C), and the strong-pool forecast that would STOP the lane is stated |
| M2 | Fixture rendering: an undeclared host decision; forms co-varying with person or givenness become pointers | FIXED | DESIGN 4.3 and the HD `decides:` line now declare the rendering: SVO; a tense frame drawn independently of the scene (past 0.6, perfect 0.2, progressive 0.2, regular verbs only); every auxiliary fused to its subject; no copula and no agreement; 'the' before every undetermined object noun and in 'the former' / 'the latter'; no 'a'/'an'. R3 is now a closed contraction list (no 'll / 'd / n't, so no well / ill / hell / shed). The toy gains the EN-matching markers ART and ASP (class W), IND (class W), and the co-varying AGR and DEF as a HOLD-COST class with a preregistered cost (T17, 8.1; Appendix A/B). The step-3 class-W check on every non-pointer EN form runs before census v2 (4.3, 8.2; PACKET). Measure 6 gains 'the' / 'are' / base-form-plural items, and 6b gains one item or more per homograph (were, mine, that, its, u, ur, em; 'us' needs none because B3 strips it). Each item's B3 extraction was checked; flag-off storage is re-checked at the merge-base. The tom-false-belief row is added to the no-regression list (8.5). Consequence, stated: present-tense verbs and copulas are unexposed in production and read none |
| M3 | Engagement: the rule arms did not have to prove plasticity ran | FIXED | DESIGN 3.4 ("Engagement of every rule arm") and 8; PACKET step 0b. Every run records gate read-back per utterance, the count of update steps, sum \|dw\| per block, the rule config read back from `core_config` (plus the runner's lam/kap/pre_floor/ybar_alpha, and the WTA weights and pool spikes), and the stream sha. A failing arm without this record is UNDEFINED, and an UNDEFINED arm earlier in the GO order blocks a later GO. 0a(i)'s step-time runs exercise the same code at learning rate 0 |
| M4 | W4 rewrites self-facts; its skip read the B3-gated `held`; W4's extractor was not a trigger | FIXED | DESIGN 2.4 splits `read_turn` into `latched` (the turn-start result) and the W1-side hold. W4, the echo and the false-belief writer skip on `latched`, whatever B3 later decides. `extract_assertion(msg) is not None` is added to the organ-read triggers (5.1, 5.2, 12, HD header). Honesty case 9 is added. H7 (8.4; PACKET Honesty) runs on curriculum self-facts with three both-form and three self-name items, each extract checked. Flag-off must show surprise engaged plus a rewrite (else the item is replaced). Flag-on requires every self-fact byte-unchanged and W4 recorded `skipped_deixis_latched`, 6/6, intact and on the lesion arms. The B3-reject control is 'the brain uses no spikes'. The removed surprise notice is declared as the one reply-text change |
| M5 | Answer rate: blind to the flip's real costs, and a probe that predictably hurts was missing from the named list | FIXED | DESIGN 8.5; PACKET. 'tell me more' and 'how do you feel' after a held statement are named in FLIP-HURTS, because it cannot be shown that they cannot hurt. A frozen natural set that this lane did not choose is added: the Stage-2 exit gate's intact teach turns plus each group's own probes, at the 11-03 pin, at both shapes. The tom `tom_fb` probe is added. The per-seed held fraction of non-census statements is reported beside the answer-rate report |
| m1 | FIX_NOTES overstated the anti-list property; R2 matched the census; EN-HO is a pin; the dose floor was missing from decides: | FIXED | FIX_NOTES C0 is corrected (struck through above). DESIGN 1, 4.3 and PACKET state that R1-R3 are a teacher scaffold chosen after the census, and that the evidence is the nonce, K = 0 and transplant arms. R2 is frozen as the complete single-word italic list of the "Complete table" in the Wikipedia article "English personal pronouns", revision 1343514077 (2026-03-14T20:17:50Z), copied verbatim into the fixture header with the page sha256. Multi-word entries are excluded by one stated rule. The census-matched round-2 list is withdrawn. EN-HO is relabelled a front-end pin (8.2; PACKET table). The dose floor is in `decides:` |
| m2 | The calibration seed was a census seed; learned weights could be seen before thresholds and conversions froze; P0 failure had no rule; step time was unmeasured | FIXED | All calibrations, conversions, the C-weight row (P0 at K = 0), per-arm step time (runner update included), two-run identity and the saturated-route check moved to 0a(i), on seeds 61-70, plasticity off, committed with sha256 before any learning arm. STOP-P0 is a RESULT of its own, with no tuning outside the frozen grid and never on census seeds. The 0b queue plan comes from the measured step times (3.3, 3.4, 13; PACKET step 0a) |
| m3 | Where-ran: fix-round evidence had no host record; host classes were mixed | FIXED | Step 0a regenerates every threshold, replica and audit number as a provenance-stamped module run on one pool node (`--expect pool`). Calibrations and identity are measured on the census host class or repeated there. Every RESULT quotes the where_ran class per cited number (8; PACKET Measures and the 0a RESULT format). Every scratch number in the design is labelled a workstation forecast |
| Packet | Restructure so Hermes runs only step 0a without a ruling | DONE | PACKET now has the "Step 0a (runs now)" part (seven files to build, the commands in order, six STOP rules, the preregistered forecast, the RESULT format) and "Steps after 0a", each opened only by a RULING. Appendix C freezes the hearing-core constants, the C-weight grid and the threshold rule |

### What remains open (not fixable on paper; each has its place)

1. **Every number is a workstation forecast.** None is a record. Step 0a turns them into pool records. In particular, the TOK -> E threshold forecast of 9 rests on a reduced prototype (five entities, four C cells); the full Appendix C circuit may differ.
2. **The C-weight AND (P0) has never been run.** The grid in Appendix C is an arithmetic guess from design 2.6. If no row passes, STOP-P0 fires and a ruling decides the next circuit. This is the most likely 0a stop.
3. **The FS constants were chosen after seeing the prototype.** The choice (the agency motif's 8/30) rests on the pool's role, but all three settings had already been seen when it was made. This is declared in DESIGN 2.6; a ruling may prefer another documented motif.
4. **R2 is pinned, not copied.** The verbatim copy of the Complete table's italic forms (and the page sha256) is made at step 3. The read quoted in 4.3 is indicative only, and an R2 entry may itself turn out to co-vary with something; the class-W check catches that.
5. **The audit's harsh level is close to its bar** (COV-PRE A1 13/15, its GO bars 13/15). If real spiking noise is near the harsh model, 0b may return 13/15, or PASS-UNEXPLAINED on one bar.
6. **AGR / DEF hold costs** are known only at rate level (AGR between the thresholds, DEF firing). They matter once the AI-teacher replacement speaks real English. EN v1 avoids them by its rendering, which is itself a declared scaffold.
7. **Carried from the fix round, unchanged:** W5/W6 stage-0 counts at the merge-base (step 3's first census line); the rulings requested (queue placement, FLIP-HURTS, the homograph cost 6b, the recall cost 7); COV-PRE at overheard 0.75 (12/15, failing bar not isolated); `moi`.
