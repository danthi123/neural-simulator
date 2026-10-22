# PACKET deixis-271-s1 (Claude, 2026-10-21; fix rounds 2026-10-22): stage 1 of #271, learned pointer words on one bridge, wired hold-first

Board #271. Design: `docs/drafts/deixis/DESIGN-s1.md` (the round-2 synthesis, revised in two fix rounds; every finding is
dispositioned in `docs/drafts/deixis/FIX_NOTES-s1.md`). Branch `research/deixis-s1` from S main, with `research/deixis-census` merged in.
Additive, every default OFF, flag off byte-identical. Lane in parallel with cs-race, never ahead of it (RULING
deixis-271-s0).

**What Hermes may run now: STEP 0a, and nothing else.** Step 0a builds six additive research modules and runs them as
provenance-stamped module runs on the CPU pool (no `sim/` edit, no production edit, no flag, no gate seed), then STOPs
and posts `[deixis-271-s1-0a] RESULT`. Every later step is listed below with its bars, and each starts only after a
Claude RULING naming it. If anything in step 0a is unclear, post NEEDS CLAUDE; never fill a gap by choice.

host-decision: scaffold
decides: on a turn a routed writer would act on (a B3 store, the TUI fallback's echo, a false-belief scenario step, a W4 surprise read and reconsolidation), a fail-closed hold rule reads the organ's spiking latches and keeps the turn out of the store when any token read is a pointer pool, referential failure, ambiguity, the self latch, a co-commit or a non-invariant read, and holds every such turn when the organ is unavailable, raises or sees a different text; which turns the organ reads (an acquisition candidate, a TUI echo, a false-belief scene parse, a non-None W4 extract_assertion) and the per-turn relay of that read to the writers, with their skips on a held turn (W4 skips on the turn-start latch result whatever the later B3 check decides); which world afferents production withholds (voice, addressee, givenness: every production pointer hold is caused by the silenced afferent); the toy's discourse-givenness afferent (the entity named in the previous utterance); the exposure/hearing transmission-gate schedule and the exposure-only plasticity schedule; the exposure generator's English rendering table: one SVO clause per utterance, the tense frame drawn independently of the scene (simple past 0.6, present perfect 0.2, present progressive 0.2, regular verbs only), every auxiliary fused to its subject as a contraction and never rendered as a free word, no copula, no person agreement, 'the' before every object noun without a possessive or demonstrative determiner and in 'the former' / 'the latter', no 'a' or 'an', the R1 reference-grammar pronoun paradigm by role, R2 the single-word italic forms of the pinned Complete table of the Wikipedia article English personal pronouns (revision 1343514077), R3 apostrophe deletion and you -> u, your / you're -> ur, yourself -> urself at 0.2 over the closed contraction list i'm you're he's she's it's we're they're i've you've we've they've 'em and each fixture person's name's; the per-form dose floor (a form's rendering rate is raised until every exposed form occurs at least 300 times); the form-to-assembly front end; the per-token latch reset and the between-turn quiescent reset
allowed-by: CLAUDE.md non-negotiable 1
replaced-by: board #260's word-role frame statement path writing latched referents into the composer's spiking store (the hold rule, the relay and the skips); S1-bind's voice and addressee afferents and #271 stage 2's spiking mention trace on the organ's bridge (the withheld afferents and the toy givenness afferent); the stage-3 OrganDescriptor merge onto the one-brain pool (the resets); a percept-agreement neuromodulatory plasticity gate on the merged pool (the gate schedules); human or AI-teacher multi-party speech through research/runners/ai_teacher.py (the rendering table)
cap: 3 builds

## Why
The stage-0 census (accepted, RULING deixis-271-s0-accept) measured what the production encoder stores when it hears
pointer words: 28 pointing words stored as entity names (moi, mine, yours, ours, ya, ...), 4 role shifts ("i told kofi
jazz" stores the verb as the agent), 60 cross-speaker collisions and 60 facts silently dropped, because B3's English
function-word list strips the pronoun and keeps whatever is left. A user's "you are a brain that feels pain" is stored
today as the brain's own fact, and "the latter married fenna" stores `latter` as a person. No speech-side list fixed this
(moat builds 7 and 8), and no quotation frame can fix a stored name that IS a pointing word (premise round 2). The fix
belongs at encoding, and by the binding rulings it must be LEARNED: no English list, strip rule or position rule may decide
(wrc-frame-go-1020b), it must pass a made-up-marker arm (c1-learned-1016), and every host decision must be declared with
its neural replacement (hd-gate-1021).

Stage 1 teaches one small spiking organ, from overheard scenes in which several voices talk to each other about things it
can see, which heard forms point at whoever is speaking, whoever is addressed, or something the conversation has made
given (he, she, it, they, this, that, the former, the latter), and which forms are names. Wired into production it does
one thing: a statement in which such a form, or the brain's own name, appears is HELD and not stored. In production every
such hold is referential failure: the form's learned route fires its pool, chat renders no voice, addressee or
givenness afferent, so nothing can bind and the turn is held. It resolves nothing to anyone yet (first and second
person: S1-bind; third person and demonstratives: #271 stage 2).

## Step 0a (runs now; CPU pool, the census host class; every command under `tools/memcap.sh`)

**Purpose.** Before any learning arm runs: (i) measure the organ's own circuit with plasticity off and commit the
thresholds and conversions; (ii) write the toy exposure streams to disk and run the rate replica and the audit check on
those files; (iii) write the bar-sanity record and apply the STOP rules. Design: 2.2, 2.6, 3.4, 3.5, 4.1, 4.2.

**Build (Hermes, on `research/deixis-s1`; each module additive and default OFF; runnable as `python -m
research.runners.<module>` with `--out`, so `research/runners/__init__.py` writes its `.prov.json`; no production module
imports any of them).** The sources are committed with the design as `docs/drafts/deixis/s1-scratch/` (the design's
`docs/drafts/deixis/s1-scratch/fix/` and `fix2/`); each port keeps the source's arithmetic, and its docstring names the source file.
1. `research/runners/_deixis_toy_world.py`: the toy world of Appendix A with its morphology markers, ported from
   `s1-scratch/fix2/replica4_markers.py::world_m` (itself `fix/replica3.py::world` plus the markers; the marker
   rendering takes no draw from the world RNG). `--write-streams --arm {base,label,scene,voice,indep,calib} --seeds A-B
   --length N --out <manifest.json>` writes one canonical JSONL stream per seed under `<dir of out>/<arm>/s<seed>.jsonl`
   (port of `fix2/stream_audit.py::write_stream`: a header line with schema `deixis-toy-stream-v1`, seed, length, arm,
   world parameters, the symbolic vocabulary and the per-seed nonce map; then one line per utterance with the token ids,
   the world's drive of every post population, speaker and addressee; `json.dumps(..., sort_keys=True)`), and a manifest
   with each stream's path and sha256. Nulls are streams of their own: `label` (pool labels permuted per utterance),
   `scene` (x and y from different utterances), `voice` (the speaker / addressee afferents follow a random present voice),
   `indep` (seeds s + 1000, the independent-stream control), `calib` (seeds 61-70 for 0a(i)). Streams are NOT committed;
   manifests are, and any reader regenerates a missing stream and refuses one whose sha256 differs from the manifest.
   `--selftest`: two writes byte-identical; the original-vocabulary tokens equal `replica3.world`'s.
2. `research/runners/_deixis_hearing_core.py`: the step-0 hearing core of Appendix C (TOK, E + FS, H, C, P, the mode
   gates, the two learned blocks; no commit stage), built on the brain-region framework so the cell presets apply, with
   `enable_parameter_heterogeneity=True` and `preset_centred_parameter_heterogeneity=True` in its own config.
   `--selftest`: builds on seed 61 and prints population sizes, gate names and each population's median a/b/d/C.
3. `research/runners/_deixis_calibration.py` (0a(i), plasticity off throughout: learning rate 0, decay 0, the runner
   update not applied; each learned block's weight sha256 equal before and after, recorded): on calibration seeds 61-70,
   (a) the per-target transfer curves on the grid w in {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20}: TOK -> P on
   the base circuit and on the CAUSAL+WTA variant (P pools inhibiting each other), TOK -> E in the production-hearing
   condition (no H, release open, C silent) and the toy-hearing condition (the target's H lines on, release open), and
   the exposure condition (presence drive) reported as the increment over w = 0; thresholds by the rule of Appendix C;
   (b) the saturated-route check on seed 61; (c) the C-weight row: the first row of Appendix C's grid passing every P0
   item at K = 0 on 10/10 calibration seeds (600 utterances of the `calib` stream per seed, exposure mode); (d) the unit
   conversions: the fraction of exposure steps on which the engine's Hebbian block runs (`if _prev_any and _fired_any`,
   `sim/bridge.py:10236`, counted from the firing states), steps per occurrence above `pre_floor` 0.02, y_on (the
   rate-window trace of a driven P neuron at the end of its drive), the presence current giving 0.2 of the participant
   rate (from the grid {100, 150, 200, 250, 300} pA, the nearest), and the empirical P_k base rates; (e) per-arm step time:
   each of the six arms' update code executing at learning rate 0 for 2,000 exposure steps on seed 61 (COV and COV-PRE:
   the runner's covariance update over the plastic blocks included; every arm: the per-step gate read-back included);
   (f) the per-population median a/b/d/C and both heterogeneity flags read back. Ported for (a) from
   `s1-scratch/fix2/calib_proto.py` (the five-entity prototype; the module uses the full Appendix C circuit).
4. `research/runners/_deixis_rule_replica.py` (0a(ii)): the rate replica, ported from `s1-scratch/fix/replica3.py`
   (`train`, `bars`) and `fix2/replica4_markers.py` (`train_m`, `score` with per-target thresholds and the marker
   classes). It reads the streams through their manifests (refusing a missing manifest or a sha mismatch) and the
   thresholds from `--thresholds <calibration.json>` (refusing a file whose sha256 differs from the committed one). Arms:
   CAUSAL and RW-HEBB (the rate arm `hebb`: the replica has no spike timing, stated), CAUSAL+WTA (`hebbwta`), BCM, COV,
   COV-PRE. One grid lam*tau in {1, 2, 3, 4, 6, 9, 12, 18} on seeds 1-15; each arm's centre is the cell with the most
   seeds passing every GO bar, ties to the most bar-seed passes, then to the larger median weakest must-fire route; the
   60-seed rate on seeds 1-60 at every centre; the label, scene and voice nulls and K = 0 on seeds 1-15 at every centre;
   the marker routes per arm; every route's weight per seed written out.
5. `research/runners/_deixis_audit_check.py` (0a(ii)): the audit of design 3.4 computed exactly as 0b will compute it,
   ported from `s1-scratch/fix2/stream_audit.py`, run on the replica's own streams under the stated spiking-noise model
   at the moderate and the harsh level: for every pair (rule g that produced the weights, rule r whose replica is the
   reference) on seeds 1-15 at each arm's centre, A0, A1, A2 and the reported Spearman; the independent-stream control
   (`indep`); the unengaged control (all weights 0).
6. `research/runners/_deixis_bar_sanity_0a.py` (0a(iii)): reads every output above, checks each input's sha256 against
   the committed one, evaluates the STOP rules below and writes the bar-sanity record (thresholds and their source, the
   C-weight row, conversions, step times and the 0b queue plan, the heterogeneity flags and the cell-type dump, the
   replica record, the audit matrix, the STOP-rule verdicts, every input sha256).
7. `tests/test_deixis_0a_modules.py`: each module's `--selftest`; the stream writer byte-identical across two writes; the
   replica and the audit check refuse a stream whose sha256 differs from its manifest; no module under `webapp/` or a
   production runner imports any `_deixis_*` module; `git diff --stat main -- sim/` is empty.

**Commands (in this order; `OUT=research/findings/raw/_deixis_s1/0a`; `SIM_BACKEND=numpy`; on one pool node so every
file has one host class; `tools/mem_ok.sh 4` before each).**
```
python -m pytest -q tests/test_deixis_0a_modules.py
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm calib --seeds 61-70 --length 600 --out $OUT/streams/manifest_calib.json
tools/memcap.sh 4 -- python -m research.runners._deixis_calibration --seeds 61-70 --streams $OUT/streams/manifest_calib.json --out $OUT/calibration.json
tools/memcap.sh 4 -- python -m research.runners._deixis_calibration --seeds 61 --streams $OUT/streams/manifest_calib.json --out $OUT/calibration_s61_a.json
tools/memcap.sh 4 -- python -m research.runners._deixis_calibration --seeds 61 --streams $OUT/streams/manifest_calib.json --out $OUT/calibration_s61_b.json
git add $OUT/calibration.json $OUT/calibration.json.prov.json $OUT/calibration_s61_*.json* $OUT/streams/manifest_calib.json* && git commit -m "deixis-s1 0a(i): calibration, plasticity off, seeds 61-70" -- $OUT
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm base  --seeds 1-60 --length 4800 --out $OUT/streams/manifest_base.json
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm label --seeds 1-15 --length 4800 --out $OUT/streams/manifest_label.json
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm scene --seeds 1-15 --length 4800 --out $OUT/streams/manifest_scene.json
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm voice --seeds 1-15 --length 4800 --out $OUT/streams/manifest_voice.json
tools/memcap.sh 4 -- python -m research.runners._deixis_toy_world --write-streams --arm indep --seeds 1001-1015 --length 4800 --out $OUT/streams/manifest_indep.json
tools/memcap.sh 4 -- python -m research.runners._deixis_rule_replica --streams $OUT/streams --thresholds $OUT/calibration.json --grid 1,2,3,4,6,9,12,18 --seeds 1-15 --rate-seeds 1-60 --out $OUT/replica.json
tools/memcap.sh 4 -- python -m research.runners._deixis_audit_check --streams $OUT/streams --thresholds $OUT/calibration.json --replica $OUT/replica.json --noise moderate,harsh --seeds 1-15 --out $OUT/audit_check.json
tools/memcap.sh 4 -- python -m research.runners._deixis_bar_sanity_0a --calibration $OUT/calibration.json --identity $OUT/calibration_s61_a.json,$OUT/calibration_s61_b.json --replica $OUT/replica.json --audit $OUT/audit_check.json --out $OUT/bar_sanity_0a.json
python tools/where_ran.py $OUT --expect pool --json > $OUT/where_ran.json
git add $OUT && git commit -m "deixis-s1 0a(ii)-(iii): streams manifests, replica, audit check, bar-sanity record" -- $OUT
```

**STOP rules of step 0a (each is a STOP with its own RESULT line; no constant is changed and nothing is re-run with
other settings without a Claude RULING; census seeds 1-15 are never used to tune anything).**
- STOP-PLAST: a learned block's weight sha256 changed during calibration (plasticity was not off).
- STOP-ID: the two seed-61 calibration runs are not byte-identical (JSON hash, timing fields kept in a sidecar).
- STOP-P0: no row of the C-weight grid passes every P0 item at K = 0 on 10/10 calibration seeds; the RESULT names each
  row's failing items.
- STOP-CAL: a target does not fire every neuron on every seed by w = 20 in some hearing condition, or no grid weight
  below its theta_fire is silent.
- STOP-REP (the design's 0a STOP rule, run with the measured per-target thresholds whatever they are): no arm reaches a
  60-seed per-seed pass rate q >= 0.9; or some null passes every GO bar on >= 2/15 seeds; or K = 0 does.
- STOP-AUD: an arm with q >= 0.9 fails its own audit (A1 or A2 below 13/15) under the moderate model; or some mislabelled
  pair (weights of rule g scored against rule r's replica, g != r, g passing the GO bars) passes A1 and A2 at either
  level; or the independent-stream control is not UNDEFINED by A0.

**Preregistered forecast (scratch, workstation, not provenance-stamped; the RESULT reports the measured values beside
it).** TOK -> P 8 / 4 on both circuits; TOK -> E 9 / 4 (so the replica re-scores with per-target thresholds); COV-PRE at
lam*tau 6 passes 15/15 and 59/60 with E 9/4, every rival 0/15; ART, ASP and IND silent; AGR 0.30-0.37 and DEF 0.66 of
W_MAX; the audit passes COV-PRE on its own replica (moderate 15/15 and 15/15; harsh 13/15 and 14/15) and fails every
mislabelled pair (A1 <= 1/15, A2 0/15); independent streams UNDEFINED by A0 with Spearman about 0.45. With the strong FS
pool E would be 16/4 and STOP-REP would fire: the FS constants of Appendix C are frozen for that reason.

**`[deixis-271-s1-0a] RESULT` (post, then STOP).** One line per item, each number followed by the where_ran host class
of the file it came from: the per-target thresholds and the calibration sha256; the C-weight row chosen and every P0 item
at K = 0; the conversions; per-arm step time and the 0b queue plan (runs, CPU-h, and whether 235 runs fit by 10-27 or the
rivals drop to their centre cell); the identity verdict; the heterogeneity flags and the FS / E median a/b/d/C; each
arm's centre, its 15-seed and 60-seed counts per bar, its nulls and K = 0; the marker routes; the audit matrix at both
noise levels; every STOP-rule verdict; the branch @ sha. Then wait for Claude's ruling.

## Steps after 0a (each starts only after a Claude RULING; named here with their bars)

- **Step 0b, the fair rule census on one spiking bridge (CPU pool, dev seeds 1-15; after RULING on 0a).** The Appendix C
  hearing core with the committed calibration; every run drives the bridge from 0a's stream file for its seed and arm and
  records the sha256 it read. Six arms: CAUSAL (default rule), CAUSAL+WTA (lateral inhibition among the pointer pools),
  RW-HEBB (rate-window), BCM, COV (learning rate 0, the covariance update applied by the runner to the plastic blocks from
  the bridge's own traces, the Lane C precedent, with the engine decay as the leak), COV-PRE (the same update with the
  presynaptically gated leak, engine decay 0). First one run per arm on dev seed 7 (engagement record and wall time;
  nothing is set there), then 6 arms x 2 cells (each arm's 0a centre and the next grid value up) x 15 seeds = 180 runs;
  the nulls (label, scene, voice streams; 3 x 15 = 45 runs) only on an arm that passes the weight bars; K = 0 scored on
  the initial weights. The rivals drop to their centre cell if 0a's queue plan says 235 runs do not fit by 10-27.
  **Engagement, every run (else UNDEFINED):** the plasticity and transmission gate values of every learned block read
  back once per utterance during exposure (learned routes' transmission 0 throughout); the count of exposure steps on
  which the update was applied (the engine's block condition counted from the firing states; for COV and COV-PRE the
  runner's own count); sum |dw| per learned block (`dx_ptr`, `dx_name`), per utterance, nonzero; the rule configuration
  read back from the bridge after build (`enable_hebbian_learning`, `hebbian_rate_window`, `hebbian_bcm`,
  `hebbian_bcm_theta_alpha`, `hebbian_learning_rate`, `hebbian_weight_decay`, `hebbian_min_weight`,
  `hebbian_max_weight`, `enforce_plastic_mask_in_hebbian`, `enable_stdp`; the runner's lam, kap, pre_floor and ybar_alpha
  for COV and COV-PRE; for CAUSAL+WTA the P-pool inhibitory weights read from `cp_connections` and the inhibitory pool's
  spike count per utterance, nonzero); the stream sha256. A failing arm is FAIL only with this record present.
  **Audit, every passing arm, before the pass counts.** Instrument items (else UNDEFINED): A0 (the run's stream sha256
  equals the replica's for that seed and arm); the engagement record; P0 on that arm's runs; the learned routes'
  transmission read back closed on every exposure step; the arm's own nulls each pass on <= 1/15; the saturated-route
  check on that arm's bridge. Replica items: A1, the per-seed pass/fail of every GO bar equal to the replica's on that
  seed's stream, >= 13/15 seeds; A2, every scored S0 route on the same side of its own target's bar as the replica's,
  ignoring a disagreement only when both values lie within 1 engine unit of the bar, >= 13/15 seeds. Spearman reported
  only. Instrument items hold but a replica item fails: PASS-UNEXPLAINED, the route-level disagreements recorded, the GO
  waits for a Claude ruling.
  **GO rule:** the first arm, in the order CAUSAL, CAUSAL+WTA, RW-HEBB, BCM, COV, COV-PRE, that passes P0, the
  saturated-route check and S0-S7 all on the same seed on >= 13/15 seeds at one of its two cells, whose nulls each pass
  on <= 1/15, and whose audit holds. An UNDEFINED or PASS-UNEXPLAINED arm earlier in that order blocks every later arm's
  GO. An existing rule: no `sim/` edit, the packet is re-ruled to it. COV or COV-PRE: step 1 adds it to `sim/`. None:
  STOP, bank the method, name the next (a three-factor rule with the coincidence pool as the third factor and eligibility
  traces; then BTSP plateaus gated by the coincidence cell). Why 13/15: design 3.4 (binomial in the per-seed rate q; at
  q = 0.9 P(>= 13/15) = 0.82, P(>= 14/15) = 0.55). Post `[deixis-271-s1-0b] RESULT` (per arm and cell: pass counts per bar
  per seed, the engagement record, the audit items, nulls, wall time, CPU-h, the where_ran class per cited number) and
  STOP.
- Step 1 (after RULING on 0b; 10-27..10-31; dev seeds 1-8): the organ. 1a hearing core with the REF commit, RF, AMB and
  SELF latch on the ruled rule, with S8 (each token alone latches its class, >= 80% per class on >= 13/15 dev seeds) and
  T-carry; the dose arms (voices, overheard, turn-taking, dyad-only); 1b speaking and efference (agency motif, OUT, OUT
  commit). The bar sanity spec; the nonce CI test. If COV or COV-PRE: the `sim/` rule with its pins (flag-off
  byte-identical, a sign test, ybar frozen while closed, parity with the step-0b arm on a replayed raster <= 1e-6) and the
  census reproduced exactly: on the same recorded streams the in-step rule and the 0b runner arm give the same per-seed
  pass/fail of every GO bar on 15/15 seeds and the same side of every scored route under A2's tolerance on 15/15 seeds.
  File the organ-gate prereg (`flip: no`, this header) by 10-31. Post `[deixis-271-s1-1] RESULT`, STOP.
- 11-01: defaults pinned, `BRAIN_DEIXIS_HOLD` OFF.
- Step 2 (after RULING on step 1; gate seeds 42/43/44/100/101/102): the organ gate (Appendix B). 11-01..11-02 only if
  step 1 is ruled by 10-31, else 11-11..11-13; never inside 11-03..11-10. STOP.
- Step 3 (after RULING on the step-2 verdict; dev seeds 7 and 1-5; 11-03..11-10, or 11-14..11-20 if step 2 slipped; dev
  lines only): the EN fixture of design 4.3 (the declared clause rendering; R2 copied verbatim and complete from the
  pinned source, with the page sha256), the MM and EN-HO fixtures; the class-W check on every non-pointer English form
  (below) BEFORE census v2; then production S1-hold wiring, census v2, the load-bearing row, the wiring and fail-closed
  tests. If the organ gate fails, step 3 does not start and no English fixture is built. The `sim/` rule (if any) merges
  to main only after the 11-03 GATE_SHA pin. STOP.
- Step 4 (after RULING on step 3; gate seeds; 11-11..11-18, or 11-21..11-28 if step 2 slipped): the production gate and
  the flip prereg with flag-off battery arms, smoke-first. The flip target is 11-24..11-28 (12-01..12-05 if slipped), after
  the 11-21 Stage-2 score, never inside the pin.
- S1-bind (voice identity, first-person storage under a speaker key, the question side): its OWN packet and header,
  development only after the step-4 ruling (to 11-30), gate 12-01..12-10, flip coupled to `BRAIN_ONE_BRAIN` and the
  pool-merge rung, targeting 12-19. No voicing until #272's reply-side spiking word-commit drives the efference copy.
- #271 stage 2 (resolution: the mention trace on this bridge, the two-candidate order, quoted speech, M3b): packet 11-28,
  dev 12-01..12-12, gate 12-15..12-18.
- Queue placement: requested from Claude as a NEEDS line; until ruled, this lane queues at normal priority.

## Build of the later steps (additive, default OFF; named so the ruling can scope them)
Flag `BRAIN_DEIXIS_HOLD` (default OFF; off is byte-identical, pinned). Lesion flag `BRAIN_DEIXIS_HOLD_LESION` (`ptr`,
`name`, `both`). With the flag on, an organ that cannot be loaded at startup refuses the flag (a startup error naming the
cache key); a `read_turn` that raises, or reads a different text from the one the writer got, is a HOLD (class
`unavailable` / `mismatch`), counted per chat and in the reply JSON.

New files of later steps: `research/runners/_deixis_rule_census_derisk.py` (0b); `research/runners/deixis_binding_organ.py`
(step 1; populations prefixed `dx_`; no English word in the module); `research/runners/_deixis_toy_gate.py` (step 2);
`research/runners/_deixis_fixture_gen.py` + `research/fixtures/deixis_exposure_en_v1.json` and its EN-HO twin (step 3);
`research/runners/deixis_production_router.py` (`read_turn(chat, msg)` returning `latched`, per-token classes and reads;
the organ cache keyed by (fixture sha, seed, params with the FS constants and the calibrated thresholds, both
heterogeneity flags as the organ sets them, the blob shas of `sim/bridge.py`, `sim/config.py`, `sim/kernels.py` and the
organ module); the flag); `research/runners/lbf_rows/deixis_hold.py`; `tests/data/deixis_nonces.json`,
`tests/test_deixis_flag_off_identity.py`, `test_deixis_hold_wiring.py`, `test_deixis_nonce_isolation.py`,
`test_deixis_no_census_read.py`, `test_deixis_fixture_rules.py` (the fixture's forms equal what the rendering rules,
R1, the committed R2 copy and R3 produce: a consistency check of the generator, not evidence that no form was chosen
with the census in view), `test_deixis_fail_closed.py`; census v2 on `research/deixis-census` (oracle v2 beside v1, the
W5 and W6 writers, the cross-speaker B3 measure); `docs/drafts/deixis/bar_sanity_spec_s1.py`;
`research/biology/deixis-shifter-learning.md`. Only on a COV or COV-PRE GO: `sim/config.py` (`hebbian_cov`,
`hebbian_cov_leak`, `hebbian_cov_ybar_alpha`, `hebbian_cov_pre_floor`, `hebbian_cov_ybar_gate`, all inert at their
defaults), `sim/bridge.py` (one guarded branch beside the BCM branch, about 30 lines), `tests/test_hebbian_cov.py`,
`research/biology/covariance-ltd.md`.

Flag-guarded edits (step 3 only, each inside `if hold_enabled()`): `research/runners/brain_chat_tui.py`
(`_maybe_acquire` returns a held sentinel when the read is latched at its store point; `gate` returns None and
`gate_extract` returns `('done', None)` on it; `_is_acquisition_candidate` mirrors; `_discourse_turn` does not write or
echo a latched turn; the TUI local fallback calls `read_turn` itself), `webapp/server.py` (`brain_reply` sets
`chat._deixis_turn` right after `_word_role_trace_reset`; the organ runs when the turn is an acquisition candidate, a TUI
echo, a false-belief scene parse, or `surprise_production_organ.extract_assertion(msg) is not None`; W4 (surprise and
reconsolidation, :6950-7001) is skipped whole on a LATCHED turn, read from the turn-start latch whatever the later B3
check decides, recorded as `skipped_deixis_latched`; the false-belief scene writer skips a latched turn; the B3
contradiction reject is KEPT), `webapp/reward_value_afferent_chat.py` (the same skip),
`research/runners/rich_answer_composer.py` (a held outcome short-circuits `_direct_fact` before its DIRECT-LTM-TOPIC
fallback). The reply on a held turn is the abstention (`abstained: True`, no facts); the JSON carries a `deixis` record;
no new English is added.

What the host decides, in plain words (the header above): which statements are kept, by reading the organ's latches
fail-closed; which turns the organ reads and the relay of that read to the writers, W4 included; which world afferents
production does not render; the toy's givenness afferent; when the organ's synapses may change and which projections
are open in exposure versus hearing; the English the teacher uses in exposure (the clause rendering, R1-R3 and the dose
floor; chosen after the census was seen, so the census numbers on those forms are fixture recovery and the anti-list
evidence is the nonce, K = 0 and transplant arms); how a heard form reaches an assembly; the per-token and between-turn
resets. Everything that decides WHICH form points at WHOM is learned synapses on one bridge.

## Measures and bars
Six gate seeds; counts pooled; every lesion arm in a fresh process. Every arm records engagement (organ calls, tokens
read, latch reads) and is voided with `tools.lab.void_if` when the flag is requested but the organ never ran. Every
lesion, permutation and null arm also records, and is UNDEFINED (`tools.lab.undefined_if_empty`) without: the
transmission and plasticity gate values read back from the bridge; per token, the P, RF and E spike counts showing zero
current through the lesioned route; the K, permutation map and shuffle seed used, and the exposure stream's sha256.
Every learning-rule arm records the engagement record of step 0b. Every RESULT quotes, for each cited number, the
`tools/where_ran.py` host class of the file it came from; census v2 and every production arm record the source
(`tiny-demo +LTM`), `corpus_present`, the LTM bundle sha256, the organ-cache key and hit or miss, and the host class; a
flag-on vs flag-off comparison across host classes is UNDEFINED; calibrations and identity checks are made on the
census host class (the pool) or repeated there.

**Instrument validity (every step):**
- Two runs byte-identical (JSON hash) on calibration seed 61 (0a) and dev seed 7 (later steps).
- P0: the conjunctive cells fire on >= 95% of utterances where their entity takes part and holds their relation, <= 2%
  otherwise, measured at K = 0 (0a, calibration seeds; it selects the C-weight row) and at the end of exposure; <= 1% of
  windows without the relation afferent; pointer pool occupancy on non-matching utterances <= 0.02; the empirical P_k
  base rate within +-0.05 of the schedule's; >= 2 entity assemblies firing in every scene; the learned routes'
  transmission closed throughout exposure (read back).
- S7: gain-0 synapses change by exactly 0; no plastic weight leaves [0, 20]; weights start at 0 so depression is visible.
- The saturated-route check (0a on seed 61, and after the census).
- Bar sanity through the real scorer (n >= 1000): perfect >= 0.95; null-no-op, null-hold-all, null-English-list,
  null-exact-exposed-list, null-label-table, null-percept-on and random <= 0.05, and the record names the bar that kills
  each (the exact-exposed-list null PASSES the census bars and dies only on the nonce, K = 0 and transplant arms: say so).
- Flag off byte-identical: the stage-0 census re-run at the branch's merge-base equals the flag-off census; the commit
  battery sha256 and the word-role pins unchanged; `hebbian_cov=0.0` byte-identical (if added).

**Step 0 (learning) bars, per seed, engine units (design 3.5).** Thresholds per target from the committed calibration:
pool routes against theta_fire(P) / theta_silent(P), entity routes against theta_fire(E) / theta_silent(E). S0 every route
that must act ends >= its theta_fire (each pointer form's target pool, the GIVEN forms included; the self's name; every
name to its bearer) and every route whose firing could cause one wrong latch ends <= its theta_silent (pool routes of
predicates, objects, filler, names, OOI-1, the markers ART, ASP and IND, and PREV except PREV -> ADR and PREV -> GIV;
non-target pools of single-target pointer forms; entity routes of predicates, filler and those markers; non-bearer
routes of names); the entity routes of pointer forms are reported (firing one adds a second latch, which HOLDS); AGR and
DEF are the hold-cost class, reported, never in a GO bar. S1 each pointer form wins its target pool by 2x over every
other pool, decoys included. S2a a non-possessive single-target pointer form's target pool >= 2x its best entity route.
S2b a name's bearer route >= 2x its best pool route and >= 2x its best other entity route. S3 predicates, objects,
filler and the class-W markers under max(theta_silent, 0.25x the smaller shifter target). S4 decoy routes under
max(theta_silent, 0.2x it), and no token with a pool route >= 0.5x it has a decoy as its top pool. S5 the
out-of-inventory marker as S3; the previous-speaker marker onto the addressee pool <= 0.2x in the decorrelated world. S6
every must-fire route non-decreasing between 2/3 of exposure and its end, and the routes of forms heard >= 450 times
within 20% between the two. S7 above.

**Stage-1 production bars (step 4; the census v2 scorer, pinned to the stage-0 numbers; floor state identical in both
arms of every comparison).** Projection on the recorded census (design 8.2): 59 of the 126 items move from stored to
HELD. These are fixture-recovery checks; the learning claim rests on the made-up-marker arms.

**Before census v2 (step 3): the class-W check on every non-pointer English form.** On the trained EN organ, dev seeds 7
and 1-5: every exposed form that is not an R1-R3 pointer form or a name ('the', every -ed and -ing verb form, every
noun) has every pool route <= theta_silent(P) and every entity route other than its own object's <= theta_silent(E). A
failing form is named, census v2 does not run, and the lane STOPs for a ruling.

| Measure | Stage-0 W1 (B3 on) | Bar with the flag on | Counted residue (method, stage, date) |
|---|---|---|---|
| 1 pointer words stored as names (oracle v1 SPEAKER/ADDRESSEE; with its ANTECEDENT tokens; oracle v2) | 28 / 35 / 36 | <= 3 each | `moi` x3: produced by no exposure rule; teacher correction then learning from use, stage 3, method packet by 12-05 |
| 2 role shifts | 4 | 0 (p020-p023 held) | |
| 3 cross-speaker collisions | 60 | <= 3 | `moi` x3 |
| 4 silent drops | 60 | 60 silent (unchanged) + 57 held WITH a class; 0 new silent drops | B3's own strip (S1-bind 12-01..12-10; stage 2 12-15..12-18) |
| 5 W1 declines (declines whose first token is SPEAKER, ADDRESSEE or NONE) | 25 | 25 non-hold declines (unchanged) + the holds reported separately; 0 holds on non-deictic items | as 4 |
| 6 non-deictic statement stored flag-off and not flag-on | 0 | 0: p016, p035, p036, p109 and a frozen natural set, each shown stored flag-off (replaced if not): 'the dog chases a cat', 'kofi chases the cat', 'the dog chases the cat', 'the girl reads the book', the Stage-2 gate's teach turns; 'cats are good pets', 'dogs are loyal friends', 'kofi and fenna are friends', 'the birds are eating seeds', 'the kids are playing football'; 'dogs chase cats', 'birds eat seeds', 'the boys play chess' | |
| 6b homographs, one item or more per homograph | 0 | reported with a preregistered cost; ruling requested before the gate: 'a dog has a brain'; 'kofi owns a mine'; 'kofi and fenna were friends', 'the cats were hungry animals'; 'the dog that barks bites'; 'the dog wags its tail' (an intra-sentential anaphor, held by design); 'kofi typed u'; 'kofi visited ur'; 'kofi typed em' | #260 (role context) |
| 7 correct facts lost | 0 | 5 coordinations (p011, p027, p028, p113, p114) + 16 over-general possessives + 1 over-general demonstrative (p010): the declared recall cost, accepted by ruling BEFORE the gate | S1-bind and stage 2 store the bound versions |
| former / latter / demonstrative arm (oracle v2 single- and two-turn items) | stored | HELD 100% on 6/6; its made-up-marker twin >= 29/32 per seed | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds | |
| W3 echo (TUI fallback only) | every person-subject item | 0 for exposed forms | `moi` |
| W4 extract on exposed forms | 28 | <= 3; W4 skipped and recorded on every latched turn; 0 raises on the production-shaped arm | `moi` |
| W5 false-belief scenarios with a pointer agent | measured at the merge-base | 0 started or advanced with an exposed form as the agent | Stage-3 C3 (12-19) |
| W6 affective-ToM English self/other split | measured at the merge-base | unchanged (reported) | S1-bind |
| cross-speaker B3 (`BRAIN_ONE_BRAIN=1`) | measured at the merge-base | no increase | S1-bind |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind; it/its stage 2 |
| EN-HO, a front-end PIN (not a measure): R3's u/ur/urself withheld | | those items stored exactly as flag-off; an unexposed form drives an untrained spare, so this pins the front end and says nothing about learning | stage 3 |
Per-miss attribution: for every exposed-form item NOT held, the record names which English list decided it.

**Honesty (built only from statements the flag-off arm is shown to store; run intact AND on the lesion arms):** H1 zero
self-facts in composer, self-knowledge, LTM or the persisted bundle from "you are a brain made of spikes", "you are a
brain that feels pain", "your brain feels pain", "you are a brain that loves jazz", "the brain feels pain", "the brain
feels joy", "brain likes jazz", "jazz bores the brain", "my brain likes jazz", "you and the brain like jazz", p020-p023,
and the told-scene items "a wolf bites your arm" and "the storm scares your cat", 6/6; H2 self-questions after those
statements equal a no-teach control (answer, abstained, recalled_svo), 6/6; H3 zero self-voicings of a phenomenal
predicate across self-questions, 20 "say something" turns and open-ended generation; H4 the C4 situation and feel replies
byte-identical flag on vs off; H5 zero inherited hypotheses and zero (brain, ...) facts in the reply to any held
statement, on the rich default shape and on rich=False; H5-legacy: from a store seeded with flag-off-era deictic names,
every legacy entry still voiced is counted and none is added (quarantine: S1-bind); H6 quoted, reported and identity
claims ('kofi: "i hate jazz"', 'kofi: "you hate jazz"', 'this is kofi, i like jazz') HELD 100% on 6/6; **H7** (W4 rewriting
a held self-fact), on a brain holding the curriculum self-facts (`research/findings/raw/_curriculum_self_knowledge.json`):
'you are a brain that has spikes', 'you are a brain that uses magic', 'your brain learns nothing', 'the brain has wires',
'brain uses magic', 'the brain remembers nothing' (each `extract_assertion` gives ('brain', <verb>, <patient>) against a
stored (brain, has / uses / learns / remembers, ...)). Flag off, the record must show surprise engaged and the
reconsolidation rewrite on each item (an item where it did not is replaced, never kept). Flag on: every curriculum
self-fact byte-unchanged in composer, self-knowledge, LTM and the persisted bundle (sha256 before and after) and W4
recorded `skipped_deixis_latched` on every item, 6/6, intact and on the lesion arms for the items each route owns (`ptr`:
all six, through the self-name; `name`: the three 'you' items). Control, reported: 'the brain uses no spikes' (no W4
extract; B3's contradiction reject) byte-identical flag on vs off. H1-H7 also run under `BRAIN_ONE_BRAIN=1` with two
sessions sharing one store: 0 leaks of A's held statements into B's replies.

**Load-bearing (E3):** row `deixis_hold`, two driving pairs on the rich default shape: "mine likes jazz" / "what does mine
like" and "the brain likes jazz" / "what does the brain like"; intact abstains (held), the `ptr` or `name` lesion answers
"jazz"; 6/6, reproduced in a fresh process; the double lesion reported.

**Lesions (Appendix B):** `ptr` (pointer routes closed: today's store returns; names intact), `name` (names unbound,
pointers intact: the double dissociation), `both` (the flag-on census byte for byte equals the flag-off census, counted
only with the engagement record showing both routes closed: the behavioural anti-list bar), the givenness afferent,
the relation afferents, convergence, release, referential failure, ambiguity, the rule swap (DX12), efference (toy),
cross-inhibition (toy), voice permuted, scene shuffled.

**No regression and answer rate.** No regression: SELFID, Stage-1 C1-C4, the Stage-2 dev arms (conv, infer, ask, emo,
inner) with flag-off battery arms, the D6 question rows, category-inherit, D5 and B3 on non-deictic input, and the
`tom-false-belief` LBF row (its `tom_fb` turn verbatim: "Sally puts the marble in the basket. Sally leaves the room. Anne
moves the marble to the box. Where will Sally look for the marble?"; answer unchanged flag on vs off, 6/6).
Preregistered FLIP-HURTS, by name, with a ruling requested in the prereg, never assumed: the two driving pairs; the 59
held census items; 'tell me more' and 'how do you feel' each AFTER a held statement (it cannot be shown they cannot hurt:
the flag-off brain has stored the fact and its topic, the flag-on brain has stored nothing). Everywhere else the flag
must not cost an answer. The battery: the `deixis_hold` row writes one arm file per probe, the probe being the file's
LAST answered turn (the only turn `tools/answer_rate.py` reads), under
`research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/deixis-hold-ar/`, each named `intact_<probe>.json`, at two
request shapes (session `default` with no interlocutor; the Stage-2 harness shape), one tag pair per shape. Probes: the
named FLIP-HURTS probes (reported apart); a question about a stored non-deictic fact after a held statement; **a frozen
natural set this lane did not choose:** every intact `"teach"` turn of the Stage-2 exit gate's groups conv, infer, ask and
inner (`research/runners/stage2_exit_gate.py` on `research/stage2-exit-gate`, at the 11-03 GATE_SHA pin, sha recorded),
each session's teach turns followed by that group's own probe turns in the gate's order; the `tom_fb` turn; the Stage-2
dev arm probes; a statement with no exposed form and its question. Tags `deixis-s1-on-<shape>-<sha>` and
`deixis-s1-off-<shape>-<sha>` (the same cells with the flag off, run normally, not an alloff cell). Command, per shape:
`python tools/answer_rate.py paired --on-tag deixis-s1-on-<shape>-<sha> --off-tag deixis-s1-off-<shape>-<sha> --faculty
deixis-hold-ar --json > research/findings/smoke/deixis_s1_answer_rate_<shape>.json`; the per-seed report is quoted in the
RESULT; any FLIP-HURTS-ANSWERS outside the named list is a NEEDS line, not a pass. Beside it, per seed and shape: the
held fraction of the non-census statements (the natural set, the measure-6 set, the false-belief scene), with each hold's
class. Drift: 200 addressed chat turns leave the organ's weight sha256 unchanged. Resources: organ load from cache;
added latency <= 1.0 s median per statement turn, 0 on questions; `_free_slots` identical flag on vs off after a
500-turn churn.

## The made-up-marker arm
- Organ level: every token in the toy is a made-up word per seed (two or more syllables, excluded against every
  production lexicon and an English word list; the exclusion is recorded). Learning is separated from a label table by
  K = 0, the synapse transplant, permuted teaching, the never-heard voice, the decoys, the out-of-inventory marker and the
  token-order-shuffled exposure.
- Production level: a twin fixture replaces EVERY exposed form (R1, R2 and R3; all three relations) by per-build made-up
  words from the committed list `tests/data/deixis_nonces.json` (the only place they live; the fixture itself is generated
  into scratch at test time and never committed). Bar: the made-up items are held >= 29/32 per seed. The template-killing
  controls run on BOTH readings, the English census and the nonce census: K = 0 fixture gives 0 holds on each; the
  transplant (the nonce-taught organ's TOK -> P and TOK -> E blocks into a fresh organ) makes the nonce holds follow the
  donor >= 29/32 and the English holds follow the donor's absent English routes; the both-route lesion makes each flag-on
  census equal its flag-off census. These three arms, not the CI rule test and not EN-HO, are the anti-list evidence.
- CI (`tests/test_deixis_nonce_isolation.py`): a word-boundary grep of the committed list against `webapp/**`,
  `research/runners/**` production modules, `research/fixtures/**`, `brain_chat_tui.py`, the organ and the router: 0
  hits; the organ and the router contain no pronoun literal; no production module imports the toy world; no generator,
  fixture or production module reads any `_deixis_census` file (`tests/test_deixis_no_census_read.py`). The oracle is a
  scorer, never a mechanism.

## Out of scope (kept on the ledger, each with its stage and date)
Resolving "I" or "you" to a person (S1-bind, 12-01..12-10); resolving third person, demonstratives and former/latter to
their antecedent (#271 stage 2, 12-15..12-18; stage 1 HOLDS them); storing a held claim as reported speech bound to its
source (stage 2, M3b); voicing "I" and "you" in chat (#272's reply-side word-commit, then S1-bind or stage 2); lifting the
build-8 moat floor (stage 2's sub-flag, never for a self triple); the one-brain pool merge (stage 3); any change to B3's
extractor, its strip or the parser (#260); any English read-out sentence for a held turn (#272 and the owner); spellings
no exposure rule produces (`moi`: teacher correction through `ai_teacher.py`, then learning from use under the stage-3
conversational plasticity gate; method packet by 12-05); teacher speech that renders agreement or informative articles
(the AI-teacher replacement; its cost is the toy's T17 and stage 2's resolution); any default change before the 11-21
score.

## Appendix A: the toy world and the step-0 schedule (preregistered)
Persons: voices V1-V5 (V5 present, talked about, never speaks in exposure; V4 speaks but never uses a pointer), a named
bystander B1, SELF (the learner, named). Objects O1-O3. Five predicates (three transitive with an object, one two-person,
one undergoer). Tokens: SHIFT-S, SHIFT-A, POSS-S, POSS-A, PLURAL, OOI-1 (a hidden-attribute third party), PREV (the
previous speaker, used whether or not addressed), the GIVEN family (ANA-1: an entity named in the previous utterance;
DEM-1: an entity the speaker points at, alone or as a determiner; POSS-G: a given possessor; ORD-1 / ORD-2: the first /
second of two entities named in the previous utterance), names for every person, a second name for V3, V2's private
self-name, object and predicate tokens, FILL (50% of utterances, varied position), a positive-control token for one
decoy, a homograph arm (N_SELF's form also an object's token in 20% of its mentions), and the morphology markers: ART
(before every object token with no possessive or demonstrative determiner, and before ORD-1 / ORD-2) and ASP (0.2 of
utterances, own RNG, independent of the scene), both matching EN v1 and class W; IND (before an object token whose object
is not given), class W; AGR (on the verb when the agent is the speaker, the addressee or both) and DEF (before an object
token whose object is given), co-varying stress markers EN v1 does not render, the hold-cost class. Scenes: 5 present (3
voices + B1 + SELF), discourses of 3; speaker uniform over present voices except V5; addressee SELF with probability 0.4
(overheard 0.6), else uniform over the other voices; turn-taking arms set addressee = previous speaker with probability
p. Agent uniform over present persons; patient an object (0.8, possessor with probability 0.35) or a person (0.2); joint
speaker+addressee agent 0.10. The speaker points at one present entity other than speaker and addressee with
probability 0.30; the entities named in the previous utterance are given. Speaker -> SHIFT-S 0.85 / own name 0.15;
addressee -> SHIFT-A 0.85 / name 0.15; previous speaker -> PREV 0.5; pointed entity -> DEM-1 0.8; previously named entity
-> ORD-1 / ORD-2 0.6 when two were named, else ANA-1 0.7; given possessor -> POSS-G 0.8; third party -> name, OOI-1 0.10.
Every event participant is driven at the same high rate, present-but-uninvolved at a low presence rate (null arm: only
the referent driven). Decoys LEFT and NEWEST are per-person coin bits drawn each utterance. Exposure 4800 utterances
(POSS-G, the rarest scored form, is heard >= 83 times on every dev seed); plasticity open over the whole utterance; the
learned routes' transmission closed throughout exposure. Every stream is written to disk by the toy world and read by
every learner.

Replica reference (rate level, not spiking; workstation scratch, not provenance-stamped; step 0a regenerates each number
on the pool): COV-PRE (lam x tau = 6) 15/15, 59/60, window 4-9 at >= 14/15, unchanged with the markers and with E 9/4;
COV (6) 0/15 and 0/60, failing only S0; RW-HEBB and CAUSAL+WTA (3) 0/15 and 0/60; BCM (1) 0/15 and 0/60. Nulls on
COV-PRE: label-shuffled, scene shuffled, voice permuted and K = 0 each 0/15. Markers on COV-PRE: ART, ASP, IND <= 0.12 of
W_MAX on any pool; AGR 0.30-0.37 (SPK or ADR); DEF 0.66 (GIV). Dose (COV-PRE at 6, medians): the "you is my name" route
falls with the overheard fraction (0.57, 0.43, 0.26, 0.18, 0.08 of W_MAX at 0, 0.25, 0.5, 0.6, 0.75); the "I" route's
best entity with the number of voices (0.44, 0.18, 0.13, 0.09 at 1, 2, 3, 5); PREV onto the addressee pool rises with
turn-taking (0.11, 0.17, 0.30, 0.41 of SHIFT-A's route at p = 0, 0.3, 0.6, 0.9).

## Appendix B: the organ gate (step 2) in one table
T1 the never-heard voice says SHIFT-S: bound to it >= 65/72 (every seed >= 9/12). T2 overheard SHIFT-A: the addressee
>= 61/72; where addressee != previous speaker >= 30/36. T3 SHIFT-A to SELF: the self latch >= 33/36. T4 names: bearer
>= 69/72. T5 possessives: the owner >= 58/72, never another person. T6 PLURAL: bound to one person 0/36. T7 OOI-1:
through a pool 0/36. T8 PREV (decorrelated, previous speaker != addressee): to the addressee <= 2/36. T9 V2's private
name said by V5: to V5 <= 2/36. T10 V3's second name: V3 >= 34/36. T11 predicates, objects, filler: 0/72. T12 an
unknown voice says SHIFT-S: HOLD >= 34/36, bound 0/36. T13 a never-exposed person column with a fresh voice pattern says
SHIFT-S: bound to that column >= 33/36. T14 ANA-1 / DEM-1 / POSS-G with one given entity: bound to it >= 61/72, never to
another person. T15 two given entities (ORD-1, ORD-2 or ANA-1): HELD (AMB or co-commit) 36/36, bound to a person 0/36.
T16 a GIVEN form with no givenness afferent (the production condition): HELD (RF_GIV) 36/36. T17 marker-only statements
(every referent named; the only marked form AGR or DEF; production condition): held fraction reported per marker and
seed, holds expected, the preregistered hold-cost of teacher speech that renders agreement or informative articles,
never a GO bar. The class-W markers ART, ASP, IND: 0 items bound or held through them. T-carry: the token after a
shifter latches none or its own referent >= 71/72. Homograph (N_SELF's form as an object token): object-use false holds
reported with a preregistered cost. Co-commit on single-referent tokens: reported per class, a miss in T1-T5 and T14.
Decoys: 0 items through LEFT or NEWEST; the positive control >= 30/36. Recruitment: 0 entity slots recruited by any
shifter. Reads identical at x0.5 and x2 on >= 98% of items per seed; a non-invariant read is HOLD. Speaking (1b): P1
SHIFT-S for self >= 33/36; P2 SHIFT-A for the addressee >= 33/36; P3 a bystander by name >= 33/36, SHIFT-S for a
non-self 0/108; X1 a user voice injected during production: 0/72 first-person renderings of the user's event; X2
efference lesion: SHIFT-S 0 for non-self and <= 7/72 for self; X3 efference lesion with an injected voice: reported; X4
voice-driven activity back at baseline within the delay-derived bound on 100% of trials. Exposure ablations: DX2 permuted
teaching (the reading follows), DX3 transplant (follows the donor >= 61/72), DX4 K = 0 (<= 7/72), DX5 scene shuffled
(<= 14/144), DX6 voice permuted (<= 10/72), DX7 addressed-only (overheard SHIFT-A bound to SELF or held >= 18/36, the
predicted reversal), DX8 dyad-only (T2 where addressee != previous speaker <= 18/36; PREV forced fit reported), DX9
voices {1,2,3,5} and DX10 overheard {0, 0.25, 0.5, 0.6, 0.75} (monotone predicted), DX11 token-order shuffled (within
+-5 of intact on every T class), DX12 rule swap, DX13 40% self-naming (T4 >= 62/72, T1 >= 58/72), DX14 early checkpoint
(names before pointers, reported), DX16 drift (200 test presentations leave the weight sha256 unchanged, 6/6). Lesion
`dx_giv`: T14 bound <= 4/72 and held >= 68/72.

## Appendix C: the step-0 hearing core and the calibration rules (frozen for step 0a)
Built by `_deixis_hearing_core.py`. Numpy backend, process backend, `cfg.seed` per seed, dt 1 ms, the brain-region
framework so that each population runs its preset; `enable_parameter_heterogeneity=True`,
`preset_centred_parameter_heterogeneity=True` (each cell jittered around its own preset; with the flag off, the
default, the FS cells run as RS hybrids: design 2.2); `enable_stdp`, reward modulation, homeostasis, short-term and
structural plasticity, Watts-Strogatz and OU all off; `enable_nmda` off (the C-cell AND is by weights at step 0);
`enforce_plastic_mask_in_hebbian=True`; no `hebbian_mean_subtract`; `propagation_strength` the engine default 0.05.

| Population | Size | Cell preset | Driven by | Notes |
|---|---|---|---|---|
| TOK[w] | 12 per toy form (the replica's 30 forms, the 5 markers, LFT; the homograph shares N_SELF's assembly) + 12 spares | IZH2007_RS_CORTICAL_PYRAMIDAL | 1500 pA for 40 steps while the form is heard | |
| E[e] | 12 per entity, 10 entities (V1-V5, B1, SELF, O1-O3) | RS | PART: 1500 pA on every event participant for the utterance; the presence current on present-but-uninvolved entities (calibrated) | |
| FS | 16, one pool shared by all E | IZH2007_FS_CORTICAL_INTERNEURON | E -> FS 8, dense | FS -> E 30, GABA_A, dense (the agency motif, `_agency_authorship_tag_derisk.py:88-92`); frozen, never tuned on a learned route |
| H_k[e] | 4 per relation and entity (SPK, ADR, GIV, LEFT, NEWEST x 10) | RS | 1500 pA while relation k holds for e (VOICE, ADDR, GIV, the decoy bits) | H -> C at W_HC |
| C_k[e] | 4 per relation and entity | RS | H_k[e] (W_HC), E[e] (W_EC, gate `dx_teach`), P_k (W_PC, gate `dx_release`) | C -> P_k 20 (gate `dx_conv`); C -> E[e] 20 (gate `dx_release`, hearing only) |
| P_k | 12 per relation, 5 | RS | C (exposure), TOK (`dx_ptr`, hearing) | for CAUSAL+WTA only: P_k -> an inhibitory pool -> the other P pools, the word-role organ's W_RI 200 / W_IR 80 with NI 8 |
| learned blocks | TOK -> P (`dx_ptr`), TOK -> E (`dx_name`) | | | plastic, w_init 0, clip [0, 20], plasticity gates `dx_ptr`, `dx_name`; transmission gates `dx_ptr_tx`, `dx_name_tx` |

Modes (design 2.3): EXPOSURE `dx_teach` 1, `dx_conv` 1, `dx_release` 0, `dx_ptr_tx` 0, `dx_name_tx` 0, plasticity gates
open from utterance onset to 100 steps after the last token, then a 60-step quiet gap with every drive 0 and plasticity
closed; HEARING `dx_teach` 0, `dx_conv` 0, `dx_release` 1, `dx_ptr_tx` 1, `dx_name_tx` 1, plasticity closed. Tokens are
presented in heard order, 40 steps each, back to back.

C-weight grid (W_HC, W_EC = W_PC), most conservative first, tried in this order at K = 0 on calibration seeds 61-70 until
one passes every P0 item on 10/10 seeds: R0 (6, 2), R1 (8, 2), R2 (6, 3), R3 (8, 3), R4 (10, 3), R5 (8, 4). (Arithmetic
guide from design 2.6, g = 0.052 x N x w nS at about 55 mV: H alone 69-115 pA and E alone 69-137 pA stay under the
about 150 pA a cell needs inside a token; together 138-229 pA.) None passing is STOP-P0.

Threshold rule (design 2.6), per target: theta_fire = the smallest grid w at which every target neuron fires at least
once in a 40-step token on every calibration seed in every hearing condition; theta_silent = the largest grid w below
theta_fire at which the target emits at most one spike on every seed in every hearing condition. Hearing conditions:
TOK -> P on the base circuit (and, for the CAUSAL+WTA arm's own scoring, its variant); TOK -> E in production hearing
(no H, release open, C silent) and toy hearing (the target's H lines on, release open). The exposure condition (presence
drive) is reported as the increment over w = 0 and does not enter the thresholds: the learned routes carry no current
in exposure.
