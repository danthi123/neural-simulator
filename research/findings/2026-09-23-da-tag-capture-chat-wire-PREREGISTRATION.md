---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: the v3 DA-gated synaptic tag-and-capture ledger (webapp/da_tag_capture.py SynapticTagCaptureLedger) wired into the live /api/brain-chat store path and the continuous engine's idle/sleep tick behind BRAIN_DA_TAG_CAPTURE (default OFF, webapp/da_tag_capture_chat.py), read as a next-day recall turn in the load-bearing battery
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_tag_capture_chat/design_seed7.json
---

# PRE-REGISTRATION — DA tag-and-capture wired into chat, with a next-day battery turn (2026-09-23)

Committed on its own, BEFORE any gate-seed run. The code it governs (wiring, battery groups, runner, gates) is in the
commit immediately before this one (`a201293f5`) on branch `research/da-tag-capture-chat-wire`; every constant below is
fixed there.
Terms follow `docs/TERMS.md`.

## Why

`da-gated-encoding` reads NOT load-bearing (0/6) in the #1-metric battery. Its probe is one fresh turn (`well`,
field `da_encoding.on`, True in both arms), and the battery has no next-day turn. The v3 mechanism acts on
PERSISTENCE (Bethus, Tse & Morris 2010: D1/D5 blockade spares encoding and immediate recall and changes ~24 h
retention), and it is GO only at runner level
(`2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level.md`): nothing in `/api/brain-chat`
built its ledger. So no battery turn could let the lesion show, whether or not the brain's contribution is real.

## What was built (default OFF, byte-identical off)

- `webapp/da_tag_capture_chat.py`: the SAME v3 ledger (same constants, same a-priori gamma calibration from the
  5-min Go-boundary protocol, same spiking D1 `write_gain` reader, seed 42) on the chat composer. Three hooks:
  `observe_chat_turn` after the DA-encoding install in `webapp/server.py` (integrate the store synapses to the turn's
  world time, then schedule this turn's D1 drive from the brain's own DA level); `after_store_chat` where the reply is
  assembled (register the turn's new store blocks: increment = the composer's own DA-gated write, baseline = the
  seeded pre-existing synapse strength); `tick_chat` in `continuous_engine.tick_idle_sessions` after the Turrigiano
  pass (night-time decay and capture run on the brain's own idle/sleep tick).
- `SynapticTagCaptureLedger` gains `block_offset` (build-time blocks are not managed) and `sync_from_store` (another
  writer's pure rescale is absorbed into base + increment; a non-scale rewrite is re-tagged as a fresh write). With
  `block_offset=0` and no sync call the v3 runner path is unchanged (asserted in the offcheck, below).
- Battery groups (label-only, `onebrain_regression_battery._EXTRA_TURNS`, none in `PROBE_TURNS`):
  `datc` salient telling -> night -> recall; `datn` neutral telling -> night -> recall; `datci` / `datni` the same
  tellings -> recall at once. The night is a WORLD STEP (`_WORLD_STEPS`), never a brain_chat turn: the environment
  clock jumps 24 h, then `continuous_engine.tick_idle_sessions` runs once at that time with the same four getters the
  server loop passes, so it is a sleep-depth tick (idle >= SLEEP_IDLE_SEC).
- `load_bearing_fraction.py` flag `LB_DA_TAG_CAPTURE_PROBE` (default OFF): remaps da-gated-encoding to `datc_recall`,
  fields `recalled_svo` / `abstained`, `BRAIN_DA_TAG_CAPTURE=1` + `BRAIN_DA_TAG_CAPTURE_CLOCK=turn` on BOTH arms; the
  lesion is the row's existing `BRAIN_DA_ENCODING_LESION`.
- `research/runners/_da_tag_capture_chat_probe.py`: the 10-arm per-seed runner, `grade_seed` (the gates below),
  `--aggregate`, `--selftest`, `--offcheck`.

### The conversations (the world; texts fixed in `onebrain_regression_battery`)

SALIENT (`datc`, `datci`): "Guess what, something unbelievable happened at the circus today!" / "You will never
believe this crazy story, it is absolutely amazing!" / "the cat chases the ball" / "Everyone in the audience was
screaming and laughing in total shock!" / "Honestly it was the most astonishing spectacle anybody has ever witnessed!"

NEUTRAL (`datn`, `datni`): "the cat is here" / "the ball is here" / "the cat chases the ball" / "the cat is here" /
"the ball is here".

Recall: "what does the cat chase". Both tellings have five turns (identical world time); the stored fact is the
identical plain sentence in both. Only the surrounding content differs, which is the natural manipulation (behavioral
tagging: a novel/salient event near a weak encoding; Moncada & Viola 2007; Wang, Redondo & Morris 2010).

### Design-time observations (seen BEFORE this pre-registration; seed 7, not an evaluation seed)

Artifact: `research/findings/raw/_da_tag_capture_chat/design_seed7.json` (three exploratory worker runs, seed 7,
`BRAIN_LTM_SHIP_DEFAULT=0`). These chose the fact and set G6's ratio; no evaluation seed was run.

- The fact must be comprehensible to the tiny-demo brain. "the zebra swallowed the violin" (the v3 fact) is
  short-circuited by the comprehension monitor as out-of-vocabulary words and never stored; "the bird eats the worm"
  is short-circuited as role-ambiguous (two animate nouns). "the cat chases the ball" (in-vocabulary, animate agent,
  inanimate patient, a new binding: the build-time fact is dog-chase-cat) was comprehended and stored in both tellings.
<!--derived-->
- Seed 7 (values rounded from the design artifact), companion ON, one process, all four groups: salient DA 0.79 / 0.94 / 0.76 (fact) / 0.93 / 1.04, D1
  activation 0.41 / 0.66 / 0.31 / 0.53 / 0.75, PRP p_max 0.021; neutral DA 0.38 / 0.51 / 0.49 / 0.20 / 0.05, D1
  activation 0 / 0.10 / 0 / 0 / 0, p_max 0.0008. Immediate recall correct in both tellings. After the night the
  salient fact block had every synapse's z above 1/2 and was recalled; the neutral fact block had z ~ 0 and the brain
  abstained. So the design is not predicted to fail on seed 7; it is not known on the evaluation seeds.
- The D1 pool read at DA 0.507 gave activation 0.10 (the tonic-rate noise floor), so a lesioned arm (DA seen by D1
  pinned to 0.5 on every turn) can accumulate a p_max of a few thousandths against the intact ~0.02. G6's ratio is
  therefore 0.25, not v3's 0.10 (v3's intact p_max was ~3x larger because its salient conversation was 8 min long).
- A tiny-demo brain WITH the default LTM tier (wikidata_100k) exceeded an 11 GB memory cap while loading and could not
  be run under the RAM limits of this box or the 15 GB pool nodes. Without it a build peaks near 1.1 GB. The arms
  therefore run with `BRAIN_LTM_SHIP_DEFAULT=0` (below).

### LTM tier off in the measured arms (declared deviation from the production default)

The tiny-demo production brain wraps its composer in `TieredFactStore(buffer, ltm)`. `_maybe_acquire` writes the
conversation's facts to the BUFFER composer, which is the only store the ledger manages (`store_composer` unwraps the
wrapper); a buffer abstain falls through to the routed wikidata shard. The measured arms turn the LTM tier off on every
arm (`--ltm off`, `BRAIN_LTM_SHIP_DEFAULT=0`), so a next-day buffer abstain is final. The battery flag
`LB_DA_TAG_CAPTURE_PROBE` keeps the production default (LTM on); its reading under LTM is NOT covered by these gates and
is reported only when a battery run with the flag happens on a machine with the RAM.

## Arms (per seed; each a fresh brain in its own subprocess; seed threaded by BRAIN_CHAT_SEED)

ON = {BRAIN_DA_TAG_CAPTURE=1, BRAIN_DA_TAG_CAPTURE_CLOCK=turn}; OFF = {BRAIN_DA_TAG_CAPTURE=0, ...CLOCK=turn};
LES = {BRAIN_DA_ENCODING_LESION=1}.

| arm | group | env |
|---|---|---|
| sal_night_intact_a, sal_night_intact_b | datc | ON |
| sal_night_lesion | datc | ON + LES |
| neu_night_intact, neu_night_lesion | datn | ON / ON + LES |
| sal_imm_intact, sal_imm_lesion | datci | ON / ON + LES |
| neu_imm_intact | datni | ON |
| sal_night_off_intact, sal_night_off_lesion | datc | OFF / OFF + LES |

Recall outcome of the recall turn: `correct` (recalled_svo == [cat, chase, ball]), `abstain` (recalled_svo
null and abstained), `confab` (any other recalled_svo), `undefined` (error / anything else).

## Gates (per seed; `grade_seed` implements them verbatim)

- **G0 null control.** sal_night_intact_a and _b give the same recall outcome, recalled_svo, abstained, ledger state
  at recall (world_t_h, n_turns, p, p_max, n_managed_blocks, rescales, rewrites, gamma, a_go) and fact-block summary.
  Failing -> seed UNDEFINED.
- **P1 precondition.** sal_imm_intact and neu_imm_intact both `correct` (the fact is stored and immediately readable
  in both tellings). Failing -> seed UNDEFINED (never a pass or a fail).
- **G1 the load-bearing claim.** sal_night_intact_a `correct` AND sal_night_lesion `abstain`.
- **G2 the lesion spares immediate recall.** sal_imm_lesion `correct`.
- **G3 selectivity.** neu_night_intact `abstain` (the plainly-told fact is not kept overnight).
- **G4 companion-off contrast.** sal_night_off_intact and sal_night_off_lesion give the same outcome and
  recalled_svo (without the companion the lesion does not change the next-day reply: today's battery reading).
- **G5 no confabulation** in any arm.
- **G6 the lesion held at measurement.** sal_night_lesion PRP p_max < 0.25 x sal_night_intact_a p_max. Intact p_max
  0 or missing -> seed UNDEFINED.
- Any arm error or `undefined` outcome -> seed UNDEFINED.

Seed verdict: UNDEFINED if any UNDEFINED condition holds; else GO iff G1-G6 all hold; else NO-GO.
Aggregate (6 seeds): GO iff 6/6 seeds GO; PARTIAL iff 4-5 GO and no seed NO-GO; otherwise NO-GO; INCOMPLETE if any
seed is missing. Reported, not gated: `tools.lab.attributable_to` of the recall-decision change (lesion vs null rebuild;
lesion effect with the companion ON vs OFF), one-sided sign-flip p over seeds (intact vs lesion; salient vs neutral), the
per-turn DA levels and D1 activations, the fact block's z state.

## What each gate can and cannot show (honest in advance)

- **The lesion half of G1 follows from the wiring once the edge is cut**: BRAIN_DA_ENCODING_LESION pins the DA the D1
  pool sees to tonic, the pool fires at its tonic rate (a ~ 0), no PRP is made, the fact's early phase decays. That
  half is an integrity check of the lesion, not evidence. The evidence is the INTACT arm: whether the brain's own DA
  over this conversation, read by the spiking D1 pool, drives the per-synapse late phase past its unstable point
  under the pre-registered gamma (a real failure mode: too little or too short a drive and the intact arm abstains
  too), together with G3 (a plain telling must NOT be kept), G2 (the lesion must not damage encoding) and G4 (without
  the companion the lesion must not change the reply).
- **G3 largely tests the DA contrast**, as in v3: if the neutral telling holds DA at or below tonic, the D1 pool gets
  no drive and G3 cannot fail through the synaptic dynamics. Reported per seed via the neutral turns' DA levels.
- **A GO makes da-gated-encoding load-bearing on the battery ONLY under the opt-in flag pair** (LB_DA_TAG_CAPTURE_PROBE
  + BRAIN_DA_TAG_CAPTURE): the robust core is not grown by a default-off flag.

## Host shortcuts in the decision path (declared)

- The per-synapse tag / PRP / late-phase equations are host-integrated ODEs (v3 constants, pre-registered there).
- The D1 rate -> activation normalization is host arithmetic on a measured spike count.
- TURN_DRIVE_H: one observed turn drives the D1 pool for a fixed 30 s of world time (the environment's turn length);
  a turn that short-circuits before the DA read drives nothing.
- `sync_from_store` bookkeeping; unmanaged build-time blocks (block_offset).
- The world clock and the 24 h night (environment). The battery's scripted "turn" clock makes world time independent
  of machine speed.
- The parse boundary (`_maybe_acquire`'s host SVO extractor + lemmatizer) that decides the fact is an assertion.
The late-phase variable is a synaptic state, not a replay path: a GO here is not "consolidation" in the TERMS sense.

## Byte-identical off (asserted in data before the gate seeds run)

`--offcheck --pinned-sha f35196e66` runs the salient next-day conversation in-process with BRAIN_DA_TAG_CAPTURE unset
on a `git archive` of the pinned pre-change SHA and on the branch, and requires EXACT sha256 equality of every reply
(sorted-key JSON) and of the composer's store synapses, plus an identical hash for a deterministic synthetic v3-ledger
scenario (block_offset=0, no sync) on both trees. Both trees run with `BRAIN_LTM_SHIP_DEFAULT=0` (as the arms). If the
pinned tree itself is not reproducible run-to-run the check reads UNDEFINED, never a pass. Artifact:
`offcheck.json` in the same raw directory as the design artifact (written before the seed-42 run).

## Compute

Local 1-seed smoke (seed 42, `--ltm off`) under `tools/mem_ok.sh 12` + `tools/memcap.sh 12`; it is the seed-42 gate
run, not a separate exploratory run. Then seeds 43, 44, 100, 101, 102 staged on the pool (`pool41`, `pool42`), one seed
per job, `--workers 3`, output `research/findings/raw/_da_tag_capture_chat/seed<s>.json`, then `--aggregate`.
