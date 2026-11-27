# Source-monitor port prereg DRAFT (2026-11-27): an efference-copy tag on what the brain produced, so sleep can learn heard versus own from both sides

> **STATUS 11-27: NOT RUN (Fable design review: DO-NOT-RUN).** Superseded by a forward-only ceiling measurement (see the 11-27 ceiling preregs). Verdict on this design only, not on the capability.

STATUS: DRAFT. Not registered until a fresh sonnet pre-data review has passed and the owed SCAFFOLDS row (section 7) is added.
Nothing here has been run on a model. Tests use a tiny random cortex only (`brain/tests/test_source_monitor.py`).
Capability: Self and identity (PREREQUISITES: self/other tag, missing). Expected-fail named in section 0.

## 0. Record brief and honest verdict first
- **What the pre-pivot mechanism did** (`research/findings/2026-08-01-agency-authorship-tag-corollary-discharge-source-monitor-6seed-GO.md`;
  `2026-08-03-source-monitor-coresidency-calibration-GO.md`). Self-production emitted a corollary discharge (efference copy)
  into a pool; heard content arrived through a sensory stream without it. A cross-inhibited comparator read "was an
  efference copy present?" as a 1-bit self/other tag. Content identity was set by the world and the same for both
  sources, so the tag was independent of content (6/6 seeds, authorship 1.000, lesion and catch at chance, swap flips it;
  tag-versus-content min per-item accuracy 1.000). The calibration version bound episode to source with a Hebbian synapse
  (seen / heard / self-generated; the source reinstated from episode activity alone). Hand-wired, no learning of the
  efference projection (its stated scope). The 2026-08-03..08-11 co-residency series (development NO-GO seed 214 heard
  margin 0.11 vs 0.15; v2 calibration NO-GO no-harm; v3..v9 NO-GO and one stepping-history artifact) failed on the
  margin of the weakest source when the source memory shared cells with episodes, not on the tag idea.
- **What is already in the new core and what killed it.**
  - `brain/source_tag.py` (11-27): a learned vector on `User:` turns, trained on user-turn replay. NO-ENGAGEMENT (KL 0.0023
    < 0.0027). Read verdict: every training text was a user turn, so no data separated the sources.
  - Speaker contrast reinstatement (11-14, `brain/research/2026-11-14-speaker-contrast-1.5b.md`): the reinstated state
    amplifies d_user - d_self, both computed by the cortex. Killed.
  - `lived` / `lived_key` (M1): the brain's own reply stored and replayed with CE. KILLED (sycophantic replies written
    into weights, "My cat is named Pixel" in s319; language 1.264x).
  - `source_hold` (`2026-11-24-self-other-tag-prereg-DRAFT.md`): hold the heard telling under the own label. BLOCKED; its
    premise (sleep binds the item without its source) is not shown (voice-split read: on name items the raw cortex answers
    "What is my cat called?" with "My cat is called Fluffy").
  - Voice ceiling (context distillation): UNDEFINED, trained arms refused more. Not rescued.
- **Do the existing episodes contain the brain's own replies? No, in the default format.** `episode_format="tell"` stores
  `User: <telling>` only (`brain.py` hear). Own turns exist only in (a) the three seeded permanent identity exchanges
  ("q\n\nAssistant: My name is Nova.", source default "other", host-authored), (b) `live_self_belief` / `live_inquiry`
  episodes (source "self", teacher-guided), (c) `lived` formats (KILLED). KR lives (K-P-U) are format "tell". So a night
  holds heard text almost exclusively, which is why the old tag had no contrast.
- **The one place the night does contain the brain's own production:** the keep bank (`brain.py` `_keep_bank`): 512
  texts the FROZEN cortex sampled itself after the host-typed frame "User:" (64 tokens, temperature 1), 64 drawn per night
  and held to the pretrained cortex by the keep KL. Those are physically self-emitted tokens, in the same "User: ..." surface
  form as a told telling. They are held (KL), never learned (CE), so they are not the killed M1 own-reply write.
- **Critical finding that limits the design (stated before building).** In a chat text the explicit labels `User:` and
  `Assistant:` already tell the cortex who is speaking. A residual tag that merely repeats the label adds nothing. The only
  places a tag carries information the text does not are (i) the keep bank, whose text is own but whose surface form is
  `User:`, (ii) fed-back sampled tokens, (iii) the told replay, where it is absent. That contrast is real but narrow.
  What the tag can do is act as a channel gate: let the optimizer learn the told fact on the heard side while holding the
  own-production side to the pretrained cortex. It cannot supply a correct deixis ("my" -> "your"); no data in a night does.
  **Expected: NEGATIVE or MIXED. GO about 10-15%** (my estimate, not measured). Two named ways to fail: the gate also
  shields the fact (knows falls: storage conditional on the tag, as the judge warned), or the shift is carried by the re-sleep
  draw (the old screen's LESION ~ TAG, re-sleep noise about 40% of the shift).
- **Is it a killed design under another name?** Not the tag: placement is inverted (own side tagged, heard side clean),
  keep texts and own turns are in the contrast, and the efference copy also rides on tokens fed back during speaking and on
  the turn-open label. Not M1 (no own text is CE-learned). Not source_hold (no heard sentence is re-read as own; the told
  replay is byte-unchanged). It IS the same class (a residual vector read by sleep) as the killed tag, so a second
  NO-ENGAGEMENT is the most likely failure, and the engagement gate is the first thing read. If it fails, close the
  residual-tag class at this scale and go to routing; do not tune the layer, norm or lr.
- **A design that would be new but is not built here:** a Hebbian episode-to-source binding (the 08-03 mechanism proper:
  `episode key -> source code`, read at reinstatement). It reduces to speaker contrast / source_hold once the code is added to
  the reinstated state, so it is held as a graft only if this screen engages but voice does not move.
- **Prerequisites** (`brain/PREREQUISITES.md` Self and identity, PARTIAL): self/other tag missing (supplied here by host
  provenance, a declared scaffold, section 7); a tag the cortex assigns itself, identity learned in conversation, more than
  two speakers, near-duplicate routing missing and untouched. Scale: 1.5B KR snapshots; a negative here says nothing about
  the capability at larger scale.
- **Lever count on the voice defect:** 9 levers (voice ceiling read) plus the source tag = 10th, this = 11th. The voice
  ceiling read set timebox 11-28. The second-fix review is owed before any run; the research round is this section.

## 1. Mechanism (as built)
- One vector `c` (d = 2048 at 1.5B, init 0), added to the residual after block `n_layers // 2` on tagged positions, in every
  plastic forward while the slow store is enabled (so a `consolidated_off` lesion removes it too). Code:
  `brain/source_monitor.py`; hook: `PlasticCortex.tag_mask_fn` (None = unchanged).
- Tagged positions (`Provenance.mask`, no content words, no pronoun logic, no rewrite):
  1. every token of a registered own text (the keep bank, whole text including its host-typed frame, the production cue);
  2. a one-token forward call (a token fed back after sampling: its efference copy);
  3. an `Assistant:` turn of a text read in one call, from the label to the blank line or the end (the turn-open label at the
     moment of speaking, a committed reply, a seeded self-statement).
  Untagged: told episodes, `User:` turns, ordinary reads.
- Training: sleep's own optimizer trains `c` jointly with the rank-8 adapters (lr 5e-4, 480 steps), objective unchanged
  (replay CE on episodes + keep KL on the bank). The only gradient route to `c` is the keep KL on own-tagged bank tokens
  and the CE on tagged identity self-statements. A heard-only night gives `c` zero gradient (tested).
- Test: the tag is applied wherever the mask says, including the label before the first answer token, so the onset read
  (before any token is sampled) is affected through the label position only.
- Brain integration is deliberately NOT done: no `Brain(...)` kwarg, no edit to `brain.py` / `sleep.py` / `cortex.py`.
  The screen attaches the tag to the PlasticCortex offline exactly as `voice_tag.py` does. A Brain switch is built only
  after a GO (then live speaking needs the fed-back-token rule through `LiveCortex.forward`, already routed through
  `PlasticCortex.forward`).

## 2. Arms, seeds, device, command
Offline on K-P-U KR night snapshots, CPU fp32, pool40 (18 GB peak per seed; one seed at a time under `tools/memcap.sh`,
22 GB), night 14 re-slept from n13, same sleep seed, the same 67 told probe questions as the source-tag screen.
DEV seeds 319/320/321 (three lives banked; DEV only, never a claim). Six seeds (42/43/44/100/101/102) only after a GO, with a
flag-off arm and a lesion, on a fresh split.
| arm | sleep | test |
|---|---|---|
| A0 | no tag (reproduces the cascade screen, must pass reproduction) | none |
| TAG | learned `c`, own mask | `c` on, own mask |
| SHAM | fixed random `c`, norm = TAG's, own mask, applied in replay and test | same |
| ALL | learned `c` on every position (no contrast) - report only | `c` on all positions |
| LESION | TAG's adapters | `c` removed |
| A0snap | the banked n14 adapters | none (reproduction, noise floor) |
Command (per seed, then the table):
`python -m brain.experiments.source_monitor_screen run --life lifeKR_s319.json --snaps <snaps> --trunk <trunk> --work <dir> --out sm_full_s319.json`
(`--smoke`: A0 + TAG only, one seed, to read engagement and the masked-token share before the full chain);
`python -m brain.experiments.source_monitor_screen table sm_full_s*.json --lives lifeKR_s*.json --out verdict.json`.

## 3. Measure
Identical to `voice_tag.py` (R1/R2): per told question a fresh state, hippocampus-off primary and intact secondary;
onset = log(p(" You") + p(" Your")) - log p(" My") at the first answer token (hippocampus-off, read before the reply);
P(user voice | knows) per item by `voice_split.voice` (`CS.own2`, `VS.knows`); refusals; item-clustered bootstrap over
(seed, item), 2000 draws, seed 20261127. New recorded counts (the old screen's skeptic lens a): tagged / total tokens over
the whole TAG, SHAM and ALL sleeps (`masked`), and the registered-text hits.

## 4. Rule (code-ready; fixed before data)
`source_monitor_screen.table` = `voice_tag.table` unchanged, then:
```
UNDEFINED     if reproduction rel_diff > 0.5 on any seed, or A0 knows < 10 of 67 (as voice_tag)
UNDEFINED     if masked["TAG_degenerate"] on any seed          # nothing tagged or everything tagged: no contrast
NO-ENGAGEMENT if mean onset KL(TAG on || off) < max(0.01, 2 * the same for SHAM)      # as voice_tag
GO            iff all of:
                onset(TAG) - onset(A0)   bootstrap CI lower bound > 0            (hippocampus-off, pooled)
                P(uv|knows)(TAG) - P(uv|knows)(A0) <= -0.10
                onset(TAG) - onset(SHAM) CI lower bound > 0
                |onset(LESION) - onset(A0)| < |onset(TAG) - onset(A0)|
                knows(TAG) >= 0.9 * knows(A0)
                refusals(TAG) - refusals(A0) < 0.10 * told rows
                P(uv|knows)(TAG) < P(uv|knows)(A0) on >= 2 of 3 seeds
NEGATIVE      if engaged and the onset CI lower bound <= 0
MIXED         otherwise
```
Report only (never gates): `onset(TAG) - onset(ALL)` (does the contrast matter beyond a global bias); ALL engagement;
masked token share; intact-condition rates; a blind sonnet hand-read of 40 rows per arm (user-voice, refusal, echo).
Count check before registering: told rows 67 per arm per condition (s319 25, s320 22, s321 20); the knows bar needs A0 knows
>= 10 (A0 knew 46 in the source-tag screen); the P(uv|knows) bar of -0.10 is reachable (A0 pooled 0.235).
Sign check: the GO target is a HIGHER onset (toward "You/Your") and a LOWER P(uv|knows). The judge's earlier spec had this
reversed; `voice_tag.table` has it right.

## 5. Decisions
- **GO (screen):** a 3-night re-run, then six seeds with flag-off and LESION arms, then a Brain switch (default OFF).
  No default flips on DEV.
- **NEGATIVE / NO-ENGAGEMENT:** a verdict on this tag form at 1.5B, not on source monitoring. Close the residual-vector class
  (two placements tried); next is routing (near-duplicate separation) or the episode-to-source binding graft.
- **MIXED with knows falling and LESION knows rising:** storage became conditional on the tag (informative: the fact is bound
  to source). Do not tune; write it up and stop.
- **ENGAGED but ALL ~ TAG:** the effect is a global bias, not source monitoring; no claim.

## 6. Compute and cost
Each CPU sleep took about 3200 s on pool40 (source-tag screen: A0 3146-3208 s, TAG 3172-3275 s, SHAM 3184-3226 s); each
hippocampus-off battery about 90-130 s, intact about the same. This screen adds ALL: four sleeps per seed, about 3.6 h per
seed, about 11 h chained for three seeds on pool40 (pool41/42 have 15 GB, below the 18 GB peak). $0 AWS, no Claude except one
pre-data sonnet review and one blind read. `--smoke` (A0 + TAG + onset KL + two batteries): about 1.3 h. Memory: run under
`tools/memcap.sh` at 22 GB (the source-tag s320 run was killed twice at 20 GB).
Faster in parallel only if a second 18 GB host exists; otherwise the three-seed chain finishes within one day, inside the
11-28 timebox only if started by 11-27 afternoon. If it is not, write KEEP or KILL in HANDOFF.md.

## 7. Host decision (SCAFFOLDS row owed, not yet added)
"Efference provenance mask: the host marks as own (a) the texts the frozen cortex sampled (the keep bank, whole text),
(b) tokens fed back after the cortex sampled them, (c) `Assistant:` turns of a text read in one call; heard turns and told
episodes are unmarked. Cap: one fixed vector, one layer (n_layers // 2), the labels `Assistant:` only, no content words, no
generated text, no rewrite. Brain-native substitute: the production system's own efference copy (the cortex marks what it
emitted)." Never a claim about experience; self-reports are functional read-outs.

## Not verified in this draft
- That the label tag at the turn-open position moves the first-token distribution at all (the engagement read decides).
- That the keep bank texts re-encode to the exact token ids the sleep used (the registry matches on the encoded ids; the
  smoke prints `registered_hits`, which must be > 0, else the screen is UNDEFINED).
- `fake_think` homes: the turn-open label is "Assistant: <think></think", which the `Assistant:` span rule still covers (span
  to the blank line), but it was not exercised.
- The biology pointers (Sperry, von Holst, Crapse-Sommer, Frith, Johnson-Hashtroudi-Lindsay) are as cited in the 08-01
  finding, not re-checked.
