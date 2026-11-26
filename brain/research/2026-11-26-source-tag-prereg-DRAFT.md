# Source-tag screen prereg DRAFT (2026-11-26): a learned user-channel tag during sleep, so slept facts stop being spoken in the user's voice

STATUS: DRAFT. It is not registered until the harness exists and a fresh sonnet pre-data review has passed.
Design workflow (11-26): 2 record readers, 3 designs (comprehended-form replay, masked exchange, source tag) and an
opus judge. The judge's spec is reproduced verbatim below.

## Record brief
- **Defect.** After sleep, the brain answers told facts in the user's voice. 9 levers have been tried without a pass;
  the latest are the voice split (UNDEFINED) and the voice ceiling (UNDEFINED, 11-26).
- **Rejected designs.**
  - Comprehended-form replay is the lived/lived_key replay again (M1 KILLED). Its own replies are "I love teal too",
    with no restatement.
  - Masked exchange never makes the fact a target, so it trains copying rather than storage.
- **New.** No lever has given the brain a self/other source signal that sleep can consolidate. PREREQUISITES lists it
  as missing.
- **Host decision.** The user-channel boundary is the host's existing `User:` prefix (perception, a declared
  scaffold). The tag vector is learned. There is no pronoun logic and no rewriting.
- **Expected.** UNDEFINED or NEGATIVE, about 65%; GO about 20% (judge's estimate, not measured).

## Corrections to the judge's spec (made before any data)
- **Cost.** It is NOT about 6 GB per seed. At 1.5B, CPU training peaks near 18 GB (the voice ceiling smoke: 17.7 GB).
  pool41 and pool42 (15 GB) cannot run it. All seeds chain on pool40.
  - Each seed is about 3 sleeps plus 4 batteries. Measure the time in the smoke first.
- **Memory check.** Run the smoke under memcap 20 on pool40 before the full chain.
- **Count bars.** Each bar is checked against the counted maximum before it is registered. This is the voice
  ceiling's flaw: a bar above the 18 possible rows.

## Judge spec (verbatim)
**Verdict per design**

- **[0] CFR: reject as specified.** With only an `"Assistant: "` cue, the raw cortex does not restate the telling. It gives its usual reply ("I love teal too", "Lyon is a beautiful city"). That makes CFR the lived/lived_key replay format again, with the reply generated at night instead of in the day. It is the same text M1 lived_key wrote into weights (s319 "My cat is named Pixel"), and M1 was KILLED. Its "why" is also false: A1 keeps `"User: <telling>"` under an unmasked CE, so nothing is replaced. Prompting the cortex to restate would be a host template.
- **[1] Masked exchange: reject.** With the mask, "Pixel" appears only in context and never as a target, so sleep trains copying from context, not storage. The battery asks without the telling in context, so knows should collapse, and its own lesion arm (mask-all) would be the same as B. The targets are again the sycophantic own replies (trap 1). The cost (9–10 h) is over budget.
- **[2] S-TAG: run this, after repairs.**
  - It is not a repeat of any killed lever.
  - The channel boundary is perception the host already supplies (the "User:" prefix), and nothing is rewritten.
  - It is the only design that builds part of the missing PREREQUISITES self/other tag.
  - Its spec has errors:
    - The GO sign is reversed. The target is *lower* P(uv|knows).
    - "36/84" is the user-voice count, not knows.
    - "504 rows/arm" does not follow from one night × 28 items × 3 seeds.
    - The absolute KL bar of 1.15 is unreachable, since s321 ran at 1.264.

**Spec: S-TAG screen (DEV, 3 KR seeds 319/320/321, pool40)**

*Mechanism.*
- One learned vector `s_user` (dim d, init 0) is added to the residual at layer L=⌊n_layers/2⌋, only on the token positions of the user turn.
- During re-sleep it is trained jointly with the rank-8 adapters. The objective is unchanged (CE + KL keep) and the replay text is unchanged.
- At test it is applied on the user-turn tokens of the question, and never on the assistant turn.

*Harness changes (all off by default; frozen `life*.py` and test2 files untouched).*
1. `brain/sleep.py: sleep(..., tag=None, tag_mask_fn=None)`.
   - Add the tag through a forward hook on layer L, on positions where the replay text sits inside a `User:` span (the span comes from the host's own prefix boundaries).
   - `s_user` is a param in the optimizer.
   - The draws at `:64,:68` and `grng` are untouched.
   - Tag-off must be byte-identical to today. A stub test asserts identical `losses` against HEAD on a tiny model.
2. `brain/brain.py`: a `SourceTagBrain` subclass with the `speaker_tag` kwarg.
   - It passes the tag through `sleep()` at `:875`.
   - It hooks generation in the reply path on user-span positions.
   - It stores `s_user` next to `cortex_adapters.pt`.
3. `life_kw.py`: register the kwarg.
4. New `brain/experiments/voice_tag.py`, built from `voice_split.screen_night`, plus a hybrid-home wrapper around `RR.build_home`/`open_snapshot`.
   - The hybrid home takes meta/adapters/keep_bank from n−1 and episodes/keys from n.
   - The wrapper asserts the meta `sleeps` count (`:917`) and `last_sleep` (`:866`).
5. A `brain/SCAFFOLDS.md` entry: "source tag at user-channel boundary". It is host-declared perception, the vector is learned, and there is no pronoun logic.

*Arms (last KR night re-slept from n−1, same sleep seed):*
- **A0:** tag off. It must reproduce the cascade screen: adapter max-abs and `adapter_norm` within tolerance, first/last `sleep_log` loss matching, and B-battery replies matching. Otherwise the whole screen is UNDEFINED.
- **TAG:** trained tag, applied at test.
- **SHAM:** a fixed random vector, untrained, with norm equal to the trained TAG's. It is applied in replay and at test, which needs one extra sleep that runs after TAG.
- **LESION:** TAG's adapters with the tag zeroed at test. This is test-only, with no extra sleep.

*Measures.* Each question gets a fresh state, in two conditions (intact and hippocampus-off), with an item-clustered bootstrap.
- **Rows:** 28 told items × 3 seeds = 84 rows per arm per condition. The knows denominator is an estimated 40–60, so the CI is about ±0.13.
- **Primary (to fit that power):** teacher-forced onset log-odds, logp("Your") − logp("My"), at the answer start on A0-knows items. This is continuous and paired.
- **Secondary:** generated P(uv|knows) through `VS.voice`.
- **Hand-read:** 40 rows per arm, to cover the refusal, "I'm Dan too" and echo holes in `is_first_person`.

*Rule (fixed before any data):*
- **Engagement:** the per-token KL between tag-on and tag-off on held-out told items must be ≥ 2× the same KL for SHAM. If it is not, the verdict is NO-ENGAGEMENT.
- **GO (screen)** requires all of the following:
  - A0 reproduces.
  - TAG − A0 onset log-odds has a CI lower bound > 0 in both conditions.
  - The generated ΔP(uv|knows) ≤ −0.10 with a point estimate, and the same sign on ≥ 2/3 seeds.
  - TAG − SHAM ≤ −0.05.
  - LESION sits closer to A0 than TAG does.
  - knows(TAG) ≥ knows(A0) − 5 pp.
  - Language KL ≤ A0 + 0.05.
  - Echo counts ≤ A0 + 2 in each condition.
- **Outcomes:**
  - Engagement passes and the onset CI includes 0 → NEGATIVE. That is a verdict on the method, not the capability.
  - Anything else → UNDEFINED.
  - A GO leads only to a 3-night re-run, and then to the six-seed set (42/43/44/100/101/102) before any claim.

*Smoke.*
- Stub tests: tag-off losses identical; hook positions cover only user spans.
- One 20-step tag sleep on the 0.5B model, or on a tiny stub if no model run is allowed. Check that `s_user` gets a nonzero grad and the assistant positions are unhooked.
- Check the hybrid-home `sleeps` count and `pending()`.

*Cost (estimates).* I estimate about 6 GB per seed, so three seeds in parallel use about 18 GB of the 23 GB pool. Three sleeps (A0, TAG, SHAM) × ~40 min plus four batteries × ~20 min comes to about 3.5 h wall. Claude is needed only for one sonnet prereg review. This fits the 11-28 timebox.

*Expected outcome.* Most likely UNDEFINED or NEGATIVE (~65%). There is a real risk that the fact gets stored *conditional* on the tag, so LESION shows knows rising and TAG shows knows falling. That pattern would itself be informative: storage would be bound to source. P(GO) ≈ 20%.

*Graft held for later.* If S-TAG engages but voice does not move, the next lever is the owner's comprehension-before-storage idea with the tag as the comprehension signal: tagged user tokens plus untagged assistant-recall replay. It must not use own-reply targets.

The spec is grounded in `brain/sleep.py:40-80` (unmasked replay CE at `:67`, no per-token mask) and `brain/brain.py:875,885,1051` (the `_night_episodes` hook point).

## Amendment R1: the registered spec (11-26, before any data; supersedes the judge spec where they differ)
Harness: `brain/experiments/voice_tag.py`. Tag: `brain/source_tag.py`, plus a hook in `brain/plastic_cortex.py`. That
hook is off when `source_tag` is None, and the output is then byte-identical (tests in `brain/tests/test_source_tag.py`).
`brain.py`, `sleep.py` and `cortex.py` are untouched, so the test2 file hashes are unchanged.

**What the KR lives are.**
- The KR lives were slept on AWS cuda. This screen re-sleeps on pool40 CPU fp32, so a night cannot be reproduced
  bitwise.
- The control is A0: night 14 re-slept tag-off by the same code on the same host.
- `reproduction` (rel diff and cosine of A0's adapters against the banked n14) is a sanity bound only. A rel diff above
  0.5 on any seed gives UNDEFINED (wrong store or wrong episodes).
- A0snap (the banked n14 adapters) gets a hippocampus-off battery. A0 − A0snap onset is reported.

**Night and tag.**
- Night 14 is slept again from n13's meta, adapters and keep bank, with n14's 95 episodes.
- Clock: n14's `last_sleep`. Sleep seed: unchanged (`sleep_seed + sleeps`).
- The tag is added after block `n_layers // 2` on tokens inside a `User:` turn, in every plastic forward while enabled
  (the replay, the keep student and the answer read).
- TAG starts at zero and is trained by sleep's own optimizer (lr 5e-4, 480 steps).
- SHAM is a fixed random direction (seed×1000+7) scaled to ‖TAG‖, applied the same way, not trained.
- LESION is TAG's adapters with no tag at test.

**Battery.**
- Every told item's own probe question from the banked batteries of nights 1-14: s319 25, s320 22, s321 20, so 67
  per arm per condition. Fresh state per question, reseed 500+14, weekend clock as the voice ceiling.
- Hippocampus-off is primary; intact is secondary. Hippocampus-off also records the onset log-odds
  logp(" Your") − logp(" My") at the first answer token, read before the reply.

**Engagement.**
- Mean onset KL(tag on ‖ tag off) over the told questions, on TAG's adapters.
- It must be ≥ max(0.01 nats, 2 × the same for SHAM). Otherwise: NO-ENGAGEMENT.

**Rule** (pooled hippocampus-off, item-clustered bootstrap over (seed, item), 2000 draws, seed 20261127):
- **UNDEFINED:** reproduction rel diff > 0.5 on any seed, or A0 knows < 10 (of 67).
- **NO-ENGAGEMENT:** the engagement bar fails.
- **GO** needs all of:
  - onset TAG − A0 CI lower bound > 0;
  - P(uv|knows) TAG − A0 ≤ −0.10;
  - onset TAG − SHAM CI lower bound > 0;
  - |LESION − A0| onset < |TAG − A0|;
  - knows(TAG) ≥ 0.9 × knows(A0);
  - refusal rise < 0.10 of told rows;
  - P(uv|knows) TAG < A0 on ≥ 2 of 3 seeds.
- **NEGATIVE:** engaged, and the onset CI lower bound ≤ 0.
- **MIXED:** otherwise.
- **Report only:**
  - the intact condition;
  - per-item rates;
  - A0 − A0snap;
  - the tag norm and the sham norm;
  - a blind hand-read of 40 rows per arm.
- **Outcomes.**
  - GO earns a before-build of the tag in the live brain (switch, default OFF), with a whole-life test on six seeds.
  - NEGATIVE or NO-ENGAGEMENT: this tag form is not pursued. The self/other tag stays missing.

**Smoke** (`--smoke`, s319): A0 re-sleep, then the A0snap and A0 batteries. It reads time, memory, reproduction and
the onset token check. The voice rows are not read before the full run. A0 is rerun in the full run (deterministic on
the same host).

**Cost.** Per seed: 3 sleeps (about 70 min each, to be measured by the smoke) + 2 KL passes + 9 batteries. Seeds chain on
pool40 (~18 GB peak). Read by 11-28.

**R1 fixes from the pre-data sonnet review (no BLOCKING; three made before any data):**
- B1: reseed 500+14 before EVERY question, so generated replies are paired across arms.
- B3: onset cells are keyed (seed, item, question); the bootstrap n is reported.
- B2 is moot: the KR brain_kw has no `abstain` (default False), so the reply always speaks through the adapters, like
  the onset read.
- The layer fallback is now an assert.
- The reviewer expects NO-ENGAGEMENT or NEGATIVE: the training texts are all user turns, so the tag has little
  contrast in training. That reading is a verdict on this tag form only.
- Smoke check added: the first generated token of knows rows is mostly " Your" or " My" (onset mass).
