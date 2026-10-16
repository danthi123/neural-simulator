---
type: finding
status: confirmed
claim_check: measured (research.runners.stage2_exit_gate --aggregate over the 64 committed handler shards, dev seeds 1-8; Claude re-ran the aggregate independently 2026-10-15 and the emo group is identical; research.runners.emotion_category_dev --flip-check over the on/off pairs)
date: 2026-10-15
lane: Stage-2 C4 (discrete-emotion read-out), integration-first (owner 2026-09-29)
mechanism: emotion_category_organ (four appraisal channels on the affect ladder, one spiking commit latches a channel; the feel read-out names it), behind BRAIN_EMOTION_CATEGORY
integration_faculty: emotion-category
seeds: [1, 2, 3, 4, 5, 6, 7, 8]
artifacts:
  - research/findings/raw/_s2_c4_build/handler/aggregate_on.json
  - research/findings/raw/_s2_c4_build/handler/flip_check.json
verdict: C4 PASS 8/8 dev seeds through the live handler; flip check clean on behaviour. BRAIN_EMOTION_CATEGORY is now default ON (=0 is the escape).
---

# Discrete emotion category on the feel read-out: default ON

**What changed.** `webapp/server.py` `_EMOTION_CATEGORY_DEFAULT_ON` is now True (docs/drafts/stage2-c4-build/HANDOFF.md
section 5). When someone asks how the brain feels, the reply keeps the existing functional read of the affect ladder
(sign, number, arousal band) and adds one sentence naming the emotion channel a spiking commit latched, e.g. "The
emotion channel that latched on my affect ladder is fear." It is a functional read-out, never a claim of felt
experience (non-negotiable 5). Before this, fear and anger read identically on the signed ladder.

**Evidence** (`research/findings/raw/_s2_c4_build/handler/aggregate_on.json`, 64 shards, dev seeds 1-8, the
Stage-2 exit-gate runner's own worker over the live `/api/brain-chat` handler, flag on):
- the four situations named joy, fear, anger and sadness correctly on 8/8 seeds;
- the affect lesion (`BRAIN_AFFECT_LESION=1`) changed the NAMED category on 8/8 (Amendment A3's clause),
  reproduced against its own repeat on 8/8;
- the permuted exposure named the swapped categories on 8/8; the null (two independent builds) was clean on 8/8.
- Claude re-ran the aggregate from the committed shards: the emo group is identical to Hermes's file.

**Flip check** (`research/findings/raw/_s2_c4_build/handler/flip_check.json`, 16 on/off pairs): no
FLIP-HURTS-ANSWERS (answer rate equal on every pair) and no FLIP-CHANGES-NONFEEL. It reports one flag,
FEEL-NOT-APPEND-ONLY, on `p_private` (seeds 3 and 7). That is the idle "say something" turn, whose OFF record carries
an inner-state read-out, so the checker treats it as a feel turn. Its ON and OFF replies are byte-identical, so the
behaviour is the most specific outcome possible; the flag comes from the checker's feel-turn classification and is
recorded, not counted. The 32 `unmeasured` entries are the non-intact_a arms, which by design have no OFF run (the
OFF half covers intact_a only).

**Caveats.**
- Dev seeds only. The gate-seed C4 measurement is the Stage-2 exit gate itself, pinned at a revision that contains
  this flip; if C4 fails there, the flip reverts.
- The 2-D sign/arousal sub-read is reported, not gated: fear's sign reads "neutral" on every seed
  (`sign_correct_all` 0/8). The category naming is what is load-bearing.
- A side fix in the same commit: `emotion_category_dev.flip_check` imported `answer_rate` after putting `tools/`
  first on sys.path, which let `tools/experiment.py` shadow the root `experiment` package; it now imports
  `tools.answer_rate`. The dev runner's OFF lines now set `BRAIN_EMOTION_CATEGORY=0` explicitly, since the default
  is now ON.
