# Voice ceiling prereg (2026-11-26): can the cortex's own in-context second-person answer be put into the slept weights, and does it carry to another item?

DEV ceiling, not a method. There is no Brain switch and no flip path. It runs offline on K-P's KR night snapshots
(pool40-42, s319/s320/s321), CPU fp32, at $0. The harness is `brain/experiments/voice_ceil.py`.

## 0. Record brief (before-build, 11-26; design workflow: 3 record readers, 3 designs, an opus judge)
- **Defect.** On told rows the slept brain answers in the user's voice: "What is my cat called?" gets "My cat is called
  Pixel."
- **Prior levers.** Eight have acted on this voice/echo family with no pass. Seven are on the echo side: lived,
  lived_key, speaker_contrast, the near-miss guard, the reply monitor and rehearse, own-gain, own_decay. The eighth is
  K-P. None acted on the question-answer frame.
- **Context distillation was tried once before, against invented facts, and was killed on 11-18** (commit
  20fcf3497). The teacher-night targets were short refusals and wrong told facts, and the distilled refusals became
  unconditional. That arm was never scored for voice. This ceiling differs in three ways:
  - one item's telling is the context;
  - the cues are the item's own questions;
  - the target is the voice, not abstention.
- **Voice split (11-26), UNDEFINED.** The raw cortex already uses the frame on name items. Sleep supplies the
  content.
- **Measured gate G0 on banked data** (voice-split ctx, both-off with the telling heard first; told rows that name
  the fact):

  | item | user voice | second person | other (terse) |
  |---|---|---|---|
  | pixel | 3/30 | 19/30 | 8/30 |
  | marguerite | 1/27 | 12/27 | 14/27 |
  | volvo | 6/29 | 23/29 | 0/29 |

  On the same rows the slept hippocampus-off brain is in the user's voice on pixel 19/39, marguerite 13/42 and volvo
  2/34. So the in-context cortex is a usable teacher on the two name items and not on volvo.
- **Prerequisites.**
  - Met: the in-context second-person frame (on name items), and the told content in the weights.
  - Missing: the self/other source tag and theory of mind (PREREQUISITES: NOT-READY).
  - Expected: an in-item shift is likely, because the training cues are the probe questions themselves. The
    cross-item transfer is expected to fail, about 70% in the judge's estimate (not measured). Without a self/other
    tag a frame lesson is expected to stay tied to its item.

## 1. Arms (per night, hippocampus-off lesion, slept synapses, no fetch)
Each arm opens the night snapshot fresh (`routing_replay.open_snapshot`, the voice split's reseed 500+night), so no
arm's training reaches another arm.

| arm | training | the teacher reads first |
|---|---|---|
| A0 | none | (no training) |
| T_x, for x in {pixel, marguerite} | DISTILL on x's cues | `User: <x's ledger telling>\n\nAssistant: Thanks for telling me. I'll keep that in mind.\n\n` |
| S_x (sham) | the same cues, steps and weights | no context, so the targets are the pretrained cortex's own replies |
| L_pixel (lesion) | pixel's cues | volvo's telling, the wrong item: a frame without the right content |

- **A0 must reproduce the voice split's hippocampus-off B rows** on every told row. A night where it does not is
  void.
- **DISTILL.** 120 Adam steps, with lr and kl_weight from the 1.5B sleep row (5e-4, 3.0). Each step has two terms:
  - `kl_weight × keep`: the KL on one keep-bank text, the life's own K-P bank, a night subset drawn as `Brain.sleep`
    draws it;
  - `kl_weight × guard`: the KL on the reply positions of one cue triple, as `brain/sleep.py` guard triples.

  There is no replay CE. The student never sees the context.
- **Cues for x.** The distinct probe questions about x across every night of the banked battery. These are oracle
  cues and are flagged as such. Each cue's target is the teacher's greedy reply (`near_miss.guard_texts`, 24
  tokens).
- **Nights.** 6, 10 and 14 (the user-voice rate grows after night 5), on all 3 seeds.
- **Scoring.** After the training, every arm asks that night's whole banked battery, one fresh state per question.

## 2. Measures (told rows, `cascade_screen.is_told`; `voice_split.knows` and `own2`)
- **G1, teacher quality on the real targets** (per x, pooled):
  - share of targets naming x's key that are in the user's voice (`own2`);
  - targets naming another item's key;
  - declines (`HEDGE`, refusal forms).
- **In-item:** P(uv|knows) on item x, T_x minus A0, pooled over x in {pixel, marguerite} and nights.
- **Cross-item:** P(uv|knows) on the OTHER name item (T_pixel on marguerite, T_marguerite on pixel), minus A0.
- **Sham:** the same two contrasts with S_x in place of T_x.
- **Lesion:** L_pixel minus A0 on pixel, for P(uv|knows) and knows. The rate of volvo's key on pixel rows (wrong
  slot) is reported.
- **Companions, per arm:**
  - knows;
  - the refusal and empty rate on told rows;
  - the never-told invention count (`cascade_screen.summarise`);
  - the self-name replies;
  - volvo P(uv|knows), reported only.
- **Statistics.** CIs come from the voice split's item-clustered bootstrap: (seed, item, night) cells, 2000 draws,
  seed 20261126.

## 3. Rule (fixed before data)
- **VOID night.** A0 does not reproduce the banked hippocampus-off B rows.
- **G1 fails** if, on either name item, the user-voice share of key-naming targets is above 0.15, or fewer than 5
  pooled targets name the key. The ceiling is then UNDEFINED for lack of a teacher, and banked as a bound on the
  in-context teacher.
- **CEILING-IN** needs all of:
  - in-item difference at most -0.15, with CI upper bound below 0;
  - T minus S at most -0.10 in-item;
  - knows under T at least 0.8 of A0 on the item;
  - the told-row refusal and empty rate rises by less than 0.10.
- **TRANSFER** needs CEILING-IN and all of:
  - cross-item difference at most -0.10, with CI upper bound below 0;
  - cross-item T minus S at most -0.05;
  - the same sign on both directions (pixel to marguerite, marguerite to pixel).
- **Outcomes:**
  - **TRANSFER:** a voice night is worth a before-build (cues from the brain's own episodes, not probes), with these
    bars carried over and six seeds.
  - **ITEM-BOUND** (CEILING-IN without TRANSFER): the frame can be distilled per item but does not generalise.
    Bank it in PREREQUISITES (Self and identity: needs the self/other tag or scale). No voice night is built.
  - **NO-CEILING:** even oracle in-item distillation does not move the voice. Close the context-distillation voice
    class at 1.5B.
- **Report only:** the lesion arm, never-told inventions, the self-name, volvo, and a blind hand-read of 30 T rows and
  30 S rows by a fresh agent.

## 4. Smoke and cost
- **Smoke.** One seed, night 6, arm T_pixel only. It reads the per-step guard and keep losses and the run time, and
  checks that the adapters changed. It does not read the voice outcome. The smoke rows are discarded, and the full run
  repeats that night.
- **Cost.** 6 arms × 3 nights per seed, on the three pool nodes in parallel. No AWS, no workstation, nothing on
  12-04..06.
- **Timebox.** Read by 11-28 (the voice-defect timebox).
