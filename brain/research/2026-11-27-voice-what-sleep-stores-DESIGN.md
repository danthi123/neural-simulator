# DESIGN (not a prereg): answered-exchange replay. Sleep also replays what the brain itself answered

Status: design only, 11-27, revised after a record check and again after the skeptic's rejection of amendment
revision 1. **Blocked by the stop rule (section 2) until `2026-11-27-voice-store-amendment.md` (revision 2) is in
force, which needs a fresh skeptic APPROVAL first. Blind hand-read done: the floor is not met on any seed, so Stage 1 cannot run on the days 1-14 store.**

## 1. Record brief (corrected)
- Defect: slept cortex answers told facts in the user's voice (hippocampus-off P(uv|knows) 0.24-0.28).
- The frame exists in context: C arm knows 55/67, P(uv|knows) 0.11 (`2026-11-27-voice-direction-ceiling-read.md`).
- What the night sees. Corrected: the earlier draft said it never sees a question or an own turn. It does, but only as
  a keep target. `ORDINARY` (`brain.py` L41-45) is three User/Assistant question-answer exchanges under whole-text keep
  KL; K-P replaces them with 64 of the cortex's own samples a night (whole-text KL; `prior_u` masks to frame turns,
  `keep_mask.py`). The replay CE gets only `User: <telling>`; questions are not stored (`_is_telling`). True gap: no
  told fact is ever a CE target in an own turn after a question.
- **Stage-0 recount, trunk only** (host only, `brain/experiments/results/2026-11-27-voice-stage0/recount.{py,md}`,
  `_scratch/stage2/read/lifeKP_s319-321.json`, kinds sameday/remention, self-name items excluded, first-token class).
  Days 1-14: 54 answered questions (18 per seed); 36 name the ledger key (319: 16, 320: 9, 321: 11): 10 start You/Your,
  19 terse or other ("Pixel."), 7 My/I (user voice plus hedges, not split); 3 of the 36 are hedged. All 36 pass a
  frame-blind "reply names the fact" filter (hedges included). Days 1-15 gives 57 / 38 (10 / 19 / 9), matching the earlier
  peek's 38 and split; its 63 answered total is not reproduced. **Stage-1 floor (>= 8 confirmed per seed): met on all
  three seeds (16 / 9 / 11) under the frame-blind filter only.** That filter is superseded by the hand-read below.
- **Blind hand-read of all 54** (labels before key, single reader; `2026-11-27-voice-stage0/handread.md`).
  LISTENER-CORRECT 6 / 1 / 4; LISTENER-CORRECT + NEUTRAL-CORRECT 12 / 5 / 9. Seed 320 has 8 hedged replies. Every
  listener, neutral and user-voice reply states the told fact. **Floor of 8 under the primary filter: not met on any
  seed; the looser filter also fails seed 320.**

## 2. Stop rule: this is a perspective lever, so it waits
HANDOFF L18, recall-voice prereg section 5: D2 lever count 5, no further perspective lever before a ROUTING method.
The headline measure here is P(uv|knows), so this is a sixth perspective lever. It is not exempt.

- Amendment revision 2 rests on one ground: routing was tried (three routing methods KILLED, arc stopped by its
  timebox 11-26), reading the rule as "before a routing method has been tried". It makes no one-cause claim and does
  not use the memory ceiling (verdict UNDEFINED).
- Nothing here is registered or run until a fresh skeptic APPROVES the amendment and only then the pointer line is
  added to the recall-voice prereg.
- Routing work is neither closed nor resumed (memory ceiling UNDEFINED; reachable strength held); the amendment only
  permits side S before a NEW routing method.
- **Binding stop line: no perspective lever 7 at 1.5B**, whatever a Stage 1 that ran returns (amendment section 5,
  copied verbatim into the Stage-1 prereg). UNDEFINED-by-floor (never ran) does not consume lever 6 and does not
  trigger the line.
- Data: Stage 1 waits for a larger store; candidate is the Stage-2 test2 lives, only after the test2 read is complete
  (amendment section 7).
- Allowed meanwhile: the host-only Stage-0 recount and hand-read.

## 3. Mechanism
When the user asks and the brain answers, that lived exchange (question and own spoken reply) is kept as an **answered
exchange** if the world lets it stand. At night it is replayed beside the tellings, with CE on the reply positions
only. The question is context, as when lived.

- Biology: retrieval practice (a retrieved memory is re-encoded in its retrieval context).
- Contrast (the banked prerequisite for a speaker signal): the fact appears in two speaker spans, `User: My cat is called Pixel.` and
  `User: What is my cat called?\n\nAssistant: Your cat is called Pixel.`.
- Storage: sleep-only, never in the hippocampus, so routing is untouched.
- World filter: the user's next turn. Offline, the blind hand-read stands in (declared teacher scaffold; labels in
  `handread_labels.json`). Only LISTENER-CORRECT exchanges enter AE-L; a second arm adds NEUTRAL-CORRECT. User-voice,
  hedged, wrong-fact, wrong-frame and other replies are excluded.
- **Code that must be written (none exists).** `sleep_extra` is a plain `(text, strength)` list and `sleep()` takes CE
  on the whole text (`sleep.py` L66-68). Only the guard KL has a `(prefix, reply)` form.
  1. `brain/sleep.py`: an episode item may be `((prefix, reply), strength)`; CE then runs from position
     `len(ids(prefix)) - 1`, the same offset rule as the guard. Replay RNG and the plain path stay byte-identical (test).
  2. `brain.py`: `Brain(answered_replay=...)`, default OFF, keeps the exchanges in their own sleep-only list, saved
     and resumed with the home; flag-off byte-identity test.
  3. Tests: mask boundary on a multi-token label, no loss on prefix positions.
  4. A SCAFFOLDS row (hand-read filter, masking rule). `local_sleep.py` and `memory_ceiling.py` are not touched.

## 4. Why it is not a killed or existing lever
| lever | difference |
|---|---|
| lived / lived_key (M1), killed | stored the reply to a TELLING, in the hippocampus, whole-text CE. Here: replies to QUESTIONS, confirmed, sleep-only, reply-masked |
| CFR, masked exchange, killed | reply generated at night; fact only context. Here: lived reply, fact is the target |
| voice ceiling T / teacher night | KL distillation, oracle cues, declines in targets. Here: CE, lived cues, no declines |
| speaker_contrast (M2), residual tags, source_hold | act at reinstatement or by an added vector. Here: data only |
| method 2 `sleep_extra` (SCAFFOLDS L22) | declared "I don't know" Q/A pairs, host-written, whole-text CE. Here: lived replies, masked |
| confab M2 (SCAFFOLDS L26) | already stores the brain's own turn as a source="self" exchange: one teacher-guided abstention, in the hippocampus, reinstated by a gate link. Here: many unguided replies carrying told content, never reinstated |
| inquiry gate (SCAFFOLDS L27, default ON) | own turn stored as a self-source episode: one guided ask, reinstated by surprisal, never replayed at night. Here: night replay only |

Honest overlap: storing the brain's own confirmed turn as its own is confab M2's and the inquiry gate's idea. New
here: night replay of it, told content, reply-only CE.

## 5. Stage 1: offline screen (KR seeds 319/320/321, re-sleep night 14 from the n13 store, as `voice_tag.py`)
Side S only; there is no reinstatement side. Floor: 8 LISTENER-CORRECT exchanges per seed, else that seed is UNDEFINED
and not run (on any arm). Today: 6 / 1 / 4, so no seed runs on the days 1-14 store. The floor and filter are not
relaxed; a later store needs a new blind hand-read.

| arm | night-14 replay adds |
|---|---|
| A0 | nothing (re-sleep noise floor) |
| AE-L (primary; "AE" below) | LISTENER-CORRECT exchanges (the You/Your replies), reply-masked |
| AE-LN (second arm) | LISTENER-CORRECT plus NEUTRAL-CORRECT, reply-masked |
| AE-LN-M | a fixed draw from the AE-LN set, equal in count to AE-L |
| AE-LOO (lesion) | AE-L without any exchange about pixel or marguerite |
| SHAM | equal count and length of lived small-talk exchanges, same masking |
| NOCON | the AE-L replies as bare text, no question, no labels, whole-text CE |
| ALL | every answered exchange, wrong ones included (control only; never carried to Stage 2) |

Measure, hippocampus off, the 67 told questions: onset (log P(You/Your) - log P(My)), P(uv|knows), knows, refusals,
language KL against A0, self-name wrong, blind 40-row hand-read per arm. Item-clustered
bootstrap; weight lesion (consolidated-off returns the both-off rows). Bars checked against counted maxima first.

**ENGAGED, all of:**
1. onset AE - A0, AE - SHAM, AE - NOCON: each CI lower bound > 0;
2. P(uv|knows) AE <= A0 - 0.10, same direction on 2/3 seeds, paired CI upper bound < 0;
3. knows >= 0.95 of A0; refusals up by at most 5 rows; language KL <= 1.10 x A0 per seed; self-name wrong <= A0 + 1;
4. ALL not better than AE on clause 2 (else drop the filter claim).

- **UNDERPOWERED-LEAD:** 1 and 3 hold, the point estimate of 2 holds, its CI crosses 0. No claim; Stage 2 allowed once.
- **KILLED:** AE does not beat SHAM on onset, or clause 3 fails.
- **AE-LOO sets the wording:** left-out items <= A0 - 0.10 reads "the frame transfers"; else "item-bound".
- **AE-L against AE-LN-M sets the second wording:** AE-L onset above AE-LN-M with CI lower bound > 0 reads "own
  correct-frame replies"; otherwise, with AE-LN engaged against A0, SHAM and NOCON, "own replies" and no frame claim.
  AE-LN against AE-L is reported only (counts differ).

Compute: 8 sleeps x 3 seeds, AWS g5 fp16 or pool40 CPU.

## 6. Stage 2 (only after ENGAGED or UNDERPOWERED-LEAD): DEV life
`Brain(answered_replay=True)` on K-P, seeds 319-321; an exchange is kept unless the user's next turn corrects it.
Sleep-only list, as section 3. A fail closes perspective at 1.5B (stop line, section 2); UNDEFINED-by-floor does not. Bars are the recall-voice prereg's, unchanged, plus hippocampus-off
P(uv|knows) <= K-P - 0.10 on the 223 told rows. Six seeds only after a DEV pass.

## 7. Expected outcome, honest odds
- Onset beats A0: about 55%; also SHAM and NOCON: about 35%.
- Voice clause with its CI: about 15%; UNDERPOWERED-LEAD another 20%.
- Transfer to left-out items: 10-15%. DEV life pass: about 10%.
- These odds predate the hand-read. The likeliest outcome now is UNDEFINED: too few LISTENER-CORRECT exchanges
  (6 / 1 / 4 against a floor of 8).
- Other failure modes: language drift (M1 s321), refusals.
- A fail is a verdict on this replay form at 1.5B with a handful of exchanges only.
