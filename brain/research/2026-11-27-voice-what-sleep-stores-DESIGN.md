# DESIGN (not a prereg): answered-exchange replay. Sleep also replays what the brain itself answered

Status: design only, 11-27, revised after a record check. **Blocked by the stop rule (section 2).**

## 1. Record brief (corrected)
- Defect: slept cortex answers told facts in the user's voice (hippocampus-off P(uv|knows) 0.24-0.28).
- The frame exists in context: C arm knows 55/67, P(uv|knows) 0.11 (`2026-11-27-voice-direction-ceiling-read.md`).
- What the night sees. Corrected: the earlier draft said it never sees a question or an own turn. It does, but only as
  a keep target. `ORDINARY` (`brain.py` L41-45) is three User/Assistant question-answer exchanges under whole-text keep
  KL; K-P replaces them with 64 of the cortex's own samples a night (whole-text KL; `prior_u` masks to frame turns,
  `keep_mask.py`). The replay CE gets only `User: <telling>`; questions are not stored (`_is_telling`). True gap: no
  told fact is ever a CE target in an own turn after a question.
- **Stage-0 peek, trunk only** (host only, first-token count, `_scratch/stage2/read/lifeKP_s319-321.json`, kinds
  sameday/remention, self-name excluded). The earlier 72 / 47 counted prunedA lines twice and mis-summed. Trunk: 63
  answered questions (21 per life), 38 name the ledger key: 10 start You/Your, 19 terse or other ("Pixel."), 9 My/I
  (user voice plus hedges, not split). Days 1-14: about 12 key-naming replies per seed before any confirmation filter,
  hedges included. Not registered; redo with `own2` and a blind hand-read.

## 2. Stop rule: this is a perspective lever, so it waits
HANDOFF L18, recall-voice prereg section 5: D2 lever count 5, no further perspective lever before a ROUTING method.
The headline measure here is P(uv|knows), so this is a sixth perspective lever. It is not exempt.

- It reads hippocampus-off, where nothing is fetched, so a routing method cannot fix this defect. That is an argument
  for amending the rule, not a way round it.
- Sequence: nothing here runs or is registered before the memory reply ceiling verdict
  (`2026-11-27-memory-ceiling-prereg.md`, running now).
  - A right fetch improves the reply: a routing method goes first; this design waits until one has been tried.
  - It does not: the stop rule needs a written amendment in the recall-voice prereg, with a fresh skeptic, before
    this is registered.
- Allowed meanwhile: the host-only Stage-0 recount.

## 3. Mechanism
When the user asks and the brain answers, that lived exchange (question and own spoken reply) is kept as an **answered
exchange** if the world lets it stand. At night it is replayed beside the tellings, with CE on the reply positions
only. The question is context, as when lived.

- Biology: retrieval practice (a retrieved memory is re-encoded in its retrieval context).
- Contrast (the banked prerequisite for a speaker signal): the fact appears in two speaker spans, `User: My cat is called Pixel.` and
  `User: What is my cat called?\n\nAssistant: Your cat is called Pixel.`.
- Storage: sleep-only, never in the hippocampus, so routing is untouched.
- World filter: the user's next turn. Offline, the ledger check "reply names the user's fact" stands in (declared
  teacher scaffold; frame-blind, so the voice scorer never selects training text). Hedges that name the key pass this
  filter: counted and reported.
- **Code that must be written (none exists).** `sleep_extra` is a plain `(text, strength)` list and `sleep()` takes CE
  on the whole text (`sleep.py` L66-68). Only the guard KL has a `(prefix, reply)` form.
  1. `brain/sleep.py`: an episode item may be `((prefix, reply), strength)`; CE then runs from position
     `len(ids(prefix)) - 1`, the same offset rule as the guard. Replay RNG and the plain path stay byte-identical (test).
  2. `brain.py`: `Brain(answered_replay=...)`, default OFF, keeps the exchanges in their own sleep-only list, saved
     and resumed with the home; flag-off byte-identity test.
  3. Tests: mask boundary on a multi-token label, no loss on prefix positions.
  4. A SCAFFOLDS row (ledger filter, masking rule). `local_sleep.py` and `memory_ceiling.py` are not touched.

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
At most about 12 exchanges per seed. If fewer than 8 pass the filter on any seed: UNDEFINED, not run.

| arm | night-14 replay adds |
|---|---|
| A0 | nothing (re-sleep noise floor) |
| AE | confirmed answered exchanges, reply-masked |
| AE-LOO (lesion) | AE without any exchange about pixel or marguerite |
| SHAM | equal count and length of lived small-talk exchanges, same masking |
| NOCON | the same replies as bare text, no question, no labels, whole-text CE |
| ALL | every answered exchange, wrong ones included |

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

Compute: 6 sleeps x 3 seeds, AWS g5 fp16 or pool40 CPU.

## 6. Stage 2 (only after ENGAGED or UNDERPOWERED-LEAD): DEV life
`Brain(answered_replay=True)` on K-P, seeds 319-321; an exchange is kept unless the user's next turn corrects it. Bars are the recall-voice prereg's, unchanged, plus hippocampus-off
P(uv|knows) <= K-P - 0.10 on the 223 told rows. Six seeds only after a DEV pass.

## 7. Expected outcome, honest odds
- Onset beats A0: about 55%; also SHAM and NOCON: about 35%.
- Voice clause with its CI: about 15%; UNDERPOWERED-LEAD another 20%.
- Transfer to left-out items: 10-15%. DEV life pass: about 10%.
- Failure modes: user-voice and hedged replies written into the weights (9 of 38 start My/I), language drift (M1
  s321), refusals, too few exchanges.
- A fail is a verdict on this replay form at 1.5B with about 12 exchanges only.
