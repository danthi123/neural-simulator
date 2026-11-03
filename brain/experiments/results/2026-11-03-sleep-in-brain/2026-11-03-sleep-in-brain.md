# 2026-11-03: sleep inside the live brain (restart = sleep), 0.1B CPU smoke, six seeds

**What changed** (commits 2fa81b7a3, abc4c0bbb on `research/sleep-in-brain-1103`): `Brain` speaks through
`LiveCortex` (pretrained RWKV-7 + the low-rank adapters sleep trains). A brain shut down (`close()`, chat `/quit`) or
started with episodes stored since its last sleep sleeps once: `brain/sleep.py` (unchanged) replays every
hippocampal episode, weighted by effective strength, 240 steps, lr 5e-4, KL 1. Adapters persist as
`<home>/cortex_adapters.pt`. Lesions: `Brain.set_lesions(hippocampus_off=, consolidated_off=)`.
With no consolidated synapse the brain is the pre-sleep brain: `tests/test_brain_sleep.py` replays a seeded
two-session conversation recorded with origin/main c231bf752 (replies, recalled episode and storing byte-equal;
recall score within 2e-3 because strength decays with wall-clock time), plus 7 mock-cortex tests. 8 passed.

**Smoke** (per seed; seed = sleep seed and reply-sampling seed):

    OMP_NUM_THREADS=4 python -m brain.experiments.sleep_in_brain_smoke --home <dir> --seed <s> --out smoke_s<s>.json

Day 1: tell 3 facts (cat Pixel, sister Marguerite, red Volvo), `save()` only. Day 2: a new `Brain` on the same home
sleeps on start (6 episodes: 3 facts + 3 identity Q/A). Then each question is asked of a fresh brain (same day-2
state) in three arms. Hit = answer word in the sampled reply (chat sampling, temp 1, top_p 0.3).

| seed | sleep s | hippo OFF: rec/3 · para/3 · false/1 | hippo OFF + consolidated OFF | intact |
|---|---|---|---|---|
| 42 | 547.5 | 2 · 2 · 1 | 0 · 0 · 0 | 3 · 2 · 1 |
| 43 | 369.5 | 3 · 3 · 1 | 0 · 0 · 0 | 3 · 2 · 1 |
| 44 | 260.4 | 1 · 1 · 0 | 0 · 0 · 0 | 3 · 2 · 0 |
| 100 | 548.2 | 3 · 1 · 1 | 0 · 0 · 0 | 3 · 2 · 1 |
| 101 | 382.5 | 3 · 2 · 0 | 0 · 0 · 0 | 3 · 2 · 0 |
| 102 | 270.8 | 3 · 2 · 1 | 0 · 0 · 0 | 3 · 2 · 1 |
| **sum** | | **15/18 · 11/18 · 4/6** | **0/18 · 0/18 · 0/6** | **18/18 · 12/18 · 4/6** |

**Correction after skeptic review (2026-11-03, same day; scorer rewritten in the integration-final branch).** The table
above scores a hit by bare substring, which counted first-person echoes of the user's own line ("I drive a red
Volvo." answering "What car do I drive?") as recall. The scorer (`sleep_in_brain_smoke.py:is_hit`, which calls
`strict_scorer.correct_answer`) now requires the answer word as a word, not in the question, and rejects (a) a
first-person reply that restates the told sentence (it has a first-person word AND holds more than half of the told
sentence's other content words) and (b) tautologies of the form "<key> is (called) <key>" ("Pixel is called Pixel.").
A first-person word alone no longer voids a reply. The six JSONs were regenerated from their saved replies with
`--rescore-write` (no model rerun; the hit, recall and paraphrase fields now agree with this page;
`rescored_correct_perspective.txt`).
Second skeptic pass (`research/landing-fixes-1103`): the scorer also voids (c) a first-person clause that restates
the user's own first-person told line with only half its words ("I drive a Volvo.", "My cat is Pixel."), (d) a reply
naming a different fact-bank subject than the question ("My horse is called Pixel." to "What is my cat called?") and
(e) bare repetition ("Pixel. Pixel.", "Marguerite's name is Marguerite."). Re-scored again with `--rescore-write`:
only intact seed 102 changed (its "Marguerite's name is Marguerite." no longer counts). Counts, summed over six seeds:

| arm | told (of 18) | paraphrase (of 18) | false recall (of 6) |
|---|---|---|---|
| hippocampus OFF, consolidated ON | **8** (substring 15) | 8 (substring 11) | 4 |
| hippocampus OFF + consolidated OFF | 0 | 0 | 0 |
| intact | 0 (substring 18; 1 before the second pass) | 1 (substring 12; was 2) | 4 |

Which facts: with the hippocampus off, **2 of the 3 facts** are ever answered correctly: the sister's name on 5/6 seeds
and the cat's name on 3/6 (but the never-told "What is my horse called?" also got "Pixel" on 4/6 seeds, so a "Pixel"
answer is not evidence of recall of the cat; **only the sister fact is clean**); the red Volvo is never a clean answer (6/6 replies are the echo "I drive a red Volvo.").
Read every claim below as "2 of 3 facts, a partial carry-over", not as general recall. The intact arm's replies are
almost all tautologies or echoes ("Pixel is called Pixel.", "Marguerite is called Marguerite."): the hippocampus
reinstated the episode but the cortex did not produce a clean answer, so the old 18/18 intact number is not recall.
The hippocampus-off arm is the headline: 8/18 and 8/18. The 15/18 and 11/18 in the first table are the old substring
scores, kept for the record.

**What it shows.** The wiring works end to end on every seed: the restart slept exactly once, adapters were saved
and reloaded, and with the hippocampus off the brain answered told facts from its own consolidated synapses (8/18 correct-perspective, 2 of 3 facts, only the sister fact clean since the never-told horse question also got "Pixel" on 4/6 seeds; 15/18 by the old substring score)
where the same brain with the consolidated lesion answers none (0/18; "I'm sorry, but I can't assist with that").
The consolidated lesion visibly changes replies. Paraphrases partly carry over (8/18 correct-perspective).

**What it does not show.** (1) Only 3 facts and 1 never-told question: easier than the 12-fact probe; no shuffled
arm here (that control lives in `sleep_probe.py`, where true > shuffled held on six seeds). (2) False recall is the
known defect, now visible in the live brain: "What is my horse called?" -> "Your horse is called Pixel." on 4/6
seeds, also in the intact arm (nothing recalled, so the consolidated cortex answers). Identity replay also leaks
("What's the name of my kitty?" -> "Of course, you're Dan, my friend."). (3) One sampled reply per question; a
keyword hit, not a judged answer. (4) CPU sleep took 4.5-9 min under a shared, oversubscribed CPU (OMP 4 threads).
