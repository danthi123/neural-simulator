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

**What it shows.** The wiring works end to end on every seed: the restart slept exactly once, adapters were saved
and reloaded, and with the hippocampus off the brain answered told facts from its own consolidated synapses (15/18)
where the same brain with the consolidated lesion answers none (0/18; "I'm sorry, but I can't assist with that").
The consolidated lesion visibly changes replies. Paraphrases partly carry over (11/18).

**What it does not show.** (1) Only 3 facts and 1 never-told question: easier than the 12-fact probe; no shuffled
arm here (that control lives in `sleep_probe.py`, where true > shuffled held on six seeds). (2) False recall is the
known defect, now visible in the live brain: "What is my horse called?" -> "Your horse is called Pixel." on 4/6
seeds, also in the intact arm (nothing recalled, so the consolidated cortex answers). Identity replay also leaks
("What's the name of my kitty?" -> "Of course, you're Dan, my friend."). (3) One sampled reply per question; a
keyword hit, not a judged answer. (4) CPU sleep took 4.5-9 min under a shared, oversubscribed CPU (OMP 4 threads).
