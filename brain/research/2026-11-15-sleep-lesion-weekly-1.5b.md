# Pre-registration: after sleep, does switching off what the night learned visibly change the weekly replies? (#281, 1.5B, six seeds, pool CPU)

Written 2026-11-15 after a 3-seed smoke (s200-s202, report-only) and before any registered seed.
Capability: Sleep consolidation; milestone day 14 item "a lesion visibly changes replies" (`brain/ARCHITECTURE.md` §4).

**Record brief.**
- The day-14 weekly (`results/2026-11-15-weekly-1.5b-milestone.md`, W2 FAILS) showed a hippocampus lesion no longer
  changes the told replies after a night (<= 2 of 4): the slept cortex holds the facts. The lesion that should show
  it now is of the night's synapses.
- `weekly.py --sleep-lesion` (default OFF, a333eb22) adds a third arm on the same shared night: hippocampus off AND
  the consolidated adapters off (`Brain.set_lesions(hippocampus_off=True, consolidated_off=True)`).
- Smoke, seeds 200-202 (pool CPU, not criteria): keyed correct intact 6/6, 6/6, 3/6; hippocampus off 6/6, 5/6, 4/6;
  hippocampus + sleep's synapses off 0/6 on all three, with fluent ordinary replies (story, feeling).
- Prerequisites met; host decision none. Engagement is recorded at the reply level only (no lesion trace field).

**Method.** `python -m brain.eval.weekly --model 1.5b --shared-sleep --sleep-lesion --seed <s>`, seeds
42/43/44/100/101/102, pool40/41/42 (`device cpu`, fp32), `jobs/2026-11-15-pool-weekly-sleep-lesion.sh`.
Scored by the weekly scorer as written (six keyed questions).

**Criteria (per seed, the third arm vs intact).**
- L0: all three arms ran after one shared night (three "### session 2" sections, sleeps == 1).
- L1 (the lesion changes the REPLY): >= 4 of the 6 keyed replies differ in text from intact, on >= 5/6 seeds.
- L2: keyed correct in the third arm <= 1 of 6 on >= 5/6 seeds (the facts are not in the pretrained synapses).
- L3: the hippocampus-off arm keyed correct >= 4 of 6 on >= 5/6 seeds (the night's synapses carry them).
- Reported, not criteria: the two ordinary replies (story, feeling) per arm, and the own name per arm.

**Decision.** L0-L3 hold: the milestone item "a lesion visibly changes replies" is met after sleep (lesion of the
night's synapses), reported in HANDOFF and #281. Otherwise not met, reported per seed.
