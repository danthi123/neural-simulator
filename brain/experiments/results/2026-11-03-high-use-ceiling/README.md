# High-use facts, part 2: in-context ceiling, reachable-only rescore, use counted inside the hippocampus (2026-11-03)

Follows `../2026-11-03-high-use-facts/README.md` (NO-GO at 0.1B, "next step 1"). Branch `research/high-use-ceiling-1103`.

## PRE-REGISTRATION (written and committed BEFORE the six-seed run; instrument fixed)
- Instrument: `brain/experiments/use_strength_probe.py` as committed with this section. Scorer = answer key as a lowercase
  substring of the greedy hippocampus-off answer (the same scorer as `sleep_probe.ask`; the questions do not contain the key,
  so first-person echo of the user's line cannot score). No change after the run; if one is needed, everything is rerun.
- Reachable fact (0.1B) = its ORIGINAL question is answered correctly in context (tell it once after the birth context, no sleep,
  greedy). Facts = the first 12 reachable in pool order (`ceiling_probe.POOL` = the 12 old facts then 24 new distinctive ones).
- Design: 12 reachable facts, 4 told k=1, 4 told k=3, 4 told k=10 times THROUGH `UseHippocampus.store` (re-telling merges into
  one episode and adds strength 1.0; nothing is hand-set; the probe asserts strengths equal k). k per fact = seeded permutation.
  One sleep (240 steps, lr 5e-4, kl 1.0), hippocampus off. Arms: no_sleep, sleep_use (replay by learned strength), sleep_equal
  (control: same episodes and steps, all strengths 1). Seeds 42 43 44 100 101 102, 0.1B.
- **GO**: sleep_use recalls 4/4 of the k=10 facts on 6/6 seeds. (The facts are reachable by construction, so 4/4 is the bar.)
- Also reported, not part of GO: use vs equal contrast on k=10 (paired by seed), k=1 and k=3 recall, paraphrase recall (read only
  against facts whose reworded question is reachable in context), false recall, drift. Anything short of the GO bar is a lead or a
  negative, whatever the contrast shows.
