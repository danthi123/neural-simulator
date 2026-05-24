# Biology references for the (c) loop: PFC response 30-50ms post-SWR; selective for trajectory not individual locations; forward + reverse replay both observed (2026-05-24)

## Direct mapping to (c) loop design parameters

| Biology finding | (c) loop implementation choice |
|---|---|
| PFC cells respond ~30-50ms post SWR | The (c) loop's "capture post-replay cortical activity" step should sample at 30-50ms post SWR trigger — matches one gamma cycle (50ms) |
| PFC cells selective for which trajectory (e.g., which Y-maze arm) | The project's parallel-matching decoder identifies the BIND (concept) at the gamma-slot position, NOT the raw activity vector — matches "trajectory-selective" coding |
| CA1-PFC replay supports BOTH recall and PLANNING for spatial working memory | The (c) loop's pre-registered test (partial-sequence completion) IS the PLANNING use case; the project's existing Phase 1.3 SWR consolidation IS the RECALL use case |
| REVERSE replay during pauses (forward + reverse both observed) | The (c) loop currently uses forward replay (Phase 1.3); reverse replay is a possible extension for the post-(c) direction (retrieving past dialog turns) |
| Cortico-hippocampal-cortical LOOP of information transmission (sensory pre-SWR → SWR → PFC post-SWR) | The (c) loop's structure matches: lang_input → CA3 → SWR → cortex/dlpfc_wm → loop iteration |

## Key biology-translatable conclusion

The 30-50ms post-SWR PFC response window is the SAME time scale as the project's gamma-slot framework (7 slots × ~14ms each = 1 theta cycle of ~100ms; one gamma slot ≈ 14ms; one gamma cycle ≈ 50ms). The (c) loop's per-iteration cycle naturally corresponds to one theta cycle in biology — within which one SWR replay event drives one PFC frame update.

This suggests the (c) loop iteration count should map to **theta cycles**: each loop iteration = one theta cycle = one SWR-driven PFC frame update. Schwartenbeck's three-stage progression (early non-selective → middle selective → late converged) takes ~3500ms in their data = ~30 theta cycles. The (c) loop should run for at least ~30 iterations to characterise full refinement.

This is a CHARACTERISATION refinement to the (c) TDD plan's pre-registered test — not a change to the PASS criterion. The PASS criterion remains: multi-seed-mean ≥ 0.80 at final iteration on every K in the ladder.

## Source

WebSearch query "prefrontal cortex working memory sharp wave ripple replay sequence completion partial cue biology"; primary references:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6005707/ — "The role of replay and theta sequences in mediating hippocampal-prefrontal interactions"
- https://www.sciencedirect.com/science/article/pii/S0896627319307858 — "Dynamics of Awake Hippocampal-Prefrontal Replay for Spatial Learning and Memory-Guided Decision Making"
- https://www.science.org/doi/10.1126/science.aax1030 — "Hippocampal sharp-wave ripples linked to visual episodic recollection in humans"
- https://kevinbinz.com/2021/12/30/sharp-wave-ripples-and-memory-retrieval/ — synthesis review

## Files

- This reference doc: `research/findings/2026-05-24-biology-references-PFC-SWR-replay-30-50ms-window-selective-trajectory-encoding.md`
- Companion: `research/findings/2026-05-24-Schwartenbeck-2023-biology-reference-for-c-generative-replay-three-stage-iterative-refinement.md`
- (c) design + plan: `docs/plans/2026-05-23-generative-replay-design.md`, `docs/plans/2026-05-24-generative-replay-implementation.md`
