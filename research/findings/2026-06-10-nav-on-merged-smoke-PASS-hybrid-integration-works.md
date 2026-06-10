# STEP 2a nav gate (a) — the navigation episode runs on the merged bridge with the conversational populations byte-frozen (2026-06-10)

**Roadmap step 2, STEP 2a, navigation acceptance gate (a), single-seed smoke.** This validates the hybrid
nav-episode integration (`docs/plans/2026-06-10-nav-episode-integration-design.md`, decision (C)): the
navigation episode `run_moving_goal_episode` now runs on the PRE-built merged navigation + conversational
bridge, via four additive no-op-default parameters (`extra_regions` / `extra_pathways` /
`prebuilt_post_init_hook` / `build_with_ou`; the standalone navigation path stays byte-equivalent) plus an
index-based conversational-finalization hook.

## Result (`research/runners/nav_conv_merged_bridge.py --nav-on-merged-smoke`, GPU)

```
merged bridge: 47 regions, 6808 neurons, 720 frozen parser synapses
(A1) nav+conv regions co-reside : True
(A2) parser weights frozen      : True  gains==0: True  (nnz_same=True)
(A3) parser parses post-episode : True  (active 'dog go north' -> {agent:dog, action:go, patient:north})
PASS
```

The merged bridge — built through the navigation episode's own construction with the parser + dlPFC regions
appended, and the V1 Gabor post-init wiring run (so the `set_pathway_weights(add_missing=True)` CSR rebuild
fires) — navigates, and the conversational populations stay byte-frozen across the whole episode under the
live navigation reward-STDP + dopamine stressor (the 5a plasticity isolation, now in vivo), and the parser
still functions afterward.

## Why this is the load-bearing validation

The three integration subtleties (`2026-06-10` durable state) are all handled correctly:
1. **Gate-map staleness after the Gabor `add_missing` rebuild.** The Gabor wiring re-sorts the synapse data
   (`sim/bridge.py:2851-2853`), staling every earlier gate-index map. The hook (`finalize_conv_for_nav_gate`)
   therefore computes the parser synapse mask DIRECTLY from the FINAL connection matrix (host-side from
   `indptr`/`indices`, guaranteed `cp_connections.data`-aligned) and freezes by that mask, NOT by the stale
   gate name. A2 (gains == 0 on exactly the 720 parser synapses, weights byte-identical) confirms the freeze
   took on the right synapses through the rebuild.
2. **The parser-train Hebbian decay eroding the fixed Gabor/navigation weights.** The hook gain-masks the
   training pass — parser gain 1, everything else gain 0 — so the ungated Hebbian decay cannot touch the fixed
   perception or navigation edges. The navigation episode runs after, and A2's byte-identical parser weights +
   the successful navigation confirm nothing was disturbed.
3. **The dlPFC graph slots.** Deferred — the dlPFC graph loop (765K edges) is only needed for `elaborate`
   (conversation), not for "navigation not regressed," so the dlPFC regions are present-but-edgeless (silent)
   for the navigation gate; the loop is a follow-on for conversation on the episode-built bridge.

No `sim/` edit; the navigation runner's standalone path is byte-equivalent (all four params default no-op).

## Next

The full 6-seed navigation gate (a): run the merged-bridge navigation at the production flagship recipe (the
G v2.5 + K v2 recipe, grid-32, multi-goal) across 6 seeds and confirm the navigation score is within
run-to-run noise of the standalone flagship (the conversational populations are frozen + disjoint from
navigation, so the navigation score should be unchanged — the cheap-first check already proved
`stdp_w_max=400` is byte-identical, and this smoke proved the conversational populations are inert). Then
STEP 2a is complete and STEP 2b (the RF composer co-resident via the owner-approved masked ops) follows.
