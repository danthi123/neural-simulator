# gap#5 one-brain MERGE — the END-TO-END CAPSTONE is a CLEAN 6-SEED GO: the real conversing brain (OneBrainComposer) sleeps, runs a CLEAN traveling CA3 replay ON ITS OWN BRIDGE, wakes, and still converses — all on one brain; the replay-quality gap was ISOLATED to short-term plasticity (STP sharpens the Ecker bump) and FIXED (2026-07-25)

## Headline
The full one-brain replay merge is demonstrated end-to-end on the **actual production conversational agent, CLEAN**. A
CA3 place-field replay track (2000 neurons + the forward-biased band) is made co-resident ON the `OneBrainComposer`'s own
bridge; the composer stores + recalls facts (WAKE), the bridge switches to AdEx/dt0.1 for a SWR/sleep phase and runs a
**clean traveling CA3 replay on the track** (DECODE_r **1.000**, width 0.4), then switches back (WAKE) and the composer
**still converses** — recall + no-confab moat perfectly preserved. All on ONE bridge. This is the substantive completion
of the one-brain replay merge on the real agent.

## Result — CLEAN 6-SEED GO (seeds 42/43/44/100/101/102)
Store 3 SVO facts; query (WAKE 1); switch to AdEx/dt0.1 + freeze STDP/Hebbian, run the CA3 replay on the track (SLEEP);
switch back (WAKE 2); re-query:
- **SLEEP replay DECODE_r = +1.000, width 0.4, range 43/100 — all 6 seeds** — a clean localized directional traveling
  replay on the composer's own bridge.
- **WAKE recall preserved + moat intact 6/6:** `['north','east','south']` before == after; never-stored cues abstain
  (`None`) — the composer's full conversational capability survives the sleep cycle.
- ⇒ **the end-to-end loop (converse → sleep + clean traveling replay → converse, one brain) is a 6-seed GO.**

## The isolated mechanism — SHORT-TERM PLASTICITY sharpens the Ecker replay bump
The first capstone attempt gave a BROAD replay (DECODE_r 0.458, width 11.6) while the conversation was already perfect.
A methodical isolation (each ruled out on a flat AdEx replay harness) traced it precisely:
- NOT OU noise (on/off both 1.000), NOT parameter heterogeneity (on/off both 1.000), NOT inhibitory neurons (0/2000),
  NOT the composer's ops (build_coresident_bridge + CA3 + no composer ops also gave 0.458), NOT flat-vs-region.
- **It is `enable_short_term_plasticity`:** flat AdEx replay STP=True → DECODE_r **1.000** (width 0.4); STP=False →
  **0.502** (width 11.6). All the clean 6-seed replay builds (`d6e140bf`, `1bdcc5a4`) left STP at its default (True);
  `build_coresident_bridge` explicitly disables it, which is why the co-resident replay was broad.
- **Mechanism (meaningful, not incidental):** STP (Tsodyks-Markram) short-term DEPRESSION weakens the trailing-edge
  synapses that just fired, so they don't re-excite the packet → the traveling bump stays razor-narrow. STP is thus part
  of the Ecker replay's sharpening/localization mechanism, alongside the band + AdEx refractoriness.
- **The FIX:** build the composer's bridge with STP enabled (replicating `build_coresident_bridge` verbatim except
  `enable_short_term_plasticity=True`). The composer's conversation is UNAFFECTED (recall + moat still perfect) — STP on
  the parser/RF synapses does not disturb the who/what pipeline.

## Assembly plumbing solved (for the record)
1. **RF kick/mask sizing:** widen `c.n_total` to `num_neurons` + pad `rf_mask` False over the CA3 slice (the composer's
   RF ops never touch the track).
2. **Band-injection ordering:** inject the CA3 band AFTER the composer wires parser/RF (else their rebuild wipes it).
3. **Hebbian band-protection:** the masked Hebbian clip protects the plastic=False band (max stays 599 through wake).

## Verdict + scope
- **The one-brain replay merge is DEMONSTRATED end-to-end, CLEAN, on the real production agent** — converse → sleep +
  clean traveling replay → converse, one brain, conversation fully preserved, DECODE_r 1.000.
- **Bonus finding:** STP is a load-bearing part of the Ecker replay-bump sharpening (STP-off broadens it 1.000→0.50).
- Remaining gap#5 closure item: the neural imaginative-replay READER (the Bayesian decode is a measurement instrument),
  naturally built alongside replay-driven consolidation. A `sim/` first-class `switch_neuron_model_phase()` +
  reserve-slice capability (additive, default-off) would replace the runner-side replication.

## Provenance
`research/runners/_gap5_onebrain_capstone.py` (logs `capstone*.log`; the diagnostic ladder documents the ruled-out
causes + the STP isolation). Reuses the phase-switch helpers + the production `OneBrainComposer`. NO `sim/` edit
(runner-side replication of build_coresident_bridge with STP on + the CA3 track). GPU. Builds on the full gap#5 arc
(`d6e140bf`→`e2b86dce`).
