# gap#5 one-brain MERGE — the END-TO-END CAPSTONE DEMONSTRATED: the real conversing brain (OneBrainComposer) sleeps, runs a REAL traveling CA3 replay ON ITS OWN BRIDGE, wakes, and still converses — all on one brain (with a characterized production-bridge replay-quality gap) (2026-07-25)

## Headline
The full one-brain replay merge is demonstrated end-to-end on the **actual production conversational agent**. A CA3
place-field replay track (2000 neurons + the forward-biased band) is made co-resident ON the `OneBrainComposer`'s own
Izhikevich bridge; the composer stores + recalls facts (WAKE), the bridge switches to AdEx/dt0.1 for a SWR/sleep phase and
runs a **real traveling CA3 replay on the track slice**, then switches back (WAKE) and the composer **still converses** —
recall + no-confab moat perfectly preserved. All on ONE bridge. This assembles the two proven halves (the co-resident
round-trip `1bdcc5a4` + the composer surviving the switch `e2b86dce`) into the complete loop.

## Result (3 seeds; DEMONSTRATED with a characterized quality gap)
- **Conversation preserved 3/3:** pre-sleep recall `['north','east','south']` == post-sleep, moat `[None,None]` intact
  before AND after — the composer's store/recall/moat survive the sleep cycle with the CA3 track co-resident.
- **Real traveling replay in sleep:** DECODE_r **+0.458** (deterministic across seeds), the bump **travels 42% of the
  track** (range 42/100). The band is intact across the composer's Hebbian wake ops (147,150 synapses, max 599).
- ⇒ the end-to-end loop (converse → sleep + real traveling replay → converse, one brain) **works in substance.**
- **QUALIFIED, not a clean GO:** the replay decode on the composer's FLAT bridge (0.458, bump width 11.6) is below the
  clean-standalone threshold (DECODE_r 1.000, width 0.4). The CLEAN replay is proven 6-seed on the region-framework
  standalone (`d6e140bf`) + co-resident (`1bdcc5a4`) bridges; the capstone's gap is specific to the flat composer bridge.

## Integration points solved (the assembly plumbing)
1. **RF kick sizing:** the composer builds RF kick vectors + its `rf_mask` at its own `n_total`; appending the CA3 slice
   enlarges the bridge, so widen `c.n_total` to `num_neurons` and pad `rf_mask` with False over the CA3 slice (the
   composer's RF ops never touch the track — its layout indices stay < the original n_total).
2. **Band-injection ordering:** inject the CA3 band AFTER the composer wires its parser/RF (else their wiring rebuild
   wipes it — the first attempt gave 0 band synapses).
3. **Hebbian band-protection:** the composer runs Hebbian (`hebbian_max_weight=400` < the band's 600) during wake, but the
   band stays at 599 (the masked Hebbian clip protects the plastic=False band; the CA3 neurons are quiescent in wake).

## The un-isolated replay-quality gap (honest, precise)
The flat composer bridge broadens the CA3 replay bump (width 11.6 vs the standalone's 0.4), capping the decode at 0.458.
RULED OUT this session: OU noise (killing/matching it — no change), parser/RF neurons firing in sleep (silencing them —
no change), inhibitory neurons in the CA3 slice (0/2000 — all excitatory, num_traits=1). The residual is a config
difference between `build_coresident_bridge`'s flat CoreSimConfig and the region-framework `_build` used for the clean
6-seed replays — NOT isolated to a specific field here. NEXT: field-by-field diff of the two configs (or build the CA3
track as a proper region on the composer's bridge, matching the region-framework setup that gives the clean 1.000).

## Verdict + scope
- **The one-brain replay merge is DEMONSTRATED end-to-end on the real production agent** — converse → sleep + real
  traveling replay → converse, one brain, conversation fully preserved. This is the substantive completion of the merge.
- **HONEST:** the production-bridge replay decode (0.458) is a QUALIFIED result, below the clean threshold that is
  independently proven (1.000, 6-seed) on the region-framework bridges — a characterized flat-bridge config gap, the one
  precise remaining tuning item.
- Remaining for a clean end-to-end GO: isolate the flat-bridge config field (or region-frame the CA3 track on the composer
  bridge). The neural replay-reader (Bayesian decode = measurement instrument) is the other gap#5 closure item.

## Provenance
`research/runners/_gap5_onebrain_capstone.py` (logs `capstone*.log`; the diagnostic ladder documents the ruled-out
causes). Reuses the phase-switch helpers + the production `OneBrainComposer`. NO `sim/` edit (runner-side monkeypatch to
enlarge the composer bridge + inject the track). GPU. Builds on the full gap#5 arc (`d6e140bf`→`e2b86dce`).
