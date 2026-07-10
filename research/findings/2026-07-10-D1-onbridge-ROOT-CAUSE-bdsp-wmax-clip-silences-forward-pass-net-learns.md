# D1 on-bridge ROOT CAUSE: the near-silent forward pass was the `bdsp_w_max=5.0` clip collapsing the forward weights. With the fix, the net LEARNS above chance.

**Date:** 2026-07-10
**Runners:** `research/runners/_d1_bdsp_wmax_clip_probe.py`, `_d1_bdsp_*_probe.py`; the fix threaded into `_d1_onbridge_learn_to_accuracy_derisk.py` (`bdsp_w_max` param, default 200). NO `sim/` edit for the fix (a runner config).
**Verdict:** ROOT CAUSE found + fixed; the on-bridge net now learns above chance. The "deep spiking-forward-net rebuild" was a mis-diagnosis.

## The root cause (a chain of a0 reads, each correcting the last)
The on-bridge learning stayed at chance because the hidden layer was near-silent. Traced by reading the substrate, not theorizing:
1. The input encoding works (input pool std 0.025) -> the bug is downstream.
2. A minimal 2-region AMPA net PROPAGATES (hidden std 0.0275) -> the propagation is NOT a fundamental wall (corrected my "deep rebuild").
3. The neuron model/profile is NOT the difference (IZH == default, both std 0.025).
4. **`enable_bdsp=True` itself suppresses the forward firing** (hidden std 0.025 -> 0.008), and the forward **weights collapse 40 -> 10 even with `bdsp_learning_rate=0` (frozen)** -> it is not the learning rate.
5. **The culprit is the clip:** `fused_bdsp_update` clips the weight to `[bdsp_w_min, bdsp_w_max]`, and **`bdsp_w_max` defaults to 5.0**. Any forward pathway weight above 5 (the runner used 6-80) is clipped to ~5 on the first BDSP step -> the forward drive collapses -> the hidden goes silent -> no bursts -> no learning. **This is exactly CLAUDE.md's documented STDP-`w_max` gotcha, in BDSP form** ("set w_max above your design weights or weights collapse silently").

## The fix (a runner config, one line)
Set `cfg.bdsp_w_max` above the forward design weight. Confirmed (`_d1_bdsp_wmax_clip_probe.py`): at `bdsp_w_max=50/200` the forward weight (40) is PRESERVED (40 -> 38.8 / 40.0) and the hidden fires input-dependently (std 0.025); at the default 5 it collapses (40 -> 7.7, std 0.007). Threaded as a `bdsp_w_max=200` default on `OnBridgeBDSPNet`.

## The payoff: the net now LEARNS above chance
emerge1 subset (96/96, chance 0.573, oracle ~0.96), 25 epochs, `bdsp_w_max=200`, fw=40, hidden=60:
- **couple ON:** hidden std 0.034, B_rises=True, held-out **0.604**
- **couple OFF:** hidden std 0.034, B_rises=False, held-out **0.677**
- (OLD `bdsp_w_max=5`: held-out 0.427, below chance -- the silent regime.)

The on-bridge net clears chance where it was stuck at chance for every prior config -- the forward pass propagates and the BDSP credit learns. D1's deferred "does the committed rule learn on-bridge?" is now a qualified YES-above-chance (the fix unblocked it).

## Honest nuances (open)
- **The signal is weak** (0.60-0.68 vs oracle 0.96): more epochs / gain tuning / full-data (not a 96-subset) needed to approach the bar. This is a tuning frontier, not a wall.
- **couple-OFF (0.677) > couple-ON (0.604) here.** With `bdsp_w_max` fixed, the apical-coupling `sim/` edit does NOT cleanly help at this operating point -- opposite to what the isolated directed-credit probe (separation 20x) suggested. So the coupling's role in END-TO-END accuracy is now genuinely open: the isolated dw-separation was necessary-not-sufficient, and the accuracy-learning here works via the BDSP burst-deviation credit even with the apical decoupled (couple OFF, B_rises=False). This needs its own investigation before claiming the coupling is load-bearing for accuracy. The coupling edit remains byte-safe + validated for what it does (apical raises B); its accuracy value is unproven.

## ⇒ the arc, honestly
The D1 on-bridge "boundary" was mis-diagnosed twice by me (deep rebuild; then config-but-uncertain) and resolved by reading the substrate to a **one-config gotcha** (`bdsp_w_max=5` silences any real forward weight). Fixed, the net learns above chance. The full-accuracy tuning + the coupling's accuracy role are the open follow-ons. Six careful-measurement self-corrections got here; each "wall" was a misread.

## Files
`research/runners/_d1_onbridge_learn_to_accuracy_derisk.py` (`bdsp_w_max` param + fix), `_d1_bdsp_wmax_clip_probe.py`,
`_d1_ampa_vs_nmda_propagation_probe.py`, `_d1_onbridge_input_propagation_probe.py`; the boundary this resolves
`2026-07-10-D1-onbridge-accuracy-blocked-on-degenerate-forward-pass-BOUNDARY.md`.
