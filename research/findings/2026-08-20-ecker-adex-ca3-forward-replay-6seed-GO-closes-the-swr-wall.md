---
type: finding
status: superseded
superseded_by: research/findings/2026-08-20-ecker-ca3-forward-band-LEARNED-by-STDP-emergent-swr-replay-6seed-GO.md
date: 2026-08-20
mechanism: swr-sequence-replay
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_adex_ca3_replay_derisk.py — an Ecker-style AdEx CA3 assembly store (banded recurrence + adaptation + refractoriness) replaying discrete forward SWR events from a non-specific prefix seed, with the full store-lesion anti-cheat battery
runner: research/runners/_gap5_ecker_adex_ca3_replay_derisk.py
external: NO-EXTERNAL-NEEDED — builds the Ecker-2022 AdEx CA3 model the boundary already named (2026-07-25 READY-TO-BUILD spec); the biology is banked, this is the build + its anti-cheat verdict.
artifacts:
  - research/findings/raw/gap5_ecker_adex/ecker_adex_ca3_replay_6seed.json
---
# GO (6-seed): an Ecker-style AdEx CA3 does DISCRETE forward SWR replay that RIDES the weight asymmetry — closing the gap#5 forward-replay wall the bistable store could not

Artifact: research/findings/raw/gap5_ecker_adex/ecker_adex_ca3_replay_6seed.json

**One line.** The SWR forward-replay arc was walled on the DECOUPLED bistable store: forward-gain + a directional cue
were validated ingredients but forward order did NOT ride the encoded chain — it SURVIVED the reverse-asymmetry lesion
([[2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology]] UPDATE-2). This builds the named fix — an
Ecker-style AdEx CA3 — and it passes the decisive test **6/6**.

## The model (new runner, NO `sim/` edit — region framework + the committed `ADEX_ECKER_CA3_PC` preset)
Six disjoint block assemblies of 80 AdEx point neurons each. MODERATE within-assembly recurrence (w=60) → a brief,
SELF-TERMINATING population volley per assembly (not a saturating attractor). STRONG forward links A_i→A_{i+1} (w=800),
WEAK reverse (w=15) — a 53× forward/reverse weight asymmetry. ECKER adaptation (spike-triggered) + AdEx refractoriness
self-terminate each volley, so the cascade moves forward and DIES → the net RESTS SILENT between events. Ignition = a
NON-SPECIFIC random-per-event prefix cue (a different random assembly each SWR period), and direction is scored
forward-FROM-SEED so forward≫reverse can ONLY come from the encoded asymmetry.

## The 6-seed verdict (all_go, 6/6) — the KEY gate is REVERSE-ASYM-LESION collapse
<!--derived-->
Forward-from-seed **0.928** vs reverse **≈0.009** vs chance 0.227; per-seed forward-frac
[0.852, 0.955, 0.864, 0.952, 0.947, 1.000], reverse ≈0 on every seed; `seed_first_frac` ~1.0 (clean cascades);
discrete (duty 0.090, rests silent); `weights_frozen=True` all seeds (no plasticity confound). The decisive property the
bistable store LACKED:
- **REVERSE-ASYM-LESION** (symmetrize the between-edges) → forward collapses to **0.219 ≈ chance** (forward ≈ reverse, no
  forward bias). The bistable store held forward at 1.00 here; the AdEx CA3 collapses. So the order genuinely RIDES the
  forward-weight asymmetry — this is what closes the wall.
- **SHUFFLED-STORE** → 0.231 (collapses). **NO-BAND** → 0.000. **PERMUTED-ASSEMBLY** → 0.000 (order is tied to physical
  identity, not labels). **NO-SEED** → 0 events, silent. **FROZEN byte-hash** True (order rides the frozen chain, not
  re-encoding). NUMPY-REFERENCE guard holds by construction (the rest loop injects only external current — order EMERGES).

## Honest scope (the agent flagged these; they are the residuals, not caveats hiding a failure)
1. **Hand-wired band, not learned.** The forward asymmetry is installed, not grown — a SCAFFOLD per the emergence bar.
   The follow-on is STDP-emergent forward links (the learned version).
2. **Synfire-chain realization, not Ecker's moving-bump.** w_fwd(800) ≫ w_within(60), so hand-off is a population relay
   (a synfire chain), structurally DISTINCT from Ecker's near-diagonal continuous-band moving-bump. A legitimate
   realization of "forward replay that rides the weight chain," but not literally the Ecker attractor — worth stating plainly.
3. **Needs a prefix cue** (the high ECKER V_T means pure spontaneous noise won't ignite) — the mission-allowed minimal
   prefix, made rigorous by randomizing WHICH assembly ignites each event.
4. **PVBC/ripple-interneuron NOT needed** — adaptation + AdEx refractoriness supply the discreteness, so the per-neuron
   AdEx `sim/` capability the spec framed was not added (an honest deviation). ADAPT-LESION survives (forward 0.66–0.88)
   but with higher duty → adaptation PARTIALLY assists; refractoriness also contributes.

## Significance + next
This closes the gap#5 SWR forward-replay wall on the decisive anti-cheat (order rides the asymmetry), which BOTH the
untargeted-noise/afferent D5 transfer AND the brain-pure sleep-replay store were blocked on. NEXT: (a) grow the forward
band by STDP instead of hand-wiring (the emergence version); (b) wire this discrete-forward-replay reactivation into the
D5 episodic organ (the learn-through-use transfer) and the spiking-CA3 sleep-replay store (the brain-pure consolidation).
Not wired live. (Agent-built; parent verified the 6/6 verdict + the reverse-asym-lesion collapse from the artifact.)
