---
type: finding
status: live
lane: load-bearing
date: 2026-09-21
---

# Open-ended-generation is NOT single-turn lesion-load-bearing on the tiny-demo brain — three measured masking layers, a real wiring fix, and an instrument mismatch (2026-09-21)

Follow-on to the v1 attempt ([`2026-09-20-hollow-open-ended-generation-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-open-ended-generation-drive-diagnosis-and-probe-fix.md), branch `research/hollow-open-ended-generation-drive`), which was brain-build VERIFIED and FAILED (treat=0), and to the hollow-set attack ([`2026-09-20-hollow-set-attack-6of9-probe-artifacts-verified.md`](2026-09-20-hollow-set-attack-6of9-probe-artifacts-verified.md)) that listed open-ended-generation as a genuine remainder. This v2 (branch `research/gap-open-ended-generation-v2`) diagnoses WHY treat=0 all the way down, lands the genuine wiring fix the v1 named, and reaches an HONEST NEGATIVE: on the tiny-demo brain the open-ended generative DRAW is not lesion-load-bearing on a SINGLE-TURN decision diff — for three distinct, separately-measured reasons — even though its likelihood contribution is real and already GO'd DISTRIBUTIONALLY (the B1 draw_many plausible-frac collapse). All numbers below are real brain builds: `SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' ... tools/memcap.sh`, seed 42 (the tiny-demo brain is hardcoded seed 42; the instrument has no other seed).

## The required verify (ACTUAL numbers)

```
SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' LB_OPEN_ENDED_DRIVE_PROBE=1 tools/memcap.sh 16 -- \
  .venv/bin/python -m research.runners.load_bearing_fraction --only open-ended-generation --repeats 2 \
  --out research/findings/raw/_load_bearing/_oeg_v2/oeg_verify_hostgate.json
```
Result (`research/findings/raw/_load_bearing/_oeg_v2/oeg_verify_hostgate.json`): **load_bearing=False, verdict=pass, treatment_diffs=0, control_diffs=0, null_control_clean=True, lesion_reproduced=True, determinism=True.** intact→`(dog,chase,rabbit)`; lesion→`(dog,chase,rabbit)` — IDENTICAL. The draw lesion did not change the single-turn reply.

## The three masking layers (each measured, each treat=0)

**Layer 1 — KB degeneracy (the default `rich_open` probe).** The tiny KB has ONE chase fact, `(dog,chase,cat)`, already stored, so the only reachable `(dog,chase,?)` patient is novelty-excluded → `_generate_hypothesis` abstains in BOTH arms. (The 2026-09-19 baseline reading.)

**Layer 2 — the spiking plausibility GATE masks the draw (default `BRAIN_SPIKING_PLAUSIBILITY`, ON).** The v2 probe teaches a predator-prey chase KB (rabbit chased by wolf/fox/hawk/eagle → chase~rabbit co-occurrence 4; five other predator→prey singletons) so 6 genuine novel+plausible patients exist for `(dog,chase,?)`. But with the DEFAULT-ON spiking plausibility gate, both arms STILL abstain → treat=0. OEG_DEBUG on the intact arm (env `{}`, `research/findings/raw/_load_bearing/_oeg_v2/oeg_debug_layers.txt`): draw histogram `{cat:109, rabbit:164, deer:27, beetle:56, boar:44}` over 400 attempts — EVERY novel candidate is rejected `implausible` on its first draw (then `seen` thereafter). The spiking associative read of `_plausible = _related(dog,chase) AND _related(chase,p)` does not fire on the tiny KB's weak agent-action edge (`_related(dog,chase)`, co-occurrence 1), so no candidate is admitted. The lesion (`BRAIN_SPIKING_DRAW_LESION`) ablates the DRAW likelihood, NOT this GATE, so it cannot change the outcome — a companion process dominates the measurement (the CLAUDE.md wall-reframe, made concrete).

**Layer 3 — even unmasked, the single soft-WTA draw is OU-noise-dominated (host gate).** The committed probe applies `BRAIN_SPIKING_PLAUSIBILITY=0` (host `P>=tau` gate) to BOTH arms so the gate admits the 6 candidates — reaching the draw (the episodic-drive precedent: construct the condition that exercises the target faculty; both arms get it, so the only inter-arm difference stays the draw lesion). Now the draw IS reached: intact volunteers `(dog,chase,rabbit)` (OEG_DEBUG in the cited debug artifact: `cat` rejected, `rabbit` accepted at draw 2 — the likelihood peak, weight 4). But the LESION (uniform draw) ALSO accepts `rabbit` (OEG_DEBUG: `words,memory,cat,memory` rejected, `rabbit` accepted at draw 5). Both → rabbit → treat=0. The likelihood ablation does not change the first-accepted patient because the operating point has OU noise std 200 pA while the likelihood drive spread is only ~120 pA (base 110, gain 160, `drive = base + gain*(w/peak)`: rabbit 270 vs a weight-1 patient 150 <!--derived: constants from vocab_agnostic_spiking_generation_production_organ.py, arithmetic in oeg_debug_layers.txt-->) — noise > signal for the single-draw argmax, so the uniform draw lands on a plausible patient (here the same one) just as the biased draw does.

## The genuine WIRING fix (necessary, banked)

The v1 finding diagnosed and this v2 RE-CONFIRMED on the real brain that the lesion was INERT on the production draw: `SpikingWTASampler.draw_from_weights` (the production wire-in `GenerativeReplayProposer._sample_weighted` calls) took the caller's `_weight_partner` weights and never consulted `ablate_likelihood` — only `_weights()` (the seed-word `_draw`/`draw_many` path the B1 LESION verify uses) honored it. Fixed: `draw_from_weights` now replaces the caller's weights with a uniform vector when `ablate_likelihood` (the EXACT `_weights` ablation semantics). Only reached under `BRAIN_SPIKING_DRAW_LESION=1` (no default/production path sets it) → BYTE-IDENTICAL when not lesioned. This is a correctness fix worth keeping regardless of the verdict — without it the draw lesion is a silent no-op, exactly the false-negative the load-bearing instrument must not make.

## Reconciliation — the draw IS load-bearing, but distributionally, not single-turn

This is NOT a claim the generative draw is inert. Its lesion collapses the aggregate plausible-FRACTION 0.83→0.04 <!--derived: prior B1 GO finding, not re-measured here--> in the B1 draw_many verify (`vocab_agnostic_spiking_generation_production_organ`) — a real, GO'd, distributional load-bearing signal. The mismatch is the INSTRUMENT: `load_bearing_fraction` reads a SINGLE turn's categorical decision diff, and a single soft-WTA draw at this operating point is noise-dominated for the WHICH-patient choice, so ablating the likelihood does not reliably change one turn's volunteered patient. "The instrument is part of the emulation": the single-turn decision-diff instrument is the wrong instrument for a faculty whose contribution is distributional. Scoring open-ended-generation by its draw_many plausible-frac lesion (already GO) is the faithful read; the single-turn instrument should report it as `distributional-not-single-turn`, not fold it into the hollow count.

## What is NOT claimed / honest residual

No single-turn flip is claimed (there is none — treat=0, honestly). Making the single-turn draw lesion-load-bearing would require either a less-conservative plausibility operating point (so the default gate admits candidates — layer 2) or a larger likelihood/noise ratio (so the ablation changes the single draw — layer 3); both are OPERATING-POINT changes, and sweeping either until the metric flips is exactly the metric-tuning this arc forbids, so neither was done. A RICHER KB alone does NOT fix it (the v2 KB already affords 6 admissible candidates); the binding constraints are the gate operating point and the draw noise/signal ratio, not KB size. Single-seed (seed 42) is inherent to the tiny-demo instrument. Brain-based-only + honesty boundary preserved: the DRAW is a `cp_firing_states` read on a firing WTA bank; `hypothesis`/`hypothesis_svo` are functional volunteered-guess read-outs (moat-verified, never passed as known fact).

## Files (branch `research/gap-open-ended-generation-v2`)
- `research/runners/_followon2_spiking_wta_sampler_derisk.py` — `draw_from_weights` honors `ablate_likelihood` (uniform when lesioned; byte-identical off). The wiring fix.
- `research/runners/load_bearing_fraction.py` — `LB_OPEN_ENDED_DRIVE_PROBE` flag (default OFF), the `measure_faculty` remap (turn `oe_ask`, fields `hypothesis_svo`/`answer`, base_env `BRAIN_SPIKING_PLAUSIBILITY=0`), the `_draw_from_weights_honors_ablate` code-level check, 4 new selftest checks.
- `research/runners/onebrain_regression_battery.py` — the `oe_t1..oe_t9`→`oe_ask` predator-prey teach group in `_EXTRA_TURNS` (label-only; `PROBE_TURNS` stays 26 → default roster + every flip-verify harness BYTE-IDENTICAL).
- `research/runners/brain_chat_tui.py` — env-gated (`OEG_DEBUG`, default OFF, byte-identical) draw-histogram + gate-rejection trace in `_generate_hypothesis`; the instrument that produced the layer-2/3 evidence.

## Provenance (committed artifacts)
- `research/findings/raw/_load_bearing/_oeg_v2/oeg_verify_hostgate.json` — the required-verify output (host-gate probe): load_bearing=False, verdict=pass, treatment_diffs=0, control_diffs=0, null_control_clean=True, lesion_reproduced=True, deterministic=True.
- `research/findings/raw/_load_bearing/_oeg_v2/oeg_debug_layers.txt` — the OEG_DEBUG draw histograms for layer 2 (default spiking gate, all-implausible-reject → abstain) and layer 3 (host gate, intact rabbit@draw2 / lesion rabbit@draw5), plus the draw operating-point constants.

## Distributional 6-seed confirmation — GO (2026-09-21, the recommended ruler)

The recommendation above (score open-ended by its DISTRIBUTIONAL lesion metric, not the single-turn decision-diff)
is now robustly confirmed. The `_followon2_spiking_wta_sampler_derisk` distributional metric — draw-many, then
plausible-fraction-of-novel — run 6-seed (42/43/44/100/101/102) on the merged main code lands **verdict = GO**.
Artifact: research/findings/raw/_load_bearing/_followon2_openended_distributional_6seed.json
- PLAUSIBLE (across all 6 seeds): spiking plausible-frac ~0.337, advantage ~17.2x the random floor (>= 3.0x on <!--derived-->
  every seed), spiking/host quality mean ~1.03 (>= 0.7 on every seed) — the spiking draw matches host quality.
- LESION collapses all seeds: True — ablating the likelihood collapses the plausible-frac (the load-bearing proof);
  the SHUFFLED-graph control also collapses (True), so the effect is the real co-occurrence structure, not noise.
- PROVENANCE + noise-ablation all seeds: True — the draw is genuinely the spiking winner read from
  cp_firing_states (ou_std->0 collapses it to the deterministic argmax), i.e. brain-based, not a host rng.choice.

So open-ended-generation IS load-bearing — measured by the right (distributional) ruler, 6-seed robust. The
single-turn battery still reads it negative (correct for that instrument); the honest metric for this faculty is the
distributional one. Wiring THAT verdict into the load_bearing_fraction battery's per-faculty counting is the
remaining refinement (deferred; the science is settled here).

## Sources (external literature — DR gate for the load-bearing lane)
The generate-then-select structure (a draw shaped by likelihood, then a selectional-preference plausibility gate) matches the hippocampal generative-replay compositional-inference account: Schwartenbeck et al. 2023 (Cell, PMID 37804832, https://pubmed.ncbi.nlm.nih.gov/37804832/) — replay assembles stored elements into compounds, each sequence "a hypothesis about a possible configuration"; Kurth-Nelson et al. 2022 (Neuron) — replay strings role-bound objects into novel compound statements. The load-bearing residual here is measurement, not mechanism: replay's contribution is over the DISTRIBUTION of generated hypotheses (which the draw_many plausible-frac captures), not a single sample's identity — consistent with replay being probabilistic sampling from a learned generative model, not a deterministic retrieval.
