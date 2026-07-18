# gap#4↔#5 unification magnitude polish — the structured-BTSP heterosynaptic-COMPETITION arm is REFUTED by the substrate: it ERODES the within-assembly weight (72→28), it does not sharpen it. The unification STANDS at cue ~0.18 (BTSP uniform, mechanism-6/6-GO — a real completion). Failing method BANKED; infrastructure retained (byte-safe, CI-guarded, may help elsewhere).

**2026-07-18.** Honest negative, read from the substrate (not theorized). Records the refutation of the "structured
one-shot storing rule = BTSP one-shot + heterosynaptic competition" hypothesis for closing the ~0.18→~0.22
completion-magnitude residual of the gap#4↔#5 unification.

## The hypothesis (from the head-to-head)
The structured-vs-uniform head-to-head (same assembly / recall / seed) localized the completion-magnitude residual to
the STORING RULE's structure:

| storing rule | seed 42 | 43 | 44 | mean |
|---|---|---|---|---|
| STRUCTURED (rate-Hebbian + heterosynaptic competition `lam_dep_wi=0.5`) | 0.218 | 0.230 | 0.230 | **0.226** |
| UNIFORM (BTSP one-shot plateau-gated) | 0.183 | 0.171 | 0.184 | **0.179** |

⇒ hypothesis: add the heterosynaptic-competition arm (Milstein-Magee 2021 bidirectional / Chistiakova-Volgushev) to
BTSP's one-shot plateau-gated rule to get the Hebbian's stronger attractor AND keep the gap#4 one-shot behavioral-
timescale credit — a SINGLE structured one-shot storing rule for both gaps.

## What was built (kept — byte-safe, CI-guarded, additive)
- `sim/kernels.py::fused_btsp_hetero_update`: `dw = eta*IS*[Etilde*(w_max-w) - lam_dep*(1-Etilde)*(w-w_min)]` (the
  competition arm; `lam_dep=0` reduces EXACTLY to the pure-potentiation `fused_btsp_update`).
- `sim/config.py::btsp_hetero_dep` (default 0.0) + the guarded bridge sub-branch (default path calls the exact
  `fused_btsp_update` → byte-identical; existing BTSP/bistability CI 11/11 green after the edit).
- `_riii_..._synchronous_assembly_derisk.run(encode_btsp_hetero=…)`.
- CI: `test_fused_btsp_hetero_update_competition_arm` (moat, potentiation, depression-of-non-coincident, eta-inert).

## The refutation — READ FROM THE SUBSTRATE (w_within after encode, seed 42, n_ca3=2000)
| encode | w_within | cue @rk40 | @rk70 | @rk110 |
|---|---|---|---|---|
| UNIFORM (hetero 0) | **71.8** | 0.178 | 0.149 | 0.117 |
| STRUCT (hetero 0.5) | **28.5** | 0.022 | 0.008 | 0.005 |

- The competition arm **ERODES** the within-assembly recurrent weight (72 → 28), it does NOT sharpen it. The
  hypothesis (competition drives within-assembly toward `w_max`) was WRONG.
- Mechanism of the erosion: in the one-shot **plateau-gated** regime, the plateau (`IS_post`) is held HIGH on ALL
  assembly cells during encode, and a co-active assembly cell's *instantaneous* eligibility dips below 1 between its
  spikes → the depression arm `lam_dep*(1-Etilde)*(w-w_min)` fires on WITHIN-assembly synapses too, eroding them. The
  competition cannot distinguish "assembly cell momentarily between spikes" from "true non-assembly input" without a
  nonlinearity on `(1-Etilde)`.
- It is NOT a recall-threshold mismatch: the collapse holds across `recall_k_thresh ∈ {40,70,110,150,200}` (the lower
  `w_within` is the cause; a weaker attractor completes weakly at every threshold). The full 3-seed × 3-dose sweep
  (h0.3/0.6/1.0 all ~0.02) is monotone-worse with more competition — consistent with erosion, not sharpening.
- ⇒ the competition arm is **not portable** from the rate-Hebbian regime (where it sharpens) to the one-shot
  plateau-gated regime (where it erodes). The Hebbian's stronger attractor comes from its rate-window LTP dynamics,
  not a competition term addable to plateau-gated BTSP.

## Status (per THE LAW: bank the METHOD, the CAPABILITY is already met)
- **Failing method BANKED:** "BTSP + heterosynaptic-competition arm" for the unification magnitude polish — REFUTED
  (erodes within-assembly). Infrastructure retained (additive, byte-safe, CI-guarded — a valid mechanism that may help
  in a different regime; e.g. a thresholded-`(1-Etilde)` depression that protects strongly-co-active pairs is a
  possible salvage, DEFERRED below higher-leverage gaps).
- **The gap#4↔#5 unification STANDS** at cue ~0.18 (BTSP uniform, mechanism-6/6-GO): BTSP one-shot plateau-gated credit
  STORES the assembly, the bistable CA3 COMPLETES it, cue-gated + specific + bistable + anti-cheat-verified, 6 seeds.
  A real completion by the gap#5 standard. The stronger 0.226 completion is ALREADY delivered by the gap#5 Hebbian
  rule (its own GO). So there is **no open capability gap here** — this was magnitude polish on an already-GO result;
  the ~0.04 is not chased further (p-hacking risk; the mission prioritizes real OPEN gaps over GO polish).
- **Next (higher-leverage):** the gap#4 KEYSTONE learning-to-accuracy run — the board's named highest-unlock OPEN item.
