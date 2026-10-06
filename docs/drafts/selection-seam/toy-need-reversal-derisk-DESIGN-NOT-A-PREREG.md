---
type: draft
status: superseded-mechanism-derisk-only
date: 2026-09-29
mechanism: need-state-reversal-bg-selection-seam
runner: research/runners/_need_reversal_selection_seam.py
lane: A (affect); plan step 3 of docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md
---

# (WITHDRAWN as a preregistration) An interoceptive need reverses the basal-ganglia winner: toy mechanism de-risk

**Withdrawn 2026-09-29 before any gate seed ran (coordinator ruling on the mid-flight audit, `.claude/worktrees/_audit_2026-09-29/selection-seam.md`).** Its commit read is the host first-crossing rule, several gates hold by construction (G3, G5, the G6 byte-identity, HS(after)=0, NS(before)=0; NS(after) is implied by G2), the need->value mapping is wired (learned-content fraction 0) and no live seam changes. It stays as a DEV mechanism de-risk only; never run it on gate seeds. The step-3 target is the live seam: `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md`.

**Filed before any gate seed (42/43/44/100/101/102) has run.** Dev seeds 1-12 were used to choose the operating
point (tune data: `research/findings/raw/_need_reversal_seam/dev_tune/`, `dev_tune2/`). Design and rationale:
`docs/drafts/selection-seam/DESIGN.md`. Review of the prior N-channel work that this lane replaces:
`docs/drafts/selection-seam/REVIEW-of-local-model-nchannel-bg.md`.

## Question

In the production basal-ganglia selector the host scores each candidate (a salience computed from what the candidate
is, injected as current), and the spiking race transmits that score. When the candidates' relative value instead
depends on an internal need state that reaches the striatum only through synapses, does the neural winner reverse with
the need, and does the host lose all causal control over which candidate wins?

## Fixed substrate and operating point (frozen)

- Gate-A v2 selector, unchanged: `research/runners/_vocal_action_selector_gate.py::build_selector_bridge` with
  `selector_config("v2")`, commit NMDA on, OU sigma 40 pA on the proposal pools, dt 1 ms, `cfg.seed` = the seed.
- Added through the builder's `extra_regions` / `extra_pathways`: `intero_satiety` and `intero_hunger`, 24
  Izhikevich RS neurons each (no recurrence), driven by the body current `400 pA x (1 - h)` and `400 pA x h`.
- Need synapses: `intero_satiety -> str_d1_0` and `intero_hunger -> str_d1_1`, density 0.5, weight 20.0, jitter
  0.05, `exc_receptor="nmda_slow"` (`enable_nmda_recurrent=True`), frozen, transmission gate `need_value`.
- Channel 0 = "rest", channel 1 = "forage", known only to the world-side `WorldCandidates` table.
- Backend numpy. Revision: the commit that adds this file; the runner is not edited between this commit and the gate
  run (any edit is an amendment filed first).

## Arms and blocks (per seed; every block is a fresh bridge built at the seed)

Extremes h = 0.0 (sated) and h = 1.0 (hungry), 60 trials per block:

| block | arm | manipulation |
|---|---|---|
| `after/intact` | synaptic_need | need gate 1, host salience (0, 0) |
| `before/intact` | host_salience | need gate 0, host salience = `host_value_rule(h, labels)` x 600 pA (the production pattern) |
| `after/need_lesion` | synaptic_need | need gate 0 |
| `after/shuffled_synapses` | synaptic_need | each interoceptive pool projects to both D1 pools at density 0.25 |
| `after/swapped_interoception` | synaptic_need | the deficit drives the satiety relay and the surplus the hunger relay |
| `after/arousal_lesion` | synaptic_need | no shared arousal current |
| `after/direct_path_lesion` | synaptic_need | D1 -> GPi transmission gate 0 |
| `after/host_intervened` | synaptic_need | world labels swapped AND host need belief = 1 - h |
| `before/host_intervened` | host_salience | host need belief = 1 - h |
| `before/neural_intervened` | host_salience | interoception swapped, host belief = h |
| `after/host_override` | synaptic_need | host injects the production salience for the non-preferred option |

Curve (reported only): h = 0.25, 0.5, 0.75 in both intact arms, 20 trials each. Ablations (section below): 60 trials
per extreme.

## Measures

- Commit rate per block; P(forage | committed).
- Reversal R = P(forage | committed, h=1) - P(forage | committed, h=0); UNDEFINED if either extreme commits on fewer
  than half its trials.
- Host share HS (intervention on host variables, neural need fixed) and neural share NS (intervention on the neural
  need, host variables fixed): HS = (P(H | intervened) - P(H | baseline)) / (1 - P(H | baseline)), H = the option the
  intervention favors, computed at each extreme.
- Byte-identity: SHA-256 digest of every trial's (winner, decision step, motor counts, region spike counts).
- Touch audit: reads of `WorldCandidates` before the commit, per block. Static AST audit of `run_need_trial`.
- Decoding (reported, not gated): sign-rule accuracy of the host variable, the interoceptive differential and the D1
  differential for the committed winner.
- Substitution (reported, not gated): P(forage) in the synaptic arm minus the host arm at matched h.
- Production anchor (reported, not gated; `production_baseline`): the same host-share intervention on the live
  `bg_action_selection_production_organ.BGActionSelector` (host salience (0, 1) vs flipped (1, 0), 60 trials each,
  fresh organ at the seed), plus which of four fixed messages the host decides without consulting the race.
- Closed loop (reported, not gated; `closed_loop_episode`): one persistent bridge, 150 trials, the deficit rises by
  0.08 per trial and a committed forage lowers it by 0.35; intact vs need synapses closed. Reported: mean deficit,
  forage rate, corr(deficit, forage), P(forage | deficit >= 0.6) and P(forage | deficit <= 0.4).
- Every trial's membrane range must stay in [-150, +100] mV (`tools/lab.py::membrane_bound_check`); a violation
  raises and the seed is UNDEFINED.

## Gates (per seed; the code is `seed_verdict`, `overall_verdict` and `overall_block` in the runner, bars in `BARS`)

- **G1 commit:** `after/intact` commit rate >= 0.90 at h = 0 and h = 1.
- **G2 reversal:** R >= 0.50, with P(forage | h=1) >= 0.70 and P(forage | h=0) <= 0.30.
- **G3 need-synapse lesion:** |R| <= 0.25 (evaluable only if both extremes commit >= 0.50).
- **G4 shuffled synapses:** |R| <= 0.25 (same evaluability).
- **G5 swapped interoception:** R <= -0.50 (same evaluability): the winner follows the neural need state.
- **G6 code path:** zero pre-commit reads in every synaptic-arm block; static audit clean; `after/host_intervened`
  digest identical to `after/intact` at both extremes; and more than zero pre-commit reads in every host-arm block
  (the audit's positive control).
- **G7 host share:** HS(before) >= 0.80 at both extremes (the instrument detects host ownership); HS(after) = 0 at
  both extremes; NS(after) >= 0.50 at both extremes; NS(before) <= 0.20 at both extremes.

Preconditions (through `tools/verdict.Verdict`; a failure makes the seed UNDEFINED, never a NO-GO): the need gate
reads back 1.0 in `after/intact`, 0.0 in `before/intact` and `after/need_lesion`; the direct-path gate reads back 0.0
in `after/direct_path_lesion`; the static audit ran; `after/intact` R is defined; each of G3-G5 is evaluable.

**Seed decision:** GO iff G1-G7 all pass and every precondition holds; UNDEFINED if a precondition fails; else NO-GO.

**Overall** (emitted as a top-level `verdict` only over exactly these six seeds; any other seed set is a
`dev_readout`): GO iff 6/6 seeds GO. PARTIAL iff G6 and G7 pass on all six seeds and G1-G5 pass on at least four.
UNDEFINED iff three or more seeds are UNDEFINED. NO-GO otherwise. A PARTIAL is not a GO.

## Mechanism necessity (non-negotiable 4; reported, not gated)

Each ablated alone in the synaptic arm, 60 trials per extreme: need synapses (gate 0); interoceptive relays silent
(no body current); shared arousal; direct path (D1 -> GPi gate 0); D2 MSNs; GPe; STN; commit FS pools (the
cross-channel commit inhibition); thalamus; commit NMDA (build without it); OU noise (sigma 0). Inactivation = a
constant -1000 pA on the region (DEV-checked on seeds 1 and 8: silences every target with V >= -95 mV; -3000 pA drove
the thalamic relay to -175 mV and into rebound firing), verified by the region firing <= 1 spike per trial.

Per seed, the capability is present iff commit >= 0.80 at both extremes and R >= 0.50. Label: NECESSARY iff absent
on >= 5/6 evaluable seeds; NOT NECESSARY iff present on >= 5/6; CONTRIBUTING otherwise; UNDEFINED if fewer than 5
seeds are evaluable (inactivation failed); NOT ENGAGED if every inactivated region already fires <= 1 spike per
trial in the intact race at both extremes on every seed.

**Expected from dev (stated before the gate run):** the GPe and STN are silent in the intact Gate-A v2 race (the GPe
receives no tonic drive), so GPe and STN are expected NOT ENGAGED, and the D2 ablation expected NOT NECESSARY because
its only target is the silent GPe. A NOT ENGAGED label is a finding about the selector's operating point, not
evidence that the indirect path is dispensable in biology.

## Run command (verbatim)

```bash
bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy python -m research.runners._need_reversal_selection_seam \
    --seeds 42 43 44 100 101 102 --allow-gate-seeds --trials 60 --curve-trials 20 --production-baseline --closed-loop \
    --out research/findings/raw/_need_reversal_seam/gate_6seed.json
```

(Per-seed invocations with `--seeds <s>` and `--out .../gate_seed<s>.json`, aggregated by `overall_verdict`, are
equivalent; the pool may run them in parallel.)

**Gates that hold by construction once the code path is clean** (stated so they are not read as independent
evidence): HS(after) = 0 and the `after/host_intervened` byte-identity (no host variable reaches the race); the
need-lesion R = 0 (with the gate closed both extremes replay the same noise); G5 (swapped interoception at h = 1 is
current-for-current the intact arm at h = 0). The gates that can fail on data are G1, G2, G4, the HS(before) and
NS(after) magnitudes, and G6 (a regression in the code path). DEV rehearsal on seeds 1-12 (before this file was
committed): 12/12 seeds pass every gate (`research/findings/raw/_need_reversal_seam/dev_rehearsal2/`).

## What each outcome would mean

- GO: for this seam the substrate owns the choice between two candidates of state-dependent value; the host-share
  measure moves from about 1 (production pattern) to 0. The need -> value mapping is wired, not learned; the claim
  is ownership of selection and commitment, not self-organized value.
- PARTIAL / NO-GO on G1-G5 with G6-G7 intact: ownership holds but the neural reversal is not reliable at this
  operating point; the residual is quantified per seed and the next method is taken (no-defer).
- NO-GO on G6 or G7: the host still reaches the race; that is the finding, and the leak is located first.
