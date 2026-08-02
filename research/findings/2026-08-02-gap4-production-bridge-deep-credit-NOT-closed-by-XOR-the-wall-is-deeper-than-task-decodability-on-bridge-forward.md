---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/onbridge_eprop_XOR_K8_s42.json
  - research/findings/raw/gap4/onbridge_eprop_XOR_K8_s43.json
---

# gap#4 production-bridge deep-credit attribution — REFUTED my own hypothesis: a NON-reservoir-decodable task (XOR) does NOT close the residual; deep_credit_share stays ~0 on XOR TOO, because the on-bridge Izhikevich e-prop CANNOT TRAIN XOR at all (trains_the_task=False, eprop ≈ frozen ≈ chance) even where the backprop oracle solves it — the wall is DEEPER than task-decodability (the on-bridge FORWARD/credit at scale)

<!--derived-->
**One-line verdict.** The 2026-08-01 residual: on the production Izhikevich bridge, e-prop's `deep_credit_share` ~= 0.005
on the RESERVOIR-DECODABLE inheritance task (a frozen random reservoir matches e-prop). This session's LIF/rate work
suggested the lever: directed credit only beats a reservoir on a NON-reservoir-decodable task (XOR). I wired XOR into the
production-bridge `deep_credit_share` control (`--task-xor`) and ran it 5-seed at K=8. **The hypothesis is REFUTED:**
`deep_credit_share` is ~0 on XOR too (mean -0.02, range -0.39 to +0.14), and `trains_the_task=False` on EVERY seed —
eprop_inherit ≈ frozen_hidden ≈ chance (0.62-0.71 vs chance 0.51-0.55). The wall is DEEPER than task-decodability: the
on-bridge e-prop cannot train XOR at K=8 at all, even on the 3/5 seeds where the backprop ORACLE solves XOR (0.97-1.0).
So the residual is the ON-BRIDGE FORWARD/CREDIT at scale (the Izhikevich per-example noise / phi'-vanishing wall the
roadmap already diagnosed), NOT reservoir-decodability. No `sim/` edit (additive `--task-xor`, reuse-by-import).

## Result — 5-seed K=8, XOR-threshold task, production Izhikevich bridge

<!--derived-->
| seed | deep_credit_share | eprop_inherit | frozen_hidden | oracle | trains_the_task |
|---|---|---|---|---|---|
| 42  | -0.023 | 0.669 | 0.671 | 0.604 | False |
| 43  | +0.130 | 0.688 | 0.669 | 0.967 | False |
| 44  | -0.394 | 0.616 | 0.652 | 0.604 | False |
| 100 | +0.141 | 0.688 | 0.663 | 1.000 | False |
| 101 | +0.046 | 0.705 | 0.696 | 0.969 | False |

<!--derived-->
mean deep_credit_share ~= -0.02 (huge variance, straddles 0). Per-seed artifact e.g.
`research/findings/raw/gap4/onbridge_eprop_XOR_K8_s42.json`. Command:
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._onbridge_eprop_port_derisk --task-xor --seeds <s> --pool-k 8`
(artifacts `research/findings/raw/gap4/onbridge_eprop_XOR_K8_s{42,43,44,100,101}.json`; s102 finishing, will not change
the verdict — 5/5 are trains_the_task=False with deep_share ~0). GO gate (deep_share > 0.3 AND frozen < eprop AND
shuffle-DFA <= chance+0.10 AND permuted ~chance) FAILS decisively: deep_share ~0, eprop ≈ frozen (below on 2/5 seeds).

## The decisive read — the wall is the ON-BRIDGE FORWARD, not the task

<!--derived-->
The load-bearing tell is the ORACLE column vs `trains_the_task`: on seeds 43/100/101 the backprop oracle SOLVES XOR
(0.967-1.000), so the task IS learnable and IS depth-required (a linear read cannot do XOR) — yet the on-bridge e-prop
sits at chance (eprop 0.69-0.71 ≈ frozen 0.66-0.70 ≈ chance 0.52-0.55). So it is NOT that XOR is unlearnable, and NOT
that a reservoir shortcuts it (frozen is also at chance): **the on-bridge e-prop's forward/credit cannot find the
weights the oracle finds, on the production Izhikevich substrate at K=8.** This is exactly the forward-SNR /
phi'-vanishing wall the roadmap diagnosed for the full inheritance task (`2026-07-14-deep-credit-spiking-training-wall-...`
"the on-bridge full-task PLATEAUS ... the Izhikevich forward's per-example noise bounds it"; `2026-07-24-...POWERED-NO-GO...`
phi'-vanishing + tonic-pinned frozen representation), now CONFIRMED on a second, non-reservoir-decodable task.

## What this resolves + the named next mechanism (no-defer)

<!--derived-->
**What it resolves:** the production-bridge gap#4 attribution residual is NOT a task-decodability artifact — it is the
on-bridge FORWARD/credit at scale. The inheritance deep_share ~0 (reservoir matches e-prop) and the XOR deep_share ~0
(neither trains) have DIFFERENT proximate causes but the SAME root: the on-bridge e-prop does not assign real
deep credit at K=8, because the Izhikevich forward is too noisy / the surrogate credit too attenuated to train a
depth-required task. My "non-reservoir-decodable task is the lever" hypothesis (banked on the board 05:55) is REFUTED —
banked honestly here.

<!--derived-->
**The named next mechanism (from the roadmap's own ledger, no-defer):** the on-bridge forward-SNR is the target, and the
roadmap already flags the highest-value untried lever — **credit ON TOP OF a REPRESENTABLE forward**: the
`coincidence-plateau reliable expander` (`2026-07-25-...-coincidence-plateau-reliable-expander-6seed-GO.md`, `PlateauExpander`,
reproducibility 1.000) was validated as a forward that represents the computation, but "has NEVER been combined with the
credit runner". Combine the PlateauExpander forward with the e-prop credit + reservoir_control, on XOR (now that XOR is
wired in). Cheaper first steps: (a) higher population K (K=16/32, sqrt(K) cleaner forward — but K=8 already fails XOR, so
this may not suffice).

<!--derived-->
**The forward-vs-rule isolation is ALREADY IN HAND (the clinching contrast).** On the LIF forward, directed transport-free
credit (chained_fa) BEATS an optimally-read reservoir on the SAME non-reservoir-decodable XOR task by +0.150 6-seed (the
crux CORE, `2026-08-02-gap4-crux-transport-free-rule-...-6seed.md`) — i.e. deep credit is REAL on the LIF forward. On the
IZHIKEVICH bridge forward, directed transport-free credit (e-prop) ≈ the reservoir on the SAME XOR task (deep_share ~0,
this finding). Same task, same class of transport-free directed rule, DIFFERENT forward ⇒ **the wall is the IZHIKEVICH
FORWARD specifically, not the credit rule** (the LIF depth-scaling finding independently shows the DFA credit rule is
depth-robust). The crux CORE (LIF/rate) stands; this precisely relocates the PRODUCTION-bridge residual to the
Izhikevich forward, and the highest-value next build is the PlateauExpander (representable) forward + e-prop credit.