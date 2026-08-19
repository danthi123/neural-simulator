---
type: plan
status: active
date: 2026-08-19
mechanism: deep-credit-on-spikes / backprop-less learning — PLAN OF RECORD + "already tested, do NOT repeat" ledger
supersedes_read_order: read AFTER 2026-08-11-gap4-ALLIN-ARC-SUMMARY (this augments it with the 2026-08-19 mouth-arc evidence + a decision)
---
# Deep-credit / backprop-less learning — plan of record (READ BEFORE RE-ATTACKING)

**Purpose.** Owner-flagged 2026-08-19: we have circled deep credit + dendrites many times; do NOT waste compute/tokens
re-running experiments or re-deriving conclusions we already have. This doc is the durable synthesis + an explicit
DO-NOT-REPEAT list + the one genuinely-open lever + the pending decision. It sits beside
[`2026-08-11-gap4-ALLIN-ARC-SUMMARY`](../../research/findings/2026-08-11-gap4-ALLIN-ARC-SUMMARY-a-spiking-deep-credit-WALL-was-a-hyperparameter-READ-BEFORE-RE-ATTACKING.md)
(the definitive arc record) and the biology catalog (`research/biology/deep-credit-on-spikes.md`,
`dendritic-plateau-coincidence-burst.md`, `urbanczik-senn-dendritic-prediction.md`).

## What is SETTLED (established in our own record — do not re-derive)

1. **The "credit collapses at depth on spikes" wall was three things wearing one label** (2026-08-11 arc summary):
   (a) on the LIF surrogate net it was a **per-arm learning-rate artifact** (a knob that stood ~10 days as fake biology);
   (b) on a **rate** net deep credit **is tractable** via *learned* transport-free feedback — Kolen-Pollack reaches the
   3rd hidden layer, closes ~66% of the depth gap; *fixed* feedback (DFA) does not; freezing the learned feedback
   collapses it; (c) on the **production Izhikevich bridge** the genuine wall is **learning-rate-invariant** — fixed-DFA
   0/6, KP 0/6, DRTP fails, AND a perfect Wᵀ oracle also fails → it is the **few-spike READ regime (multiplexing SNR)**,
   not the feedback rule and not the step size.
2. **The feedback-alignment family is EXHAUSTED** (2026-07-12). The spiking port fails at the multiplexing SNR; the
   named unmet piece is "BurstCCN's two spiking mechanisms our port lacks."
3. **A clean depth-obligatory *spiking* test is not constructible** on a point-neuron spike-count read (both the
   generalisation route — Q5 matched-width — and the fit route — Telgarsky sawtooth — collapse under the finite-spike
   read). Do not spend time building one; it is foreclosed for a documented reason.
4. **Deep credit is NOT the load-bearing blocker on fluent conversation** (owner-verified 2026-08-11). The working
   conversation faculties use zero deep credit; the mouth (speak-with-own-neurons) is bridged by the teacher/BPTT
   scaffold. Owner re-pointed "the crux" away from gap#4 on 2026-08-11.
5. **NEW confirming evidence (2026-08-19 mouth arc).** The mouth read-out e-prop-through-the-substrate-forward plateaus
   at ~0.37 recovery **even at 5× coverage (40k positions)**, while the matched-coverage host-linear-proxy forward
   reaches ~0.90 — so **coverage is excluded** and the bottleneck is the substrate forward's few-spike read, exactly as
   (1c) predicts. (Seed-42 trajectory decisive; full 3-seed `sub_learned_recov` confirmation pending the running job —
   `research/findings/raw/_wkv_readout_eprop_substrate_coverage40k_3seed.json`.) The mouth read-*window* lever
   (120→360) was also tested this session and did NOT move it — so integration-time is excluded too.

## DO NOT REPEAT (each already concluded; re-running wastes compute/tokens)
- FA-family variants (DFA / KP-as-fixed / DRTP) on any substrate — exhausted-negative.
- Per-arm-lr sweeps on the LIF surrogate net — the wall there is dissolved (it WAS the lr).
- KP learned feedback on a **rate** net — already GO (66% gap-close); re-running proves nothing new.
- Depth-obligatory **spiking** task construction (parity/mux/nestedxor/Telgarsky-on-spikes) — foreclosed.
- The mouth read-**window** / integration-time lever — tested-negative 2026-08-19.
- Treating "leaves majority-class / beats a reservoir on XOR" as deep credit — it is not (any fair-lr rule does it).
- Gesturing at "dendrites / burst-multiplexing" as a fresh idea — spec'd 2026-07-01, FA-family exhausted, biology
  already catalogued; the *specific* unmet quantity is the multiplexing SNR, below.

## The ONE genuinely-open lever (not yet done)
Port KP **learned** feedback to the **production Izhikevich bridge** together with a **read-SNR manipulation that is NOT
integration-window** — i.e. raise the effective spike count of the read: higher firing-rate/gain, an **ensemble** read
(average over a population), or a **multi-compartment / dendritic** read (the BurstCCN "two mechanisms our port lacks";
Urbanczik-Senn soma-vs-dendrite). This is one bounded, single-variable experiment. **Honest expectation:** given the
window lever already failed and this is the field's unsolved holy-grail (backprop-less at depth on spikes: the rate
version works, the spiking port hits SNR), the likely outcome is to **characterise the read-SNR wall precisely, not
beat it.** Backprop-less-at-depth is not something we should promise to solve.

## The decision (pending owner — 2026-08-19)
- **Option 1 — one bounded read-SNR run** (spike-count/ensemble/dendritic read, NOT window) on the Izhikevich mouth
  read-out, to definitively close/characterise the residual, then accept the scaffold. A few GPU-hours, single
  variable, no re-tread.
- **Option 2 — accept the scaffold-bridge now** (as 2026-08-11 already decided) and keep the crux on the conversation
  frontier (drive-couplings, memory, the faculties that actually gate talking). Deep-credit stays a documented, mapped
  boundary with its one open lever recorded here.

**DECISION (2026-08-19, owner-delegated best-judgment): Option 2.** Accept the scaffold-bridge; do NOT run a new
deep-credit experiment; keep the crux on the conversation frontier. Rationale: deep credit does not gate conversation
(verified three ways now, incl. the 2026-08-19 mouth arc confirming coverage AND window excluded → the read-SNR is the
wall), and Option 1 would most likely CHARACTERISE the read-SNR wall, not beat it — not worth the GPU on a non-blocking
residual. **Option 1 stays available as a single bounded read-SNR run (spike-count / ensemble / dendritic read, NOT
window) if/when we ever want to formally close gap#4 — never as "the crux."** The running mouth job finishes + lands
its finding as the last read-regime datapoint; no further deep-credit compute is queued.
