# D3 EVENT → the FULLY SELF-SUPERVISED transition δ (6-seed GO, adversarially verified): the running meaning EMERGES from prediction, with NO state label

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_selfsup_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102) — **on a gate hardened by three independent adversarial skeptics**, all of whom failed to refute the mechanism but overturned four of my reported margins.

## The crux this closes
Every prior event rung learned the transition δ from a **host `(agent, patient)` state label** (per-step, or end-state-only in the weak-supervision rung). **A real brain is never told who the agent is.** That label was the last host supervision inside the event composition — a residual shortcut against the master directive. This rung removes it: **δ is learned from an agent-emission cross-entropy ALONE.**

## The self-supervised signal
Each clause EMITS a symbol drawn from the **current agent's** distribution `theta[a_t]`. The emission is a **TARGET ONLY, NEVER AN INPUT** (independently audited by two skeptics: `EMIT` appears solely as the backward CE target; forward + rollout read `SUBJ`+`OBJ` only; `TA` never touches the gradient). So the model cannot read the agent off the current observation — to predict the emission it must MAINTAIN the running agent:

| op | utterance | what the K-way slot must do |
|---|---|---|
| INTRODUCE | "s V o" | the subject NAMES the agent → **SET** |
| AGENT-COREF | "he V o" | the utterance does NOT name it → **PERSIST** (deep) |
| PROMOTE | "it V o" | agent ← the previous patient → **BIND** the observed object |

Trained on lengths (2,3,4); evaluated on held-out-DEEPER (6,7,8) — genuinely disjoint (0/1500 sequence or suffix overlap, verified).

**Eval = a frozen-state linear probe** (state → agent identity). Labels are used ONLY to *read* what the unsupervised state encodes, never to learn δ. A skeptic confirmed a 6-dim linear probe cannot fabricate this: shuffled-(state,label) null **0.181**, random-gaussian-state null **0.165** (chance 0.167).

## Result (6-seed, hardened controls; `theta_peak=3.0`, emission purity 0.73)
| | mean | range |
|---|---|---|
| **SELF-SUP (overall)** | **0.940** | 0.905 – 0.985 |
| **SELF-SUP on coref-DEEP (≥3 trailing corefs)** | **0.841** | 0.725 – 0.962 |
| **SELF-SUP on promote-bound finals** | **0.949** | 0.912 – 0.987 |
| FAIR reservoir (ESN + ridge) | 0.652 | 0.613 – 0.681 |
| **FAIR reservoir on coref-DEEP** | **0.147** | 0.111 – 0.172 |
| honest label-free floor `last-named-subject` | 0.592 | 0.569 – 0.608 |
| …the same floor on promote-bound finals | 0.169 | 0.151 – 0.191 |
| emission-severed (`random_emit`) | 0.257 | 0.232 – 0.277 |
| no-recurrence | 0.368 | 0.354 – 0.391 |
| *(weak reference)* recency | 0.167 | 0.153 – 0.180 |

**The two decisive contrasts:**
1. **coref-DEEP: 0.841 vs a fair reservoir's 0.147** — the reservoir is *at chance* (0.167) exactly where deep coref tracking is required, while the trained model holds.
2. **promote-binding: 0.949 vs the honest floor's 0.169** — `last-named-subject` structurally *cannot* bind a promoted patient, and promote-bound finals are ~51% of the test set. This is the genuine novel capability.

## Adversarial verification (three skeptics, run BEFORE this entered the record)
All three: **the mechanism is real and could not be refuted.** All three independently verified the emission never enters the forward pass and the labels never enter δ's gradient. Between them they overturned four of my reported margins, and every fix is applied above:

1. **My `untrained=0.24` control was a degenerate reservoir** (it probed only the collapsed K-dim softmax slot of a random-init net), inflating the "learning over architecture" margin ~3×. **Fixed:** replaced with a *fair* echo-state network (512-dim recurrent, ridge read-out) at 0.652 — which nonetheless collapses to 0.147 on deep coref.
2. **`recency` (0.167) was a weak strawman.** The honest label-free floor is **`last-named-subject`** (latch the last INTRODUCE subject; reads only observables, zero labels) at 0.592. **Fixed:** it is now the gated floor; recency is demoted to a reference.
3. **`random_emit` and `untrained` are ONE control axis, not two** (they land within 0.01 of each other; test-state cosine 0.84). **Fixed:** reported as a single emission-severed line.
4. **The 0.909 aggregate was shallow-dominated** (~half of finals have the agent set at the last clause), so "coref-DEEP" was overstated. **Fixed:** the depth-conditioned and promote-conditioned numbers are now the headline.

**A genuine bug they found:** `theta[e]` peaks on `e % M`, so **K > M aliases agents** (at `--K 10` with the fixed `M=8`, agents 8,9 collide with 0,1). Now auto-enforced: `M = max(M, K)`.

## The cited-literature discrepancy — adjudicated, and my stated reason was WRONG
The project's TEM/HAE read says *"loss = L_rec + γ·L_pred; **prediction-alone collapses to identity**, so the reconstruction anchor is load-bearing."* This runner uses prediction alone, no anchor, and it works. I had attributed that to "the target-only emission plus the hard K-way bottleneck." **A skeptic refuted my explanation:** handing the model a copy path (feeding the previous emission as an input) makes the probe *rise* to **0.96**, not collapse — so the bottleneck is **not** the load-bearing factor and the identity solution simply does not exist here.

**The correct reason:** the emission target **moves across the discourse** — introduce/promote switch the agent, while coref leaves the *subject* agent-independent but the *emission* agent-dependent — so there is **no static input→target map to collapse onto**. The target itself requires memory, which makes prediction alone a *sufficient* self-supervised signal in this setting. (Robustness confirms it isn't riding a near-noiseless label: **0.87** at a 49%-modal emission.)

## A defective control of my own, caught before the skeptics
The first run used a **within-sequence emission shuffle** as the "destroy the structure" control. It scored **0.625** — because on a coref run *every clause emits from the same agent*, permuting within the sequence destroys almost nothing. Exactly the failure the project's `anti-cheat-control-validity-methodology` finding warns about: **match the control to what the mechanism computes.** Replaced with `random_emit` (emissions independent of the agent).

## Honest scope
- **K=6 at the shipped capacity.** Scaling K needs proportional capacity (K=10 wants `n_hid=256`, `epochs≥80` → 0.91–0.95; at the shipped defaults it under-fits to 0.72) **and `M ≥ K`** (now enforced).
- Robust to emission noise (0.87 at 49%-modal) and to deep coref (`p_coref=0.8` → 0.88).
- The K-way slot is an architectural prior (the D3 discrete-attractor agent register). The claim is that **δ over that slot is learnable without state labels** — not that the slot itself emerged.
- The probe reads the agent as a learned *permutation* of slot dimensions (not the identity permutation); "the slot encodes the agent" is true in the linear-decodable sense the probe measures.

## Next
The SPIKING self-supervised δ (re-discretize the self-supervised slot on the FS-WTA substrate); feeding the self-supervised state into the RANK-3 QA (a fully-emergent situation model answering questions); discourse connectives.

## Files
`research/runners/_d3_event_selfsup_derisk.py`; the supervised rungs `2026-07-09-D3-event-*.md`; the QA rungs `2026-07-09-D3-event-QA-*.md`; multi-turn `2026-07-10-D3-event-multiturn-coherence-GO.md`.
