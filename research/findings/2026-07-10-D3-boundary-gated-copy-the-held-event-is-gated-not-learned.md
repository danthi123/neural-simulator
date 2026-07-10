# D3 EVENT PAIR → the BOUNDARY-GATED COPY: a prior event is **not learned, it is gated** — and my "one-emission ceiling" was mechanism-bound

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_gated_copy_derisk.py` (numpy; NO `sim/` edit — the rate de-risk of a route `sim/` **already supports**).
**Verdict:** the mechanism is real and adversarially verified. **The headline is corrected downward by the skeptics: honest held-out selection gives a 6-seed mean ≈ 0.63 (0.52–0.69)**, not the 0.693 obtained at the test-set-optimal constant.

## The reframe that unlocked it
The self-supervised pair rung established an honest **NEGATIVE** (forward prediction gives the held `a_prev` slot no gradient — chance, 0.226) and a mechanism that surpassed it (**REPLAY** → 0.597 at γ=3), with a **"one-emission ceiling" of 0.755** — because the replay *target* is a single noisy emission symbol.

But `a_prev` never needed to **infer** an agent from an emission. At the boundary `a_curr` already holds one. The held slot needs a **COPY**, not an inference. I had forced it to *learn* that copy through a softmax head taught only by a lossy symbol — **so the "ceiling" was the ceiling of the wrong mechanism, not of the substrate.**

## The biology — which this repo already implements
A brain does not learn that transfer by prediction: an event boundary **opens a gate** and the working-memory content transfers (BG output gating, O'Reilly & Frank PBWM 2006; thalamocortical gating, Logiaco-Abbott-Escola 2021). Reading my own substrate found it already there:

> `sim/regions.py: RegionPathway(transmission_gate=...)` + `sim/bridge.py: set_transmission_gate(...)` — *"pre-wire a route with a fixed weight, hold it normally **CLOSED**, and OPEN it on command — binding = which gate is open, not which weight grew."*

```
g_t    = sigmoid(w_g · clause_code + b_g)        # the boundary marker is OBSERVABLE, in the utterance
a_prev = g_t · a_curr_prev + (1 - g_t) · a_prev  # OPEN -> shift the event; CLOSED -> hold it
```
The copy is **structural**; only *when* to open is learned. **No state label anywhere** — the gate reads the clause code; the emission is a target-only observable; `(agent, patient)` labels are used solely by a frozen-state probe on the informative subset.

## Result (6-seed)
| | value |
|---|---|
| prediction-only NEGATIVE (prior rung) | 0.226 |
| REPLAY γ=3 (prior rung) | 0.597 |
| **GATED COPY, honest held-out gate_cost selection** | **≈0.63** (0.52 – 0.69) |
| GATED COPY at `gate_cost=0.01` (test-set-optimal) | 0.693 (min 0.649) |
| **ORACLE gate** (observable marker read perfectly) | **0.738** |
| **random-schedule gate, open-rate matched** | **0.316** |
| one-step-lag ORACLE ceiling ("copy every clause") | 0.456 |
| gate-lesion (never opens) | 0.207 |
| recency | 0.164 |

**The mechanism is real, and it is *comparable to* replay (0.597) — not reliably "far past" it.** What is decisive is *why* it works:
- **Learning WHEN to open is worth +0.38.** A clean random gate with the *same open-rate* reaches only 0.316 (0.149 above chance) while the learned gate is 0.526 above chance — ~3.5× the signal. The gate genuinely reads the marker (which is 0.9996 linearly decodable from the observed clause code).
- **Holding is load-bearing even against an oracle.** A perfect "copy every clause" one-step-lag reader tops out at **0.456**, far below the gated copy.
- ⇒ **the "one-emission ceiling" (0.755) does not apply**: a copy inherits identity rather than inferring it.

## Adversarial verification — two skeptics, both SURVIVE-WITH-SCOPE-FIXES
**Skeptic 1 (leakage / oracle validity).** The oracle is **not** a label leak: `1[op==BOUND]` is linearly decodable from the observed code at **0.9996**, and structurally the oracle sets only a **0/1 timing gate** — the identity copied is the model's own `a_curr`, so it *cannot* inject the probed label even if BOUND were unobservable. Gate and probe are leak-free; all arms share one informative subset.
- **Scope fix applied:** my claim "`a_prev` is bounded by `a_curr`'s fidelity" is **wrong**. Under the oracle, `a_prev` **exceeds** `a_curr` on all 6 seeds (0.738 vs 0.695) — it is a **frozen snapshot** taken at the boundary while `a_curr` keeps churning under corefs/promotes.

**Skeptic 2 (tuning / control validity).** Its own decisive attack **failed to refute** the mechanism (the random-schedule gate result above). But it corrected three things:
1. **The headline was selection-optimistic.** `gate_cost=0.01` is exactly the test-set argmax. My own clearing check used **one** held-out triple (7/8/9 → 0.01) — not enough. Across **five disjoint** held-out triples selection splits 0.01 (×2), 0.006 (×2), 0.003 (×1), giving an honest reported-six mean **≈0.63 (0.52–0.69)**.
2. **"Beats replay on every seed" is retired.** True only at `gate_cost=0.01` (min 0.649). At the frequently-selected 0.006 the min is **0.480 < 0.597**.
3. **`marker_scramble` was a leaky control** (it still fits `w_g` on permuted codes, plus positional confounds), which is why it scored 0.49–0.57 on two seeds and dragged them to PARTIAL. The clean random-schedule gate (0.316) is the correct control — and it **understates nothing**: the scramble was flattering the null, not the mechanism.
4. **`lr_gate` is less load-bearing than I framed.** It is a plateau, not a knife-edge (x1→0.571, x20→0.675, x100→0.693; degrades past x200). And the "shared-lr → 0.348 collapse" I reported is only true at `gate_cost=0`; with the opening cost present, shared-lr already reaches 0.571.

## Two mechanism constants — measured, and honestly scoped
1. **The gate is normally CLOSED and opening COSTS.** Measured: without the cost the gate drifts open everywhere on bad seeds (seed 101: BOUND 0.930 but COREF 0.906 — separation only **+0.162**, prev 0.273), because opening on a COREF copies `a_curr` into `a_prev` when the agent hasn't changed — nearly harmless to the loss. With it, seed 101 is repaired (prev 0.683, separation +0.88). This is the documented semantics of the route being de-risked, and of tonically-inhibited BG gating. **The effect is large and real for every `gate_cost ∈ [0.003, 0.015]` (all ≫ the 0.226 negative), but the window is delicate and seed-dependent** — 0.01 is the *intersection* of the per-seed windows, and 0.02 slams the gate shut everywhere. The opening cost is **dense** (every clause) while the gradient rewarding opening is **sparse** (~20% BOUND).
2. **The gate has its own faster plasticity channel** (PBWM: the gating net is trained by a separate phasic dopamine signal). Robust across `lr_gate ∈ [1,5]`; not the load-bearing ingredient the first draft implied.

## ⇒ the claim (corrected)
**A brain does not learn to remember a just-ended episode. A boundary opens a gate, and the content transfers.** Making that copy structural turns a slot that forward prediction could not teach *at all* (0.226) into one learnable from **prediction alone**, with **no replay and no state label**, reaching ≈0.63 under honest selection (0.693 at the tuned constant, 0.738 with an oracle gate) — while a random gate of the same open-rate gets 0.316 and a perfect copy-every-clause reader gets 0.456.

## Honest scope + next
- Rate de-risk. The **spiking port** is next, and it now rests on a slot that actually remembers (`2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`: HOLD 6/6, and a gate must CLEAR longer than τ_NMDA before it LOADS).
- The learned gate reaches 0.693 of the oracle's 0.738 at the tuned constant; gate learnability under honest selection is the residual.
- Then: re-deploy the self-supervised pair register on the gated copy (its BEFORE answer was 0.367 with replay).

## Files
`research/runners/_d3_event_gated_copy_derisk.py`; the negative + replay mechanism `2026-07-10-D3-event-pair-selfsup-NEGATIVE-then-replay-mechanism.md`; the persistent spiking slot `2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`.
