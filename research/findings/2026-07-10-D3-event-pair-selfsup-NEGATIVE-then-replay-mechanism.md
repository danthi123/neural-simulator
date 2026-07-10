# D3 EVENT PAIR, self-supervised: forward prediction does **not** teach a brain to hold a prior event — **replay does**

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_selfsup_pair_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** an honest **NEGATIVE** (6/6 consistent) followed by the **mechanism that surpasses it** (5/6 GO; seed 43 at 0.366 vs the 0.40 gate).

## The problem this exposes
The connectives rung learned its event-pair δ from per-step `(agent, patient)` **state labels**. Removing them (the master directive) exposes a genuine gap:

> Predicting the **current** agent's emission gives the `a_prev` slot **no gradient at all**. Nothing in a purely forward-predictive objective teaches a brain to hold a *prior* event.

## THE NEGATIVE (6/6 seeds, well-controlled)
With a fair probe (fitted and read on the **informative subset**: the prior event is real and differs from the current agent):

| | mean | range |
|---|---|---|
| prediction-only — **held (prev) agent** | **0.226** | 0.204 – 0.258 |
| prediction-only — *current* agent | 0.698 | 0.649 – 0.743 |
| chance | 0.167 | |

**Forward prediction learns the current event fine and teaches the held one nothing.** I first hypothesised that language's own **discourse pop** (Grosz & Sidner: a "meanwhile / again" clause that RETURNS to the prior protagonist) would supply the missing pressure — after a RETURN, the emission *depends* on `a_prev`, so holding it becomes necessary to predict. It doesn't work: the credit must travel back through many softmax slots to the boundary where `a_prev` was set, and the model simply eats the loss.

## THE MECHANISM: replay / retrodiction
The prior event's emission **was itself observed** when that event was current. The biological signal for consolidating a just-ended episode is **replay** (hippocampal sharp-wave ripples — machinery this project already has). So the held slot is taught by **retrodiction**: reconstruct the just-ended event's last **observed** emission from `a_prev`. No state label — the target is a symbol the brain heard.

| arm (held/prev agent) | mean | range |
|---|---|---|
| prediction-only (the negative) | 0.226 | 0.204 – 0.258 |
| **+ REPLAY (retrodiction)** | **0.492** | 0.366 – 0.553 |
| + REPLAY, EMISSION-SEVERED | 0.223 | 0.212 – 0.233 |
| + REPLAY, SINGLE-SLOT (no prev slot) | 0.230 | 0.197 – 0.265 |
| RECENCY | 0.164 | |
| *one-emission decode ceiling* | *0.755* | |

Replay **doubles** the held-slot decode; severing the agent→emission link or removing the prev slot both collapse it back to chance.

## My own hypothesis, refuted by my own control
**Replay *without* any discourse pops scores 0.531 — as good as or better than replay *with* them (0.492).** So the discourse pop is **not** what teaches the held slot; **replay is.** The RETURN op makes the prior event *useful*, but usefulness is not a training signal. Reported as a refutation of the framing I built the rung around.

## This resolves the cited HAE/TEM discrepancy — from both sides
The project's TEM/HAE read says *"loss = L_rec + γ·L_pred; prediction-alone collapses, so the reconstruction anchor is load-bearing."* Earlier (`2026-07-10-D3-event-selfsupervised-delta-GO.md`) I showed prediction-alone works fine for the **current** slot, and an adversarial skeptic proved my explanation for *why* was wrong (a copy path makes the probe rise, not collapse). The correct account, now complete:

- **CURRENT slot:** the emission target **moves** across the discourse, so no static input→target map exists; the target itself requires memory ⇒ **prediction alone suffices**, no anchor needed.
- **HELD slot:** nothing in the forward objective ever queries it ⇒ **prediction alone fails**, and a reconstruction/replay anchor **is** load-bearing.

The anchor is not universally required — it is required exactly where the objective supplies no gradient.

## Honest reporting
- **5/6 seeds GO; seed 43 lands at 0.366 vs the 0.40 gate** (its replay-no-return arm reaches 0.379). Reported, not rounded.
- **Replay recovers ~65% of the achievable ceiling** (0.492 vs 0.755). The replay target is a *single, noisy* emission sample (~72–79% agent-modal), so the held slot cannot identify the prior agent better than one sample affords. The gap to ceiling is real and unexplained.
- **Two self-caught defects before any of this was believable:** (1) an early RETURN could pop to the *empty* initial state, making `ident` a 42.5% majority class that every arm — including an untrained one — predicted; found because `P(a_prev==ident)` exactly matched all three arms' scores. (2) The probe, fitted on the full split, collapsed to that majority class and scored a meaningless 0.000 on a subset excluding it. Both fixed (a return now requires a prior event; the probe is fitted *and* read on the informative subset).

## Next
The replay signal on spikes (the project's SWR machinery is the natural substrate); closing the gap to the one-emission ceiling; a deeper event stack.

## Files
`research/runners/_d3_event_selfsup_pair_derisk.py`; the labelled pair `2026-07-10-D3-event-discourse-connectives-GO.md`; the single-slot self-sup rung `2026-07-10-D3-event-selfsupervised-delta-GO.md`.
