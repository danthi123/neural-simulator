---
type: finding
status: live
date: 2026-07-31
mechanism: deep-credit-on-spikes
runner: research/runners/_gap4_onbridge_spiking_selfpredict_derisk.py
artifacts:
  - research/findings/raw/gap4_ceiling/AGG_ceiling_precondition.json
  - research/findings/raw/gap4_crux/AGG_crux_stopped_UNDEFINED.json
---

# gap#4: the crux was never askable — the idealised ceiling cannot fit its own training set

**Verdict: UNDEFINED, and now with a located cause.** This is not a negative about deep credit. It is a
measurement showing the experiment could not have answered the question it was asked.

## The question

The crux ran five credit arms for nine hours and every one landed at or near chance. Before spending more
seeds on it, one precondition had to hold: **is there ANY configuration in which the idealised bound learns
at all?** The `transport_ceiling` arm is that bound — weight transport is ALLOWED (`nwt False`), which no
biological rule gets. If the ceiling cannot beat chance, nothing beneath it is interpretable and more seeds
buy only more noise.

Two ceiling-only cells, `--core-arms-only`, on GPU, differing in task difficulty. Aggregate:
`research/findings/raw/gap4_ceiling/AGG_ceiling_precondition.json`; the crux it reframes is
`research/findings/raw/gap4_crux/AGG_crux_stopped_UNDEFINED.json`.

| config | chance | ceiling held-out | ceiling **train** | oracle |
|---|---|---|---|---|
| n_prop=2 | 0.278 | 0.111 | **0.239** | 1.000 |
| n_prop=3 | 0.167 | 0.148 | **0.129** | 1.000 |

## The answer is no, and the training column says why

Neither configuration lets the ceiling beat chance. But the decisive number is not the held-out score — it
is the **training** score. In both cells the idealised net fails to reach chance **on the data it was
trained on**. That is not a generalisation failure, an overfitting story, or a credit-assignment story.
**The network cannot fit.**

And the task is not the problem: a fenced-backprop oracle scores **1.000** on both configurations, on the
same data, in the same runner. So the difficulty is fully expressible — just not by this substrate.

## What this localises

The failure is **not** the credit rule. Every arm the crux compared — microcircuit, Kolen-Pollack,
fixed-feedback, reservoir — sat beneath a bound that itself could not learn. Ranking them was meaningless,
which is why the crux verdict is UNDEFINED rather than "kp fails".

The failure is **not** task difficulty. The oracle solves it perfectly.

What remains is the **on-bridge spiking forward and read-out**. Even granted weight transport, the spiking
implementation cannot express a mapping that fenced backprop expresses at 1.000. That is where the next
work belongs, and it is a substrate question, not a learning-rule question.

## Cost, recorded because it drove a decision

Each ceiling cell took **4824 s / 4853 s** with `--core-arms-only`. The same runner WITH its four anti-cheat
nets took ~23 h per cell — the arithmetic that stopped the crux at nine hours rather than at its ~136
GPU-hour tail. The precondition that reframed the whole arc cost about eighty minutes on one GPU.

Note also that `n_prop=3` reproduced the crux's ceiling **exactly** (0.148), on a fresh run at a fifth of the
cost. The crux's numbers were real; only their interpretation was wrong.

## Honest scope

One seed per configuration. This is a precondition test, not a 6-seed result, and it is reported as one — a
precondition that fails at one seed is enough to stop an arc, but it would not be enough to *close* a
question. The next step is a substrate probe, not more seeds of this.
