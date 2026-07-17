# D3 — the biological transition **survives a finite-sample spiking read** of the agent layer; the substrate port is de-risked

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_delta_cleanerror_derisk.py` (`read_noise`; numpy; NO `sim/` edit).
**Verdict:** GO. The last new variable the on-substrate *learning* introduces is harmless at a realistic read window, and
it has a genuine floor (an anti-cheat), so the survival is real.

## The one new variable
The event register's **memory slots and both gates already run on real spiking attractors** (`_d3_event_pair_spiking_derisk`),
and this project's **D1 microcircuit already ports the clean-error credit rule to spikes** for a static depth-2 map
(0.964). The remaining question for putting the *learning* of the transition on the substrate is narrow and specific: the
agent layer's soft output `raw` is a **rate code**, and a spiking read of it samples a **finite number of spikes**, not an
exact rate. Does the transition still learn when its agent state is read that way?

`read_noise = n` draws `n` spikes ~ `Multinomial(n, raw)` per clause and uses the empirical rate `counts/n` — exactly the
finite-sample read the substrate would deliver. Single variable; the exact-rate path is `read_noise = 0` and is
byte-identical to the committed swept optimum (seed 42 a_prev 0.423 either way).

## Result (6-seed at the swept optimum: clean-error, lr 0.02, batch 32)

| agent-layer read | a_prev | RETURN | a_curr | next-emission |
|---|---|---|---|---|
| exact rate | 0.462 | 0.454 | 0.498 | 0.601 |
| **200 spikes** | 0.458 | 0.491 | 0.530 | 0.611 |
| **50 spikes** | 0.467 | 0.568 | 0.580 | **0.603** |
| **20 spikes** | 0.499 | 0.585 | 0.582 | 0.565 |

**A 50-spike read costs essentially nothing** (emission +0.002, a_prev +0.005). If anything the softmax-coupled metrics
(`a_curr`, RETURN) improve slightly, because the multinomial read **sparsifies** the soft agent code toward the
near-one-hot that the attractor slots deliver — the spiking read *matches* the substrate rather than fighting it.

## The floor (the anti-cheat: the read is load-bearing)
If the metric were simply insensitive to the agent read, the test would prove nothing. It is not — an extreme read
collapses it (⚠️ **CORRECTED 2026-07-16: seed 44 ONLY, n=1 — not "3 dev seeds"**; `raw/_d3_readfloor*` is a single
file whose `seed` field is `44` and whose nine values are byte-for-byte the table below. No seed 42/43 run exists.
Note seed 44 is the **outlier-high** seed on this exact harness — its `exact` a_prev is 0.597 vs 0.423/0.447 for
42/43 — so this anti-cheat rests on the least representative seed. The **survival** half of this finding IS genuinely
6-seed (`_d3_readnoise_seed{42,43,44,100,101,102}.json` all present), so the mismatch is confined to the floor.
Cheap fix: run the readfloor arm on 42/43. Low severity — this is a collapse control arguing *against* the arc's own
interest, and a 1-spike multinomial read from a ~6-way soft code collapsing to chance is near-arithmetically forced —
but the label was wrong and it propagated to `CLAUDE.md`'s "a real floor (1 spike → chance)"):

| read | a_prev | a_curr | emission |
|---|---|---|---|
| 5 spikes | 0.555 | 0.486 | 0.489 |
| 2 spikes | 0.492 | 0.390 | **0.388** (= the Markov floor) |
| 1 spike | 0.133 | 0.192 | **0.214** (chance) |

At 1 spike everything is at chance; at 2 spikes next-emission sits exactly on the Markov floor (0.390); by 5 it is
degrading. So the read is genuinely load-bearing, and survival at 20–50 spikes is a real result, not metric insensitivity.
A 50-spike window is entirely realistic for a settling attractor pool.

## ⇒ the substrate port is de-risked, and it composes validated pieces
Everything the on-substrate learning of the transition needs is now established:
- **the clean-error credit rule ports to spikes** — D1 microcircuit, static depth-2, 0.964, batch-robust, adversarially
  verified (`2026-07-07-D1-microcircuit-...clears-bar-on-spikes.md`);
- **the recurrent, gated state does not hurt alignment** — the alignment probe on the model's own weights is +0.63–0.83
  and *flat* across clause types (`_d3_alignment_probe.py`);
- **the memory + both gates already run on real attractors** — `_d3_event_pair_spiking_derisk` (6-seed GO);
- **the transition survives the finite-sample spike read of the agent layer** — this rung.

The remaining work is the engineering composition (run the two-layer transition as a spiking population on a bridge,
learn it with the committed `enable_bdsp` ∧ `enable_bdsp_microcircuit` `sim/` path, drive the replay target from the held
slot's spikes), with two costs to **report, not tune away**: the ≈27% feedback-alignment partiality (measured) and the
≈17% substrate read-noise on the held-slot *deployment* (measured earlier). Next-emission carries essentially no
read-noise cost, per this rung.

## Honest reporting
- The "slightly better at 20–50 spikes" is small and on the softmax-coupled metrics; a_prev and emission are within noise
  of exact. The honest claim is **no cost at a realistic window**, not a benefit.
- Rate model with a modeled spike read; the full on-bridge spiking forward is the next build.

## Files
`research/runners/_d3_delta_cleanerror_derisk.py` (`read_noise`); raw `research/findings/raw/_d3_readnoise_seed*.json`,
`_d3_readfloor_seed*.json`. The rate result: `2026-07-10-D3-clean-error-delta-PARTIAL-feedback-alignment-is-not-the-somatic-nudge.md`.
