# The reservoir fading-memory ceiling is SURPASSED by a FIXED structured multi-timescale (SSM/HiPPO-extract) recurrence — 6-seed GO on the memory-horizon task, anti-cheat-confirmed; the mission-relevant language escalation is next

**Date:** 2026-07-13
**Runner:** `research/runners/_ssm_fixed_structured_reservoir_derisk.py` (self-contained rate reservoirs; numpy-CPU; NO `sim/` edit). Gate: `2026-07-13-fresh-gate-spiking-SSM-...md`.
**Status:** ✅ CHEAP-FIRST GO (6-seed, anti-cheat-confirmed) on the MEMORY-HORIZON task — the honest surpass of the fading-memory ceiling; the language/generation + spiking escalations named.

## The result (6-seed 42/43/44/100/101/102; cue-decode after N distractors; chance 1/12=0.083)
A cue token, then N disjoint-vocab distractors, decode the cue from the reservoir state after N (a LOCAL linear read-out). DEEP = gap 80-150.
| arm | deep-gap cue-decode (mean over 6 seeds) |
|---|---|
| **MULTI-TIMESCALE diagonal** (SSM extract: `x←a_i·x + W_in·u`, `a_i` spanning τ∈[1.5,1000]) | **1.000** (all 6 seeds) |
| RANDOM reservoir (ESN, tanh, spectral radius 0.95) | 0.084 (≈ chance — FADED) |
| FAST-ONLY diagonal (τ∈[1.5,4], no slow units) — **anti-cheat** | 0.081 (≈ chance — FADED) |
| GATE (mt > random+0.15 AND mt > fast-only+0.15 AND mt > chance+0.15) | **GO — 6/6** |

## Why this is a genuine surpass (both controls fade → the structure is load-bearing)
- The **random reservoir** (the standard echo-state reservoir the whole reslm ladder uses) fades to chance by gap 40 — this IS the fading-memory ceiling that bounds Rung 6 (1.0→0.83 by gap 6 on the discourse task) and the R1 long-range boundary (16% deep capture).
- The **FAST-ONLY diagonal control fades identically** — so it is NOT "diagonal" alone; the **SLOW time-constants** (a range up to τ=1000) are the load-bearing element. The multi-timescale structure holds the cue unbounded.
- **Mechanism:** a diagonal set of leaky integrators spanning a log range of time constants retains a linear input in its SLOW units WITHOUT the random reservoir's recurrent MIXING (which scrambles/dilutes the signal). This is the emergence-compatible EXTRACT of the spiking-SSM class (the structured multi-timescale FORWARD state = the SSM/HiPPO long-range), realized with a FIXED recurrence + a LOCAL read-out — **no BPTT**, consistent with the R3 reframe (fixed reservoir + learned read-out wins) and the multi-timescale-reservoir GO (the principled version). **Biology:** diverse neuronal/synaptic/dendritic time constants ARE the substrate for multi-timescale memory (SpikingSSMs' dendritic-inspired framing; arXiv:2408.14909).

## Honest scope (what this is + is NOT)
- **Linear memory-horizon task** (hold a cue). It DEMONSTRATES the core mechanism (structured multi-timescale surpasses the random-reservoir fade, 6-seed, anti-cheat-confirmed) — NOT yet a language/generation surpass.
- **The mission-relevant escalation (next build):** does the fixed multi-timescale reservoir improve DEEP-CONTEXT LM CE on real text — where the reslm's random reservoir fades (the by-context-depth metric of `_recurrent_lm_ceiling.py`, reusing `load_stream` + bigram baseline)? If the deep-context structure in language is linearly decodable from the multi-timescale state (which the random reservoir cannot expose), the generation ladder's fading-memory wall is surpassed with an emergence-compatible fixed structure.
- **The spiking realization:** LIF neurons with a RANGE of membrane/adaptation time constants (or a spiking diagonal SSM) as the fixed reservoir — the fully-on-substrate version (biology's diverse time constants), NO learned recurrent credit.

## ⇒ Why this matters (the fresh mechanism class, working)
Every prior long-range attempt (reservoir-scale, e-prop, learn-W_in, dendritic-gain, cross-neuron decorrelation) was exhausted at a boundary. The fresh-gate (external: spiking SSMs) surfaced a genuinely-new class whose emergence-compatible extract — **STRUCTURE the fixed recurrence (multi-timescale), don't learn it** — cleanly surpasses the fading-memory ceiling on the memory task, 6-seed. This is the workflow (exhausted ladder → fresh mechanism-class gate → cheap-first GO) turning the long-standing fading-memory wall into a path. NEXT: the language-CE escalation, then the spiking realization.

Runner: `_ssm_fixed_structured_reservoir_derisk.py`. NO `sim/` edit.
