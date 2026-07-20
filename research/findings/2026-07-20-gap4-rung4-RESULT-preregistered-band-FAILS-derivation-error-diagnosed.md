# gap#4 RUNG 4 — RESULT: the pre-registered band FAILS. Recording the failure, not tuning it.

Pre-registration committed at `356223d0` **before** this run. Predictions P1-P4 as filed. **They FAIL.**

## Result (fresh seed 300, both arms in one invocation)

| arm | c_adj | c_far | `map_ok` | dw |
|---|---|---|---|---|
| `P4_bandOFF` (attribution control) | **1.213** | **2.609** | 1 | 445.5 |
| `MAIN_bandON` | 1.000 | 1.000 | **0** | **0** |

**P4 PASSES** — band-off exactly reproduces the recorded values (1.21x adjacent, 2.60x far), so the comparison is
sound and the arms are not drifting.

**P1, P2, P3 FAIL.** With the band ON, `map_ok=0`: stage 1 never forms the CA1 map at all, so there is nothing to
measure downstream.

## Diagnosed to a specific cause — my DERIVATION, not the kernel

Verified in isolation first: the kernel does exactly what it should. In-band synapses depress (−0.003), the
out-of-band extremes are protected, and high-eligibility synapses potentiate **identically** to band-off. So `dw=0`
was never the kernel refusing to update.

Then measured the actual eligibility distribution over the lap (3995 steps):

| statistic | value |
|---|---|
| max reached | 0.022681 |
| p90 (mean over lap) | 0.017585 |
| **median (mean over lap)** | **0.007665** |
| my derived band | **0.006958 .. 0.015484** |

**The peak IS correctly protected** (0.0227 > band_hi = 0.0155; only 1.85% of the lap has max-E inside the band).
**But the band's lower edge sits directly on the MEDIAN of the distribution** — so it depresses the bulk of the
`pos->ca1` synapses, CA1 never reaches threshold, no field forms. With CA1 silent, L2 receives no presynaptic
activity, so its eligibility is zero and `dw` is **exactly** 0 — which is what the exact zero was telling me.

**The derivation error, precisely:** I placed the band using the relative curve `E(D) = exp(-D/tau)` scaled by
`E_max`, which correctly locates the ADJACENT LAG in eligibility units — but I never checked **how much synaptic
MASS sits at that eligibility**. Locating a lag is not the same as selecting a population. The adjacent-lag
eligibility band happens to contain the median synapse, so "depress the adjacent lag" became "depress most of the
layer".

## ⛔ What I am deliberately NOT doing

The band has two free numbers and one run showing exactly what is wrong. Nudging `band_lo` up until stage 1 survives
would almost certainly produce a passing result within a few attempts — and it would be **worthless**, because the
band would then be fitted to the outcome it is supposed to predict. That is the precise failure mode this session
already committed once (moving a scoring window 22 minutes after banning myself from moving it) and had withdrawn by
an external audit.

**So: P1-P3 are recorded as FAILED as filed.** Any corrected band must be derived from the eligibility
**distribution** (not only the lag curve), **pre-registered again in a separate commit**, and tested on seeds not
yet used (306+).

## What this run genuinely establishes

1. **The band mechanism is live and correctly implemented** — kernel verified in isolation, byte-identical off,
   provably non-inert on, and its effect at the bridge level is real (it changed stage 1 outcome decisively).
2. **`P4_bandOFF` reproduces the recorded contrast values exactly**, so the instrument is sound and the
   1.21x/2.60x adjacent/far asymmetry replicates on a fresh seed.
3. **A quantitative constraint for the next derivation:** the band must depress the adjacent lag while leaving
   enough drive for CA1 to reach threshold. Given median eligibility 0.0077 and max 0.0227, a viable band must sit
   **above the median** — i.e. `band_lo > ~0.008`, with `band_hi` still below the peak (~0.020). That interval is
   narrow, and whether any band in it both spares stage 1 and lifts adjacent contrast is now the open question —
   to be pre-registered, not tuned.

## Honest status

gap#4's blocker remains **adjacent-band contrast**. The first mechanism aimed at it failed on its first
pre-registered test for a diagnosed reason. The mechanism is not refuted; **this particular band placement is.**
