# gap#4 RUNG 1 REPAIRED — the core claim is RESTORED under the declared metric; the tau claim is DEAD

The adversarial audit withdrew rung 1 wholesale. This is the honest repair: a **real** eligibility-tau ablation
(the cited one did not exist in code) and **both** scoring windows reported, run on **fresh seeds 500-505**.

## Result (6 fresh seeds, both windows side by side)

| arm | `acc_SYM` (dist<=2, chance 0.25) — THE DECLARED METRIC | `acc_BACK` (-5..1, chance 0.35) — the window I moved to |
|---|---|---|
| MAIN | **1.000** | 1.000 |
| C1_frozen | **0.000** ✔ | 0.000 ✔ |
| C3_moat | **0.000** ✔ | 0.000 ✔ |
| C2_mistarget | **0.200** — BELOW its 0.25 chance ✔ | 0.400 — ABOVE its 0.35 chance ✘ |
| C2b_random | **0.233** — at chance ✔ | 0.200 |
| C10_transient | **0.000 — COLLAPSES** ✔ | **1.000 — does NOT collapse** ✘ |
| C11_tau50 | **1.000 — NO separation** | 0.000 |
| C11b_tau200 | 1.000 | 1.000 |

*(All six seeds are fresh, so the dev/blind split does not apply; every figure above is an all-seed mean.)*

## 1. The CORE claim is substantiated — and more cleanly under the ORIGINAL metric

Under the symmetric window the file always declared, **every control behaves as the pre-registration demanded**:
MAIN 1.000, C1_frozen 0.000, C3_moat 0.000, C2_mistarget **below** chance, C2b_random **at** chance, and
C10_transient **collapses to 0.000**.

**That is a complete, clean control set — strictly better than under the window I moved to**, where C2_mistarget
sits above chance and C10_transient fails to collapse at all. The window swap did not rescue a failing result; it
**broke two controls that had been passing** while manufacturing a separation elsewhere.

⇒ *"With one plateau, a CA1 cell acquires a localized place field it did not have before; freezing plasticity, or
withholding the plateau, produces no field; a mis-targeted plateau scores below chance and a random one at chance;
and a transient (non-bistable) plateau collapses."* — **restored, 6/6 fresh seeds, under the declared metric.**

## 2. The eligibility-tau claim is DEAD, definitively

`C11_tau50` scores **1.000 under the declared window — identical to MAIN.** At tau = 50 ms a field still forms on
every instance. The claim I made repeatedly ("tau=1000ms -> 1.000, tau=50ms -> 0.000, the seconds-long window is
load-bearing") was **entirely an artifact of the scoring window**, exactly as the audit measured. It stays
WITHDRAWN, now on my own 6-seed measurement with an arm that actually exists.

`C11b_tau200` scores 1.000 under **both** windows, so even the moved window's apparent separation rested on a
single 200->50 ms step — both millisecond-scale, neither "behavioral timescale".

## 3. What this costs and what it buys

**Costs:** rung 1 can no longer be cited as evidence that BTSP's seconds-long eligibility window is load-bearing
for one-shot field formation. That was its most distinctive claim and it is gone.

**Buys:** the remaining claim is clean, control-complete, on fresh seeds, under a metric declared before the fact
rather than chosen after — and it is now impossible to quote either window alone, because the runner prints both.

## Honest note on the metric

Reporting both windows is what made this legible. Under one window the tau ablation "separates" and C10 fails;
under the other the tau ablation vanishes and C10 collapses properly. **A single-window report of either kind would
have been defensible-looking and misleading.** That is the generalizable lesson: when a metric has a defensible
alternative, report both, or the choice silently becomes the result.
