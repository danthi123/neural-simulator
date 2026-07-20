# gap#4 — the ranked contrast candidates are EXHAUSTED: three families, three distinct diagnosed failures

The research gate ranked four mechanisms for raising adjacent-band contrast. Three have now been built and tested,
each failing for a **different and precisely identified** reason. None was abandoned on a hunch.

| # | mechanism | outcome | root cause |
|---|---|---|---|
| Rank 3 | Milstein split-threshold **band** | FAILED, 2 pre-registered attempts, cap fired | **geometric collision**: field spacing (4 bins) == backward shift (4-6 bins), so "the adjacent field" and "where THIS field forms" are the SAME lag. No band in eligibility space can separate what lag space does not. |
| Rank 2 | **zero-DC** difference-of-exponentials | FAILED pre-registered test | **trace-amplitude mismatch**: validated on EQUAL-amplitude idealized traces, but deployed traces are normalized EMAs with increments 1e-3 vs 3.33e-4 → amplitude ratio 0.36, so `a_dep*slow` cannot cancel the fast DC. Over-potentiates 3x and saturates stage 1. |
| Rank 4 | **mean-subtracted increment** (Miller-MacKay) | REFUTED PRE-FLIGHT (no seeds spent) | **the w_min floor breaks the structural guarantee**: mean subtraction makes ~half the increments negative, and **4240 / 8320 weights (51%) clip at w_min = 0**, so the surviving positive increments drift the mean UP. Total dw RISES (+342k vs +206k) — the opposite of the intended effect. |
| Rank 5 | replacing traces | not applicable here | spike count has **zero variance** in this task (every pool fires exactly 250), so the count-vs-timing concern cannot bite; measured corr(eligibility, lag) = **-0.9445**. |

## The process point: Rank 4 cost ZERO seeds

Rung 5's root cause was that I validated a kernel on inputs the implementation never generates. The correction —
**verify the claimed property on the DEPLOYED traces before pre-registering** — was applied immediately to Rank 4
and caught its failure pre-flight, at the cost of two short probes instead of a 6-seed run plus a pre-registration
plus a retraction. One cycle, one lesson, immediately repaid.

⚠️ It also caught a smaller error of the same family: I wrote a print statement concluding *"the clip breaks the
zero-sum"* while that same test showed **0/64 clipped and sum(dw) ≈ 0**. I had written the explanation before
reading the numbers. The real cause (the w_min floor, 51% of synapses) came from actually measuring which bound
was being hit.

## What the three failures have in common

Every mechanism tried so far manipulates the update as a function of **eligibility magnitude** on the synapse's own
afferent. Each fails at a different place, but the pattern is that the adjacent-lag population is not separable
from the field-forming population **by any per-synapse scalar available at the time of the update** — the band
tried to select it, the DoG tried to subtract it, and the mean subtraction tried to normalize it away.

## Where this leaves gap#4 — unchanged blocker, much better characterized

**ROBUST ANCHOR (the most reproduced measurement in this arc):** adjacent contrast **1.213x** vs far **2.609x**,
reproduced 6/6 on fresh seeds across THREE separate runs (rung 4, 4b, 5). The input is not the blocker (afferent
adjacent-bin cos 0.0000 at L1, 0.7436 at L2). The rule already decorrelates as well as the repo's engineered grid
code (0.7436 vs 0.7379).

**NEXT: a second research gate, now armed with three diagnosed failure modes** rather than a general question. That
is a materially better-posed gate than the first one, and it should be asked as: *given that adjacent-lag and
field-forming synapses are not separable by any per-synapse scalar at update time, what mechanism separates them —
or what changes the geometry so that they are separable?* The geometric route (field spacing > backward shift) is a
live, falsifiable option the second failure handed us and which no mechanism above addresses.

**NOT claimed:** that adjacent-band contrast is unachievable. Three methods failed; the capability is untouched.

---

## PF-2 (transfer-loss route) — probe INVALID, but the route is metric inflation and is dropped

**The probe is invalid and its numbers are not cited.** It returned CA1 peaks `[0, 0, 3, 7, 11]` — a duplicate, so
stage 1 produced a degenerate map — because I omitted the `map_ok` guard that the runners themselves carry. Its
`c_adj = 1.003` is therefore not comparable to the recorded 1.213. My own omission, of the exact assertion class
that has caught several errors today.

**One structural point survives, and it kills the route independently of the probe:**

- **Divisive normalization cannot change a contrast ratio at all** — dividing every bin by a constant leaves ratios
  identical (the probe shows 1.003 -> 1.003, as it must).
- **A pointwise expansive read maps `c -> c^p`** (verified in the probe's own numbers: 1.003, 1.006, 1.009 = exactly
  1.003^1,2,3). Applied to the real 1.213, `p=3` gives **1.785 > the 1.60 target**.

**And that is precisely why the route is rejected.** Clearing the bar that way changes no learned weight and adds no
information — it inflates every ratio equally, including noise and the already-healthy far contrast (2.609 -> 17.8).
It is the same family as the read-out-sensitivity lever the record already excluded analytically, arriving by a
different door.

⇒ **The transfer loss is NOT a route to the goal.** The 1.5x compression is real and worth knowing as a constraint
on how much weight contrast any fix must deliver — but recovering it by read-out exponent would be gaming the
metric, not solving the problem. Dropped, and recorded as dropped so it is not revisited as a fresh idea.

---

## Rank 4 CLOSED in BOTH forms — subtractive normalization is structurally incompatible with a Dale substrate

The gate's named correction for the mean-subtracted increment was **`w_min < 0`**. That is **not available on this
substrate**: polarity is assigned PER PRESYNAPTIC NEURON (`exc_fraction` / `inhibitory_indices`), so a negative
weight would flip an excitatory synapse inhibitory and violate Dale's law — and the repo's own decorrelation
research explicitly works under "non-negative (Dale), point-neuron" constraints. Checked BEFORE building it.

The Dale-compliant alternative is an **active-set projection**: subtract the mean only over synapses not pinned at
a bound, so `sum_j dw_ij = 0` holds among exactly those that can move. Implemented and pre-flighted (seed 800):

| | naive mean-subtract | active-set projection |
|---|---|---|
| clipped at `w_min` | 4240 / 8320 (51%) | **5040 / 8320 (61%) — WORSE** |
| total dw | +342,212 | +137,299 |
| per-post-cell mean \|sum(dw)\| | ~= typical \|dw\| | **4767.8 vs typical 4893.8 — still no cancellation** |

**Both forms fail, and the reason is structural rather than a tuning matter.** Even with an exact per-step zero-sum
over the free set, the free set **shrinks monotonically**: each step some synapses pin at the floor and stop
contributing negative increments, so the accumulated total drifts positive regardless. Subtractive normalization
requires the weight vector to be able to absorb negative mass; a hard non-negative floor — which Dale's law makes
mandatory here — removes exactly that capacity.

⇒ **Miller-MacKay subtractive normalization is closed on this substrate, in both its naive and Dale-compliant
forms, at ZERO seed cost.** This is a general result, not a property of one parameterization: any rule whose
guarantee is "the increments sum to zero" is defeated by a bound that absorbs one sign.

## Running tally: five mechanism families closed, five distinct causes, three at zero seed cost

| mechanism | cause | seeds spent |
|---|---|---|
| Milstein split-threshold band | geometric collision (spacing == backward shift) | 12 (2 pre-registered attempts) |
| zero-DC difference-of-exponentials | trace-amplitude mismatch (validated on inputs never generated) | 6 |
| Milstein two-sigmoid on `ET*IS` | no separating axis (1.001x; the gate predicted this itself) | **0** |
| Miller-MacKay subtractive (both forms) | hard floor absorbs negative mass; `w_min<0` violates Dale | **0** |
| expansive/normalizing read-out | metric inflation — `c -> c^p` adds no information | **0** |

The pre-flight rule (*verify the claimed property on DEPLOYED inputs before pre-registering*), learned from the DoG
failure at a cost of one 6-seed run plus a retraction, has now closed three candidates for nothing.
