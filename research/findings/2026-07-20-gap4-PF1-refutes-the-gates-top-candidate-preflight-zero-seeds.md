# gap#4 PF-1 — the second gate's TOP candidate is refuted PRE-FLIGHT, at zero seed cost

The second research gate ranked **Milstein two-sigmoid bidirectional BTSP on the `ET*IS` overlap** first, resting on
two claimed separating axes. It also — to its credit — stated its own skeptical objection and designed the offline
test that would falsify it. I ran that test before pre-registering anything.

## Result: neither claimed axis separates, in the config where the deficit lives

Measured on `_gap4_btsp_multicell_map_derisk` (seed 600, 24000 synapse-samples taken DURING the final induction),
comparing afferents at the ADJACENT field's lag against those that FORM the field:

| axis | gate's claim | measured |
|---|---|---|
| current weight `w` | 1.3-1.8x (adjacent elevated) | **1.093x** |
| overlap `ET*IS` | the genuinely new separating axis | **1.001x — none** |
| `ET` alone (baseline) | 1.213x deficit | 0.999x |

**The gate's own skeptical objection is confirmed.** It wrote: *"in this task the plateau drives all CA1 cells, so
`IS(t)` is near-uniform across synapses at any instant. Therefore at a single update step, ranking synapses by
`ET*IS` is identical to ranking by `Etilde`."* Measured: the overlap ratio is 1.001x against `ET`'s 0.999x. **The
product creates no instantaneous separating axis, exactly as predicted.**

Its residual hope was that the *temporal integral of a nonlinearity* would separate where the instantaneous product
does not. That remains formally untested here — but with the instantaneous inputs differing by 0.1%, any separation
must come entirely from the sigmoid's curvature acting on near-identical trajectories, which is a far weaker claim
than the one that earned it the top rank.

## A discrepancy I am flagging rather than resolving

The gate measured `ET*IS` percentiles of p50 **0.040** / p90 0.186 / p99 0.833, and built its strongest argument on
those straddling Milstein's thresholds (alpha_dep 0.09, alpha_pot 0.24). My probe measures p50 **0.4068** / p90 2.47
/ p99 4.36 — about **10x higher**, with **63% ABOVE alpha_pot** rather than straddling.

**The likely explanation is that we measured different runners** (the gate used the single-cell
`_gap4_btsp_oneshot_place_field_task_derisk`; I used the 4-cell `_gap4_btsp_multicell_map_derisk`, which is where
the adjacent-contrast deficit is actually measured). **I am NOT claiming the gate is wrong** — I am recording that
its load-bearing distributional argument does not reproduce in the configuration the mechanism is meant to fix, and
that this must be resolved before that argument is relied on.

## The one claim that partially survives

The **weight axis** is real but weak: 1.093x, not the claimed 1.3-1.8x. Adjacent synapses ARE elevated at induction
time, just far less than the post-hoc +80%/+33% figure suggested — because that figure was measured after ALL
inductions, whereas what the rule can read is the state DURING induction. **A 1.093x axis cannot deliver the >2x
weight contrast the transfer loss demands**, alone or conjunctively with a 1.001x axis.

## Process

This is the **second mechanism refuted pre-flight at zero seed cost**, both by the same correction: *verify the
claimed property on the DEPLOYED inputs before pre-registering*. That rule was learned from the DoG failure, which
cost a 6-seed run, a pre-registration and a retraction. It has now saved two.

It also demonstrates something worth keeping about the gate: **its most valuable output was the objection to its own
recommendation**, not the recommendation. A gate that hands you a falsification test alongside its top pick is worth
more than one that hands you confidence.

## Standing

Four mechanism families now refuted, each with a distinct diagnosed cause. The blocker is unchanged and remains the
best-characterized object in the arc: **adjacent contrast 1.213x vs far 2.609x, 6/6 on fresh seeds across three
independent runs.** Remaining ranked candidates (untested): weight-keyed depression in isolation, STC winner-take-all
capture over a finite cell-wide pool, corrected Miller-MacKay with `w_min < 0`, and the population kWTA read-out that
attacks the 1.5x transfer loss rather than the weight contrast.
