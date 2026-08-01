---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/dfc_plateau/dfc_opsweep_seed42_aggregate.json
  - research/findings/raw/gap4/dfc_plateau/dfc_dt015_6seed_aggregate.json
---

# gap#4 crux: transport-free DEEP FEEDBACK CONTROL (DFC) overfits the movable plateau hidden — an INSTRUMENT-VERIFIED negative across 11 operating points (best merely TIES the frozen reservoir)

<!--derived-->
**One-line verdict:** the record's explicitly-named-untested crux lever — closed-loop Deep Feedback Control
(Meulemans et al.) with **transport-free Kolen-Pollack feedback learning** — does **NOT** add held-out credit
beyond unsupervised sharpening on the movable plateau hidden. Across an **11-config controller operating-point
sweep (seed 42)**, DFC beats both baselines by the 0.05 margin on **0/11** configs: it consistently **overfits**
(train ~0.79, held-out capped at the FROZEN reservoir 0.537), and the two best operating points (dt015, steps8)
merely **tie** frozen (deep_credit_share_dfc = 0.000); every other config is below it. Crucially, this is an
**instrument-VERIFIED** negative, not a broken-instrument false negative: the Kolen-Pollack rule genuinely aligns
the feedback matrix transport-free (Q^T·W_out diagonal +0.54 to +1.11 across configs, Q not a copy), the
closed-loop controller reduces error and fits train, the control sign is load-bearing, and the no-weight-transport
anti-cheat holds. Consistent with the DFA + larger-task nulls: three credit routes tried on the movable hidden,
only the label-free unsupervised rule helps.

Artifact: `research/findings/raw/gap4/dfc_plateau/dfc_opsweep_seed42_aggregate.json` (backend numpy/CPU). The
runner is `research/runners/_gap4_dfc_plateau_credit_derisk.py`.

## The instrument, and why the FIRST version was a false negative (the verification that mattered)

<!--derived-->
The first DFC build used a **fixed random** feedback matrix Q. Verifying the instrument before banking (silent-
failure rule #3: a negative needs the instrument checked as much as a positive) caught that this is **weak
feedback-alignment — the exhausted FA family, not real DFC**: with random Q, Q^T·W_out diagonal was **−0.76**
(misaligned), so the closed-loop controller barely reduced error (CE 10.25→10.23) and its "SMOKE NEGATIVE" was an
artifact of a broken controller. The mechanism was PROVEN to work when the feedback is aligned — a W_out^T
controller reduces CE 10.25→3.91. The fix is a **transport-free feedback-LEARNING rule (Kolen-Pollack**, Akrout et
al. 2019 "Deep Learning without Weight Transport"): the same local pre×error signal that trains W_out also updates
Q (with weight decay), so Q ALIGNS to W_out **without ever reading it** (alignment ≠ copy: cos 0.57, diagonal
+1.01 in the standalone probe; controller accuracy 0.119→0.381 ≈ the 0.392 full-transport bound). Only with this
corrected, verified instrument does the negative below mean anything.

## Result — 11-config operating-point sweep, seed 42 (frozen reservoir = 0.537)

<!--derived-->
| config | DFC held-out | dcs_dfc | DFC train | Q align-diag | beats both +0.05 |
|---|---|---|---|---|---|
| dt015 | 0.537 | 0.000 | 0.812 | 0.942 | no (ties frozen) |
| steps8 | 0.537 | 0.000 | 0.812 | 0.891 | no (ties frozen) |
| epochs10 | 0.519 | −0.040 | 0.812 | 0.724 | no |
| epochs15 | 0.519 | −0.040 | 0.812 | 0.919 | no |
| fbwd05 | 0.500 | −0.080 | 0.812 | 0.537 | no |
| baseline / lrdfc01 / alpha1 / gentle | 0.481 | −0.120 | ~0.79 | 0.74–1.11 | no |
| lrdfc002 | 0.463 | −0.160 | 0.625 | 0.732 | no |
| reg_combo | 0.444 | −0.200 | 0.710 | 0.841 | no |

`0/11` beat both baselines. DFC held-out **max = 0.537 = frozen exactly**; unsupervised ≈ 0.48–0.52. Every config
that regularizes (early-stop, gentle/weak control, stronger feedback decay) still tops out at the frozen reservoir.

## 6-seed confirmation (best config dt015)

<!--derived-->
**6-seed confirmation (dt015, the best operating point):** 0/6 beat both baselines by 0.05 (a clean NO-GO). DFC beats the unsupervised rule on only **1/6** (below it on 5), and beats the frozen reservoir (dcs_dfc > 0) on 4/6 -- so the aligned controller adds over no-credit but **overfits BELOW label-free sharpening**. Mean DFC held-out **0.460** vs unsup **0.522** vs frozen **0.417**; DFC train **0.791** (the overfit). no-transport holds 6/6; Q-alignment (KP) mean diagonal **0.773**. Artifact: `research/findings/raw/gap4/dfc_plateau/dfc_dt015_6seed_aggregate.json`.

## What this settles for the crux

<!--derived-->
The gap#4 movable-hidden arc has now tried **three** credit routes on the same substrate + sweet-spot task:
(1) **unsupervised** local covariance — the only positive (beats frozen 5/6, dcs +0.139 at 6-seed);
(2) **supervised fixed-DFA** directed error — trains but overfits (held-out null, and a larger task does not
rescue it — the local-vs-oracle gap widens);
(3) **DFC + Kolen-Pollack** closed-loop control — overfits: at 6-seed it beats the frozen reservoir on 4/6
(dcs_dfc > 0, so the aligned controller does add over no-credit) but stays **below the unsupervised rule on 5/6**
and clears the beat-both margin on 0/6. So the residual is **not the credit route** (a linear projection and a
closed-loop controller both overfit below the label-free rule), and **not the operating point** (11 tested). The movable hidden's held-out ceiling under *directed*
credit is the frozen reservoir; only label-free sharpening adds a little. The overfit is robust: the controller
reliably fits train (0.79) and the aligned feedback works, but that training signal does not convert to held-out
generalization on this rate/analytic task. No capability abandoned — a verdict on the METHOD.

## Next
The directed-credit routes are exhausted on this rate substrate; the honest next levers are the ones this arc has
NOT yet reached: (a) the **on-bridge SPIKING** port of the unsupervised movable-plateau rule (the only positive
signal — does it hold on real spikes, the actual mission target, where the rate result is only a stand-in?);
(b) a **task where directed credit's advantage is representable** but the reservoir cannot already solve it — the
current sweet spot may be a regime where a random projection + a little sharpening is near-optimal, capping every
directed rule. Both move away from "find a better directed rule for this task" (three tried, all overfit) toward
the substrate/regime where deep credit would actually pay.
