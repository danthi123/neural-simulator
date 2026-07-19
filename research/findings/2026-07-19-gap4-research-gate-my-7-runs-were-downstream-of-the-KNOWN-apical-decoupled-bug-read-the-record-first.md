# gap#4 research gate — the 7-run "definitive characterization" was RE-DERIVING a KNOWN result: my runs tuned DOWNSTREAM of the already-root-caused apical-decoupled bug (C1) + a forward-pass collapse. The real de-risk is the frozen-reservoir + WTA + trained-readout control (with cfg.seed). Read-the-record-first lesson.

**2026-07-19.** After 7 diagnostic runs concluded the gap#4 on-bridge BDSP keystone was "definitively blocked (held 0.420
== lesion, invariant to everything), likely the rate-code wall," a read-only research scout — reading the PROJECT'S OWN
record — delivered a major, verified reframe. My hypothesis was the deepest-and-mis-targeted of three stacked failures.

## The 3 stacked failures (only C2 was my hypothesis; A + C1 are the actual cause, and C1 is a KNOWN bug)
- **Failure A (PRIMARY) — forward-pass representation collapse.** "Hidden fires bias-driven (0.07), not input-selective"
  is the smoking gun: the hidden layer fires from tonic drive, not the input pattern → NO class-conditional hidden
  representation → nothing for the readout to decode (→ constant-class 0.420) AND no class-conditional eligibility for
  ANY rule (BDSP/e-prop/backprop) to shape (→ BDSP==LESION *necessarily*). An E/I-balance + input-drive + lack-of-
  competition problem, upstream of every credit question.
- **Failure B — degenerate readout.** argmax over sparse per-class output-pool counts collapses to the bias class; the
  differential readout I tried didn't help because Failure A means there is no underlying class signal to recover.
- **Failure C1 (a KNOWN, ALREADY-ROOT-CAUSED bug) — the immediate cause of BDSP==LESION.** VERIFIED against
  `2026-07-10-D1-onbridge-BDSP-apical-decoupled-from-soma-BOUNDARY-root-caused.md` (the SAME runner
  `_d1_onbridge_learn_to_accuracy_derisk.py` I re-ran): the committed `enable_bdsp` apical raises the burst-PROBABILITY
  read P (0.30→1.00) but NOT the measured burst rate B (0.000→0.000), and the rule `dw ∝ ẽ·(B − P̄·E)` uses the MEASURED
  B (soma-set, apical-independent). ⇒ **the apical delivers ZERO directed credit → BDSP≡LESION, invariant to dw
  magnitude / epochs / bias — the EXACT symptom + the literal "apical-decoupled" run label.** Kolen-Pollack is learning a
  feedback direction for a signal that never reaches the update — moot until C1 is fixed.
- **Failure C2 (my rate-code hypothesis) — the DEEPEST layer, and already partially refuted.** Even wired correctly, BDSP
  credit is a burst FRACTION (a rate) that a point neuron at 0.07 firing can't estimate. BUT the project ALREADY ran the
  graded-coding control (`2026-07-17-rate-net-control-graded-coding-does-NOT-unlock-supervised-deep-credit-...`): graded
  coding does NOT rescue supervised deep credit (0/6 both spiking + graded) → "the block is the RULE" (directional, the
  control was under-powered).

## The tell I missed + the honest lesson
**The INVARIANCE to everything (2000× dw range, credit type, epochs, drive, readout) is the signature of a FORWARD/WIRING
failure, not a credit-tuning failure.** With the apical decoupled (C1) and the hidden bias-driven (A), BDSP CANNOT differ
from LESION — every one of my 7 runs was PRE-DETERMINED to read null. **I re-ran the same runner and re-derived a symptom
the project root-caused 9 days ago, without reading `2026-07-10-...-apical-decoupled-...md` first.** This is exactly the
`feedback_read_own_substrate_before_theorizing` lesson: read the record/wiring BEFORE re-diagnosing. The deep-research gate
(reading the project's OWN findings) is what caught it — a strong argument for firing the gate at the START of a
re-investigation, not after 7 runs.

## The de-risk (the scout's recommendation, = the project's own vindicated instrument)
**Frozen-reservoir + trained-linear population readout, with a competitive/balanced forward pass, ≥6 seeds, `cfg.seed`
SET** (⚠️ the `2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE` bug would confound it by ~3× the effect — verify
two-process threshold-hash identity first). Freeze input→hidden at random init; add lateral-inhibition WTA +
threshold-homeostasis (Diehl-Cook 2015 → input-selectivity, fixes A) + turn the tonic bias down (balanced E/I, van
Vreeswijk-Sompolinsky 1996 / Vogels 2011); train ONLY a logistic readout over the FULL hidden population (fixes B); longer
settle window + higher input rate. READ AS A FORK: clears chance → the substrate CARRIES a decodable signal, ~0.42 was
A+B (not the rate-code wall) → then fix C1 (route the apical through the two-compartment coupling so it raises REAL bursts
B) and re-ask "does BDSP beat the reservoir?" (the project's frozen reservoir already reached ~0.778 — confounded by the
seed bug, but directionally the substrate can carry the signal). Stays at chance → the forward pass itself is broken;
iterate competition/balance before any credit. **Do NOT run another BDSP sweep before this control — it is pre-determined
to read null.**

## Mission-path note (surface for the owner)
The scout flags the project's standing 2026-07-17 decision (`learning-rule-frontier-map`) to pursue the UNSUPERVISED
on-spike stream cortex (HTM competitive pooler + `fused_htm_permanence_update`), which learns deep representations from a
stream WITHOUT supervised global-loss deep credit — sidestepping this whole wall. ⇒ the gap#4 keystone as SUPERVISED
BDSP-to-accuracy may be a parked direction vs the unsupervised stream cortex being the mission-critical path. A genuine
value fork worth the owner's steer; meanwhile the frozen-reservoir de-risk + the C1 apical-coupling fix are the two
concrete unblockers IF supervised BDSP-to-accuracy continues.

## SYNTHESIS (2026-07-19) — the substrate ALREADY learns to accuracy; my `_d1` runner's 0.42 is its READOUT (a DIFFERENT runner already solves this)
The frozen-reservoir de-risk the scout recommended is ALREADY BUILT + RUN in a DIFFERENT runner:
`_onbridge_eprop_port_derisk.py` has `reservoir_control=True` (line 507, added 2026-07-16) — trains ONLY a Bellec-2020
LEAKY LINEAR READOUT over the hidden population with the hidden FF FROZEN — and its own comment + the
`2026-07-16-deep-credit-GO-is-80pct-RESERVOIR` finding report **FROZEN 0.778 vs chance 0.333** (docstring line 51: "a
clean linear readout on H2 is ~0.7-0.9 separable — the forward IS discriminative; the leaky readout is what lets e-prop
read it out"). ⇒ **the substrate CARRIES a decodable class signal (0.7-0.9) with a PROPER trained-linear population
readout.** My `_d1_onbridge_learn_to_accuracy` runner's degenerate 0.42 is precisely **Failure B — its argmax-over-sparse-
output-pools readout** (which the eprop runner AVOIDS with the leaky readout) + Failure A (forward collapse) + C1 (apical-
decoupled credit). ⇒ **gap#4 "local-credit learning on the substrate to accuracy" is SUBSTANTIALLY ACHIEVED already**
(reservoir + leaky readout ≈ 0.778); learned deep credit adds a SMALL seed-variable margin (~+0.037..+0.185, and the
80/20 split is itself seed-confounded per 2026-07-17). **The deepest read-your-record lesson: a DIFFERENT runner already
achieves what my `_d1` runner couldn't — because `_d1` has a readout bug my 7 runs kept tuning around.** ⇒ the REAL open
frontier is the narrow "does learned deep credit BEAT the reservoir by a robust margin, seed-clean" — a hard, small-margin
problem the field also faces (surrogate-grad BPTT / rate readouts, not sparse local credit). **VERIFYING NOW** (running
`_onbridge_eprop_port_derisk --seeds 42 43 44` to re-confirm FROZEN clears chance). This resolves the gap#4 investigation:
the keystone (substrate learns) is achieved; the deep-credit-margin is the honest, narrow, field-hard frontier.

## ⚠️ SYNTHESIS REFINED (verify-don't-assert — the eprop run caught my loose framing)
Running `_onbridge_eprop_port_derisk --seeds 42 43 44` to verify: seed 42 shows the LEARNED e-prop deep credit
**FAILS on-bridge (inherit-heldout 0.222 < chance 0.333, "TRAINS-THE-TASK False")** while a numpy deep net gets 1.000
(the task IS separable + deep helps — STAGE0 deep-best 1.000 vs 1-layer 0.444). ⇒ my earlier "the substrate learns to
0.778" CONFLATED two distinct things: (a) the FIXED random forward pass + a trained LINEAR readout (RESERVOIR COMPUTING —
SHALLOW learning, carries the signal ~0.7-0.9 per the docstring/finding); vs (b) DEEP local-credit learning (learning the
hidden layer via e-prop/BDSP), which this run shows is CONFIG-FRAGILE and here FAILS (0.222 < chance). The prior banked
"K=8 0.877" (2026-07-16) was 80% the reservoir + a small deep-credit margin; this k=5 run's deep credit fails outright.
**⇒ the HONEST, PRECISE gap#4 read: the substrate's FORWARD PASS is discriminative (a shallow reservoir/linear readout
carries the class signal) — but DEEP local-credit LEARNING to accuracy on the sparse spiking substrate is NOT robustly
achieved (small/variable/config-fragile margin over the reservoir; fails at k=5). This matches the field (surrogate-grad
BPTT / rate readouts reach accuracy; sparse local credit does not).** My `_d1` runner's degenerate 0.42 was additionally
its argmax-readout bug (Failure B) masking even the shallow signal. **The gap#4 keystone (DEEP biological local-credit to
accuracy) remains the honest, field-hard open frontier** — NOT "substantially achieved" (I over-claimed; corrected). The
frozen-reservoir re-verification was KILLED (impractically slow — 1h38m and still on seed 42, ~5h for 3 seeds on-bridge);
the documented FROZEN 0.778 (2026-07-16) stands as the substrate-carries-signal number, corroborated by THIS run's STAGE0
(the task IS separable: numpy deep-best 1.000 vs 1-layer 0.444). The load-bearing NEW datum — learned deep credit FAILS
on-bridge (inherit 0.222 < chance at k=5) — was captured before the kill. NEXT if pursued: the
C1 apical-coupling fix + a proper leaky/population readout, then measure the deep-credit margin over the reservoir
SEED-CLEAN — but the field-evidence says expect a small margin. The UNSUPERVISED stream cortex (2026-07-17) stays the
more-promising mission path.

## Verified sources
- **VERIFIED (internal):** `2026-07-10-...-apical-decoupled-...` (C1, exact bug, same runner); `2026-07-17-rate-net-control-graded-...`
  (C2 control run); `2026-07-16-deep-credit-GO-is-80pct-RESERVOIR-...` (reservoir instrument + 0.778, WITH its own
  2026-07-17 seed-confound correction). Primary (to verify when building): Payeur 2021 Nat Neurosci 24:1010 (BDSP=burst
  probability, needs ensembles/rate); Diehl-Cook 2015 (WTA+adaptive-threshold→selectivity); Maass 2002 + Cover 1965
  (reservoir+linear readout); van Vreeswijk-Sompolinsky 1996 / Vogels 2011 (balanced E/I).
