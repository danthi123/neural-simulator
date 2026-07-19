# gap#4 keystone — the apical→soma COUPLING flag (`--soma-g >0`) flips the microcircuit BDSP run from BOUNDARY (apical-decoupled, B=0, credit≈lesion) to PIPELINE-VALIDATED (directed hidden-layer credit 415 >> lesion 198, P0 moat holds). My earlier "field-hard/deep-credit-fails" verdict THIS session was on the path with the coupling OFF. Accuracy-to-0.75 is now the never-run SCALE sweep, not a wall.

**2026-07-19.** Ran the board's named gap#4 next action (`_d1_onbridge_learn_to_accuracy --microcircuit` + the
bistability flags, vs the 0.75 accuracy bar, on the DENSE-redundant task = the trainable spiking-deep-credit regime,
NOT parity/emerge1). A verify-the-flag catch changed the whole picture.

## The catch (silent-failure discipline: an ABSENT flag means DEFAULT, not "the fix is on")
- **First smoke** (`--microcircuit --graded-credit --apical-bistable --apical-self-regen 2 --apical-kir-g 3
  --differential-readout`, seed 42, dense): **VERDICT BOUNDARY — APICAL DECOUPLED.** Task valid (oracle 0.800 ≥ 0.80,
  single-layer floor 0.731 = chance), but held-out **0.731 == LESION 0.731 == wrong-sign 0.731 == chance**, and the
  smoking gun: **in→hid dw 0.059 ≈ LESION 0.045** (hidden layer gets NO directed credit) while **hid→out dw 6.073 >>
  lesion 0.036** (the OUTPUT layer, which has direct target access, DOES learn). The apical raises the burst-PROB read
  P (0.14→1.00) but NOT the measured burst rate B (0.000→0.000) = the C1 apical-decoupled signature.
- **Root cause of the null:** the run had `--soma-g` at its **default 0.0** → `couple_soma=(args.soma_g>0.0)=False`
  (runner line 592) → `cfg.bdsp_apical_couples_soma=False` → the apical drives `cp_v_apical` (→P) but there is NO
  electrotonic apical→soma coupling (→B stays 0) → the FF `dev=B−Pbar·E` gets no apical-directed credit. **The board's
  named action listed the bistability flags but OMITTED `--soma-g`, which is the actual C1-fix switch.** The runner's own
  help says it: *"--soma-g >0 couples apical→soma so bursts B rise (DIRECTED credit); 0=decoupled."*
- **Second smoke, add `--soma-g 8.0`** (everything else identical): **VERDICT flips to PIPELINE-VALIDATED (smoke)** —
  **both pathways MOVE under credit (in→hid 415.4, hid→out 332.4); the P0 MOAT holds (apical-lesion hidden-dw 198.5 <<
  credit 415.4 = the hidden layer NOW gets ~2× directed credit); no weight transport (True); task valid.** Held-out is
  still 0.731 (chance) **at this smoke scale (hidden=12, epochs=3)** — the verdict is explicit: the ≥0.75 accuracy bar
  "needs width + epochs + drive tuning the smoke deliberately omits."

## The load-bearing correction (to my OWN session verdict)
Earlier this session I concluded gap#4 supervised BDSP-to-accuracy is "field-hard / deep credit fails" from 7 runs +
an e-prop k=5 fail. **Those runs had the apical→soma coupling OFF** (`--soma-g 0`), so they were the *pre-determined-null*
path the research gate warned about (apical-decoupled C1 + forward). WITH the coupling ON, the MECHANISM is correct:
directed credit reaches the hidden layer and the moat holds. **⇒ gap#4 is NOT at the "deep-credit-fails" wall I stated;
the mechanism pipeline-validates, and held-out ACCURACY is the never-completed SCALE sweep (width + epochs) — a lever,
not a wall.** This is the read-your-record + verify-the-flag lesson: the board's summary of the next action dropped the
load-bearing knob, and an absent flag defaulted to OFF.

## One yellow flag to clear before any GO (silent-failure discipline)
At the coupled smoke, **wrong-sign held 0.731 (chance), it did NOT anti-learn (< chance)** — if the credit were fully
sign-informative for accuracy, negating the top error should push below chance. So "moat holds + credit directed"
(dw ≠ lesion) is necessary but NOT yet sufficient for accuracy. The scale sweep must show held-out clearing 0.75 with
**BDSP >> lesion AND wrong-sign < chance** (the sign-informative gate), 6-seed, `cfg.seed` set (verified: runner line
254 `cfg.seed = int(seed)` — NOT the 8-of-93 bug).

## Status + next
- **DONE:** ran the board's named gap#4 action; found + fixed the missing `--soma-g` coupling flag; the microcircuit +
  bistable + **soma-coupling** BDSP pipeline-validates on the dense task (directed hidden credit, moat holds).
- **IN FLIGHT:** two single-seed scale-up smokes (hidden=48, epochs=12; raw+coupling vs graded+coupling) to confirm the
  held-out accuracy MOVES off chance before committing the 6-seed GPU sweep.
- **NEXT (if the scale-up moves):** the full learning-to-accuracy sweep (wider hidden + more epochs + drive tuning),
  6-seed, vs the 0.75 bar + the depth-helps + wrong-sign-anti-learns gates — the never-completed run the board names.
- NO `sim/` edit (the coupling is the committed 2026-07-10 `sim/` mechanism, default-off; a one-line runner flag turns
  it on). Diagnostics: `_d1_onbridge_learn_to_accuracy_derisk.py --soma-g`.
