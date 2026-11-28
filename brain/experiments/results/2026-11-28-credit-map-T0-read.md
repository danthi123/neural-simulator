# T0 credit map read (11-28): CONFIRMED (misdirected credit)

Registered prereg `brain/research/2026-11-28-credit-map-T0-prereg.md`; run unattended by tools/engine on pool40, 0.1B seed 7,
commit f543cc32 (results under `results/credit_map/`). Backprop used as an instrument only.
- Recall carriers (gradient sleep restricted to one layer at a time): layers 0, 4, 5 (a minority of 12).
- Cosine of each rule's accumulated adapter update with the gradient-sleep update, over the carrier layers: v1 (transport
  feedback) 0.036, v3 (layerwise targets) 0.012; positive control, an independently reseeded gradient sleep, 0.58.
- Verdict by the registered rule: CONFIRMED (|cos| < 0.1 in every carrier layer, positive control >= 0.5).
Reading: the four local rules failed because their updates are nearly orthogonal to the direction that gives recall, in the
few layers that carry it; not because of step size or drift budget. This favours rules whose expected update is the true
descent direction (three-factor / reward-gated perturbation, running on pool41) and writes aimed at layers 0, 4, 5.
Also 11-28: v3q (question-form targets) NEGATIVE-DEV at 0.4B; v3q 0.1B stopped at its G0Q gate (2/12 brain-sourced cues).
