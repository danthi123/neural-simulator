# Seed-7 dev pick (0.1B), by the prereg rule (drift <= 1.25 x gradient drift), written before any six-seed data

Gradient reference (seed 7): recall 3/12, drift KL 0.032 -> drift cap 0.040. No-sleep recall 0.
- mixed: no config meets the cap (lowest drift lr 0.01 dec 1e-3: KL 0.069, recall 1).
- transport: lr 0.01 meets it (KL 0.031 dec 0 / 0.026 dec 1e-3), recall 0 both. Tie broken by lower drift (not
  registered; disclosed): lr 0.01, decay 1e-3.
- local = transport lr 0.01 dec 1e-3 (mixed has no config under the cap, so transport is the only constrained pick).
- local_random: no config meets the cap (KL 12.6-16.3, every lr: the strict random feedback diverges); lowest-drift
  config by the rule: lr 0.3, decay 1e-3.
Report only (not a pick): transport lr 0.1-0.3 reached recall 2/12 at KL 0.11-0.12 (3.5x gradient drift). The local rule
learns the replay (loss 2.44 -> 0.31) but disturbs generic text more per unit of learning than the gradient sleep.
Expected six-seed outcome at this pick: NEGATIVE (seed-7 recall 0 < bar no_sleep + 2).
