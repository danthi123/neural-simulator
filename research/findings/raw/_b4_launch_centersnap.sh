#!/bin/bash
# B4 cooling: CENTER the snap. The merged critic snaps the cue burst @ absolute trial ~5-6 with the strong
# derivative (gain2/tau250); at n_train=30 that is ~17% of the window -> step-r -0.66. The standalone snapped
# at ~30-40% of its n_train=50 window -> -0.80. Two legitimate ways to put the snap near the window midpoint:
#  (A) lower td_csc_to_strio_weight (the value starts lower -> the cascade fires later -> snap delayed), strong
#      derivative KEPT (gain2/tau250); n_train=30.
#  (B) match the measurement window to the transition (n_train ~12-15 so trial ~5-6 is the midpoint). The migration
#      FUNCTION (value grows, US shrinks, dip) is complete by trial ~6; this measures r over the window where the
#      transition happens (the standalone bar was migration over the training window).
cd /e/Documents/Projects/sim
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_opsearch \
  --seed 42 --n-train 30 --pass center \
  --op '{"td_csc_to_strio_weight": 10.0, "td_stdp_w_max": 40.0, "td_to_fs_weight": 30.0, "td_fs_to_strio_weight": 20.0, "td_gabab_prop": 0.04, "td_derivative_gain": 2.0, "td_slow_tau_ms": 250.0}' \
  --label "G1_csc10_clip40_gain2_tau250_nt30" \
  --op '{"td_csc_to_strio_weight": 8.0, "td_stdp_w_max": 40.0, "td_to_fs_weight": 30.0, "td_fs_to_strio_weight": 20.0, "td_gabab_prop": 0.04, "td_derivative_gain": 2.0, "td_slow_tau_ms": 250.0}' \
  --label "G2_csc8_clip40_gain2_tau250_nt30" \
  --out research/findings/raw/_b4_oppoint_center_s42.json
