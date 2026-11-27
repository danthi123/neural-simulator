# Local rule v2 (layerwise anchoring off the keep-bank subspace) read (11-27): NEGATIVE-DEV at 0.1B and 0.4B

Registered prereg `brain/research/2026-11-27-local-rule-v2-prereg.md`; run end to end by `tools/engine` on pool41/42
(automatic real smoke: 0.1B peak RSS 2.9 GB -> memcap 5 GB; 0.4B 6.8 GB -> 10 GB), commit 34325395; results and
pick.json under `results/local_rule_v2_01b/` and `results/local_rule_v2_04b/`. No dev (lr, tau) cell met the pick rule
(recall >= floor under the drift caps), so by the rule the six seeds were not run.
Seed 7 dev: 0.1B recall 0-1/12 in all nine cells (floor 2), generic KL 0.00003-0.032 (cap 0.040); 0.4B recall 0 except
lr 0.3 tau 0.9 (2/12, floor 2.5, held-out KL 0.115 under its cap 0.133); lr 1.0 at 0.4B diverges (KL 16-18).
Reading: the anchoring removes drift (v1's binding constraint) and with it most of the learning. With v1 (learns, drifts
3.5x), SHY (cut keeps some recall, below the bar) and v2 (no drift, no learning), the local-rule family at this scale is
limited by credit assignment, not by drift: the fixed feedback paths do not put the update where the fact lives. Verdict on
this method; the 12-31 milestone stays open (next: a rule whose error signal is local but target-aligned, e.g. layerwise
targets from the cortex's own in-context activity, judged by the record check first).
