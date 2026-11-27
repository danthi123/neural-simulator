# SHY seed-7 dev pick (0.4B), by the registered rule, written before any six-seed data

Cap 0.04043 (1.25 x ref gradient KL 0.03234), recall floor 2, g = 1 excluded.
Candidates under the cap with recall >= 2: transport lr 0.3, g 0.6 (recall 4, KL 0.0352). lr 0.1: g 0.6 KL 0.053 (over),
g 0.4 recall 1 (under floor). lr 0.3: g 0.8 KL 0.055 (over), g 0.4 recall 1.
**Pick: lr* = 0.3, g* = 0.6.** Report only: the cut kept recall (4 at g 1, KL 0.089 -> 4 at g 0.6, KL 0.035).
Matched control at g 0.6 on seed 7 (gradient@0.6): recall 3, KL 0.0169; clause (e2) local drift <= 1.25 x 0.0169 = 0.021 is
not met on seed 7 (0.035), so the six-seed verdict is expected NEGATIVE on (e2) unless seeds differ. Dev RSS: gradient 7.1 GB,
local 3.3 GB, 8-14 min per config, so the six seeds fit pool41/42 under memcap 13.
