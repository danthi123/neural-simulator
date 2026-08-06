---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v6
runner: research/runners/_laneC_source_monitor_coresidency_gate_v6.py
aggregator: research/runners/aggregate_source_monitor_seeds.py
artifacts:
  - research/findings/raw/source_monitor_v6_generalization/development_verdict.json
  - research/findings/raw/source_monitor_v6_generalization/development_652.json
  - research/findings/raw/source_monitor_v6_generalization/development_653.json
  - research/findings/raw/source_monitor_v6_generalization/development_654.json
---

# v6 development NO-GO: silent-by-construction generalizes, the redistributive-win criterion does not

<!--derived-->
**Verdict: NO-GO at v6 development.** Running the FROZEN v6 mechanism and the FROZEN acceptance rule on the three
unseen development seeds 652/653/654 (aggregator artifact
`research/findings/raw/source_monitor_v6_generalization/development_verdict.json`), two seeds `DEVELOPMENT_PASS` and
one, seed 654, `DEVELOPMENT_FAIL`. GO requires all three, so development is a NO-GO. The failure is isolated to ONE of
the twenty preregistered components, `weakest_source_margin_strictly_improved`. The circuit, thresholds, and acceptance
rule were held frozen; only the seed partition advanced. Held-out seeds 655/656/657 stay sealed (the runner's
`validate_phase_seed` rejects them until development records a GO), and were NOT run.

## What DID generalize

<!--derived-->
The v6 headline fix generalizes cleanly on all three seeds. `learning_off_has_no_source_recall` reads zero on every
seed (the v5 leak stays closed), `recall_settle_reaches_quiescence` holds (settle terminated in 160 steps on every
seed), `unseen_episode_has_no_source_recall` is zero, and every source margin clears the fixed 0.15 floor on all three
seeds. Silent-by-construction recall is robust to unseen seeds. Nineteen of twenty components pass on seed 654 as well.

## What did NOT: the redistributive win on the weakest source

<!--derived-->
Per-seed numbers from the four cited artifacts. `M` intact margin, `L` matched competition-lesion margin, floor
`F = 0.15`. The failing criterion requires `min(M) > min(L)` strictly (competition must lift the weakest source).

| seed | M seen/heard/self | L seen/heard/self | min M | min L | learning-off | status |
|---:|---|---|---:|---:|---:|:---:|
| 652 | .2225/.1842/.1958 | .2225/.1775/.1958 | .1842 | .1775 | 0 | DEVELOPMENT_PASS | <!--derived-->
| 653 | .1758/.1608/.1975 | .1675/.1442/.1975 | .1608 | .1442 | 0 | DEVELOPMENT_PASS | <!--derived-->
| 654 | .2425/.1825/.3075 | .2300/.1825/.3075 | .1825 | .1825 | 0 | DEVELOPMENT_FAIL | <!--derived-->

<!--derived-->
On seed 654 the weakest source is `heard` at margin .1825, and competition leaves it EXACTLY unchanged (gain 0.0):
`min M = min L = .1825`, so the strict-improvement test fails by exactly 0.0. Competition on seed 654 is active and it
DOES help — but it lifts `seen` (.2300 -> .2425, +.0125), the second-strongest source, not the weakest. On the two
passing seeds competition happened to lift the weakest source (652: heard +.0067; 653: heard +.0167). So the
redistributive win is seed-dependent: fixed symmetric lateral inhibition suppresses each rival in proportion to its own
activity, which does not preferentially rescue the least-active source. On 654 the geometry of the learned pattern put
the competitive benefit on a non-weakest source, and the weakest got zero.

## Why (mechanism-level), and the next mechanism

<!--derived-->
The v2 competition circuit is a FIXED local fast-spiking GABA-A biased-competition module: symmetric lateral
inhibition with hand-set weights. Such a circuit implements divisive suppression of rivals, not a guarantee that the
weakest population is boosted; whether the weakest source gains depends on the per-seed rate/timing configuration.
Requiring `min(M)` to STRICTLY exceed `min(L)` on every seed asks the circuit to reliably up-regulate the LEAST-active
source, which a fixed symmetric inhibitory ring does not do by construction.

The biological mechanism that specifically boosts under-active populations is intrinsic-excitability homeostasis
(activity-dependent threshold adaptation; Turrigiano). The v3 line already implements region-scoped threshold
homeostasis (`settle_homeostasis` / `source_threshold_vector`), which lowers the firing threshold of an under-firing
population and so relatively lifts the weakest source. **Next mechanism (v7 candidate): merge v6's silent-by-
construction settle-to-quiescence recall with v3's intrinsic threshold homeostasis on the source-memory populations,
so the weakest source's excitability is up-regulated and the strict-improvement criterion holds on every seed rather
than by seed luck.** This is a new method for the SAME frozen criterion — the criterion is not loosened.

## What was NOT done, on purpose

<!--derived-->
The frozen criteria were not touched. `weakest_source_margin_strictly_improved` (strict `>`) is the P3 functional-role
deliverable — the whole-brain honesty path needs competition to protect the weakest source, not merely the average —
so relaxing it to `>=` to force a pass on seed 654 would be gaming the gate, and would erase exactly the redistributive
property the criterion exists to certify. The NO-GO is reported as a verdict on the FIXED-competition method's
robustness, not on the capability. Held-out seeds remain sealed.

## Deliverables added this arc

<!--derived-->
A `--phase {calibration,development,held_out}` mode on the v6 runner (identical evaluator + preregistered controls;
held_out sealed until dev GO via `validate_phase_seed`), and a reusable hands-off self-sweep + aggregator
`research/runners/aggregate_source_monitor_seeds.py` that runs every seed of a phase in ONE process, collapses per-seed
PASS/FAIL into one GO/NO-GO, and writes one aggregate verdict artifact. Launch-then-hands-off:
`SIM_BACKEND=numpy python -m research.runners.aggregate_source_monitor_seeds --phase development`.

## Provenance

All three development seeds ran locally on the NumPy backend, deterministic across re-runs (seed 654 re-ran identical:
min M = min L = .1825). Runner `research/runners/_laneC_source_monitor_coresidency_gate_v6.py`; aggregator
`research/runners/aggregate_source_monitor_seeds.py`; artifacts and `.prov.json` sidecars listed in the front matter,
stamped from git `94e7e72c399f7020bc9549978e577f25caca946b`. Pool nodes 40/41/42 are SSH-reachable but
`~/derisk-pool/sim` is not a git checkout on any of them (needs `tools/pool_provision.sh`); local was correct for this
6-seed sweep.
