---
type: finding
status: boundary
date: 2026-08-17
mechanism: wave1-banking
---
# REPLAY-V3 calibration: UNDEFINED (BOUNDARY) -- replay fires, cortical recovery is zero, target FS loop never recruits

seed-waiver: 2-seed bounded-isolation CALIBRATION by design; dev seeds 230/231/326 + held-out 327/328/329 are LOCKED until the target-FS-loop precondition is met — a >=6-seed run is not the current experiment.
NO-EXTERNAL-NEEDED: the BOUNDARY is an internal unmet-precondition (target FS inhibitory loop non-recruitment during sleep), not a capability wall; the named surpass (recruit the FS loop) is an in-engine mechanism, no external literature is load-bearing.

**Artifact:** research/findings/raw/parallel_gates/replay_v3_calibration.json (+.prov.json). On main (da5f0de60); NOT on codex/gap4-axon-capd-derisk. Runner _replay_cortical_consolidation_gate_v3.py, git_sha d24548b6, numpy, --phase calibration --seeds 228 229 (2026-08-03).

**Result:** The run's OWN gate reports calibration_status=UNDEFINED on both seeds -- not a scored 0, an UNDEFINED (a require() precondition failed). The failing precondition: intact sleep must contain uncued replay + index relay + BOTH inhibitory loops; the local TARGET feedforward-inhibition loop never recruited during sleep (cortical_target_fs sleep spikes=0, both seeds). Replay DID fire (reactivated_events=24, replayed_A/B>0, cortical_index 156/240 spikes, index_fs 21/33).

**Load-bearing metric at floor:** intact_mean_recovery=0.0 (both seeds); both-memory correct-recall A/B=0.0/0.0; margin 0.0 / -0.0014; false_recall 0.0 / 0.5 (s229). Every recovery/load-bearing check fails. <!--derived--> (-0.0014 is the s229 rounded margin; full value in the cited calibration JSON)

**Controls (present, honest):** 8 lesion/shuffle conditions wired; all lesion transmission-gate preconditions pass; single-bridge-persists, wake-never-drives-index, content-preserving order-shuffle all pass -- the harness is sound, the capability is absent. Bounded isolation: STDP/reward/homeostasis/STP/structural plasticity disabled; host-defined populations + fixed anatomy + rate-window Hebbian (5 named scaffolds remain).

**Verdict: BOUNDARY.** Not a GO; calibration produced no measurable replay-driven cortical recovery and declined to score. Dev seeds (230/231/326) and held-out (327/328/329) LOCKED / never run.

**Residual to clear before promotion:** (1) recruit the target FS inhibitory loop during sleep so the precondition is MET; (2) achieve nonzero intact recovery that beats no-sleep + shuffled-order + Schaffer/plasticity lesions; only then run development seeds.
