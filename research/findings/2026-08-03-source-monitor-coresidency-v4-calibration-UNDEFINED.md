---
type: finding
status: undefined
date: 2026-08-03
mechanism: source-monitor-coresidency-v4
runner: research/runners/_laneC_source_monitor_coresidency_gate_v4.py
artifacts:
  - research/findings/raw/source_monitor_coresidency_v4/calibration.json
  - research/findings/raw/source_monitor_coresidency_v4/calibration.json.prov.json
  - research/findings/raw/source_monitor_coresidency_v4/source_manifest.sha256
  - research/findings/raw/source_monitor_coresidency_v4/source_manifest_precheck.log
  - research/findings/raw/source_monitor_coresidency_v4/source_manifest_postcheck.log
  - research/findings/raw/source_monitor_coresidency_v4/source_revision.txt
---

# Adaptive source inhibition calibration is undefined

<!--derived-->
**Verdict: UNDEFINED.** Both formal calibration rows failed a validity
precondition because the runner's source-blind interface guard expected `self`
in the signature of a bound recall method. Python correctly omitted `self`, so
the guard reported failure even though the recorded interface contained no
source argument. Seeds `601` and `607` are consumed and will not be rerun.

## What ran

The preregistered v4 candidate used local inhibitory STDP on all six
fast-spiking-to-rival-source routes. It compared intact learning, a matched
learning lesion, and an expression lesion after an exact 5,200-step rehearsal
with 1,040 plasticity-open steps. Each comparison restored a matched
post-rehearsal network state before recall.

The real circuit engaged on both seeds. Every source-memory and fast-spiking
pool fired during learning, every intact inhibitory route changed, all lesion
routes stayed fixed, and excitatory weights and thresholds remained unchanged.
All routing, state-matching, finite-value, and fixed-budget preconditions other
than the interface guard passed.

## Why no scientific conclusion is claimed

The recorded recall parameters were `episode_pattern`, `source_path_lesion`,
and `acc_lesion`. The runner compared them with a list that also contained
`self`, although `inspect.signature()` was called on the already bound method.
This mechanical error forced both rows to `UNDEFINED`. The guard is corrected
and regression-tested for future mechanisms, but changing the evaluator after
seeing these formal rows would turn the same seeds into tuning data.

The measurements also give no reason to preserve this candidate. On seed 601,
the intact seen/heard/self-generated margins were `0.20417`, `0.21917`, and
`0.23000`; on seed 607 they were `0.27000`, `0.23417`, and `0.23167`. Every
margin exactly matched its learning-lesion value. Rival spike burden was `0.0`
in intact, learning-lesion, and expression-lesion arms on both seeds. Thus the
plastic inhibitory routes changed physically but did not improve the measured
source attribution or reduce an active rival burden.

## Provenance

The aggregate ran on the NumPy backend on `pool40` from immutable Git archive
commit `6986109804925cd49d40d6127fb2662c24df4762`, manifest
`a1292cdca4f1234023316b69da1c15948ae846235b15412978ed3816aa2aba74`,
run ID `1785787233-501434`. The exact ordered calibration pair was `601/607`.
Artifact: `research/findings/raw/source_monitor_coresidency_v4/calibration.json`
with its `.prov.json` sidecar.

An independent audit discovered after launch that the old formal gate trusted
the manifest metadata without hashing deployed source. The provisioned archive
was therefore checked directly before accepting this result and again after
the run with `sha256sum -c`; all 1,514 listed source files passed both times,
and the manifest file's own digest matched `.source_revision`. The central
provenance door now performs equivalent start and exit checks automatically and
new pool archives make listed source files read-only.

## Decision

Do not open v4 development or held-out seeds. Do not rerun `601/607` with the
fixed evaluator. Retire this adaptive-inhibition candidate because the formal
records are invalid and its intact and learning-lesion behavior was identical.
A successor needs a fresh preregistration and fresh seed partition, and should
first establish a nonzero rival burden that a local competition mechanism can
causally reduce without weakening already strong sources.
