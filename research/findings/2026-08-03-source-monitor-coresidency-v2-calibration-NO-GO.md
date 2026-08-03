---
type: finding
status: negative
date: 2026-08-03
mechanism: source-monitor-coresidency-v2
runner: research/runners/_laneC_source_monitor_coresidency_gate_v2.py
artifacts:
  - research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed216.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed216.json.prov.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed217.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed217.json.prov.json
---

# Local source competition clears the margin floor but misses its no-harm control

<!--derived-->
**Verdict: NO-GO at v2 calibration.** Local fast-spiking competition kept
seen, heard, and self-generated source margins above the unchanged `0.15`
floor on both fresh seeds. Seed 217 nevertheless lost `0.0092` of its
self-generated margin versus the competition lesion, violating the fixed
requirement that stabilization must not weaken any source.

## Mechanism Change

V1 missed development because one heard-source population reached only a
`0.11` margin. V2 keeps the source drive, Hebbian rate, and margin floor
unchanged. Each source-memory pool recruits a six-neuron fast-spiking
interneuron population, which inhibits the other two source pools through
GABA-A pathways on the same bridge. There is no source-specific host gain or
stronger labelled input.

## Result

Artifacts:
`research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed216.json`
and
`research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed217.json`.
Both cluster runs used clean source commit `7e925b766`, the NumPy backend, and
one revision-addressed source manifest. All nine validity preconditions passed.

| seed | seen margin | heard margin | self-generated margin | competition gains: seen / heard / self | competition spikes / lesion | result |
|---:|---:|---:|---:|---:|---:|:---:|
| 216 | 0.1683 | 0.2400 | 0.1508 | +0.0225 / +0.0217 / 0.0000 | 18 / 0 | pass | <!--derived-->
| 217 | 0.1733 | 0.1583 | 0.2067 | +0.0408 / +0.0225 / -0.0092 | 18 / 0 | fail | <!--derived-->

<!--derived-->
Both seeds passed every inherited source-memory requirement: all source
margins exceeded `0.15`, episode-to-source and source-to-ACC attribution were
`1.0`, unseen and learning-disabled episodes produced zero source spikes,
source swapping followed physical afferents, mixed-source recall reinstated
both sources, and inference accepted episode activity without source metadata.

Competition was active and causal. Its interneurons produced 18 spikes during
the measured seen recall on each seed and zero after their transmission gate
was lesioned. On seed 217, the lesion margins were approximately `0.1325`
seen, `0.1358` heard, and `0.2158` self-generated. <!--derived--> Competition
therefore rescued the two weak sources above the fixed floor while modestly
reducing the already strong source.

## Decision

Do not open development seeds 218, 219, or 314, and do not erase the no-harm
failure after seeing it. Preserve the local competition circuit as a promising
mechanism. Any successor gate must be filed on fresh seeds and justify, before
running, whether the functional requirement is zero degradation for every
source or bounded tradeoff with all absolute margins remaining robust. That
decision must be based on the role of source monitoring in the whole brain,
not chosen to make these two outcomes pass.

Remaining scaffolds are caller-supplied episode activity, predefined source
afferents and competition wiring, externally timed learning windows,
competition disabled during source-free rest to prevent rebound carry-over,
and host spike-count evaluation.
