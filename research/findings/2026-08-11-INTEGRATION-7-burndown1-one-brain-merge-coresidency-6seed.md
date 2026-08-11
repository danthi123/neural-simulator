---
type: finding
status: go
date: 2026-08-11
mechanism: co-resident one-brain merge — the e-prop acquisition net becomes disjoint slices of the single conversational SimulationBridge (append-LAST), sharing one cp_connections
lane: E-language / INTEGRATION (the "one brain" non-negotiable)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/i7_burndown1_one_brain_6seed.json
runner: research/runners/_i7_burndown1_one_brain_merge_derisk.py
instrument: `build_one_brain(co_resident_eprop=True, ...)` appends the e-prop acquisition slices (eprop_in/h1/out, ~80 neurons) LAST to the conversational bridge + injects the plastic FF pathways into the same union; a CoResidentEpropNet binds to the merged bridge (SEAM-A read discipline, sparse position-map, slice-scoped wash-out). Byte-identity requires the per-parameter heterogeneity reseed (the merged `sim/` cfg flag `per_parameter_heterogeneity_seed`, prototyped here as `--hook`). SIM_BACKEND=numpy.
---

# INTEGRATION #7 burn-down #1 — the ONE-BRAIN merge: the plasticity-learned-facts chat now runs on ONE spiking bridge (6/6 GO)

INTEGRATION #7 (+ burn-downs) made the plasticity-learned-facts chat's ACQUISITION (e-prop) and MOAT (spiking gate) the
brain's own — but the e-prop acquisition net ran on its OWN co-resident `SimulationBridge`, a SECOND bridge. This
burn-down merges it: the acquisition net's ~80 neurons (eprop_in 34 | h1 40 | out 6) become DISJOINT REGION SLICES of
the SINGLE conversational `build_one_brain` bridge (append-LAST, the proven nav+conv-merge pattern), and the two plastic
FF pathways are injected into the SAME union → ONE `SimulationBridge`, ONE `cp_connections`. This is the mission's core
"one brain" non-negotiable, at the substrate level.

## Result — 6/6 GO (`research/findings/raw/lanes/stageA/i7_burndown1_one_brain_6seed.json`: `GO_6of6: true`, `n_byte_identity_held: 6`, `n_smoke_go: 6`)

<!--derived-->

Across seeds 42/43/44/100/101/102, with the per-parameter heterogeneity reseed enabled (the merged `sim/` cfg flag;
`--hook` prototype in this de-risk):
- **BYTE-IDENTITY held 6/6** — appending the e-prop slices LAST leaves every pre-existing conversational neuron's
  thresholds + izh params AND the full decision transcript bit-identical (the reseed makes the append-LAST invariant;
  without it the single-RNG-stream draw shifted izh b/d/C and flipped the near-tie arbiter on 2/14 turns — no
  reply/moat change, but not byte-identical).
- **SMOKE_GO 6/6** — the merged one-brain #7 chat still GO: taught-recall 3/3, FROZEN-readout 0 (content rode the weight
  change), moat false-accepts 0, lesion-gate load-bearing.
- **ONE-BRAIN TEETH (per the earlier build report, held here):** `net.br IS comp._merged IS bridge` (a SINGLE
  `SimulationBridge`); the 1600 e-prop FF synapses live in the SAME `cp_connections` array as the 91333 conversational
  synapses; an e-prop teaching pass moves ONLY the eprop FF slots — the conversational weights are byte-unchanged.

## Scope / honesty

<!--derived-->

- Reaches **substrate-level ONE BRAIN — CO-RESIDENCY**: disjoint slices of ONE bridge object sharing ONE neuron pool and
  ONE connection matrix. This is a real step up from two separate bridge objects → one bridge hosting both.
- **NOT yet cross-region synaptic INTERACTION** — there is zero conv↔eprop synapse; co-location without cross-synapses is
  not full one-brain (per `project_one_brain_substrate_vs_functional`). The FURTHER step (a synaptic pathway conv-cue →
  eprop_in and eprop_out-spikes → composer render) is the `codex/cross-region-one-brain` arc (in flight).
- **The `sim/` enabler** (per-parameter heterogeneity reseed, ADDITIVE/DEFAULT-OFF, merged `1f4ea9f5`, determinism suite
  9/9) is what makes byte-identity hold; the merged bridge sets the flag. Runner-side NO `sim/` edit in this de-risk.
- Remaining scaffolds (named): the numpy familiarity gate (spiking v320 is burn-down #2, done separately), the argmax
  patient read-out (now on spikes via the separate neural-patient-readout merge), the host leaky-readout integration,
  the AI-teacher presentation. Throughput: e-prop teaching now steps the whole ~26K-neuron bridge (slower — inherent to
  co-residency; speed is secondary).
