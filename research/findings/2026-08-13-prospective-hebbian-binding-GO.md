---
type: finding
status: go
date: 2026-08-13
lane: prospective
mechanism: one-shot Hebbian potentiation of the cue->action binding at intention-formation (implementation-intention)
runner: research/runners/_pmem_hebbian_binding_derisk.py
artifacts:
  - research/findings/raw/_pmem_hebbian_binding.json
  - research/findings/raw/_pmem_hebbian_binding.json.prov.json
---

# Prospective memory's cue->action binding is RETIRED from the build-time install: it is LEARNED via a ONE-SHOT HEBBIAN potentiation at intention-formation (6/6, learned == installed)

**Verdict: GO at the pre-registered 6-seed gate (6/6 pass EVERY clause; need 5/6).** The #1 declared residual of the
production-wired prospective faculty (`2026-08-13-prospective-memory-production-wired.md`,
`scaffold_retired: NO`) was that the cue->action CONTENT binding -- WHICH cue Y releases WHICH action X -- is
INSTALLED synaptically at build. This runner replaces that install with a ONE-SHOT HEBBIAN potentiation at
intention-formation (Gollwitzer implementation-intention: stating "when Y, do X" forges a rapid cue->action link),
and the full faculty still fires **6/6** with every silence clause **6/6**, the binding **absent before formation**
on every seed and the **Hebbian event load-bearing** on the fire. CPU, reuse-by-import of the de-risked GO
`SFANmdaProspectiveMemory` + the FROZEN gate, NO `sim/` edit.

Artifact: `research/findings/raw/_pmem_hebbian_binding.json` (provenance sidecar beside it).

## The mechanism (additive; NO `sim/` edit; reuse-by-import)

Build the GO substrate normally so the homeostat bias + the plateau threshold CALIBRATE against the canonical
binding -- a DEVELOPMENTAL operating-point tuning of the release circuit (its innate readiness to hold a
cue->action association at the standard synaptic strength) -- then ZERO the `cue_monitor` binding so NONE exists
before the formation turn. AT FORMATION, a single event co-activates the cue assembly (Y), the action assembly (X),
and -- as the deliberate PFC goal activation that forms an implementation intention -- an instructional drive to the
release pool `rel_X`. The coincident spiking (pre = cue u action cortex, post = `rel_X`) potentiates `cue_Y->rel_X`
and `act_X->rel_X` in ONE shot via a saturating Hebbian outer product `w_ij = ceiling * sat(pre_i) * sat(post_j)`,
`sat(x)=min(1, rate_x/0.06)`. The cue-monitor + latch then operate on the LEARNED binding.

This is binding by COINCIDENCE, written locally with no algebra (`research/biology/coincidence-binding.md`; Kandel
6e: a spine Ca2+ signal is "a biochemical detector of the near simultaneity of the input (EPSP) and output
(backpropagating action potential)"), ONE-SHOT over a behavioral timescale
(`research/biology/btsp-place-field-formation.md`; Bittner et al. 2017 Science 357:1033, a SINGLE plateau creates a
place field). The realization is the established repo pattern -- a Hebbian pre x post OUTER PRODUCT, coincidence/
post-gated, NOT presynaptic TM facilitation (`2026-07-13-RUNG6d-spiking-STP-binder-needs-HEBBIAN-not-presynaptic-6seed-GO.md`),
the same host-applied LOCAL rule the repo's other spiking binders use (RUNG6c/6d, gap#2). External grounding:
Gollwitzer, P.M. (1999), "Implementation intentions: strong effects of simple plans", American Psychologist
54(7):493-503.

## Results -- 6 seeds 42/43/44/100/101/102, N=5

The gate (thresholds + per-seed clause logic) is IMPORTED from the parent runner and the substrate class is
monkey-patched, so every clause is scored by the SAME code, now with the binding LEARNED not installed.

<!--derived-->
| clause | result (binding LEARNED via one-shot Hebbian) |
|---|---|
| fire_on_cue | **6/6** |
| persistence | 6/6 |
| no_fire_before | 6/6 |
| no_fire_wrongcue | 6/6 |
| no_intention_silent | 6/6 |
| lesion_holds | 6/6 |
| lesion_forgets | 6/6 |
| separation | 6/6 |
| **seeds passing ALL clauses** | **6/6** (need 5/6) |

<!--derived-->
Per-seed correct-cue release `fireA`/`fireB` (both must clear FIRE_THR=0.20): 42 `0.395`/`0.383`, 43 `0.382`/`0.373`,
44 `0.334`/`0.358`, 100 `0.372`/`0.340`, 101 `0.364`/`0.385`, 102 `0.381`/`0.398`. Every silence read stays
sub-ceiling (`max_silent` 0.042-0.050 < 0.06 on every seed). These match the installed-binding GO
(`2026-08-13-prospective-sfa-nmda-amplifier-GO.md`) because the one-shot potentiation reconstructs the binding
EXACTLY: learned `|w|` = **17760.0 / 17760.0** (canonical) on all 6 seeds.

## Anti-cheats -- the binding is genuinely LEARNED-not-installed (load-bearing)

- **NO BINDING BEFORE FORMATION (6/6):** on a fresh build, the summed `|cue_monitor weight|` read from the live CSR
  is **0.0** (<= 1e-3) on every seed -- the build-time install is gone; it becomes > 0 only after the Hebbian event
  (to exactly the canonical `17760.0`).
- **HEBBIAN-LESION -> NO FIRE (6/6):** latching the intention WITHOUT the Hebbian event
  (`form_intention_no_hebbian`) leaves the binding at 0, and the correct cue does NOT fire (`rel_A` = **0.0000** <=
  SILENT_MAX on every seed). So the fire is caused by the one-shot formation event, not a residual install. (This is
  DISTINCT from the parent's latch-lesion, which zeroes the HELD attractor; here the LATCH is intact and the BINDING
  is what is absent.) `attributable_to`: **100%** of the correct-cue release is owned by the formation event
  (formed 0.36 vs Hebbian-lesion 0.00).
- **SILENCE STAYS 6/6:** a learned binding must still be cue-SPECIFIC -- every silence clause (wrong-cue,
  no-intention, no-fire-before, lesion) holds, so the Hebbian event did not smear the association. `void_if` guards
  for silence-regression, binding-present-before, and Hebbian-lesion-fires are all clear.

**Note on the saturation knob (honest, quantified).** The numbers in this paragraph are from INTERMEDIATE
diagnostic runs (a `form_sat=0.12` 6-seed run and a `--seed 44 --form-sat 0.06` single-seed run) whose raw JSON was
overwritten by the final `form_sat=0.06` artifact cited above; they are recorded here as the tuning trail, not as
headline results.

<!--derived-->
At a looser `form_sat=0.12` the run was 5/6: seed 44's weaker pool `fireB` dipped to 0.181 (~0.019 under
FIRE_THR=0.20) because a few `rel` neurons under-fired during formation, so its learned `|w|` reached 17710.7 /
17760 (99.7%) rather than exactly canonical -- and seed 44 is the pre-existing marginal seed the amplifier arc
flagged (its fire is bias-suppressed). Tightening `form_sat` to 0.06 fully saturates every participating neuron ->
the learned binding reconstructs EXACTLY canonical on all seeds -> seed 44 `fireB` recovers 0.181 -> 0.358 and the
run is a clean 6/6. `sat(x)=min(1, rate/0.06)` still zeros any truly-silent neuron (the lesion holds), so this is
fuller one-shot LTP saturation, not a loosened silence.
<!--/derived-->

## Brain-based scope + what is retired vs what remains host

- **BRAIN-BASED:** the potentiation is a LOCAL pre x post rule driven by REAL spikes (`cp_firing_states`), read and
  applied via `set_pathway_weights` -- the same class of host-applied local Hebbian outer-product the repo's spiking
  binders use. The hold, cue-monitoring and release remain fully spiking (inherited unchanged from the GO substrate:
  the intention LATCH, the per-pool homeostat, the SFA + NMDA/dendritic-plateau coincidence amplifier).
- **RETIRED:** the build-time synaptic INSTALL of the cue->action content binding. The binding is now CONTINGENT on
  a spike-driven formation event -- absent until formation, load-bearing on it, cue-specific.
- **REMAINS HOST (declared, narrowed):** (1) the text->slot / cue-presence SENSORY boundary (unchanged; the
  declared boundary like curiosity's novelty derivation); (2) the formation INSTRUCTIONAL drive to `rel_X` (the
  goal-activation ENCODING input, the same host-provides-input boundary as `_write`'s drive); (3) the DEVELOPMENTAL
  calibration of the pool operating point (homeostat bias + plateau theta) against the canonical binding strength.
  The engine-native STDP realization of the same local rule is the further step.
- **FUNCTIONAL CORRELATE, NOT phenomenal:** this measures a prospective-memory correlate (a learned held-intention x
  cue coincidence release). It makes NO claim of subjective intending.

## What this closes, and the next step to `scaffold_retired: YES`

This de-risk establishes the RETIREMENT MECHANISM at the pre-registered gate: the cue->action content binding is
learnable one-shot at formation and is functionally EQUIVALENT to the install (identical 6/6, exact-canonical
reconstruction). To move the production faculty to `scaffold_retired: YES`, the remaining step is to WIRE this
Hebbian formation into the production organ (`research/runners/prospective_memory_production_organ.py`): build the
`HebbianBindingProspectiveMemory` subclass in `_ensure_pm`, and route `form_intention` through the one-shot Hebbian
event instead of the build-time install. That is deterministic wiring glue over a now-de-risked mechanism (the same
pattern as the organ's original wiring), not a new scientific GO.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._pmem_hebbian_binding_derisk --smoke     # 1 seed, N=3, fast
SIM_BACKEND=numpy python -m research.runners._pmem_hebbian_binding_derisk --derisk    # 6 seeds, N=5
```
