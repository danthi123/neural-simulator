# Cheat-removal #2 RESOLVED: genuine basal-ganglia disinhibition opens the gate — 2026-06-04

**One line:** The gated-composition demos opened the thalamic gate-control pools by driving them with **direct
input current** — a stand-in for basal-ganglia disinhibition. That stand-in is now removed: the gate is opened by
a **genuine direct-pathway cascade** `cortex/striatum D1 → (GABA) GPi → (GABA) thalamus`, where a tonically-pacing
GPi normally silences its thalamic relay and a D1 "go" signal silences that GPi, **disinhibiting** the relay so its
firing opens the cortical route transmission gate. Validated 11/12 across 3 seeds with the inhibition mechanism
isolated and confirmed.

## What was the cheat

`gated_compose_bg_demo` (`bind_via_bg`) set `cp_external_input_current` on the thalamic gate-control pools directly,
as a proxy for "the basal ganglia selected this gate." No striatum, no GPi, no disinhibition — the thalamus was
just switched on by hand. Honest, labelled, but not the biology (Logiaco-Abbott-Escola 2021; Kandel ch 38 direct
pathway "go").

## The genuine replacement (`research/runners/gated_compose_bg_genuine_demo.py`)

Per (verb, motor) binding, in the core-sim brain-region framework:

- `d1_v_m` — striatal D1 MSN pool, all-GABAergic (`exc_fraction=0.0`). The "select this binding" signal.
- `gpi_v_m` — GPi output pool (`IZH2007_GPI_OUTPUT`), all-GABAergic, held tonically firing by a 2200 pA pacemaker
  drive (the reduced model's stand-in for the STN excitation that paces real GPi).
- `thal_v_m` — thalamic relay (`IZH2007_THALAMIC_RELAY`) carrying a 600 pA tonic excitation it can only express
  when GPi releases it.
- Pathways: `d1 -| gpi` (GABA), `gpi -| thal` (GABA), and the cortical route `verb -> motor` gated by the
  per-binding `transmission_gate` (the validated core-sim `cp_transmission_gain`). `couple_gate_to_pool` opens
  each route gate from its `thal_v_m` firing.

Flow: drive `d1_v_m` → D1 silences `gpi_v_m` → the relay `thal_v_m` is disinhibited → its firing opens the route
gate → the verb routes to its motor. Non-selected bindings: GPi keeps pacing → relay stays silent → gate closed.

## The non-obvious blocker: synaptic WEIGHT SCALE, not cascade structure

The first hand-built attempt failed in a way that looked like a deep framework bug: **driving D1 RAISED its GPi**
(0.276 → 0.465) — inhibition appeared inverted. A minimal control isolated the cause and it was **not** a sign bug
and **not** the trait routing:

- The brain-region framework builds every cross-region pathway as `E_TO_MIX` with positive weights; inhibition is
  applied at runtime from the **presynaptic neuron's trait** (set via `output_inhibitory_indices → cp_traits=1`).
  Introspection confirmed D1 was correctly trait=1, in the inhibitory mask, `inhibitory_propagation_strength=0.105`.
- The conductance kernel sign is correct: `I_syn = g_e·(E_e−V) + g_i·(E_i−V)`; positive `g_i` with `E_i=−75`
  hyperpolarizes.
- **The bug was the weight.** At `weight_mean≈300` (the value the stand-in used elsewhere), the inhibitory
  conductance `g_i` accumulated to **~2300** — vs a physiological O(1–10). Such a conductance **clamps the membrane
  to the −75 reversal** and drives the Izhikevich neuron into a paradoxical rebound-firing regime, so the
  "inhibition" reads as excitation.

### Decisive control (`research/runners/_framework_inhibition_minimal_probe.py`)

One all-inhibitory region → one excitable region, default Izhikevich, no neuron-type confound:

| weight | tgt rate, src OFF | tgt rate, src ON | verdict |
|---|---|---|---|
| 300 | 0.057 | **0.462** | EXCITES (g_i≈2300 breaks numerics) |
| 2 | 0.057 | 0.010 | INHIBITS |
| 5 | 0.057 | 0.006 | INHIBITS |
| 10 | 0.057 | 0.004 | INHIBITS |
| 20 | 0.057 | 0.006 | INHIBITS |

At g11_bg's validated weight scale (D1→GPi=15, GPi→thal=8) the conductance stays physiological and inhibition
works cleanly. **This is why a too-strict read almost shipped a false "framework inhibition is inverted" finding —
the smell-test (scrutinise a surprising result harder; the project has extensive *validated* inhibitory results)
forced the control that corrected it.**

## Result

```
D1 silences GPi: True   thal released: True   other stays silent: True   => CLEAN
D1->GPi ISOLATION: gpi_GO_N(no d1)=0.276  gpi_GO_N(d1 driven)=0.068  -> inhibition WORKS
seed 42: GO->N COME->S STOP->W LOOK->E  = 4/4
seed 43: GO->N COME->S STOP->W LOOK->E  = 4/4
seed 44: GO->N COME->N STOP->W LOOK->E  = 3/4   (COME->S is the one seed-fragile binding)
                                           ----
                                           11/12
```

The single miss (seed 44 COME→S) is a verb→motor **decode** fragility of the underlying gated-compose substrate
(documented per-seed), not the BG cascade — the cascade correctly disinhibits the COME→S channel; the gated route
decodes to N at that seed. The disinhibition **mechanism** is clean at all three seeds (isolation + diagnostic).

## Biology-translatable insight

Conductance-based synaptic inhibition is only inhibitory within a **physiological conductance range**. An oversized
synaptic weight does not "inhibit harder" — past the point where `g_i` dominates the membrane equation it pins V to
the GABA reversal and the cell's intrinsic dynamics (here Izhikevich recovery) produce rebound firing, so the
projection flips to net-excitatory. Real GABAergic synapses operate at conductances small relative to the leak;
the model must too. This is the concrete reason the direct-current stand-in existed (it sidestepped tuning the
cascade) and the concrete fix (g11_bg-scale weights) that makes the genuine cascade work.

## Scope / honesty

- Mechanism (D1 ⊣ GPi ⊣ thal disinhibition opening the gate) is genuine and validated multi-seed.
- WHICH D1 pool is driven is still **commanded** (a direct current on the selected D1 pool). That is the *separate*
  cheat #3 (commanded vs learned selection) — the next item. #2 is specifically about the gate being opened by
  genuine disinhibition rather than direct thalamic current, which is now true.
- The GPi pacemaker baseline is a tonic current (reduced-model stand-in for STN drive); a fuller model would pace
  GPi via an STN region.

## Files

- `research/runners/gated_compose_bg_genuine_demo.py` — the genuine cascade + in-demo D1→GPi isolation test.
- `research/runners/_framework_inhibition_minimal_probe.py` — the decisive weight-scale control.
- `research/runners/_bg_inhibition_probe.py` — probe of the bare g11_bg cascade (showed it needs the runner's
  tonic/STN drives to be observable; quiescent on its own).
