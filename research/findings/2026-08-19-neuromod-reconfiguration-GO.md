---
type: finding
status: live
date: 2026-08-19
mechanism: neuromodulator-reconfigures-effective-connectivity
lane: neuromodulation
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
board_task: 64
instrument: put the fixed-anatomy nav basal-ganglia spiking substrate under two DOPAMINERGIC states (high-DA Go / low-DA NoGo) via ONE neuromodulator on the subsystem bus (dopamine_mode: excitability_drive on D1(+) and D2(-) via NeuromodulatorManager), measure the FUNCTIONAL connectivity matrix F in each state with the board-#63 perturb-and-measure probe, and test whether the PATTERN reconfigures (not merely rescales). GO gate (all, 6 seeds): SAME WIRING max|W_go-W_nogo|==0; RECONFIG spearman(F_go,F_nogo)<0.9 max over seeds (a pure gain change is rank-preserving ⇒ 1.0); DOUBLE DISSOCIATION ≥1 edge active only in Go AND ≥1 active only in NoGo, every seed; MOD-DRIVEN without the modulator F_go==F_nogo byte-for-byte (max|dF|==0); determinism byte-identical F at fixed cfg.seed.
artifact: research/findings/raw/neuromod_reconfig/nr_6seed.json
---

# A neuromodulator RECONFIGURES the effective circuit on fixed wiring — not just its gain (6-seed GO)

**Verdict: GO (6/6 seeds).** On ONE fixed connectome, a single dopaminergic neuromodulator switches the
navigation basal-ganglia spiking substrate between two DIFFERENT functional circuits: a **Go-mode** in
which the DIRECT pathway carries signal (cortex→str_D1→thal/motor, thalamus released) and a **NoGo-mode**
in which the INDIRECT pathway carries signal (cortex→str_D2→gpe→stn, thalamus clamped). The functional
connectivity PATTERN reconfigures — specific region-crossing edges OPEN in one state and CLOSE in the
other — rather than uniformly rescaling. This realizes the Bargmann/Marder thesis that the connectome
UNDERDETERMINES behaviour: neuromodulatory state, not wiring, selects which functional mode runs.
All-spiking / all-synaptic; reuse-by-import of the board-#63 probe; NO `sim/` edit.

## Question + sources (external — deep-research gate)

Bargmann, "Beyond the connectome: how neuromodulators shape neural circuits", **BioEssays 34:458–465
(2012)** (doi:10.1002/bies.201100185). <!--derived--> Marder, "Neuromodulation of Neuronal Circuits:
Back to the Future", **Neuron 76:1–11 (2012)** (doi:10.1016/j.neuron.2012.09.010). <!--derived--> Both
argue a fixed synaptic wiring diagram does not determine the circuit's output: neuromodulators
reconfigure the EFFECTIVE connectivity moment-to-moment, so one anatomical network supports multiple
functional circuits (Marder's stomatogastric ganglion is the canonical demonstration; Bargmann
generalizes it). Our neuromodulators to date mostly change GAIN (scale activity up/down). The target
here is a PATTERN change — some pathways opened, others closed — so one substrate supports multiple
functional modes. The dopaminergic direct/indirect switch tested here is the canonical mammalian
instance (Albin–DeLong–Penney model; Gerfen & Surmeier, "Modulation of striatal projection systems by
dopamine", Ann. Rev. Neurosci. 34:441–466, 2011). <!--derived-->

## Mechanism (all spiking / all synaptic; on the neuromodulator bus)

Dopamine acts through D1 receptors (Gs-coupled, EXCITATORY) on direct-pathway MSNs and D2 receptors
(Gi-coupled, INHIBITORY) on indirect-pathway MSNs. We express exactly this with ONE modulator on the
subsystem's own bus (`sim.neuromodulators.NeuromodulatorManager`): a `dopamine_mode` config with two
`excitability_drive` targets — `scope="group:str_D1"` (sensitivity +1000 pA/unit) and
`scope="group:str_D2"` (sensitivity −1000 pA/unit). The concentration IS the state, so at
(conc−baseline)=±0.5 the per-neuron bias is ±500 pA:

- **Go-mode** (DA high, conc=1.0): D1 +500 pA / D2 −500 pA → direct pathway primed, indirect silent.
- **NoGo-mode** (DA low, conc=0.0): D1 −500 pA / D2 +500 pA → indirect pathway primed, direct silent.

The per-neuron drive is produced by the subsystem's OWN
`NeuromodulatorManager.compute_excitability_drive_per_neuron()` (a concentration→current mapping — the
biological action of a receptor-coupled conductance on membrane excitability); the runner only ADDS that
bias to the neurons' input current, exactly as the bridge does internally when the subsystem is enabled.
The reconfiguration itself — which pathways propagate — is done entirely by neurons firing and synapses
transmitting. No host arithmetic stands between the modulatory state and the measured propagation.

## Read-out (board #63, reuse-by-import)

For each state we run the perturb-and-measure probe (`research/runners/_perturb_and_measure_derisk.py`):
perturb every region TYPE A (+400 pA), settle, record the signed Δ firing-rate of every other TYPE B →
a state-specific FUNCTIONAL matrix F. Anatomy is aggregated to the same 12 TYPES. All plasticity + noise
+ heterogeneity OFF (isolation: measure propagation on a FIXED graph). `cfg.seed` seeds the substrate
(the seed trap — not `actual_seed_used`). Operating point lowers the striatal tone to 40 pA (vs the #63
probe's 250) so the D1/D2 MSNs sit near threshold and the dopaminergic drive GATES which pathway a
cortical drive recruits; at the #63 tone the MSNs fire regardless and DA only rescales (spearman ~0.89,
the gain-only regime — see residuals). The gpi keeps its strong pacemaker so the disinhibition route can
carry signal. NOT tuned to flatter the probe: the modulator-lesion and W-unchanged controls guard it.

## Result — the PATTERN reconfigures (not gain), robustly across 6 seeds

Pooled (per-seed spearman = 0.7989, 0.7751, 0.8033, 0.7488, 0.7587, 0.8122):

| metric | pooled | reading |
|---|---|---|
| spearman(F_go, F_nogo), union-nonzero edges | **mean 0.7828, max 0.8122** | RANK correlation < 1 on every seed ⇒ NOT a pure gain change (gain is rank-preserving ⇒ 1.0) |
| edges OPENED only in Go (min / mean over seeds) | 7 / 7.0 | direct-pathway edges present only in Go |
| edges OPENED only in NoGo (min / mean) | 4 / 4.667 | indirect-pathway edges present only in NoGo |
| spearman(F_go, F_nogo) with the MODULATOR ZEROED | **1.000** | without DA the two states are the SAME network |
| max\|F_go − F_nogo\| with the modulator zeroed | **0.0** | byte-for-byte identical ⇒ 100% attributable to the modulator |
| max\|W_go − W_nogo\| (anatomy) | **0.0** | wiring is IDENTICAL across states — a purely functional switch |
| determinism: byte-identical F on re-run @ fixed seed | True | — |

**The DOUBLE DISSOCIATION (edges common to ALL 6 seeds):**

- **Opened only in Go (DIRECT pathway):** `cortex→thal`, `cortex→motor`, `str_D1→thal`, `str_D1→motor`,
  `gpi→thal`, `gpi→str_PV_FSI`, `str_PV_FSI→gpi`. The direct route disinhibits the thalamus (str_D1
  inhibits gpi → gpi releases thal), so a cortical drive reaches thal/motor.
- **Opened only in NoGo (INDIRECT pathway):** `str_D2→gpe`, `str_D2→gpe_arky`, `str_D2→str_PV_FSI`. The
  indirect route engages (str_D2 inhibits gpe), and the thalamic disinhibition edges are absent.

<!--derived-->
The unperturbed BASELINE regimes corroborate this independently (seed 42, rates rounded for readability;
full precision in the artifact's `baseline_rates_go`/`baseline_rates_nogo`): **Go** gpi=0.078 (suppressed)
→ thal=0.041 (released), str_D2=0.0 (silent); **NoGo** gpi=0.247 (~3× higher, pacemaker restored) →
thal=0.001 (clamped), str_D1=0.002 (silent). The same anatomy sits in two whole-circuit regimes — the
thalamus is released in Go and clamped in NoGo — set only by the dopamine concentration.

## Why this is reconfiguration, not gain (the anti-cheats ARE the result)

1. **spearman is scale-invariant.** A pure gain change (F_go = k·F_nogo) preserves edge RANK-ORDER ⇒
   spearman = 1.0. spearman ≤ 0.8122 on every seed means the rank-order changed: edges moved between
   active and inactive. The opened/closed edges make the mechanism concrete (direct vs indirect).
2. **Modulator-driven (zero-lever control).** With both targets' sensitivity set to 0 the per-neuron
   drive is identically 0 at every concentration, so the two "states" collapse to the same network:
   F_go == F_nogo byte-for-byte (spearman 1.0, max|dF|=0). The reconfiguration is 100% attributable to
   the modulator — no confound in the two-state protocol survives.
3. **Same wiring.** max|W_go − W_nogo| = 0 exactly — pathways are never touched. This is a functional
   reconfiguration on FIXED structure, which is the whole claim.

## Honest residuals / scope

- **The DA level is host-set (manual concentration), not produced by a spiking dopaminergic nucleus.**
  This is the standard neuromodulator-subsystem boundary: the modulator LEVEL is a scalar the runner
  sets (the "state"), analogous to VTA/SNc tonic DA. A full closure would drive `dopamine_mode` from a
  spiking dopaminergic population reading the behavioural context (the subsystem already supports
  production rules such as `from_reward` / `from_region_firing_signed` for exactly this). What is
  demonstrated here is that the bus, once set, RECONFIGURES the circuit — the capability the board task
  asked for — not a self-driven mode switch.
- **De-risk, not production integration.** This is additive / default-off (pure runner reuse, no `sim/`
  edit); it shows the CAPABILITY on the existing substrate. Wiring `dopamine_mode` into the live brain's
  default config so it reconfigures moment-to-moment during a conversation (rest/focus/arousal modes) is
  the next rung, not done here.
- **Operating-point dependence is itself the Bargmann/Marder point, and is bounded.** At the #63
  striatal tone (250 pA) the MSNs fire regardless of DA and the switch degrades to gain (spearman
  ~0.89, few opened edges) — the honest gain-only regime. The reconfiguration requires the MSNs to sit
  where DA can gate their recruitment (tone 40 pA here). That neuromodulation matters MORE near
  threshold is expected, not a cheat; the controls (modulator-lesion, W-unchanged) hold regardless.
- **The perturbation is +400 pA (vs #63's +800):** lower so recruitment is set by DA excitability, not
  a saturating drive. At +800 the switch still reconfigures (spearman ~0.83) but the canonical
  direct-vs-indirect edge dissociation is cleaner at +400.

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -m research.runners._neuromod_reconfiguration_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/neuromod_reconfig/nr_6seed.json
# exploration (one seed, operating-point sweep):
#   ... --explore --seeds 42 --str-tone 250 --s-d1 1000 --s-d2 1000   # the gain-only regime
```

Runner: `research/runners/_neuromod_reconfiguration_derisk.py`. Raw + provenance sidecar:
`research/findings/raw/neuromod_reconfig/nr_6seed.json(.prov.json)`.
