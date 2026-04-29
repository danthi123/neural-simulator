# Catalog-Driven Remediation Pass

**Date:** 2026-04-29
**Source:** `references/feature-catalog.md` + `references/biology-buildout-roadmap.md` (catalog-build branch)
**Purpose:** Fix inconsistencies, incorrect implementations, and incorrect naming in the simulator before implementing new clusters.

The textbook catalog (Kandel 6e + 4 supplemental texts) flagged ~13 sim-level corrections to existing implementations. Each is grounded in textbook citations. Tier R1 = bugs in implemented machinery; Tier R2 = naming/conceptual; Tier R3 = architectural extensions to existing components.

## Priority order (executed)

| ID | Title | Tier | Status |
|---|---|---|---|
| R1.1 | Per-region E_inh override (MSN ~−60 mV, SNc DA ~−55 mV) | R1 | _in progress_ |
| R1.2 | FSI cross-action wiring (replaces MSN→MSN as biologically-correct WTA) | R1 | _pending_ |
| R3.5 | Sparse + decorrelated cortex→MSN (Bolam-2000) | R3 | _pending_ |
| R3.10 | SNr→SNc disinhibition pathway (DA-burst driver) | R3 | _pending_ |
| R3.7 | GPe split into PV+ (prototypic) / PV− (arkypallidal) | R3 | _pending_ |
| R3.11 | Striosome/matrix split (limbic→striosome→SNc; matrix→SNr) | R3 | _pending_ |
| R3.6 | D1 dynorphin/SP + D2 enkephalin neuropeptide arms | R3 | _pending_ |
| R3.8 | GPi/SNr NaP + SK + Ih channel enable for tonic 40–80 Hz | R3 | _pending_ |
| R2.4 | Tonic DA + aversive=phasic-depression coding (Schultz98/16) | R2 | _pending_ |
| R3.9 | MSN KIR2/Kv2 voltage-dependent leak (Up/Down bistability) | R3 | _pending_ |
| R2.3 | Striatal interneuron taxonomy doc clarification | R2 | _pending_ |
| R3.12 | CA3 SWRs intrinsic, NREM as gate (framing fix) | R3 | _pending_ |

## Catalog citations

Catalog lines (`references/feature-catalog.md` on `catalog-build` branch):

- **R1.1** — line 434, 520: PBR-160 ch 6/11. MSN GABA_A reversal ~−60 mV (depolarizing-but-shunting); SNc DA neurons lack KCC2 → −55 mV. Sim `cfg.syn_reversal_potential_i = -75` is wrong for these regions.
- **R1.2** — line 466: TK-2017 pp 161–163; Tepper-2018 pp 8–9. MSN→MSN paired recordings show <0.5 mV IPSPs, ~14–25% probability, fails often. FSI→MSN feedforward IPSPs are larger and more reliable. v3 `--bg-lateral-inhibition` is anatomically backwards but functionally equivalent. Recommended v3-bis: cortex_X → FS_X → MSN_Y for Y≠X (cross-action only).
- **R2.3** — line 412: TK-2017 ch 8; Tepper-2018. Striatum has 8 distinct GABAergic interneuron classes (PV-FSI, NPY-LTS, NPY-NGF, CR, TH/THIN, FAI, SABI, ChI/TAN), non-isomorphic to cortical taxonomy.
- **R2.4** — lines 947–948: Schultz98 / Schultz16 / Fiorillo 2013. Aversive DA "activations" are physical-impact artifacts; biology is phasic depression below tonic. Sim's signed-scalar reward conflates these.
- **R3.5** — line 167: Bolam-2000 / Kincaid 1998. Cortical→MSN: 1–2 synapses per pre/post pair, close-neighbour MSNs have disjoint cortical inputs. Current density 0.3–0.5 is too dense.
- **R3.6** — lines 98, 113, 453, 486: PBR-160 ch 16 McGinty. D1 MSNs co-release dynorphin (+SP) with GABA → KOR (suppress DA + Glu) + NK-1 (excite ChI). D2 MSNs co-release enkephalin → DOR (excite DA, inhibit ChI) + MOR (mixed).
- **R3.7** — line 213: PBR-160 ch 7 Kita; Mallet 2008. GPe = PV+ (prototypic, → STN/GPi/SNr) ∪ PV− (arkypallidal, → striatum). Currently single mixed pool.
- **R3.8** — line 140: PBR-160 ch 9 Deniau. SNr 40–80 Hz tonic pacemaker driven by NaP + SK + Ih. Kernels exist (`fused_hh_NaP_current_update`, `fused_hh_h_current_update`) but not enabled for SNr/GPi presets.
- **R3.9** — line 433: PBR-160 ch 6 Wilson. MSN bistability requires KIR2 (RMP clamp at −80 to −95 mV) + Kv-1.2/Kv-2.1 (−60 mV deactivation) → input resistance peaks 6× higher at −60 mV. KIR2 expressed P25–P28 only.
- **R3.10** — line 520: PBR-160 ch 11 Tepper & Lee. SNr→SNc axon-collaterals provide disinhibitory drive for DA bursts; major in-vivo DA driver.
- **R3.11** — line 519: PBR-160 ch 9. Striosomes project to BOTH SNc and SNr (not just SNc); striosome/matrix split aligns with SNc/SNr at the output level.
- **R3.12** — line 1003: Bz Cycle 12. SWRs are intrinsic CA3 events — present in fetal hippocampus, transplanted hippocampi, and CA3 slices in vitro. NREM is a gate, not a generator.

## Notes

- All changes preserve `--bg-lateral-inhibition` as a deprecated alias to keep flagship configs working during transition.
- All new neuromodulators (R3.6) opt-in via existing `NeuromodulatorConfig` framework; default off.
- KIR2/Kv2 kernel (R3.9) is the only item that requires new GPU code; rest reuse existing primitives.
- Final validation pass runs flagship smoke + all unit tests.

## R3.12 — CA3 SWRs intrinsic, NREM as gate (forward-looking)

The catalog (Bz Cycle 12 p. 344, Leinekugel et al. 2002) flagged that sharp-wave-ripple events (SWRs) are **intrinsic CA3 self-organized events** that *also* happen to occur preferentially during NREM. Specifically, sharp waves are present in fetal/neonatal hippocampus before any sleep-stage architecture exists; they persist in transplanted hippocampi cut off from all afferents; they emerge in CA3 slices in vitro.

**Implication for any future SWR / replay implementation in this project:**

- SWR generation must live in **CA3 region intrinsic dynamics** (recurrent excitation + adaptation thresholds), NOT in a bridge-level sleep-stage scheduler.
- The role of NREM (slow oscillation up-states + spindle troughs) is to **gate** which SWRs have downstream effect — passive selection, not active generation.
- Empirical validation: with the NREM scheduler disabled, the CA3 recurrent network should still produce intermittent sharp-wave-like population bursts during quiet rest. Re-enabling the slow-oscillation gate biases bursts toward Up-state troughs *passively* — no scheduler needs to impose this.

The current project does not yet have a CA3 region, an SWR scheduler, or sleep-replay infrastructure. R3.12 is a **forward-looking framing fix**: when T1.B (SWR-driven sequential replay, biology-buildout-roadmap.md month 2) lands, it must place the SWR generator inside CA3 intrinsic dynamics, with NREM as a passive gate.

This also informs Cluster D (hippocampal trisynaptic loop, T1.A, month 1): CA3 should be configured with sufficient recurrent density + spike-frequency adaptation to self-organize sharp-wave-like population bursts on its own.

Flagging here so it isn't forgotten when sleep-replay work begins.
