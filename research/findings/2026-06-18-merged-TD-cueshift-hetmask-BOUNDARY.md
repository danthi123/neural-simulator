# Merged-bridge TD cue-shift — per-region heterogeneity-mask GO-fix attempt = continued BOUNDARY (the `sim/` edit is clean infrastructure; the full migration is a deeper op-point arc) (2026-06-18, CYCLE 224)

**Follow-on to** the #3 consolidation BOUNDARY (`2026-06-18-merged-TD-cueshift-consolidation-BOUNDARY.md`, commit `1f470737`),
which localized the partial migration (r=−0.43) to the merged het-OFF config saturating the `td_striosome` MSN-D1 critic
~6× (V saturates instead of grading), with the `--global-het-test` diagnostic confirming het-ON restores the critic's
graded band. The named GO fix was a **per-region heterogeneity mask** (give the td critic the het-ON graded band while
nav/conv stay het-OFF deterministic). This is the result of building + validating that fix.

## VERDICT: continued BOUNDARY — the `sim/` edit is CORRECT, CLEAN, byte-reviewed infrastructure, but it did NOT lift the migration to GO at the op-points tested

The per-region heterogeneity mask is the right MECHANISM (it restores the critic's het-ON grading, as the diagnostic
showed) and ships as correct default-preserving infrastructure, BUT at the gain co-tune grid tested it did **not** lift
the full merged-bridge peak-migration to the r<−0.7 bar — and the full migration remains a deeper, multi-knob
op-point co-tune arc (heterogeneity × GIRK-cap × derivative-gain × reward-relay) that this CPU de-risk did not crack.

### The `sim/` edit (CONTROLLER BYTE-REVIEWED — clean, additive, default-preserving)
A per-region parameter-heterogeneity mask, mirroring the existing `cp_nmda_neuron_mask` / `cp_homeostasis_neuron_mask`:
- `sim/regions.py`: `BrainRegion.enable_heterogeneity: Optional[bool] = None` (new field; None = follow the global flag).
- `sim/bridge.py`: `self.cp_heterogeneity_neuron_mask` (init None) + built ONLY if ≥1 region opts in (else stays None) +
  the B2 apply-gate widened to `(cfg.enable_parameter_heterogeneity or mask is not None)` + the apply itself:
  `if cfg.enable_parameter_heterogeneity or mask is None: target_array[:] = samples  else: cp.where(mask, samples, target_array)`.
- `nav_conv_merged_bridge.py`: `enable_heterogeneity=True` on the 12 `td_*` regions (the `co_resident_td_cueshift` slice).

**Byte-identity when off — VERIFIED (controller + subagent):** when no region opts in (mask None), BOTH the apply-gate
AND the `cp.where` fall back bit-for-bit to the legacy `target_array[:] = samples` (global-ON) / the method-not-called
(global-OFF). The **nav-byte gate is bit-for-bit identical (`n_mismatch=0`, `nav_byte_identical=true`)** and the mask
covers exactly the 354-neuron td slice (no non-td neuron). Biology: cell-type-specific intrinsic heterogeneity
(Marder-Goaillard 2006; Tripathy 2013) applied to one population while another stays homogeneous is legitimate.

### Results (seed 42, CPU/numpy)
| op-point (het-mask ON) | migration r | r<−0.7 GO? | dir |
|---|---|---|---|
| clip40/girk0.5/fs30/gain1 | **+0.100** | ✗ | wrong-way |
| clip40/girk0.5/fs30/gain4 | **−0.243** | ✗ | ✓ |
| girk0.5/gain2 | **−0.312** | ✗ | ✓ |
| girk0.5/gain3 | **−0.133** | ✗ | ✓ |

All 4 fail the −0.7 bar; the best (−0.31) is WORSE than the no-het-mask baseline (−0.43). The het-mask restores the
critic's *grading components* (per the diagnostic) but the specific op-point that lands the full *peak-migration* on the
merged bridge was not in this de-risk's gain×GIRK grid (and some settings hurt — the co-tune is non-monotone).

### The two consolidation gates STILL PASS (the het-mask co-resides cleanly)
- **MOAT byte-intact** = PASS (`moat_intact: true` — stored fact retrieves, the 3 abstentions hold; the het-mask + the DA broadcast do not perturb conversation).
- **NAV byte-identity** = PASS (`n_mismatch=0`, the het-mask is byte-identical for all non-td regions).

## Honest scope + decision

- **The dendrite question stays CLOSED-NEGATIVE.** This is a merged-bridge OPERATING-POINT engineering arc, not a
  substrate/biology one — the standalone `2026-06-10-N9-TD-cue-shift-A-CSC-GO` already proves point neurons do the full
  cue-shift. The science is settled; this is transplant-tuning on a busy shared bridge.
- **Decision: do NOT chase the migration op-point indefinitely (low scientific stakes).** The het-mask ships as correct,
  clean, default-preserving infrastructure (a generally-useful per-region heterogeneity primitive); the full r<−0.7
  merged migration is a **named, bounded op-point follow-on** (a proper multi-dimensional co-tune of het ×
  `critic_value_to_snc` GIRK-cap × `td` derivative-gain × the reward-relay weight, ideally with a small search rather
  than a hand grid), to be picked up if/when the merged cue-shift is prioritized over the consolidations already GO.
- **Roadmap status unchanged:** the TRUE-ONE-BRAIN roadmap is MECHANISM-COMPLETE; #3's standalone cue-shift is GO, its
  merged consolidation is co-residence-clean (both gates) + partial-transfer with the full peak-migration a bounded
  op-point follow-on. The shipped het-mask is the infrastructure that follow-on will build on.

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk --hetmask --seed 42 \
    --girk-cap 0.5 --deriv-gain 2   # (the gain×GIRK grid; none reach r<-0.7)
```
