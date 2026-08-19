---
type: finding
status: no-go
date: 2026-08-19
runner: research/runners/_replay_dg_pattern_separation_homeostat.py
artifacts:
  - research/findings/raw/kwta_homeostat/homeostat_6seed.json
---

# An intrinsic-excitability HOMEOSTAT does NOT stabilize the DG k-WTA — a per-cell firing-RATE set-point is ANTI-SPARSE and defeats the competition; the residual needs a POPULATION-competitive set-point

**Board #73** — "stabilize the memory-separator so one memory can't swallow another." Attacks the k-WTA-STABILITY
residual localized by the #71 bridge finding
(`2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-single-recall-ceiling-kWTA-stability-residual.md`):
on the production Izhikevich substrate one memory's DG engram collapses to near-dense (150-200 of 200 granules),
that dense engram SUBSUMES the other memory's sparse engram, and the dense memory's answer wins BOTH probes
(both_win 0/6). The #71 finding named, as its next mechanism 1, "a slow per-granule adaptive threshold (intrinsic
excitability homeostasis; Turrigiano) that drives every granule toward a target firing fraction ... would cap the
dense-collapse." This finding BUILDS that mechanism and shows it does the OPPOSITE.

## Verdict

**NO-GO, and the negative is FUNDAMENTAL, not a tuning miss.** A per-cell firing-RATE homeostat (adaptive spike
threshold driven toward a target activity set-point) cannot produce or stabilize a sparse pattern-separated code.
Its fixed point is a UNIFORM target-rate code in which EVERY drivable granule fires at the set-point — the DENSEST
possible code — so it actively RECRUITS silent cells (lowers their threshold) and, run alongside the k-WTA basket,
DEFEATS the inhibition that would silence them. Enabling it on `dg` drives the code fully dense on 6/6 seeds
(worse than the 4/6 lesion), leaving both_win at 0/6. This corrects the #71 mechanism-1 hypothesis. The k-WTA
residual instead needs a set-point on the POPULATION (total activity ≈ k), realized competitively — not a per-cell
rate. No `sim/` edit (the mechanism is the engine's own per-region homeostat, scoped via `BrainRegion.enable_homeostasis`).
Deterministic (`cfg.seed`); the LESION arm reproduces the #71 table byte-for-byte.

## What was built (the named mechanism, on-substrate, no `sim/` edit)

The production substrate already carries a per-neuron intrinsic-excitability homeostat: `cp_neuron_firing_thresholds`
adapted by `fused_homeostasis_update` (`ema <- (1-a)·ema + a·fired; err = ema − target; thr <- thr + err·adapt`,
clipped to a band). `BrainRegion.enable_homeostasis=True` scopes it to ONE region via `cp_homeostasis_neuron_mask`
while the GLOBAL `cfg.enable_homeostasis` stays False (every other region keeps its normal vpeak spike threshold).
We enabled it ONLY on `dg`, at the biological DG sparsity set-point (`target_rate=0.05`), fast (α=0.20, adapt=1.5,
so it acts within a replay event), threshold band ENTIRELY sub-vt `[-56,-44]` (granule RS: vr=-60, vt=-40, vpeak=+35).
The sub-vt band is required for the adaptive threshold to have ANY authority: at/above vt the Izhikevich quadratic
`k(v−vr)(v−vt)` runs v away past any finite threshold ≤ vpeak, so threshold detection cannot gate a driven cell;
sub-vt, detection fires the cell before the quadratic engages (a LIF-like regime). Runner reuses the #71 runner by
import (its shunting reversal + transmission-gated write + all measurement/probe/scramble/direct-readout machinery);
it runs a homeostat-ON arm and a LESION arm (`dg_homeostat=False`, byte-identical to #71) for the dissociation.

## Results — 6 seeds (42/43/44/100/101/102), ON vs LESION (`homeostat_6seed.json`)

<!--derived-->
_Numbers from the cited artifact._

| seed | ON sizes (m0,m1) | ON dense | ON sel (m0,m1) | LESION sizes | LESION dense | LESION sel (m0,m1) | single | scramble |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|
| 42  | (200,199) | yes | (+0.00,+0.03) | (200,44)  | yes | (−0.17,+0.15) | +1.00 | −1.00 |
| 43  | (200,200) | yes | (+0.00,+0.00) | (39,55)   | no  | (+0.55,−0.59) | +1.00 | −1.00 |
| 44  | (200,200) | yes | (+0.00,+0.00) | (56,198)  | yes | (−0.11,+0.06) | +1.00 | −0.99 |
| 100 | (200,200) | yes | (+0.00,+0.00) | (199,60)  | yes | (+0.09,−0.13) | +0.98 | −0.98 |
| 101 | (200,200) | yes | (+0.02,−0.04) | (46,39)   | no  | (+0.36,−0.36) | +1.00 | −0.92 |
| 102 | (200,200) | yes | (+0.00,+0.00) | (199,68)  | yes | (+0.44,−0.44) | +1.00 | −1.00 |

**Pooled:** ON dense-collapse **6/6** (LESION 4/6), both_win **0/6** (ON and LESION), ON mean selectivity +0.001
(chance), LESION mean −0.013. **The homeostat made the code DENSER, not sparser**, and reduced the per-memory
selectivity to ≈0 (the lesion's anti-symmetric signal, |sel| up to 0.59, is ERASED — every granule now fires for
every memory). The LESION arm's engram sizes reproduce the #71 finding's table EXACTLY (byte-faithful baseline).

**Controls (PASS — the runner is honest, not broken):** single-memory recall stays at ceiling (+0.98…+1.00, 6/6);
scramble-teach still inverts (−0.92…−1.00, 6/6); the LESION reproduces the #71 residual (dense-collapse returns
4/6, both_win 0/6). Determinism: seed-42 ON `(200,199)` reproduced across independent runs.

## Why it fails — three measurements, one root cause (the fixed point is anti-sparse)

The residual is not that the homeostat was mis-tuned; a per-cell RATE set-point is the wrong TOOL. Three sweeps
(scratchpad probes on seeds 42/43/44) localize it:

1. **It RECRUITS silent cells (anti-sparse fixed point).** Where the raw drive leaves DG silent or minimal
   (`input_to_dg_weight=25, drive=500`: LESION engram = (0,0)), the homeostat LOWERS thresholds to pull cells up to
   the target rate → ON engram = (67,120). A rate homeostat's stable state is "every drivable cell fires at target"
   = the densest code; it cannot leave most cells silent, which is what sparsity requires.

2. **It DEFEATS the k-WTA inhibition.** Strengthening the feedback basket (the mechanism that SHOULD select k
   winners) makes the homeostat code MORE dense, not less: `fs_to_dg_weight` 15→30→60 (with a sub-vt band + strong
   hyperpolarizing reversal) took DG from ~(130,150) to (200,200) on all three seeds. When inhibition silences a
   granule, its EMA falls below target, so the homeostat lowers its threshold and re-recruits it. The homeostat
   homeostatically UNDOES the competition.

3. **The Izhikevich quadratic bounds threshold authority.** At/above vt=-40 the regenerative `k(v−vr)(v−vt)`
   drives v past any threshold ≤ vpeak, so an adaptive spike threshold cannot silence a driven cell there; only
   sub-vt (LIF-like) does the threshold gate — and sub-vt, the recruitment of (1) dominates. No band position
   escapes: at/above vt → no authority; sub-vt → anti-sparse recruitment.

Across the full drive × weight × fan-in × inhibition-reversal × basket-gain × target-rate sweep, NO operating point
produced two sparse, symmetric, mutually-non-nested engrams under the homeostat. Every regime was dense or
nested-asymmetric.

## The mapped next mechanism (this banks a METHOD; the capability stays open)

**The selection set-point must live on the POPULATION, not the cell.** DG's ~1-2% sparsity is a k-of-N COMPETITION
(only the k best-matched granules fire; the rest are actively silenced by feedback interneurons), and its stability
is a set-point on TOTAL activity (~k), not on each cell's own rate. A per-cell rate homeostat optimises the wrong
objective (every cell hits its own rate ⇒ all active). The companion process the #71 residual actually needs — the
one we keep replacing with a constant — is an **adaptive INHIBITORY GAIN that regulates total DG activity toward
k** (a divisive-normalization / feedback set-point on the dg_fs pool, e.g. a competitively-plastic or
activity-scaled fs_to_dg gain), which raises inhibition when too many granules fire and thereby SHARPENS rather
than erases the competition. A secondary, complementary route the #71 finding also named — a DEVELOPED (competitively
learned) perforant projection — sharpens the per-pattern drive gradient so a fixed threshold selects k winners
(here the fixed random projection drives ~all granules above any sub-vt threshold, which is why (3) has nothing to
select on). **Per-cell firing-rate homeostasis is banked as insufficient for k-WTA selection: wrong locus (per-cell,
not population) and wrong sign (recruits, not selects).** Its correct biological role is slow (hours-days)
operating-point maintenance subordinate to the fast lateral inhibition that does selection — not a per-pattern
sparsifier.

## Tracked scaffolds (host, not brain)

Inherited from the #71 runner: host-defined input patterns and answer assemblies; host reinstatement of each
memory's input AND answer during replay (hippocampal index / SWR trigger); scheduled down-states; the WRITE/READ
transmission-gate phase (host-scheduled sleep/wake gate); a rate-window Hebbian coactivity write; an argmax over
answer spike counts for MEASUREMENT only; a fixed random perforant projection and fixed FS anatomy (not developed).
The homeostat itself is on-substrate (the neuron's own adaptive threshold); its operating point (target/α/adapt/band)
is host-set, like any config.

## Reproduce

    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_homeostat \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/kwta_homeostat/homeostat_6seed.json

The `--target-rate/--ema-alpha/--adapt-rate/--thresh-min/--thresh-max` flags sweep the homeostat operating point;
the runner always runs the LESION (`dg_homeostat=False`) arm alongside for the dissociation.

## Sources

EXTERNAL-SEARCH-RAN: intrinsic plasticity / firing-rate homeostasis set-point and its distinction from competitive
k-WTA selection (logged to the corpus-check record, 2026-08-19).

- Turrigiano, G. (2011). Too many cooks? Intrinsic and synaptic homeostatic mechanisms. Annu Rev Neurosci
  34:89-103. — intrinsic-excitability homeostasis toward a firing-rate set-point (the mechanism built here); it is
  a SLOW operating-point regulator, not a fast selector.
- Desai, N.S. (2003). Homeostatic plasticity in the CNS: synaptic and intrinsic forms. J Physiol Paris 97:391-402.
  — intrinsic (threshold/excitability) homeostasis, named as #71's next mechanism 1.
- Carandini, M., Heeger, D.J. (2012). Normalization as a canonical neural computation. Nat Rev Neurosci 13:51-62. —
  divisive normalization / population-gain set-point (the mapped next mechanism).
- Marr 1971; O'Reilly & McClelland 1994; Leutgeb 2007; Bakker 2008 — DG sparse-expansive separation as a k-of-N
  competition (the target the homeostat cannot supply).

Internal: builds on `2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-single-recall-ceiling-kWTA-stability-residual.md`
(the #71 residual, reproduced here as the LESION arm) and the per-region-homeostasis precedent
`2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md` (where the SAME mechanism, used to RAISE excitability
of a driven region, works — the sign that fails here).
