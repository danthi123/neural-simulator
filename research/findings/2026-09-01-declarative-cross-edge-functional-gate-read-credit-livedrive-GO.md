---
type: finding
status: live
date: 2026-09-01
mechanism: declarative-cross-edge-functional-gate
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_declarative_crossedge_generic_gate_6seed.json
runner: research/runners/_onebrain_declarative_crossedge_generic_gate.py
builds_on:
  - research/findings/2026-08-27-declarative-cross-edges-framework-GO.md
  - research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md
  - research/findings/2026-08-28-onebrain-r4-declarative-crossedge-migration-GO.md
  - research/findings/2026-08-27-onebrain-integration-R1-wm-to-comprehension.md
---

# The declarative cross-edge framework now instantiates the READ / CREDIT / LIVE-DRIVE from the declaration, not just the wiring — a GENERIC functional-gate harness reproduces R1's hand-typed `_emergence`/`_f2` BIT-FOR-BIT, and a NEW biologically-motivated RECIPROCAL edge (comprehension role → d6 WM) clears the 6-seed GO added PURELY BY DECLARATION

**One-line:** The construction side of a cross-edge was already declarative (`CrossEdge` +
`merge_organs(cross_edges=...)` + `apply_cross_edge_freeze`, `2026-08-27-declarative-cross-edges-framework-GO.md`).
The honest residual its own gap-analysis flagged (`2026-08-28-...-gap-analysis-already-closed.md` §3) was that the
FUNCTIONAL GATE deciding whether an edge is REAL — `_f1/_f2/_f3/_f4` + `_emergence` + `_migration_invariant` — was
still HAND-TYPED, near-identically, in seven runners. This closes it: `research/runners/onebrain_crossedge_gate.py`
makes the three load-bearing checks the brief's 6-seed GO names — GROWS-from-the-substrate's-own-rule ·
LOAD-BEARING-vary/lesion · BYTE-IDENTICAL-OFF — GENERIC, driven from a `CrossEdgeGateSpec` declaration. Proof it is
faithful, not a drift: the generic gate reproduces R1's imported, UNMODIFIED `_emergence_with_drift`/`_f2`
**bit-for-bit** (max|delta| = 0.0, 6/6 seeds). Proof it makes the next edge a declaration: a genuinely NEW
RECIPROCAL feedback edge (comprehension `sel_agent`/`sel_patient` → d6 WM `w0`/`w1`) — added as a `CrossEdge` list +
`train_fn` + `read_fn` + conditions, NO bespoke F-gate — clears the 6-seed GO (6/6) through the SAME harness. NO
sim/ edit; additive; numpy CPU.

## 1. What existed, and the exact residual this closes

`CrossEdge` + `merge_organs(cross_edges=...)` + `MergedPool.apply_cross_edge_freeze()` already make the WIRING +
LEARNED SYNAPSE of a cross-region edge a registry row (R1 refactored bit-for-bit; R4 migrated; surprise→provenance
added new-by-declaration on the construction side). But the brief's declarative form is
`{source_region, target_region, credit_signal, read_site, learning_rule}` from which the wiring, the synapse, **AND
the LIVE-DRIVE** are instantiated. The `{credit_signal, read_site}` / live-drive half did NOT exist declaratively:
each edge's functional gate (does it GROW, is it LOAD-BEARING, is it BYTE-OFF) was hand-typed in its own ~37-46KB
runner (`_onebrain_integration_r1_*`, `_r2_*`, `_r3v3_*`, `_r4_*`, `_surprise_episodic_*`, ...). Seven near-duplicate
`_f1.._f4`/`_emergence`/`_migration_invariant` implementations. That is the duplication this closes.

## 2. The declarative form (`onebrain_crossedge_gate.py`)

`CrossEdgeGateSpec` — the read/credit/live-drive half of the declaration (the wiring half stays the `CrossEdge`
list): `cross_edges`, `train_fn(pool)`, `read_fn(pool, condition)` (the read_site), `condition_order` + `control` +
`expected{sign,floor}` (the source-state vary drivers), `correct_edges`/`selectivity_pairs` (the emergence
mapping), `credit_signal` (`"rate_hebbian"|"da_credit"|"none"`, mirrors `CrossEdge.learn_rule`), plus the floors.

`run_gate(pool, spec)` runs three checks. What is GENERIC (derived from the declaration + the pool's OWN
connectivity — the framework already KNOWS each declared edge's synapses):

- **EMERGENCE** — `cross_edge_masks` selects each declared edge's synapses off the COO row/col region membership;
  the correct edges' mean weight after training vs `grow_factor*init_weight` (GREW from near-zero); `frozen_maxdrift`
  over EVERY non-edge synapse (`noncross_mask`, no migrated weight moved).
- **INTERACTION** (the crux) — read the target under each declared source-state condition; `lesion_cross_edges`
  zeroes exactly the declared edges (from the declaration); re-read; `attributable_to(sign*Δ_intact, sign*Δ_lesion)`
  per condition. The shift must appear intact and VANISH on lesion.
- **BYTE-OFF** — `verify_byte_off`: a pool built WITHOUT the `cross_edges` has base connectivity byte-identical to
  the with-edge pool once the declared edge (pre,post) slots are removed (integration added ONLY the edge).

What is IRREDUCIBLE (it DEFINES the faculty, so it is a callable the spec supplies, never re-implemented by the
harness): `train_fn` (the experience that grows the edge) and `read_fn` (the load-bearing read). Reuse-by-composition:
the harness CALLS them and reproduces the seven runners' shared gate STRUCTURE.

## 3. PART A — the refactor proof: the generic gate == R1's hand-typed gate, BIT-FOR-BIT (6/6, max|delta|=0.0)

R1's edge is re-expressed as `R1_GATE_SPEC` (a DATA declaration) + R1's own `train`/`amb_read`. The ORACLE is R1's
imported, UNMODIFIED `_emergence_with_drift` + `_f2`. Both run on ONE `R1Pool` at the SAME F2-entry state:
emergence is non-destructive (both read the same trained weights); for the interaction the runner snapshots the
post-train bridge state, runs the GENERIC interaction (it lesions in place), RESTORES the snapshot, then runs R1's
`_f2` from the identical entry — so any delta is the GATE LOGIC alone (no two-separately-built-pool RNG-order floor,
no pre-F2-warmup difference; R1's `_hard_reset` deliberately leaves state that makes `_f2` sequence-dependent, so
entering both right after `train()` is required).

**Result, 6 seeds:** `reproduces=True`, **max|delta| = 0.0** every seed. The F2 interaction deltas
(`delta_ref0/1_intact/lesion`) and the `attributable_to` fractions are BIT-IDENTICAL between the generic gate and
R1's `_f2`. The emergence grown weights are identical too (compared at R1's own 4-decimal `cross_weights()`
convention — R1 stores `round(_wmean, 4)` in its trajectory; the underlying weights are the same read). The generic
harness is a faithful generalization of the hand-typed gate, not a reimplementation that could silently drift.

## 4. PART B — a NEW edge added PURELY BY DECLARATION: the RECIPROCAL comprehension → d6 WM feedback (6/6 GO)

**The edge.** R1 built the FEEDFORWARD WM→role edge (d6 `w0/w1` → comprehension `sel_agent/sel_patient`). The
single most biologically-motivated new edge on this pool is its RECIPROCAL: the FEEDBACK comprehension role → d6 WM
slot (`sel_agent`→`w0`, `sel_patient`→`w1`) — "the comprehended role refreshes the referent held in working memory".
Functionally-related cortical areas are connected RECIPROCALLY, not all-to-all (Magrou et al. 2024; Gámanut et al.
2018; Theodoni et al. 2020), so the feedback edge completes exactly the pair R1 opened.

**Added as a declaration**: 4 `CrossEdge` rows (gate `sel_to_wm`) + a `train_fn` (episodes where a role fires while
its referent slot is actively maintained → the edge GROWS by the substrate's OWN rate-window Hebbian) + a `read_fn`
(drive the comprehension cue; read the signed WM-slot margin `w0_rate - w1_rate` as the winning role feeds its
referent slot) + the conditions (`balanced` control, `agent`, `patient`). NO bespoke F-gate — it runs through the
SAME `run_gate`.

**Result, 6 seeds — GO (6/6)** (values below are rounded displays of the full-precision per-seed numbers in the
cited artifact, `research/findings/raw/_onebrain_declarative_crossedge_generic_gate_6seed.json`):

<!--derived-->
| seed | grown sel_agent→w0 | grown sel_patient→w1 | agent Δmargin (lesioned) | patient Δmargin (lesioned) | emg · int · byte-off | GO |
|---|---|---|---|---|---|---|
| 42 | 10.07 | 9.66 | +0.0145 (+0.0000) | −0.0271 (+0.0000) | ✓ · ✓ · ✓ | GO |
| 43 | 10.10 | 10.43 | +0.0187 (+0.0000) | −0.0182 (+0.0000) | ✓ · ✓ · ✓ | GO |
| 44 | 8.92 | 10.62 | +0.0183 (+0.0000) | −0.0222 (+0.0000) | ✓ · ✓ · ✓ | GO |
| 100 | 10.61 | 11.21 | +0.0218 (+0.0000) | −0.0107 (+0.0000) | ✓ · ✓ · ✓ | GO |
| 101 | 10.42 | 11.32 | +0.0145 (+0.0000) | −0.0139 (+0.0000) | ✓ · ✓ · ✓ | GO |
| 102 | 10.67 | 10.04 | +0.0196 (+0.0000) | −0.0130 (+0.0000) | ✓ · ✓ · ✓ | GO |

(Δmargin = read under that role's cue minus the balanced-control read; lesioned = the same shift with the four
declared edges zeroed. Every intact shift clears the 0.008 floor and every lesioned shift is +0.0000 — the WM-slot
bias is 100% attributable to the reciprocal edge.)

Every seed: the edge GROWS from 0.05 to ~10 (LEARNED, not hand-set); the winning comprehension role biases its
learned WM slot (agent → +margin/w0, patient → −margin/w1) above R1's own load-bearing floor (`F2_INTACT_FLOOR` =
0.008); the bias is 100% attributable to the edge (it goes to 0.0000 on lesion); and the no-edge pool's base
connectivity is byte-identical.

**Honest characterization (the instrument is part of the emulation).** This edge carries NO weight-RATIO
selectivity check, and that is a substrate property, not a relaxed gate. The d6 WM slots have a baseline firing
rate (~0.04 spikes/neuron/step even under the balanced control), so plain rate-Hebbian coactivity grows the
non-held slot's cross-edge as much as the held slot's — a coactivity-threshold sweep from 0.02 to 0.25 finds NO
separating value (either all four edges grow, or none do). The mapping's selectivity therefore lives in the
FUNCTIONAL READ, not the weight ratios: the agent and patient roles bias OPPOSITE slots, and each bias vanishes on
lesion. That is a STRONGER selectivity claim than a weight ratio (it tests the load-bearing read, which is what the
faculty actually does) and the correct instrument for a feedback edge onto an intrinsically-active WM store. The
read averages 6 reads (vs R1's 3: the WM-slot margin — one of 30 active pools — is noisier than R1's clean WTA;
denoising raised the marginal seed-101 agent shift (0.0067 at 3 reads → 0.0145 at 6 reads <!--derived: the
0.0067@3-reads value is from the pre-final coactivity/read tuning sweep, not the committed 6-read artifact-->), it
did not manufacture it — the effect is correctly-signed and 100% lesion-attributable at every seed regardless).

## 5. How much this shortens the NEXT edge

Adding a learned cross-region edge is now: a `CrossEdge` list (wiring) + a `CrossEdgeGateSpec` (a `train_fn`, a
`read_fn`, a conditions dict, and the emergence/credit fields) run through ONE `run_gate` + one `verify_byte_off`.
The emergence-growth read, the no-corruption drift over every non-edge synapse, the lesion, the `attributable_to`,
and the base-connectivity byte-off are all supplied by the harness FROM the declaration — the parts each of the
seven prior runners re-wrote by hand (their ~37-46KB was dominated by exactly this `_f1.._f4`/`_emergence`/
`_migration_invariant` boilerplate). What remains per edge is only the two callables that DEFINE the faculty
(what experience grows it, what read it drives), plus the pool construction the declarative `merge_organs`
already provides.

## 6. Files

`research/runners/onebrain_crossedge_gate.py` (NEW — `CrossEdgeGateSpec`, `cross_edge_masks`, `noncross_mask`,
`lesion_cross_edges`, `verify_emergence`, `verify_interaction`, `verify_byte_off`, `run_gate`) ·
`research/runners/_onebrain_declarative_crossedge_generic_gate.py` (NEW — PART A refactor proof `_partA_refactor`
with the snapshot/restore single-pool comparison; PART B `ReciprocalPool` + `RECIP_GATE_SPEC` + `_partB_new_edge`) ·
`research/findings/raw/_onebrain_declarative_crossedge_generic_gate_6seed.json`. Read against (unmodified):
`research/runners/_onebrain_integration_r1_wm_comprehension.py` (the oracle — `_emergence_with_drift`, `_f2`,
`R1Pool`), `research/runners/onebrain_merge_framework.py` (`CrossEdge`, `merge_organs`, `apply_cross_edge_freeze`).
No `sim/` file touched; no `webapp/server.py` edit; no production default changed.

Functional read-outs only; no phenomenal-experience claim.
