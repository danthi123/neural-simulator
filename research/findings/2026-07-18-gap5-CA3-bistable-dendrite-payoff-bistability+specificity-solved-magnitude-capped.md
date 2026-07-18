# Gap #5 CA3 payoff — the bistable dendrite RESOLVES the bistability + specificity horns (impossible on a point soma); MAGNITUDE at the joint bar is capped by an assembly-level recurrent loop

**2026-07-18.** The completion trilemma (magnitude vs specificity vs bistability) was root-caused to the point soma's
lack of intrinsic bistability. This session BUILT intrinsic dendritic bistability (self-regenerating NMDA plateau + KIR
down-state stabilizer; single-cell latch-and-hold demonstrated + CI), wired it into the CA3 completion network, and ran
the payoff test (FROZEN recall + OU off + the mandatory no-cue + permuted anti-cheats) across ~9 sweeps.

## The advance — two horns the point soma could NEVER reach, now solved on the bistable dendrite

| horn | point-soma state (retracted) | bistable-dendrite state |
|---|---|---|
| **BISTABILITY** (silent rest) | ALWAYS self-sustains (nocue = cue; a strong attractor can't have a silent rest) | **nocue 0.005-0.024** WITH a completing cue — a genuine bistable low state |
| **SPECIFICITY** (cue ≫ perm) | perm ≈ cue (any cue completes; ratio ~0.9-1.0) | **ratio up to 3.36** (high recall_k_thresh: only the strong learned within-assembly coincidence latches; the permuted cue's generic coincidence can't cross) |

These are decisive: a point-neuron recurrent attractor strong enough to complete self-sustains AND completes from any
input; the bistable dendrite gives a silent rest AND cue-specific completion. The magnitude-vs-bistability opposition
that made the strict result impossible on a point soma is broken (sustaining is intrinsic per-cell).

## The residual — MAGNITUDE at the strict joint bar (cue≥0.20 AND cue≥3×perm AND nocue≤0.10)

Best specific+bistable config (structural_sep + high recall_k_thresh, apical_gc=1): **cue 0.156, nocue 0.005, perm
0.081, ratio 1.94** — a genuine held, cue-triggered, silent-at-rest completion, but cue capped ~0.16. The cap is NOT
the within-assembly weights (stronger encoding w 218→511 did NOT lift cue; hebb_lr=5 hurt it) — it is the **READ**:
completion is read from SOMA firing, and lifting the soma read couples back into the network.

**The decisive diagnosis (asymmetric coupling + full isolation experiments):** a strong apical→soma read (for higher
cue) reintroduces self-ignition (nocue 0.19), and **it persists even under FULL bidirectional assembly isolation**
(member↔non-member zeroed). ⇒ the self-sustain is NOT spread to non-members — it is the **ASSEMBLY'S OWN within-member
recurrent loop**: member soma fires → member→member recurrents (the completion path itself) → re-trigger member apical
latches → the mutually-recurrent set of bistable cells self-sustains. **Per-cell intrinsic bistability does NOT
decouple completion from self-sustain at the ASSEMBLY level** — a set of mutually-recurrent bistable cells has its own
network bistability, and the strong read needed for high cue re-closes that loop. (This nuances the research-gate
claim that intrinsic bistability decouples the two: true per-cell, but the assembly re-couples them through mutual
recurrence + the soma read.)

## Next mechanisms (per THE LAW — the residual LAUNCHES the next method, it is not a wall)

1. **A DECOUPLED read-out** — read completion from the held APICAL PLATEAU state (cp_v_apical, intrinsically bistable)
   with a WEAK apical→soma coupling, so the plateau HOLDS (completion) without the soma firing hard enough to drive the
   recurrent loop. Trades the "soma spikes" completion signal for the "cell is in the UP state" signal (a valid
   memory-holding read; biologically the apical UP state IS the held memory, the soma spike is the output). Cheapest.
2. **Sub-critical within-assembly W_rec with a matched lower trigger** — weak recurrents (no self-sustain) + a trigger
   the weak recurrents can still cross; the tension is that a lower trigger also lets the permuted cue in (specificity).
3. **The emergent DG-selected assembly** (the standing follow-on) — replaces the pre-assigned mask; orthogonal to this.

## Honest status
- **CLOSED/MAJOR:** intrinsic dendritic bistability BUILT + single-cell validated + CI (the keystone; resolves the
  point-soma limit, serves gap #4). On CA3: the bistability + specificity horns are SOLVED (impossible on a point soma).
- **NOT closed:** the strict joint bar (cue≥0.20) — capped by the assembly-level recurrent-loop/read tension. A genuine,
  precisely-characterized residual with the decoupled-read-out next mechanism specified. NOT a wall.
- Infra banked (all default-off / byte-identical): the bistability kernel + KIR + asymmetric coupling + structural_sep
  (1=non-member→member, 2=full) + selective_inhib + recall_k_thresh + rate_homeo + plasticity-freeze + enable_ou +
  ca3_density. Runners: `_riii_ca3_synchronous_assembly_derisk.py`, `_gap5_ca3_bistable_*_sweep.py`.
