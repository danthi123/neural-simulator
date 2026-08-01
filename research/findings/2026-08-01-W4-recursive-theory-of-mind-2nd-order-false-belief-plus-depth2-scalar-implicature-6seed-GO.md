---
type: finding
status: contributing
date: 2026-08-01
mechanism: recursive-theory-of-mind
artifacts:
  - research/findings/raw/lanes/recursive_tom_6seed_aggregate.json
---

# W4 (lane C): RECURSIVE theory of mind — 2nd-order false belief on a WM-buffer STACK + depth-2 scalar implicature (RSA) — 6/6 GO on the spiking substrate

**One-line verdict:** the recursion rung above W3 (agent-keyed false belief, GO) and W5 (affective ToM, GO)
resolves to **GO on all 6 seeds {42,43,44,100,101,102}**, computed on the spiking `SimulationBridge`
(reuse-by-import, NO `sim/` edit). Two faculties, both with their moat intact: (A) a **2nd-order false belief**
(Perner–Wimmer ice-cream-van) realized as a STACK of W3 belief stores chained by `sim/`'s own
`transmission_gate` witnessing — frame_2 dissociates from BOTH reality AND frame_1, so a 1st-order reader FAILS;
(B) **scalar implicature to depth 2** (Frank–Goodman RSA L0→S1→L1) emerging from the substrate's FS **divisive
normalization** (Carandini–Heeger) — lesion the normalization and the implicature vanishes. Ran concurrently with
the other roadmap lanes (the parallelism directive), on the CPU pool (numpy).

Artifact: `research/findings/raw/lanes/recursive_tom_6seed_aggregate.json` (backend numpy/CPU; per-seed raw under
`research/findings/raw/lanes/rectom/`).

## Result — 6 seeds, both parts GO, `moat_intact=True`

| read-out | value | gate | note |
|---|---|---|---|
| **A** 2nd-order false-belief acc (depth 2) | **1.000** | ≥0.85 | frame_2 predicts M's FALSE model of J |
| A 1st-order baseline (false trials) | 0.000 | ≤0.20 | a 1st-order reader is WRONG → genuinely 2nd-order |
| A reality baseline (false trials) | 0.00694 | ≤0.20 | the world-read is WRONG → not answering reality |
| A 2nd-order TRUE-belief acc | 1.000 | ≥0.85 | frame_2 UPDATES when M witnessed — not "always old" |
| A depth profile (false belief) | 1/2/3 = 1.0/1.0/1.0 | — | holds through the tested nesting depths |
| **B** depth-2 implicature acc (L1) | **1.000** | ≥0.85 | L1("some") ranks SBNA > all |
| B literal L0 acc | 0.000 | ≤0.40 | L0 shows NO implicature → it is DEPTH-created |
| — flatten-lesion (A) | 0.000 | collapse | force all witness gates open → frames mirror reality |
| — buffer-scramble (A) | 0.36111 | collapse | read a random stack frame → chance |
| — permuted-premises (A) | 0.25694 | collapse | shuffle premise tuples → chance |
| — normalization-lesion (B) | 0.000 | collapse | FS inhibition → 0 kills the implicature |
| — permuted-lexicon (B) | 0.29167 | collapse | shuffle the truth matrix → chance |

`overall verdict = GO (6/6)`, `moat_intact = True`. Every anti-cheat separates from the intact read (|sep| ≥ 0.64
on A's controls, 1.0 on the two lesions).

## What the substrate actually does (brain-based, not a symbolic solver)

**Part A — nested belief frames on a WM stack.** W3 gave one agent-keyed belief store ("J believes L"). W4 STACKS
them: frame_d is a W3 belief store (a GNW single-content attractor = a WM-buffer slot) written FROM frame_{d-1},
gated by `transmission_gate = witness_d` ("did the level-d agent observe the level-(d-1) update"). The 2nd-order
ice-cream-van signature (J saw the van move; M did NOT see that J saw) is `witness_1=1, witness_2=0` → frame_2
HOLDS the old placement while frame_1 (J's real belief) and reality are both the new one. The decisive read:
frame_2 dissociates from both, which a 1st-order reader cannot produce (baseline 0.000). The witnessing gate is
`sim/`'s own `transmission_gate`, reused verbatim from W3 — the ToM-specific neural work is the *stack of gated
frames*, not a host computation.

**Part B — implicature from divisive normalization.** Each RSA distribution is the graded firing rates of a
competitive assembly whose shared FS pool performs divisive normalization. At rationality α=1 the depth-2 recursion
L0→S1→L1 is three rounds of proportional normalization — the substrate's operation — and the "some → not all"
implicature is a *consequence*: the single-item state (all|"all") fires harder than each two-item state, and that
informativity gap, propagated through the iteration, yields L1(SBNA|"some") > L1(all|"some"). Lesion the FS
inhibition → rates ride raw truth → the gap vanishes → no implicature (0.000). The literal-truth lexicon is
legitimate linguistic input (as W5's situation→valence appraisal is legitimate world input).

## Honest scope (carried from the runner, not softened)

Both parts are **FUNCTIONAL mentalizing/pragmatics correlates**: a substrate that REPRESENTS and OPERATES ON nested
belief frames + the pragmatic recursion, dissociable from reality and from the lower nesting levels, collapsing
under lesion/scramble/permute. This is **NOT a claim of phenomenal access to another mind** — the honesty boundary
is the deliverable. Plasticity is off (STDP/Hebbian/homeostasis/STP/structural/reward/OU disabled): the belief
stores and RSA normalizer are read at a fixed operating point, exactly as in the W3/W5 GO's — an operating-point
read of the mechanism, not a learned-from-experience result. Unbounded recursion is OPEN (humans plateau at ~2–3
embeddings too); we report the depth profile and do NOT force a GO past where it works — the tested 1/2/3 all hold.

## Runner bugs fixed this cycle (both in the aggregator, caught by verifying the instrument)
1. `build_summary` referenced a non-existent `thr["rsa_collapse_max"]` → the aggregator **crashed after printing
   the verdict but before writing the JSON**. Fixed to the per-seed keys `rsa_lesion_max` (0.40) / `rsa_perm_max`
   (0.65), mirroring the per-seed `goB` logic exactly.
2. The docstring GO-gate line stated `literal_depth0_acc in [0.35, 0.65]` while the code enforces `<= 0.40`.
   Aligned the docstring to the code (the measured 0.000 is the flat-L0 distribution scored by a strict SBNA>all
   readout — the control is satisfied more strongly than the loose band anticipated, not violated).

## Next
The ToM ladder now has W3 (agent-keyed false belief), W5 (affective ToM), and W4 (recursion: 2nd-order + depth-2
pragmatics) all GO as fixed-operating-point functional correlates. The standing residual for the whole cluster is
the same one the honesty boundary names: these are *read at a fixed operating point with plasticity off* — the
learned-from-experience version (the frames + normalizer emerging from a training stream rather than wired) is the
open rung, shared with W3/W5. No capability abandoned; the recursion depth is characterized, not walled.
