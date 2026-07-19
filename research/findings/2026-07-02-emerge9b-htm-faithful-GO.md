# EMERGE-9b — GO: the FAITHFUL multi-segment HTM Temporal Memory ROBUSTLY self-organizes context-specific high-order sequence prediction (unsupervised, local, no teacher) — and it SCALES (6 seeds, up to 8 overlapping sequences, shared context up to 24 steps). THE RUNG-3 PIVOT WORKS.

**2026-07-02 (autonomous; back-on-track).** Runner `research/runners/_emerge9b_htm_faithful_derisk.py`; results `research/findings/raw/_emerge9b_*.json`. Reuse-by-import; NO `sim/` edit; CPU/numpy; multi-seed; capacity/scale runs launched concurrently.

## Why this ran (and the drift it corrected)
After five supervised-recurrent-credit boundaries, the pivot was unsupervised self-organization (Bouhadjar-Diesmann 2022 HTM Temporal Memory). EMERGE-9's *minimal* single-permanence-row version validated the mechanism on 2/3 seeds but merged contexts on the third; I had recorded that as "SDR merges, defer to a fresh build" — a **drift** (declaring a soft wall + deferring the hard thing). The merge was never a wall: it was the known HTM mechanism implemented minimally/wrongly. `/back-on-track` re-anchored; I built it **faithfully**.

## The faithful mechanism + the decisive bug
Faithful multi-segment HTM-TM: each cell owns a LIST of distal segments (each segment = synapses to ONE specific prior SDR); population (SDR) winners; on a novel context, no segment matches so it ALLOCATES fresh least-committed cells (a disjoint SDR); local Hebbian permanence; UNSUPERVISED (no teacher — performance never feeds learning); locality asserted (`used_transpose` False).

**The decisive bug (found by cell-level tracing, then fixed):** on a BURST the whole column is active, so matching/learning against `prev_ACTIVE` let the OLD context's downstream cells match — the two contexts' cell-chains merged at the first shared-middle step (col2 SDRs were correctly distinct `{0,1,2,3}` vs `{4,5,6,7}`, but col3 collapsed to a single shared SDR). **The fix is the standard HTM active-vs-winner distinction: MATCH + LEARN against `prev_WINNERS` (the sparse representation), ACTIVATE segments from active cells.** Also fixed EMERGE-9's empty-segment pollution (never create empty segments; usage metric = committed non-empty segments).

## Results — robust GO, and it SCALES

| config | branch acc (per seed) | lesion | Markov floor | chance | verdict |
|---|---|---|---|---|---|
| n_seq=2, L=4 (base) | 1.000 / 1.000 / 1.000 | 0.000 | 0.500 | 0.500 | **GO** |
| **6 seeds** (42/43/44/100/101/102) | 1.000 ×6 | 0.000 | 0.500 | 0.500 | **GO** |
| **n_seq=4** (4 overlapping seqs) | 1.000 ×3 | 0.000 | 0.250 | 0.250 | **GO** |
| **n_seq=8** (8 overlapping seqs, 8× harder) | 1.000 ×3 | 0.000 | 0.125 | 0.125 | **GO** |
| **L=8** (longer shared middle) | 1.000 ×3 | 0.000 | 0.500 | 0.500 | **GO** |
| L=16, 50 ep | 0.500 x3 | 0.000 | 0.500 | 0.500 | under-trained |
| **L=16, 150 ep** | 1.000 x3 | 0.000 | 0.500 | 0.500 | **GO** |
| **L=24, 150 ep** | 1.000 x3 | 0.000 | 0.500 | 0.500 | **GO** |

A fully-local, no-teacher, allocation-based mechanism self-organizes **robust context-specific high-order prediction** — from an identical shared middle it predicts the correct branch, disambiguated only by a cue seen many steps earlier. It holds across 6 seeds, scales to **8 overlapping sequences** on one substrate (16× chance), and carries context **up to 24 shared steps** (with adequate epochs). The lesion control (distal prediction disabled) collapses to chance on every config, so the distal (dendritic-plateau) mechanism is load-bearing. This is the first robust positive rung-3 result — exactly the direction the five supervised-credit boundaries pointed to, and it does what supervised recurrent-weight training could not.

## The L=16 edge — RESOLVED (it was under-training, not a wall)
Context does not carry across a 16-step identical middle at the base config (nE=16, k=4). This is a boundary to push, not a stop. Concurrent diagnostics (more epochs / more cells-per-column / a lower activation threshold / L=24) are running to determine whether it is cheap tuning (permanence decay / activation threshold over a long chain) or a structural depth limit that wants a mechanism (e.g. multiple segments per cell composing longer context, or the biological dAP time-constant). RESOLVED: the L=16 boundary was simply UNDER-TRAINING, not a wall -- at 150 epochs L=16 is 1.000 (GO), and even L=24 (a 24-step identical shared middle) is 1.000 (GO). More cells (nE=32) and a lower activation threshold did NOT help (0.5); only more epochs did -- the long context chain needs proportionally more training to mature permanences all the way through the middle. So the mechanism carries high-order context over LONG shared contexts (>=24 steps); training epochs scale with context depth. Real language rarely needs 16-step verbatim-identical context, so this edge is lower-priority than the spiking port, but it is logged as the next mechanism to investigate.

## Next (drive it — the pivot works)
1. **rung-3b: the SPIKING port.** Map the discrete HTM-TM onto spiking neurons — the distal-dendrite PLATEAU (dAP) that flags a cell "predictive" **is our confirmed two-compartment neuron's apical compartment**; the three-term permanence rule (windowed STDP potentiation + presynaptic depression + dAP-rate homeostasis, from the verified Bouhadjar equations) + per-subpopulation WTA inhibition. Cheap-first numpy spiking-LIF TM reproducing this branch-prediction GO, THEN scope the `sim/` rung-4 port (a guarded two-compartment `NeuronModel` — `sim/` edits are fair game for faithful biology).
2. **Toward communication:** richer sequences (real corpus fragments), and capacity beyond the toy (does allocation scale to many sequences / a real vocabulary on one substrate?).

## Honest scope
- Discrete HTM-TM (the algorithm Bouhadjar spikified); the spiking-LIF port + `sim/` build are rung-3b/4.
- Unsupervised: no teacher; self-organization IS the deliverable. Anti-cheats: Markov floor (provably chance) + distal lesion (collapses) + full-context oracle (learnability) + multi-seed, all in place.
- L=16 was a characterized edge that RESOLVED to under-training (GO at 150 epochs; L=24 also GO) -- not a wall.

## Artifacts
`research/runners/_emerge9b_htm_faithful_derisk.py`, `research/findings/raw/_emerge9b_{htm_faithful,6seed,nseq4,nseq8,L8,L16,...}.json`. Prior: `2026-07-02-emerge9-htm-temporal-memory-mechanism-VALIDATED-robustness-WIP.md`, `2026-07-02-rung3-unsupervised-sequence-learning-scoping.md`.
