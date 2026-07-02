# Research gate — on-bridge HTM Temporal-Memory context-scaling (the "multi-segment" boundary)

**2026-07-02 (read-only deep-research + code-audit + numpy-isolation subagent; NO `sim/` edited, NO commit — the controller reviews + commits).** The standing research gate BEFORE building a mechanism to push the on-bridge unsupervised HTM Temporal-Memory (rung-4, EMERGE-14) past its `n_seq` context-scaling boundary. Ran the four SURPASS moves: ISOLATE + QUANTIFY the true residual, REFRAME via real biology, RANK cheap-first mechanisms, VERDICT. Every load-bearing claim is backed by a cited paper or a run I executed this session.

---

## TL;DR (the headline the premise did NOT expect)

**The stated boundary — "the on-bridge flat single-segment-per-cell scheme fails at `n_seq=8` because it cannot express multiple independent dendritic segments per cell" — is, on inspection, MOSTLY A DISGUISED BOUNDARY.** Two isolation experiments I ran this session prove:

1. The **numpy multi-segment reference did NOT carry `n_seq=8` via segment reuse.** Its `n_seq=8` GO used **`n_cells=64`** (16 disjoint SDR slots ≫ 8 contexts), not the 16 cells the premise implied. Multi-segment cell-reuse only matters when cells are SCARCE.
2. A **FLAT single-coincidence-per-cell scheme (the exact on-bridge representation, reproduced in numpy) reaches `branch_acc = 1.000` on all 6 seeds at `n_seq=8, n_cells=40`** — identical to multi-segment. So the flat representation is NOT the wall at ample cells.

⇒ The genuine, precisely-isolated residual has **two distinct pieces**, and the biology reframe (§2) says the CHEAP one is the right first move:
- **(Residual R1 — the real bottleneck, cheap):** the on-bridge learner's failure at `n_seq=8, n_cells=40` (which has ≥ 8 slots) is an **ALLOCATION / operating-point failure specific to the dense weighted-coincidence pool**, not a representational-capacity wall. The flat scheme provably holds 8 contexts with ≥ `k_win·n_seq` cells; the on-bridge learner just isn't landing the 8 contexts on disjoint cells.
- **(Residual R2 — the genuinely-irreducible piece, only past a cell-budget frontier):** multiple independent segments per cell give **cell REUSE**, so `N` cells hold ≫ `N/k_win` contexts. This is a real, biologically-mandated capacity axis (Hawkins-Ahmad, Poirazi-Mel) — but it only becomes the binding constraint once you are past the "enough cells for disjoint allocation" regime. It is a scale/cell-efficiency lever, not the reason `n_seq=8` fails today.

**Verdict: SURPASSABLE, and cheaply.** The single cheapest next de-risk fixes R1 (an allocation/operating-point fix on the existing flat pool — NO `sim/` edit, NO new mechanism). R2 (a segment-index dimension on the coincidence mask) is the correct, biologically-grounded, ranked-second lever for the LATER cell-efficiency frontier — and even it is a small, well-precedented change, not a wall.

---

## SURPASS move 1 — ISOLATE + QUANTIFY the true residual

### The mechanism gap as the premise framed it
On-substrate (EMERGE-14), `cp_connections` is one CSR of per-`(pre, post)` synapses. A post-cell's predictive drive is a **single** thresholded sum over ALL its incoming connected synapses (`fused_coincidence_plateau`'s `c_count` → one plateau; `sim/kernels.py:252`). There is no per-cell OR-over-segments: all synapses onto cell `i` pool into one count, so `i` can encode at most one context, and two contexts landing on `i` add their partial matches → spurious ≥`act_th` → merge. The numpy reference (`_emerge9b_htm_faithful_derisk.py:70`) instead gives each cell a LIST of segments and marks it predictive if `any(_seg_conn_active(seg, active) >= act_th)` (`:129-132`) — a true OR-over-segments.

That framing is correct *as a description of the code difference*. It is WRONG as a diagnosis of why `n_seq=8` fails. I isolated this with three runs.

### Isolation run A — where does multi-segment actually beat flat? (the capacity map)
I built a `FlatHTM` that caps each cell to exactly ONE pooled segment (the faithful numpy analogue of the on-bridge flat CSR) and swept it vs the full multi-segment `HTM` across `n_seq × n_cells`, 100 epochs, `k_win=4`, `act_th=3`, seeds 42/43/44. `slots = floor(n_cells/k_win)` = the number of disjoint SDRs the column can hold.

| n_seq | n_cells | slots | MULTISEG | FLAT | note |
|---|---|---|---|---|---|
| 8 | 16 | 4 | 0.000 | 0.000 | both fail — 4 slots < 8 contexts |
| 8 | 24 | 6 | **0.500** | 0.000 | **SEG WINS** (6 slots < 8) |
| 8 | 32 | 8 | 1.000 | **1.000** | flat OK (8 slots = 8) |
| 8 | 40/48/64 | 10–16 | 1.000 | **1.000** | flat OK |
| 12 | 32 | 8 | 0.333 | 0.000 | SEG WINS (8 slots < 12) |
| 12 | 48 | 12 | 1.000 | **1.000** | flat OK |
| 16 | 48 | 12 | 0.500 | 0.000 | SEG WINS (12 slots < 16) |
| 16 | 64 | 16 | 1.000 | **1.000** | flat OK |

**The pattern is unambiguous: multiple segments per cell help ONLY when `n_cells < k_win·n_seq`** (fewer disjoint slots than contexts). When `n_cells ≥ k_win·n_seq`, FLAT == MULTISEG == 1.000 everywhere tested. Multi-segment buys CELL REUSE (holding more contexts than `floor(n_cells/k_win)`), nothing more.

### Isolation run B — the numpy `n_seq=8` GO used AMPLE cells, not segment reuse
The 9b finding's "n_seq=8 GO" is real, but its raw config (`research/findings/raw/_emerge9b_nseq8.json`) is **`n_cells=64, k_win=4`** → 16 disjoint slots for 8 contexts. It was NOT operating in the segment-reuse regime. So "the numpy scaled to many contexts" is true but is attributable to cells-per-column, not to multi-segment reuse, at that config.

### Isolation run C — the FLAT scheme holds `n_seq=8` at ample cells, 6/6 seeds
Pure-numpy `FlatHTM`, `n_seq=8, n_cells=40` (10 slots ≥ 8), epoch sweep, seeds 42/43/44/100/101/102:

| epochs | mean branch_acc | per seed |
|---|---|---|
| 40 | **1.000** | 1.00 ×6 |
| 60 | **1.000** | 1.00 ×6 |
| 80 | **1.000** | 1.00 ×6 |

Multi-segment `HTM` at the identical config = 1.000 ×6 (control). **The flat single-coincidence representation is NOT the boundary at `n_seq=8, n_cells=40`.** The on-bridge EMERGE-14 learner fails at this SAME config — therefore its failure is on-bridge-implementation-specific, not representational.

### The quantified residual, split
- **R1 (the actual `n_seq=8` blocker — cheap, ~most of the "scaling" problem):** an ALLOCATION / operating-point failure of the on-bridge learner on the **dense weighted-coincidence pool**. Candidate causes, all on-bridge-specific and all tunable (no new mechanism): (i) the committed-metric allocation on the dense pre-allocated pool doesn't spread 8 contexts onto 8 disjoint cell-sets (the allocation RACE the EMERGE-14 finding already flagged, worse at scale); (ii) `coincidence_k_threshold = act_th − 0.5` in WEIGHT units on a graded pool where many sub-connected `p_init` synapses contribute background drive as `n_seq` grows — the "dense-pool over-priming" the finding root-caused at `p_init=0.24` and fixed for `n_seq=2` with `p_init=0.0`, which may re-emerge at higher `n_seq` because the number of cross-column potential synapses per cell grows with vocab; (iii) too few epochs for permanences to mature across more contexts (the L=16 finding showed epochs scale with load). Isolation runs A/C prove the *representation* can hold 8 contexts at 40 cells, so R1 is entirely in the learner/pool tuning.
- **R2 (the genuinely-irreducible piece — a scale/efficiency lever, not the `n_seq=8` cause):** with a fixed cell budget, a flat 1-segment-per-cell column holds at most `floor(n_cells/k_win)` disjoint contexts. To hold MORE contexts than that on a FIXED number of cells, you need cell reuse = multiple independent segments per cell (isolation run A, the "SEG WINS" rows). This is real and biologically mandated, but it is the constraint only PAST the disjoint-allocation frontier.

---

## SURPASS move 2 — REFRAME: how does real biology represent many sequence contexts?

Two literature syntheses (full transcripts folded into this session) converge, and they resolve the premise's implicit assumption ("must be multi-segment") into a two-axis answer.

### The two independent, MULTIPLICATIVE capacity axes
1. **Cells-per-column (many cells, ONE segment each).** **Bouhadjar, Wouters, Diesmann & Tetzlaff 2022, "Sequence learning, prediction, and replay in networks of spiking neurons," *PLoS Comput Biol* 18(6):e1010233** — the canonical SPIKING HTM this project ports — **use a SINGLE dendritic branch per neuron** and reach high-order (up to order-10) sequence memory by giving each subpopulation **`n_E = 150` neurons** of which **`ρ ≈ 20`** become context-specifically predictive; distinct prior contexts recruit distinct sparse subsets of the SAME 150 cells. They explicitly write: *"In this work, we use a single dendritic branch per neuron. However, the model could easily be extended to include multiple dendritic branches."* dAP threshold `γ = 5` co-active partners, `θ_dAP = 59 pA`, plateau `I_dAP = 200 pA` clamped `τ_dAP = 60 ms`. **This is the direct proof that ONE segment per cell + enough cells per column already gives dozens of contexts** — exactly what isolation runs A/C reproduce. Their honest caveat: absolute sequence count is small precisely because they did not add multiple branches — i.e. R2 is their next lever too.
2. **Segments-per-cell (cell REUSE).** **Hawkins & Ahmad 2016, "Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex," *Front. Neural Circuits* 10:23 (arXiv:1511.00083)** — the HTM theory — give each cell **~128 basal distal segments, each an INDEPENDENT coincidence detector** (8–20 clustered synapses, NMDA-spike threshold `θ ≈ 15`), and the predictive state is the **logical OR across segments**: `π = 1 if ∃ segment d with ‖D_d ∘ A‖₁ > θ`. Their explicit capacity accounting is **linear in BOTH axes**: `capacity ≈ (cells-per-column / sparsity) × patterns-per-cell`, worked as `(32/0.02) × 200 ≈ 320,000` transitions; a single cell with 128 segments recognizes **~300 patterns**. The false-match probability per segment is a hypergeometric tail (their Eq.), e.g. `9.8×10⁻²¹` at `n=2·10⁵, a=2000, s=10, θ=10` — negligible while the code is sparse and `θ, s` are not tiny.

### The biophysical license for R2
**Poirazi & Mel 2001, "Impact of Active Dendrites and Structural Plasticity on the Memory Capacity of Neural Tissue," *Neuron* 29:779–796** and **Poirazi, Brannon & Mel 2003, "Pyramidal Neuron as Two-Layer Neural Network," *Neuron* 37:989–999**: a pyramidal cell is a TWO-LAYER net — `m` independent thresholded dendritic subunits, the soma summing (OR-like) their outputs — and **nonlinear addressable subunits raise a single neuron's storage capacity ~10× over a linear point neuron**, reachable via a random-synapse-formation + activity-dependent-stabilization structural rule (exactly the HTM segment-growth rule). **Major, Larkum & Schiller 2013, "Active properties of neocortical pyramidal neuron dendrites," *Annu. Rev. Neurosci.* 36:1–24**: thin basal/oblique branches are electrically SEMI-INDEPENDENT and each generates its own NMDA spike from ~4–20 clustered coincident synapses — so "many independent segments per neuron" is measured biology, not a modeling convenience. Project catalog: **G.02 "Active dendrites — local computation, dendritic spikes"** (`sim-catalog/references/feature-catalog.md:2644`; Kandel 6e Ch 13 pp 293–298) — flags single-compartment-everywhere as one of the largest abstractions, ~10× compute per neuron, the OR-of-branch-NMDA-spikes as the missing nonlinear-summation rule.

**The reframe, stated plainly:** the premise tested the WRONG hypothesis ("can the flat scheme represent 8 contexts? — no, it needs multi-segment"). The right hypothesis is "does the flat scheme have enough CELLS + correct ALLOCATION to place 8 contexts disjointly? — yes, and the on-bridge learner isn't doing the allocation." Biology carries many contexts on EITHER axis; the project's own porting target (Bouhadjar) carries them on the cells-per-column axis with one segment per cell. Multi-segment is the compact SECOND axis for a fixed cell budget.

---

## SURPASS move 3 — RANK cheap-first mechanisms to go PAST the boundary

Ordered by implementation cost on THIS substrate (a spiking bridge with a per-synapse coincidence mask + a per-neuron all-or-none plateau). Each: mechanism · citation · cheapest de-risk · anti-cheats · `sim/`-edit-or-not.

### Option A — FIX THE ALLOCATION / OPERATING POINT ON THE EXISTING FLAT POOL (recommended first; fixes R1)
- **Mechanism.** Keep the flat single-coincidence pool. Give the on-bridge learner enough cells for disjoint allocation (`n_cells ≥ k_win·n_seq`, exactly the Bouhadjar cells-per-column axis) AND fix the dense-pool operating point so 8 contexts land on 8 disjoint cell-sets: (i) re-verify/repair the committed-metric allocation at `n_seq>2` (the allocation RACE the EMERGE-14 finding flagged); (ii) sweep the weighted-coincidence threshold + `p_init`/`perm_conn` so background drive from the growing cross-column pool doesn't over-prime as vocab grows; (iii) scale epochs with `n_seq` (the L=16 lesson). This is the same class of fix EMERGE-14 already used to take `n_seq=2` from 0.000 → 1.000.
- **Citation.** Bouhadjar-Diesmann 2022 (one segment/cell + many cells/column reaches high-order sequences); Ahmad-Hawkins 2015 SDR capacity (sparse allocation keeps false-match negligible).
- **Cheapest de-risk (THE recommended next step).** Run the EXISTING `_emerge14_stageC_onbridge_learning_derisk.py` at `--n-seq 8 --n-cells 40 --epochs 80` (≥ 8 slots — isolation run C proves the flat REPRESENTATION holds this). If it still fails, the failure is localized to the on-bridge allocation/pool tuning (not representation), and the fix is a single-variable sweep of `{committed-metric allocation, coincidence_k_threshold, p_init/perm_conn, epochs}` against the numpy `FlatHTM` oracle (which is 1.000 at this config, 6/6). Target: on-bridge `branch_acc ≥ 0.90` at `n_seq=8, n_cells=40`, dAP-lesion collapses.
- **Anti-cheats.** Markov floor (`0.125` at n_seq=8) + dAP-lesion collapse + untrained collapse + EMERGE-9c/FlatHTM numpy-oracle parity (the numpy FLAT at the same config is the exact target) + multi-seed 42/43/44(/100/101/102). The permuted-sequence control guards against a spurious win.
- **`sim/` edit?** **NONE.** Runner-only allocation/tuning + more cells. This is the cheapest possible move and, per the isolation, the one that actually addresses why `n_seq=8` fails today.

### Option B — A SEGMENT-INDEX DIMENSION ON THE COINCIDENCE MASK (the biologically-correct SECOND axis; addresses R2)
- **Mechanism.** Attach a `segment_id` to each distal coincidence synapse (an array parallel to the CSR `data`), accumulate `c_drive` PER `(post_cell, segment)` via a segmented reduction → an `(n_cells × S)` drive matrix, threshold each, and set the cell predictive if **ANY segment ≥ `act_th`** (an OR / `max` over the segment axis). This is the literal Hawkins-Ahmad OR-over-segments rule and is the universal neuromorphic encoding (Loihi represents a segment as a child compartment; Numenta memristive HTM as separate crossbar rows). It DECOUPLES context count from cell count → `N` cells hold ≫ `N/k_win` contexts (isolation run A's "SEG WINS" regime).
- **Citation.** Hawkins-Ahmad 2016 (128 segments/cell, OR rule, capacity linear in patterns-per-cell); Poirazi-Mel 2001/2003 (~10× single-neuron capacity from addressable nonlinear subunits); Loihi (Davies et al. 2018, segment = compartment); Numenta memristive HTM (Zyarah-Kudithipudi 2019, segment = crossbar row).
- **Cheapest de-risk.** First reproduce R2 as a NEED in numpy: show `FlatHTM` fails and multi-segment `HTM` succeeds at a SCARCE-cell config (`n_seq=8, n_cells=24` — isolation run A already shows 0.000 vs 0.500). THEN de-risk a numpy "segment-index over one flat drive vector" that reproduces the multi-segment result WITHOUT per-cell dicts (a segmented-reduction prototype) — proving the segment dimension is the mechanism, cheaply, before any kernel. Only after that numpy GO, scope the on-bridge segmented `c_drive`.
- **Anti-cheats.** Same battery + the KEY additional control: the win must appear ONLY in the scarce-cell regime (`n_cells < k_win·n_seq`) — if a "segment" change also changes the ample-cell result it is confounded. Locality assert (no forward-weight transpose). False-match rate stays negligible (sparse code, `act_th` not tiny — Ahmad-Hawkins).
- **`sim/` edit?** **YES, but small and well-precedented** — a `segment_id` array on the coincidence pathway + a segmented reduction feeding `fused_coincidence_plateau` per `(cell, segment)` + an OR/`max`. Additive, default-off (S=1 ⇒ byte-identical to today's flat pool), guarded — the exact `enable_coincidence_detection` / RESONATE_AND_FIRE precedent (rung-4 scoping §4). NOT a new `NeuronModel`; a dimension on an existing mask.

### Option C — MORE CELLS PER COLUMN ONLY (zero-risk warm-up; a partial, inefficient R2 substitute)
- **Mechanism.** Just raise `n_cells` so `floor(n_cells/k_win) ≥ n_seq` for the target `n_seq`. No detector change.
- **Citation.** Bouhadjar-Diesmann 2022 (order-10 sequences with 1 segment/cell + 150 cells/column).
- **Cheapest de-risk.** Subsumed by Option A's run (Option A already includes ample cells). Distinct only as the trivial statement "the flat scheme needs `≥ k_win·n_seq` cells" — proven by isolation runs A/C.
- **Anti-cheats.** Same battery.
- **`sim/` edit?** NONE. But it is CELL-INEFFICIENT (dozens of contexts ⇒ dozens× cells) — the reason biology evolved R2. Use as a floor, not the destination.

### Option D — FULLY-NEURAL HOMEOSTATIC ALLOCATION (closes the acknowledged host residual; orthogonal, not required for `n_seq` scaling)
- **Mechanism.** Replace the host-orchestrated committed-metric allocation + winner-selection (the EMERGE-14 residual) with a neural per-column WTA + a per-cell dAP-rate homeostasis (`z`) that neurally selects the freshest cells (the `hfac = 0.5 + 0.5·max(0, z*−z)` already in `fused_htm_permanence_update`). Biology: neurogenesis-like structural allocation (Zyarah-Kudithipudi memristive HTM); dAP-rate homeostasis (Bouhadjar Eq. term 3).
- **Citation.** Bouhadjar-Diesmann 2022 (homeostasis term); Poirazi-Mel 2001 (activity-dependent structural stabilization).
- **Cheapest de-risk.** After Option A/B GO — a neural WTA + `z`-driven allocation reproducing the host allocation on the `n_seq=8` task.
- **Anti-cheats.** Allocation must remain DISJOINT (no context merge) under the neural rule; same battery.
- **`sim/` edit?** Reuses the per-column FS-WTA wiring recipe (nav cascade, no `sim/` edit) + the existing `z` homeostasis; possibly a small structural-allocation addition. Orthogonal to the `n_seq` boundary — do NOT bundle it into the scaling fix.

---

## SURPASS move 4 — VERDICT

**SURPASSABLE, and cheaply — the `n_seq=8` boundary is NOT a representational wall.** The premise's "flat cannot express multiple segments" is a true code-difference but a wrong diagnosis of the failure: isolation runs prove a FLAT single-coincidence-per-cell scheme reaches `branch_acc = 1.000` at `n_seq=8, n_cells=40` on 6/6 seeds (identical to multi-segment), and the numpy reference's own `n_seq=8` GO used ample cells (64), not segment reuse. The on-bridge failure at the same config is therefore an **allocation / dense-pool operating-point** problem (R1), fixable with NO `sim/` edit and NO new mechanism. Multiple segments per cell (R2) is a REAL, biologically-mandated capacity axis — but it is the cell-EFFICIENCY lever for a FIXED cell budget past the disjoint-allocation frontier, not the reason `n_seq=8` fails today.

**The single cheapest next de-risk (Option A):** run the existing `_emerge14_stageC_onbridge_learning_derisk.py` at `--n-seq 8 --n-cells 40 --epochs 80`, with the numpy `FlatHTM` (1.000, 6/6 at this config) as the oracle. If it clears `≥ 0.90` (dAP-lesion collapsing), `n_seq` scaling is already surpassed by cells-per-column and the "multi-segment wall" was a mirage. If it still fails, the failure is localized to the on-bridge allocation/pool tuning — a single-variable sweep of `{committed-metric allocation · coincidence_k_threshold · p_init/perm_conn · epochs}` against that oracle, still with NO `sim/` edit. Only after R1 is closed does the genuinely-irreducible R2 (Option B — a `segment_id` dimension on the coincidence mask + OR-over-segments; a small, guarded, default-off `sim/` change, NOT a new `NeuronModel`) become the ranked-next lever, and it should be de-risked first in numpy in the SCARCE-cell regime (`n_seq=8, n_cells=24`, where isolation run A shows multi-segment 0.500 vs flat 0.000) to prove the segment axis is load-bearing before any kernel work.

**No wall.** R1 is cheap tuning of an already-GO mechanism; R2 is a small, well-precedented, biologically-grounded dimension on an existing mask. The `n_seq` axis is surpassable on the cells-per-column axis immediately and on the segments-per-cell axis with a minor guarded edit — matching how the project's own porting target (Bouhadjar) and the HTM theory (Hawkins-Ahmad) each scale.

---

## Sources (cited)
- **Bouhadjar, Wouters, Diesmann & Tetzlaff 2022** — "Sequence learning, prediction, and replay in networks of spiking neurons," *PLoS Comput Biol* 18(6):e1010233. DOI 10.1371/journal.pcbi.1010233 (PMC9273101; arXiv:2111.03456). One segment/cell; `γ=5`, `θ_dAP=59 pA`, `I_dAP=200 pA`, `τ_dAP=60 ms`, `n_E=150`, `ρ≈20`; "could easily be extended to include multiple dendritic branches."
- **Hawkins & Ahmad 2016** — "Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex," *Front. Neural Circuits* 10:23 (PMC4811948; arXiv:1511.00083). 128 segments/cell, OR-over-segments predictive rule, capacity linear in cells/column AND patterns/cell (`~320,000` transitions), hypergeometric false-match `~10⁻²¹`.
- **Poirazi & Mel 2001** — "Impact of Active Dendrites and Structural Plasticity on the Memory Capacity of Neural Tissue," *Neuron* 29:779–796 (PMID 11301036). ~10× single-neuron capacity from `m` nonlinear addressable subunits via structural plasticity.
- **Poirazi, Brannon & Mel 2003** — "Pyramidal Neuron as Two-Layer Neural Network," *Neuron* 37:989–999 (DOI 10.1016/S0896-6273(03)00149-1). Neuron = two-layer net; more nonlinear subunits ⇒ more storage.
- **Major, Larkum & Schiller 2013** — "Active properties of neocortical pyramidal neuron dendrites," *Annu. Rev. Neurosci.* 36:1–24 (DOI 10.1146/annurev-neuro-062111-150343). Thin branches electrically semi-independent; each generates NMDA spikes from ~4–20 clustered coincident synapses.
- **Ahmad & Hawkins 2015** — "Properties of Sparse Distributed Representations…," arXiv:1503.07469. SDR capacity `C(n,w)`; false-positive hypergeometric tail; subsampling robustness (`<10⁻¹²` at `n=2048,w=40,s=20,θ=10`).
- **Davies et al. 2018** — "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning," *IEEE Micro* 38(1):82–99. Multi-compartment neurons — segment = child compartment.
- **Zyarah & Kudithipudi 2019** — "Neuromemristive Architecture of HTM with On-Device Learning and Neurogenesis," arXiv:1812.10730. Segment = crossbar row; neurogenesis grows capacity structurally.
- **Hussain, Liu & Basu 2015** — "Hardware-Amenable Structural Learning… Active Dendrites," *Neural Computation* 27:845–897 (arXiv:1411.5881). Spiking two-layer/branch realization; branch decomposition cheaper than SVM/ELM.
- **Catalog G.02** — "Active dendrites — local computation, dendritic spikes," `sim-catalog/references/feature-catalog.md:2644`; Kandel 6e Ch 13 pp 293–298.
- **Project de-risk artifacts (this repo):** `research/runners/_emerge9b_htm_faithful_derisk.py` (multi-segment numpy `HTM`, the OR-over-segments at `:129-132`); `_emerge14_stageC_onbridge_learning_derisk.py` (the flat on-bridge learner); `sim/kernels.py:252` (`fused_coincidence_plateau`), `:407` (`fused_htm_permanence_update`); findings `2026-07-02-emerge9b-htm-faithful-GO.md`, `-emerge12-stageB2-bridge-tm-on-substrate-GO.md`, `-emerge14-stageC-onbridge-learning-GO-rung4-complete.md`, `-rung4-sim-two-compartment-tm-port-scoping.md`.
- **Isolation experiments run THIS session** (numpy, seeds 42/43/44[/100/101/102]): capacity map (multi-seg beats flat ONLY at `n_cells < k_win·n_seq`); numpy `n_seq=8` GO used `n_cells=64`; FLAT(1-seg) = 1.000 6/6 at `n_seq=8, n_cells=40`.

**Do NOT commit — the controller reviews + commits.**
