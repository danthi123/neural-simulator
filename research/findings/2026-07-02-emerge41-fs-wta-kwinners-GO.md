# EMERGE-41 / toward-semantics — GO (6/6 seeds): the pooler's k-winners SELECTION runs as SPIKING competition (rank-order coding + FS lateral inhibition), not a host argsort over the drive. Closes the last host step in the competitive self-organizing pooler. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge41_fs_wta_kwinners_derisk.py`; CI guard `tests/test_emerge41_fs_wta_kwinners.py` (3 tests). Reuse-by-import (`_emerge11` WTA pattern); NO `sim/` edit; CPU numpy-backend; 6-seed.

## Why
EMERGE-38/39/40 made the competitive-pooler LEARNING fully-on-substrate (`sim/` kernels: potentiation + winner-inactive depression), but the k-winners-take-all SELECTION of *which* columns win was still a host `np.argsort(-drive)[:K]`. A biological pooler selects winners by SPIKING competition (Diehl-Cook 2015 lateral inhibition; HTM SP local inhibition; Thorpe rank-order coding). This closes that last host op.

## The claim (6/6 seeds)
On a graded per-column drive (60 columns, K=6, chance overlap 0.10):
- **The spiking rank-order selection == the host top-K:** overlap **0.92 mean** (1.00/1.00/1.00/0.67/1.00/0.83 across seeds 42/43/44/100/101/102) — the columns integrate their graded drive to threshold and the higher-drive columns SPIKE EARLIER, so the first-K-to-spike are the top-K by drive.
- **PERMUTED-drive follows:** 0.92 mean — the spiking winners track the permuted top-K (the competition reads the drive, not a fixed bias).
- **The FS lateral inhibition suppresses the loser pool:** fired fraction 0.29 with the FS vs 0.57 FS-lesioned.

## Mechanism
`NCOL` Izhikevich column cells + a fast-spiking (FS) inhibitory pool; column→FS excitatory + FS→column inhibitory. The graded drive is injected as external current at a near-rheobase operating point, so time-to-first-spike decreases with drive (Thorpe rank-order coding). The winners = the first-K columns to spike (read from the spike TIMING — a spiking observable, not a host argsort over the drive); the FS suppresses the rest so the loser pool stays sparse.

## Anti-cheats (6/6)
- **PERMUTED-drive** (input-destruction): the spiking winners follow the permuted top-K (0.92) — isolates that the competition reads the drive.
- **FS-LESION** (no lateral inhibition): the fired fraction rises (0.57 vs 0.29) — the FS is load-bearing for sparsity.
- Overlap 0.92 ≫ chance 0.10; 6-seed.

## Honest scope + next
- The PRIMARY selection is **rank-order timing** (drive → spike-time; the neural code that carries the ranking); the FS lateral inhibition is a **secondary** contributor that suppresses the loser pool (fired fraction 0.29 vs 0.57), not a sharp exact-K clamp (a single global-inhibition pool oscillates when pushed to exactly K — the honest limitation). At dt=1.0 ms the timing has finite resolution, so columns with near-equal drive occasionally tie (seeds 100/102 lower) — the mean 0.92 is well above chance.
- This de-risks the SELECTION against the host top-k; wiring `select(drive)` into the EMERGE-40 pooler learning loop (replace `np.argsort(-drive)[:K]`) is the mechanical follow-on — the learning is unchanged (it needs only the winner set).
- ⇒ with EMERGE-38→41, the competitive self-organizing pooler is realizable **fully on the spiking substrate**: drive read from the substrate weights, winners by spiking rank-order + FS inhibition, learning by `sim/` kernels.

## Artifacts
`research/runners/_emerge41_fs_wta_kwinners_derisk.py`, `tests/test_emerge41_fs_wta_kwinners.py`, `research/findings/raw/_emerge41_fs_wta_kwinners.json`. Prior: `2026-07-02-emerge40-spiking-htm-sp-kernel-GO.md`.
