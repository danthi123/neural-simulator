# EMERGE-41 / toward-semantics — GO (reframed): the pooler's k-winners SELECTION runs as SPIKING RANK-ORDER (latency) coding, not a host argsort over the drive. The FS lateral inhibition is INERT for the selection (it only sparsifies the loser pool). Closes the last host step in the competitive self-organizing pooler. NO `sim/` edit.

**2026-07-02 (autonomous; reframed after an adversarial audit — see "Audit correction" below).** Runner `research/runners/_emerge41_fs_wta_kwinners_derisk.py`; CI guard `tests/test_emerge41_fs_wta_kwinners.py` (4 tests). Reuse-by-import (`_emerge11` WTA pattern); NO `sim/` edit; CPU numpy-backend; multi-seed.

## Why
EMERGE-38/39/40 made the competitive-pooler LEARNING fully-on-substrate (`sim/` kernels: potentiation + winner-inactive depression), but the k-winners SELECTION of *which* columns win was still a host `np.argsort(-drive)[:K]`. A biological pooler selects winners by SPIKING dynamics (Thorpe rank-order / latency coding; Diehl-Cook 2015 lateral inhibition sparsifies). This closes that last host op.

## The claim (reframed, honest)
On a graded per-column drive (60 columns, K=6, chance overlap 0.10):
- **The spiking rank-order (latency) selection == the host top-K:** overlap **1.00 mean** (1.00 / 1.00 / 1.00 across seeds 42/43/44) — the columns integrate their graded drive to threshold at a near-rheobase operating point and the higher-drive columns SPIKE EARLIER, so the first-K-to-spike are the top-K by drive. (Prior wider-seed spread noted below.)
- **FLAT-drive (input-destruction) collapses:** overlap **0.17 mean** (0.17 / 0.17 / 0.17) — a uniform, non-graded drive removes the ranking signal, so every column integrates identically and the first-K is decided by the randomized tie-break → the overlap with the host top-K collapses to the tie-break floor (~chance 0.10). Isolates that the SELECTION reads the GRADED drive, not a fixed structural bias.
- **The FS lateral inhibition is CAUSALLY INERT for the selection:** the winner set is **identical** FS-on vs FS-lesion (winner-overlap **1.00 / 1.00 / 1.00**). The FS does NOT choose which columns win.
- **The FS's ONLY effect is loser-pool SPARSITY:** fired fraction **0.28 mean** with the FS vs **0.57 mean** FS-lesioned (0.48 / 0.60 / 0.63) — the FS suppresses how many columns fire at all, a post-hoc effect that does not touch the winner set.

## Mechanism
`NCOL` Izhikevich column cells + a fast-spiking (FS) inhibitory pool; column→FS excitatory + FS→column inhibitory. The graded drive is injected as external current at a near-rheobase operating point, so time-to-first-spike decreases with drive (Thorpe rank-order / latency coding). The winners = the first-K columns to spike (read from the spike TIMING — a spiking observable, not a host argsort over the drive). Ties on the first-spike step are broken by a randomized, drive-independent key (so ties do not default to a fixed low-index bias). The FS suppresses the rest so the loser pool stays sparse — but it does not participate in the selection.

## Anti-cheats (multi-seed)
- **FLAT-drive** (input-destruction): a uniform drive collapses the overlap to the tie-break floor (0.17 mean vs 1.00 graded) — isolates that the SELECTION reads the graded drive.
- **FS-LESION winner-set identity** (reported control): the winner set is identical FS-on vs FS-lesion (1.00) — the FS is inert for the selection; the pure rank-order integrator reproduces the overlap.
- **FS-LESION sparsity**: the fired fraction rises (0.57 vs 0.28) — the FS's only effect is loser-pool sparsity.
- Overlap 1.00 ≫ chance 0.10; multi-seed.

## Audit correction (why this was reframed)
An adversarial audit confirmed the on-substrate rank-order SELECTION result is real, but caught two overclaims in the original write-up, both now fixed (each verified to preserve the GO):
1. **The FS is causally inert for the selection.** The winners are byte-identical FS-on vs FS-lesion on every seed, and the pure integrator without FS reproduces the overlap exactly. Winners are selected purely by rank-order spike TIMING (higher drive → earlier spike). The FS only sparsifies the loser pool (post-hoc). The finding/runner/CI no longer claim "k-WTA" / "winners-take-all" / "FS competition" for the SELECTION — it is spiking **rank-order (latency) coding** (Thorpe); the FS provides loser-pool sparsity, not selection.
2. **The permuted-drive control had no independent power.** Permuting a drive over an index-agnostic mechanism is just an index relabeling, so `permuted_overlap == overlap` by construction — it proved nothing. It is REPLACED with a genuine input-destruction control: a FLAT (uniform, non-graded) drive, which collapses the overlap toward the tie-break floor (measured 0.17). The tie-break in `select()` is now randomized (was a stable argsort with a low-index bias); the gate uses the flat-drive collapse.
3. The FS-lesion winner-set-identity is now an explicit reported control (winner-overlap ~1.00) that DEMONSTRATES the FS is not doing the selection.

## Honest scope + next
- The SELECTION is **rank-order latency timing** (drive → spike-time; the neural code that carries the ranking). The FS lateral inhibition is NOT part of the selection on this single-global-FS-pool substrate — it only suppresses the loser pool (fired fraction 0.28 vs 0.57). A local / structured inhibition (vs one global pool) would be required for the FS to *shape* the winner set; that is not what this de-risk shows, and is not claimed.
- At dt=1.0 ms the latency has finite resolution, so columns with near-equal drive can tie; the randomized tie-break keeps ties unbiased. On the 3 re-run seeds (42/43/44) the overlap is a clean 1.00. **Honest wider-seed disclosure:** an earlier (pre-reframe) 6-seed sweep saw overlap 1.00/1.00/1.00 on 42/43/44 but 0.67 and 0.83 on two of the extended seeds (100/102) — near-equal-drive ties at the dt=1.0 timing resolution — so the mean over 6 seeds was ~0.92. The finite-resolution tie is the honest limitation; the effect is far above chance (0.10) at every seed.
- This de-risks the SELECTION against the host top-k; wiring `select(drive)` into the EMERGE-40 pooler learning loop (replace `np.argsort(-drive)[:K]`) is the mechanical follow-on — the learning is unchanged (it needs only the winner set).
- ⇒ with EMERGE-38→41, the competitive self-organizing pooler is realizable **fully on the spiking substrate**: drive read from the substrate weights, winners by spiking rank-order (latency) timing (FS providing loser-pool sparsity), learning by `sim/` kernels.

## Artifacts
`research/runners/_emerge41_fs_wta_kwinners_derisk.py`, `tests/test_emerge41_fs_wta_kwinners.py`, `research/findings/raw/_emerge41_fs_wta_kwinners.json`. Prior: `2026-07-02-emerge40-spiking-htm-sp-kernel-GO.md`.
