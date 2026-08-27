---
type: finding
status: partial
lane: memory-learn-through-use
date: 2026-08-27
mechanism: BTSP-eligibility-gated directional write (seconds-long causal presynaptic eligibility x an all-or-none plateau instructive post signal, reusing sim.kernels.fused_btsp_update) replacing replay-time ms-coincidence STDP on the Ecker AdEx CA3 forward-replay store
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py (--write-rule btsp)
runner: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py
external:
  - Gonzalez, Lacefield et al. 2023 bioRxiv "Synaptic Basis of Behavioral Timescale Plasticity" (https://consensus.app/papers/details/fd7ad7e9446c55469c76953511ddf985/) — BTSP is an ASYMMETRIC plasticity kernel of BIDIRECTIONAL weight changes around a plateau; inputs active in a seconds-long window PRECEDING and FOLLOWING the plateau potentiate. The bidirectional/before-and-after structure is exactly why a too-long plateau re-symmetrizes.
  - Bittner, Milstein, Magee 2017 Science (doi science.aan3846) + Milstein-Magee 2021 eLife 73046 — the eligibility x plateau, pure-potentiation BTSP rule the substrate's fused_btsp_update implements; Magee 2026 Nat Neurosci review confirms BTSP is strong, bidirectional, seconds-scale, single-plateau-induced.
artifacts:
  - research/findings/raw/gap5_ecker_adex/btsp_directional_write_ltu_6seed.json
  - research/findings/raw/gap5_ecker_adex/btsp_directional_write_oppoint_sweep_s42.json
  - research/findings/raw/gap5_ecker_adex/btsp_directional_write_oppoint_sweep_s42.py
---
# The BTSP-eligibility write (causal seconds-long eligibility x plateau, pure potentiation) HALVES the reverse excess and flips the forward edge from DEPRESSING to POTENTIATING with recall preserved — but does NOT reach net-directional consolidation (0/6 directional; dw_rev still ~1.9x dw_fwd). A 14-point op-point sweep never flips it: the residual blocker is the VOLLEY OVERLAP, not the write window — next is conduction-delay-separated volleys

Artifact: `research/findings/raw/gap5_ecker_adex/btsp_directional_write_ltu_6seed.json` (6-seed decisive) +
`research/findings/raw/gap5_ecker_adex/btsp_directional_write_oppoint_sweep_s42.json` (seed-42 14-point plat_tau x elig_tau x eta sweep).
Runner: `research/runners/_gap5_ecker_replay_learn_through_use_derisk.py --write-rule btsp` (additive, default-OFF; `--write-rule stdp` is byte-identical to the banked [[2026-08-27-ecker-adex-store-learn-through-use-NOGO]]; reuse-by-import of the substrate's `fused_btsp_update`; NO `sim/` edit).

## Question
The [[2026-08-27-ecker-adex-store-learn-through-use-NOGO]] wall was THE WRITE RULE: replay-time ms-coincidence STDP SYMMETRIZES the Ecker store's forward band (reverse potentiates ~6x, forward DEPRESSES −11.0). Ecker 2022 traced it to a temporally-SYMMETRIC SWR-STDP rule. Does a DIRECTIONAL write — BTSP's SECONDS-long CAUSAL presynaptic eligibility gated by an all-or-none plateau instructive POST signal, fed to the substrate's own pure-potentiation `fused_btsp_update` — deepen forward-only (dw_fwd > dw_rev, weak-cue forward rises) where ms-STDP could not?

## Result 1 — the BTSP write is a real, seed-robust ADVANCE, but still not net-directional (6-seed, plat_tau=0.1ms elig_tau=15ms eta=0.02)
<!--derived-->
Forward now POTENTIATES: dw_fwd **+25.5** (was −11.0 under STDP), band adj_fwd 319.4 → **344.9**. The reverse excess more than HALVES: dw_rev **+49.4**, ratio **1.94x** (was ~6x). Recall is PRESERVED, not degraded: full-cue forward 0.931 → **0.962**, weak-cue forward 0.846 → 0.834 (flat; up/flat on 3/6 seeds) — vs STDP's 0.87 → 0.775 collapse. But the write is still not forward-selective: **0/6 directional** (dw_fwd < dw_rev every seed). The write is fully lesion-controlled: NO-SEED (STDP-off host BTSP, clock advancing, OU on, only ignition removed) → dw = 0.000 on ALL 6 seeds, 100% attributable; determinism seed-hash identical all 6.

## Result 2 — a 14-point op-point sweep NEVER flips it (seed 42; the STDP-sweep analog)
<!--derived-->
Across plat_tau {0.1, 0.5, 1, 2, 4, 8} ms x elig_tau {15, 40} ms x eta {0.005, 0.02, 0.05}, dw_rev > dw_fwd at EVERY point. Shorter plateau shrinks the reverse excess MONOTONICALLY (diff dw_fwd−dw_rev: −286 at plat=8ms → −18.7 at plat=0.1ms) but NEVER crosses zero. Longer eligibility helps forward but helps reverse equally. No setting is directional — exactly like the banked STDP b x tau sweep (shrinks, never flips).

## Root cause (proved, not asserted): the VOLLEY OVERLAP, not the write window
At plat_tau=0.1ms the plateau is a ~1-step pulse (no tail), so a reverse edge B→A can only potentiate if the pre (B) is already eligible AT the step A fires — i.e. A fires WHILE B is already active. dw_rev is still > 0 there, so the assemblies OVERLAP in time: the leading/driven assembly A keeps firing after B ignites (its within-assembly reverberation outlasts B's onset), and the pure-potentiation rule reads that overlap as bidirectional coincidence. A causal eligibility + a no-depression rule structurally CANNOT make dw_rev >> dw_fwd (the STDP failure), but it also cannot make dw_fwd > dw_rev while the reactivation itself is not temporally SEPARATED. Literature-consistent: BTSP's kernel is asymmetric but BIDIRECTIONAL (potentiates before AND after the plateau; Gonzalez 2023), so it does not, on its own, impose forward order on an overlapping cascade.

## Verdict + next mechanism (a wall defers a METHOD, not the capability)
PARTIAL: the BTSP-eligibility write materially improves the replay-time write (forward flips from depressing to potentiating, reverse excess ~6x → ~1.9x, recall preserved) but does NOT achieve directional forward-only consolidation (0/6; 14 op-points). BANKED positives carried forward: the Ecker store segments, replay drives durable lesion-controlled 100%-attributable plasticity, and pure-potentiation removes the STDP reverse-runaway. The residual is now precisely isolated and STRUCTURAL — the reactivation VOLLEYS OVERLAP, so no coincidence-read write can be forward-selective. Named next mechanism (the finding's alternative #1): a forward-edge CONDUCTION DELAY that SEPARATES the volleys (B ignites only AFTER A terminates → clean pre-before-post, zero overlap → directional under either BTSP or STDP) — Ecker's own separation; then Braun 2022 inhibitory gap-coding (sparse sequences with no overlapping synfire excitation). Production D5 wire-in stays separately blocked at the AdEx soma-recurrence seam ([[2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read]]).
