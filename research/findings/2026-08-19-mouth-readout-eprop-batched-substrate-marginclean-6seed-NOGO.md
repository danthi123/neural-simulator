---
type: finding
status: live
date: 2026-08-19
verdict: NO-GO
mechanism: mouth-readout-eprop-batched-substrate-forward
lane: mouth / gap#4-read-regime
artifacts:
  - research/findings/raw/_wkv_readout_eprop_batched_substrate_marginclean_6seed.json
---
# Mouth read-out via e-prop through a batched-substrate FORWARD — margin-clean 6-seed: NO-GO on parity

The session-long GPU crux (board #37 / Gate-B): does the word/mouth read-out, learned by e-prop with the FORWARD pass
computed through the real spiking substrate (batched graded-conductance read, NOT a host-linear proxy), reach the
copied-weight ceiling? Runner `_wkv_mouth_readout_eprop_batched_substrate_derisk` (cupy, 6 seeds 42/43/44/100/101/102,
`--batch 48 --n-train-pos 8000 --epochs 10 --lr 0.3 --sub-read-window 360`), elapsed ~20.8 ks. This is the margin-clean
follow-up to the 2026-08-17 PARTIAL — it SUPERSEDES that with a clean 6-seed verdict.

## Verdict: NO-GO (go_count 0/6, 5-of-6 false).

<!--derived-->
The bottleneck is the LEARNING, not the read. All numbers below are aggregate means quoted from the `summary` block
of the cited artifact `research/findings/raw/_wkv_readout_eprop_batched_substrate_marginclean_6seed.json`:
- **The READ is faithful.** Copied-weight recovery `sub_copied_recov_mean` = **0.979** — with the correct weights, the
  batched-substrate FORWARD read reaches the ceiling. The forward is genuinely substrate: `forward_is_substrate_all`
  true, `host_matmul_on_forward_max` = 0, `no_transport` = true, `no_host_grad` = true.
- **e-prop LEARNING plateaus at the host-linear proxy.** Substrate-learned recovery `sub_learned_recov_mean` = **0.371**
  = the host-linear baseline `hostlinear_recov_mean` **0.3705** — i.e. learning the read-out weights by e-prop through
  the substrate forward reaches the SAME level as the linear proxy and stays far below the 0.979 achievable ceiling.
  `sub_recov_ratio_mean` 0.379 (min 0.345). Learned weights barely align with the target: `weight_cosine_mean` 0.136.
- **Anti-cheats clean (collapse 6/6).** Shuffle recovery `sub_shuffle_recov_mean` = **0.0027**; host-linear FLOOR
  `hostlinear_floor_recov_max` 0.062; weight-cosine floor 0.005; per-seed gains ~9.3-9.6; `seed_hash_check` seeded
  (thr hashes identical). So the 0.371 is a real learned signal that collapses under the controls — not an artifact.

## Caveat (honest)
`verify_first_all_ok` = false: **5/6 seeds passed the instrument verify-first; seed 100 failed it.** The headline rests
on the 5 verify-clean seeds (all of which land at the same ~0.37 plateau), so the NO-GO is not a seed-100 artifact,
but the seed-100 pre-check failure is disclosed, not hidden.

## What this localizes (the next mechanism, not a wall)
The substrate FORWARD read is faithful (copied → ceiling); the deficit is entirely in CREDIT ASSIGNMENT — e-prop
through the batched-substrate forward learns no better than a linear proxy (0.37, cosine 0.14). This pins the mouth
frontier to the **deep-credit / read-regime** the gap#4 arc named (see
[[project_gap4_wall_was_a_hyperparameter]]): the read is not the wall, the LEARNED credit through the spiking read is.
Next levers to isolate the residual: (1) a richer per-read credit signal (dendritic/burst-multiplexed vs the current
e-prop eligibility) so the learned weights approach the copied ceiling the read already supports; (2) confirm the
seed-100 verify-first failure is a per-seed operating-point issue, not a systematic instrument gap; (3) the Izhikevich
read-regime operating point flagged by the gap#4 summary. The copied-vs-learned gap (0.98 vs 0.37) is the exact,
quantified residual to close.
