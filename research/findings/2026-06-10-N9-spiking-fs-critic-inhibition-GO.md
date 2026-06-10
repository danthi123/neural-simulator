# 🎉 N9 (B): spiking FS→critic feedforward inhibition is the ROOT fix for the over-firing critic + binary δ — GO

**Date:** 2026-06-10
**Type:** runner-only build (no sim/ edit) + GPU Stage-B de-risk.
**Owner directive it serves:** "anything that CAN be spiking is made spiking." The over-firing MSN critic was being masked downstream by the GIRK conductance cap (`gabab_conductance_max`) — legitimate biophysics but a symptom-patch. This makes the **root** fix spiking.

## The fix (catalog B.06, Lee 2017; research 2026-06-10-N9-spiking-reward-and-critic-normalization)

The FS-PING pool `place_fs` already gamma-synchronizes the place volley but inhibited **only** `place`. Added a single `place_fs → striosome_value` GABA_A feedforward-inhibition pathway (`--enable-critic-fs-inhibition`, runner-only, default OFF = byte-equivalent). Because the FS pool's firing **scales with the volley size**, a hotter draw recruits **more** FS inhibition → a **divisive-leaning** clamp that holds the MSN critic in a physiological rate band across draws. The honest hybrid (per the research): FS clamps the **rate**; the **weighted coincidence plateau** (already shipped) grades **V** (the all-or-none plateau can't be smoothly divided).

## De-risk: Stage-B `--critic-fs-weight` sweep on the hot draw (seed 44, deterministic-selforg)

| critic_fs_weight | critic@near | critic@far | grade | GABA_B gap (pred→unpred) |
|---|---|---|---|---|
| 0 (off, GIRK-cap regime) | **125.97 Hz** (over-firing) | 40.42 | 3.1× | 0.00 → 0.00 (**binary, both clamped**) |
| 8 | 47.22 | 5.83 | 8.1× | 0.00 → 72.50 (de-saturated) |
| **16 ★** | **8.19 Hz** (physiological) | 0.56 | **14.6×** | **30.0 → 112.5 = 3.75× (GRADED, pred>0)** |
| 24 | 0.56 (over-clamped) | 0.00 | fail | 105 → 125 = 1.19 (collapsing) |

**`critic_fs_weight=16` (now the default) is the validated sweet spot:** the spiking FS inhibition brings the over-firing **126 Hz → 8.19 Hz** (squarely the MSN physiological range 1–20 Hz, Wilson & Kawaguchi), keeps strong value-grading (near 14.6× far), and produces a **GRADED Eshel arithmetic δ** (pred=30 < unpred=112.5, ratio 3.75× — pred>0, NOT the binary clamp) — **all via a real spiking inhibitory pathway, with the GIRK cap OFF.** This is the root fix the owner asked for: the critic is physiological because a *neuron* (the FS pool) inhibits it, not because a conductance bound clips the GIRK at the SNc.

## Status

- **GO** on the hot draw. `critic_fs_weight=16` default. The GIRK cap is now a redundant guardrail (relegated, not the operative mechanism), per the research's B-3.
- Anti-cheat to confirm at multi-seed: place-shuffle must still break V (grading rides on learned value, not "fired-on-any-drive"); cap-binding frequency should be ~0 with FS on.
- Next: the (A) spiking-reward de-risk, then the FULLY-biologized nav A/B (`--spiking-reward-us --enable-critic-fs-inhibition --perceived-approach-reward`) vs the host baseline + multi-seed.
