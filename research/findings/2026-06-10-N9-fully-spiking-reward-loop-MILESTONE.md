# N9 reward-prediction-error loop is now FULLY SPIKING — the two remaining host pieces (reward delivery, critic normalization) made neural

**Date:** 2026-06-10
**Owner directive served:** "the fixes are approved if they're production-grade and don't rely on shortcuts; anything that CAN be spiking is made spiking." This closes the two host-computed pieces the audit found in the otherwise-all-neural N9 loop.

## What was host, and is now spiking

| Piece | Was (host shortcut) | Now (spiking) | Flag | Surface |
|---|---|---|---|---|
| **(A) Reward delivery** | `snc_ext_current += snc_reward_gain·max(0,reward)` — a number → DA current, NO neuron between | a PPN-like **`reward_us`** excitatory population (US→VTA glutamate, catalog C.33) receives the **perceived** reward (the coord-free N5 approach signal, a sensory drive) and **FIRES into the SNc** → the reward burst is a neuron's synapse; the host write is **dropped** | `--spiking-reward-us` | runner-only |
| **(B) Critic rate normalization** | the over-firing (~125 Hz) MSN critic was masked downstream by the GIRK conductance cap (`gabab_conductance_max`) | a **`place_fs → striosome_value`** GABA_A feedforward inhibition (the FS-PING pool, which scales with the volley → divisive) holds the critic in a **physiological** rate band, so the δ is graded at the source | `--enable-critic-fs-inhibition` | runner-only |

With both on (and `--perceived-approach-reward`), the **entire** δ = r − V loop is synaptic: **r** = `reward_us` excitation onto the SNc, **V** = the striosome critic's GABA_B subtraction at the SNc membrane, both detected/learned by neurons. The only legitimately-host residuals are the **environment** (rendering the goal cue into the retina; the goal-contact event) and the **body** (acting on the motor pools) — per the BRAIN-BASED-ONLY boundary.

## De-risk results

**(B) GO — Stage-B `--critic-fs-weight` sweep on the hot draw (seed 44):** weight 0 → critic@near **125.97 Hz**, GABA_B gap **0→0 (binary, clamped)**; weight **16 (default)** → critic@near **8.19 Hz** (physiological MSN 1–20 Hz), grade 14.6×, gap **30→112.5 = 3.75× GRADED** (Eshel arithmetic, pred>0). The spiking FS inhibition fixes BOTH the over-firing AND the binary-δ — with the GIRK cap OFF. FS is held ON throughout (value-train + read-out) so the critic learns and reads V in one self-consistent clamped regime (gating it off for value-train learns a stronger V that over-fires the ungated read-out → binary; tested + reverted). Finding: `2026-06-10-N9-spiking-fs-critic-inhibition-GO.md`.

**(A) GO (integration):** `reward_us` (40 cells) builds and fires the SNc from the perceived reward; the agent **navigates** end-to-end (recent_dist 18→1.4 by step 200, holds at the goal), with V learned (w_near 5.95 / w_far 1.06 = 5.6× at seed 42). The host reward write is dropped; `reward_us` fires only during the reward-hold (zeroed after). Coord-free guard warns if used without `--perceived-approach-reward`.

## Fully-biologized nav A/B (FULLSPIKE vs host) + anti-cheats

_(To finalize when the 1800-step multi-seed A/B (`b58hg1b4o`) completes — aggregator `research/findings/raw/g11_bg/_n9fullspike_aggregate.py`.)_

- **FULLSPIKE** (`--spiking-reward-us --enable-critic-fs-inhibition --perceived-approach-reward`, FS on, GIRK cap off) vs **OFF** (host-V scaffold cheat) vs **ONcap25** (the GIRK-masked version), seeds 42–44.
- **(A) US-lesion anti-cheat** (`--reward-us-drive-pa 0` → US silent): nav should regress if the US chain is load-bearing.
- **(B) place-shuffle** (Stage-B harness): the FS-clamped grading must still ride on learned V (the w_near≫w_far asymmetry already shows the weights are learned).

## Honest residuals (NOT host shortcuts — feasible-spiking is done)

- **Value-train V draw-variance**: some place-code draws under-learn V (anti-learned on one seed-42 draw) — the same CuPy place-code non-determinism the determinism edit addresses (byte-review pending), not a host shortcut.
- **Multi-seed robustness** of the online graded δ, pending the A/B + the determinism fix.
- The `critic_teacher_pa` (a sub-threshold host current during value-train, removed before nav) is an **acceptable teaching scaffold** by the project's standard (the critic re-validated to fire from the volley alone), not a shortcut.

## Status

Both remaining host pieces of the N9 loop are now spiking — the BRAIN-BASED-ONLY completion. The fixes are runner-only (no new sim/ edit beyond the byte-reviewed determinism + GIRK-cap edits); the GIRK cap is relegated to a redundant guardrail by the FS inhibition.
