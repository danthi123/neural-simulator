---
type: finding
status: no-go
lane: memory-learn-through-use
date: 2026-08-27
mechanism: replay-driven coincidence-STDP consolidation on the Ecker AdEx CA3 discrete-forward-replay store
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py
runner: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py
external:
  - Ecker, Bagi, Vamosi et al. 2022 eLife 11:e71850 (https://elifesciences.org/articles/71850) — SWR sequence replay emerges from cellular ADAPTATION + a temporally-SYMMETRIC STDP rule over chain recurrence (not bistable completion); implies replay+coincidence-plasticity relaxes toward a SYMMETRIC weight structure, not a forward-only synfire asymmetry.
  - Braun et al. 2022, sequence generation via inhibitory gap coding / selective disinhibition WITHOUT synfire feedforward excitation — locate https://scholar.google.com/scholar?q=Braun+2022+inhibitory+gap+coding+sequence+generation (named fallback, avoids the overlapping-volley problem).
  - Widloski et al. 2025 Nat Commun, replay and ripples are DISTINCT/separable — locate https://scholar.google.com/scholar?q=Widloski+2025+replay+ripples+distinct+Nature+Communications (load-bearing element is the forward-ordered REPLAY SEQUENCE, the ripple envelope is optional).
artifacts:
  - research/findings/raw/gap5_ecker_adex/ecker_replay_learn_through_use_6seed.json
  - research/findings/raw/gap5_ecker_adex/ecker_replay_ltu_oppoint_sweep_s42.json
---
# The Ecker AdEx store SEGMENTS into discrete forward replay (the bistable wall is surpassed), but OFFLINE replay-driven coincidence-STDP does NOT strengthen forward-ordered recall — it SYMMETRIZES the band (reverse potentiates ~6x forward) — 6-seed NO-GO on the method; capability continues via a separated-volley / gap-coding / BTSP-eligibility write

Artifact: `research/findings/raw/gap5_ecker_adex/ecker_replay_learn_through_use_6seed.json` (6-seed decisive) +
`research/findings/raw/gap5_ecker_adex/ecker_replay_ltu_oppoint_sweep_s42.json` (seed-42 8-point b x tau op-point sweep).
Runner: `research/runners/_gap5_ecker_replay_learn_through_use_derisk.py` (reuse-by-import of the banked Ecker STDP-band build/encode/replay; NO `sim/` edit).

## Question
The [[2026-08-27-swr-envelope-learn-through-use-NOGO]] wall was the bistable STORE ARCHITECTURE (never segments — co-fires 0.97). Using the Ecker AdEx CA3 store INSTEAD (self-terminating volleys + forward/reverse asymmetry + adaptation): does it segment, and does OFFLINE discrete forward replay DURABLY strengthen the replayed sequence with lesion-the-replay controlling it?

## Result 1 — the Ecker store SEGMENTS (the bistable wall is genuinely surpassed)
<!--derived-->
Re-confirmed on the real learn-through-use instrument, 6-seed: the STDP-grown band replays DISCRETE forward SWR events from a non-specific prefix seed, full-cue forward-from-seed **0.914** (chance 0.2524), rests silent between events. This is the discrete forward-ordered replay the bistable co-firing store could NOT produce. YES on segmentation.

## Result 2 — but replay-driven STDP does NOT consolidate forward recall: 6-seed NO-GO 0/6
<!--derived-->
Turning the substrate's OWN STDP ON during the offline replay bouts (clock advanced, plastic between-edges), the self-generated replay durably moves the weights — but in the WRONG direction. 6-seed mean: **dw_fwd −11.0** (forward edges DEPRESS) vs **dw_rev +65.6** (reverse POTENTIATES ~6x); adj_rev 11.4 → 77.0. Forward-ordered recall does not strengthen: weak-cue forward 0.87 → 0.775 (robustness DROPS). Every seed: directional=False.

The strengthening IS load-bearing and lesion-controlled: **NO-SEED (lesion-the-replay: STDP on + clock advancing + OU noise, only ignition removed) gives dw=0.000 on ALL 6 seeds, 100% attributable** — so the plasticity requires the replay, it is just misdirected. Determinism verified (seed-hash identical all 6).

## Root cause (mechanistic + literature-confirmed), and why no op-point rescues it
The population hand-off needs a within-assembly VOLLEY, and the strong forward weight igniters the next assembly FAST, so consecutive volleys OVERLAP during the self-driven cascade. For the reverse edge B→A, B fires while A's volley tail still fires → pre-before-post → LTP on reverse. (The ENCODE phase avoided this: its moving cue temporally SEPARATED activations.) The seed-42 op-point sweep (b 120→1400 = 7x Ecker's default, tau 4→20; 8 points) shows faster self-termination MONOTONICALLY shrinks the reverse excess (dw_rev−dw_fwd 62→4) but NEVER flips it — directional=False everywhere. Literature-consistent (Ecker 2021): coincidence plasticity + adaptation relaxes toward a SYMMETRIC (bidirectional-replay) structure; a forward-only synfire asymmetry is not a stable fixed point of its own replay.

## Verdict + next mechanism (a wall defers a METHOD, not the capability)
NO-GO for coincidence-STDP-during-replay as the forward-consolidating write — refuted robustly (6 seeds x 8 op-points). BANKED positives to build on: the Ecker store segments, and replay drives durable, lesion-controlled, 100%-attributable plasticity. Named next mechanisms: (1) temporally-SEPARATED volleys via a forward-edge conduction DELAY (B ignites after A terminates → clean pre-before-post) — Ecker's own separation; (2) Braun 2022 inhibitory gap-coding sparse sequences (no overlapping synfire excitation); (3) a BTSP-eligibility-gated write (seconds-long window) instead of ms-coincidence STDP. Separately, wiring replay INTO the production D5 organ stays blocked at the AdEx SOMA-recurrence seam ([[2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read]], NO-GO), needing the dendritic-dAP-latch composition.
