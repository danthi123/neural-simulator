---
type: finding
status: nogo
lane: memory-learn-through-use
date: 2026-08-27
mechanism: Braun-2022 inhibitory GAP CODING as the volley-separation route for replay-driven learn-through-use on the Ecker AdEx CA3 store — dense basket inhibition on the pyramidal g_i channel (GABA-A, E_inh=-75mV) with feedback recruited BY the pyramidal volley (self-locked gaps), reusing the substrate's real inhibitory conductance + the directional BTSP write (sim.kernels.fused_btsp_update); default-OFF flag --gap-coding, byte-identical off
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py (--gap-coding)
runner: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py
external:
  - Braun, Memmesheimer 2022 "High-frequency oscillations and sequence generation in two-population models of hippocampal region CA1", PLoS Comput Biol vol18 issue2 e1009891, https://consensus.app/papers/details/e471764bac0c59ddb40ca5ebc11b6111/ (PubMed PMID 35176028) — sequence replay from "alternating excitatory pulse and inhibitory gap coding: phases of silence in specific basket cell groups induce selective disinhibition of groups of pyramidal neurons", giving sparse pyramidal + dense basket spiking, NOT a synfire chain.
artifacts:
  - research/findings/raw/gap5_ecker_adex/gapcoding_mildtonic_learn_through_use_6seed.json
  - research/findings/raw/gap5_ecker_adex/gapcoding_learn_through_use_6seed.json
---
# Braun-2022 inhibitory GAP CODING does NOT unblock replay-driven learn-through-use on the Ecker AdEx CA3 store. Decisive NO-GO 0/6 (mild inhibition, store still segments): the un-separated write stays REVERSE-dominant and DEGRADES weak-cue recall. Strong inhibition (the config that WOULD separate the volleys) instead EXTINGUISHES the fragile traveling-bump replay entirely (an UNDEFINED, instrument-compromised arm — full-cue forward drops to 0). Either way gap coding fails, and the read-saturation the conduction-delay PARTIAL isolated is diagnosed as a METRIC artifact (binary forward-order), not a separation problem inhibition can fix.

Artifact: `research/findings/raw/gap5_ecker_adex/gapcoding_mildtonic_learn_through_use_6seed.json` (DECISIVE 6-seed NO-GO, mild tonic gi_base=4 — store segments so the precondition holds) + `research/findings/raw/gap5_ecker_adex/gapcoding_learn_through_use_6seed.json` (feedback gi_base=8 fb_gain=0.6 — UNDEFINED: the inhibition extinguishes the replay, compromising the read).
Runner: `research/runners/_gap5_ecker_replay_learn_through_use_derisk.py --gap-coding` (additive, default OFF = byte-identical: no g_i is ever set, the STDP/BTSP/conduction-delay paths are untouched, the gap-coding functions are never called; reuse-by-import of `fused_btsp_update`; NO `sim/` edit). g_i suppression verified directly: g_i~50 partial, g_i~200 full silence of a 9000pA drive.

## Question
[[2026-08-27-conduction-delay-directional-replay-learn-through-use-PARTIAL]] SOLVED write-directionality (6/6, forward-edge conduction delay separates the volleys) but recall did not durably strengthen: the long-period regime REQUIRED to separate the delayed volleys drove the recall read to ceiling (weak-cue forward_frac ~1.0 BEFORE — no headroom). Named next: Braun-2022 inhibitory gap coding — sparse volleys that never overlap, separated by inhibition, so the read runs at a NORMAL regime and keeps headroom. Does it unblock?

## Result — NO-GO 0/6: no regime gives BOTH a directional write AND a live/headroom read
<!--derived-->
Two failure modes bracket the whole inhibition range (calibration sweep + two 6-seed runs). (1, UNDEFINED — instrument compromised) Inhibition strong enough to SEPARATE (feedback basket recruited by the volley, or tonic g_i>=~30) EXTINGUISHES the traveling bump: full-cue forward replay collapses to 0.000 with n_multi=0 on all 6 seeds (the Ecker bump has almost no margin — the very inhibition meant to sparsify it kills the forward hand-off), so the store no longer segments (the read precondition fails -> UNDEFINED, not a negative) and the write is reverse-dominant (dw_fwd 62.7 < dw_rev 79.6; only 2/6 directional; adj_rev 11.5->91.2). (2, the DECISIVE NO-GO) Inhibition mild enough to preserve the replay (weak tonic gi=4; store SEGMENTS, full-cue forward 1.000 vs chance 0.3126 -> precondition HOLDS) does NOT separate the volleys, so the write stays REVERSE-dominant exactly as the pre-delay BTSP PARTIAL (dw_fwd 306 vs dw_rev 583, 0/6 directional) and DEGRADES the very recall it should strengthen: weak-cue forward 0.958->0.236. Lesion-controlled throughout: dw_noseed 0.000, weak-noseed unchanged (0.958) on all seeds.

## Why — the read saturation is a METRIC artifact, not a separation problem inhibition can fix
<!--derived-->
`forward_frac` = forward-ordered events / multi-assembly events. It only drops on ORDER ERRORS or reverse intrusions; a truncated forward run (A->B->C then stop) still scores forward. A clean encoded band makes almost no order errors, so forward_frac ~1.0 (0.96 weak-cue) at every non-extinguishing inhibition level — separating the volleys cannot open meaningful headroom in THIS metric, and the marginal residual only shrinks under the reverse-dominant write. The only large sub-1.0 forward_frac reachable was via EXCITATORY background (pc_tonic 200-400 -> forward 0.67/0.18), which is disorder, not order-recoverable structure, and is not gap coding. So the residual the PARTIAL called "recall at ceiling" is the INSTRUMENT (the read is binary-order), consistent with "the instrument is part of the emulation".

## Verdict + next mechanism (a wall defers a METHOD, not the capability)
NO-GO — Braun gap coding is REFUTED as the volley-separation route on this substrate: dense basket inhibition on the Ecker AdEx bump extinguishes forward replay before it sparsifies it, and turns the directional write reverse-dominant. Banked negative; the delay PARTIAL's separated-write win stands (that is the right separation route). Scope note: this reduces Braun's GROUP-SPECIFIC structured basket->pyramidal gap wiring to a global feedback-inhibition gap + learned-band selection; the group-specific structured version is not tested (and would put the sequence in the inhibitory wiring, not the learned band under test).
Named next: (1) a GRADED recall instrument with headroom, decoupled from volley separation — sequence-COMPLETION DEPTH (how far a weak cue replays) or recall LATENCY, which rise as the band deepens even when order is already perfect; this directly attacks the isolated instrument cause. (2) Widloski-2025 replay-without-ripples (a separation route that does not require the read-saturating regime).
Production D5 learn-through-use is CLOSED + on-by-default separately ([[2026-08-21-d5-learn-through-use-flip-GO-per-topic-strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact-not-substrate-crosstalk]]); this NO-GO is the TOY-STORE replay-strengthens-recall research thread only.
Source: Braun & Memmesheimer 2022, PLoS Comput Biol, doi:10.1371/journal.pcbi.1009891 (PubMed PMID 35176028). <!--derived-->

