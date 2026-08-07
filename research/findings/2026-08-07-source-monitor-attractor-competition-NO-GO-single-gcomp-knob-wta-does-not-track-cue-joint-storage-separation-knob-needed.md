# Source-monitor RECALL-SIDE CA3-attractor-competition: single-g_comp knob NO-GO (smoke) — joint storage+separation knob needed

Status: NO-GO (smoke, needs no further validation at this locus — the decisive anti-cheat fails robustly; routes to the joint knob per the scoping doc).
Backend: numpy (deterministic). Runner: `research/runners/_laneC_source_monitor_attractor_competition.py`.
Artifact: `research/findings/raw/laneC_source_monitor_attractor_competition/smoke_650_651_overlap0.2.json`.
Scoping: `research/findings/raw/_source_monitor_attractor_competition_scoping.md`.

## What was built
Two encoding-side levers (hetero-depression `8aca3c62`, conjunctive-tag `1a5d2db6`) and four recall-activity levers
all recorded NO-GO: a source-blind recall drive over a fully-shared core reactivates every committed subset, so the
rival burden persists. All six were LINEAR / per-cell / FEEDFORWARD. This de-risk adds the first NONLINEAR,
attractor-level, recall-time competition — a CA3-style autoassociator among the three `source_memory_{s}` pops:
(i) within-population recurrent EXCITATION (slow-NMDA, `exc_receptor="nmda_slow"`; Wang 2001/2002; Rolls CA3 recurrent
collaterals), (ii) between-population lateral INHIBITION (the v2 fast-spiking interneuron circuit, GABA-A). Both scale
with ONE knob `g_comp` at a fixed recurrent:lateral ratio, share `SOURCE_COMPETITION_GATE`, are SYMMETRIC across
sources (no source-specific term), and reuse the v6 silent-by-construction recall (`reset_dynamical_state` +
settle-to-quiescence per recall). NO `sim/` edit — pathways added via the bridge build (`RegionPathway`), reused by
config. Assembly identities are the pre-defined region memberships; discrimination still comes only from the learned
`episode→source` fan-out.

## Smoke result (calib seeds 650/651, g_comp {0, 0.5, 1, 2}, overlap 0.2, core=2/12)
The recall-time margin floor is `min_s M_s ≥ 0.15` AND `min_s M_s > min_s L_s` AND `all_dominant_correct` True.

| seed | g_comp | min_margin_M | min_margin_L | clears 0.15 | all_dominant_correct |
|------|--------|--------------|--------------|-------------|----------------------|
| 650  | 0.0    | +0.0367      | +0.0367      | no          | True (null)          |
| 651  | 0.0    | −0.0567      | −0.0567      | no          | False (null)         |
| 650  | 0.5    | −0.1025      | −0.0442      | no          | False                |
| 651  | 0.5    | −0.1617      | −0.1417      | no          | False                |
| 650  | 1.0    | −0.1050      | −0.0025      | no          | False                |
| 651  | 1.0    | −0.1675      | −0.1383      | no          | False                |
| 650  | 2.0    | −0.1033      | −0.0025      | no          | False                |
| 651  | 2.0    | −0.1058      | −0.1383      | no          | False                |

`min_margin_M` never clears the 0.15 floor at any `g_comp > 0`, and — the DECISIVE anti-cheat — `all_dominant_correct`
is False at EVERY `g_comp > 0` on both seeds. Per-source (seed 650, g_comp 1): the competition amplifies
`self_generated`'s attractor across ALL cues (own-rate 0.237→0.304 with competition ON) so it becomes the dominant
winner even when `seen` or `heard` is cued (`seen`/`heard` margins go −0.105/−0.088). This is the WTA-does-not-track-the-
cue failure the scoping doc named: a strong competition picks ONE source (here a bias/attractor winner, not the strong
mixed-boosted source as feared, but equally wrong) regardless of correctness. Robust across a recurrent:lateral ratio
grid (recurrent base 50–400 × lateral base 0.5–3.0 at g_comp 1, overlap 0.2, seed 650): `all_dominant_correct` False in
ALL 9 cells — not a bad-ratio artifact.

## Anti-cheats (all reported, all hold)
- (a) `g_comp=0` builds NO competition pathway → a TRUE feedforward null; its M arm reproduces the lesion arm EXACTLY
  (`byte_identical_null_at_g0=True`, exact compare, both seeds), and `min_margin_L=+0.0367` (seed 650) reproduces the
  overlap NO-GO's own L arm at overlap 0.2.
- (b) HONESTY: at recall the source-afferent external current is 0 AND source-afferent firing is 0 on every source
  (`afferent_current_zero`/`afferent_firing_zero`=True); the competition module is parameter-symmetric across sources
  (one scalar wires every source — structurally cannot encode which is cued). Non-vacuity: at the feedforward null a
  forced wrong-source afferent MOVES the dominant winner (`forced_afferent_moves_winner`=True) — the guarded path is
  real. (Under strong competition on seed 651 the attractor can override the forced afferent; expected, not a broken
  guard.)
- (c) no source's own-recall rate collapses (`no_own_rate_collapse`=True everywhere).
- (d) zero-learned-weight instrument control stays strict=False (`control_zero_weight_strict`=False) — no
  stepping-history artifact.

## Verdict + next locus (do NOT defer the capability)
The single-`g_comp` recall-only knob, at any fixed recurrent:lateral ratio, cannot thread winner-bias vs weak-margin at
overlap 0.2 — it fails the decisive `all_dominant_correct` anti-cheat robustly. This is exactly the scoping doc's
`all_dominant_correct=False ⇒ go joint` branch: the attractor amplifies whichever assembly reaches its basin first, and
at co-residency the shared-core / mixed-episode asymmetry means that is NOT reliably the cued source. The capability is
not abandoned; the METHOD (recall-only competition gain) is banked. The next method is the JOINT knob the scoping doc
prescribes — competition gain × storage-side separation (larger uniq fraction / sparser core) co-tuned so the correct
source's uniq-coincidence crosses the basin before a rival's boosted core does. That joint version is NOT built here
(flagged only, per the smoke discipline).

## Honest caveat
Across DIFFERENT `g_comp` builds the neuron set shifts (adding `enable_nmda_recurrent` + the competition pathways moves
the seeded RNG), so `min_margin_L` is not comparable across g_comp rows. Within each build M vs L is clean (same
neurons, gate-only difference), and the decisive `all_dominant_correct` is a pure M-arm within-build fact — so the
verdict is unaffected. The g_comp=0 null is validated independently by `byte_identical_null_at_g0`.
