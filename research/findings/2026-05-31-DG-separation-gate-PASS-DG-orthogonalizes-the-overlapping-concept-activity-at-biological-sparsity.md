# DG pattern-separation gate: PASS. The hippocampal DG orthogonalizes the overlapping concept-pool activity (between-concept cosine 0.806 -> 0.30 / 0.17) in its biological sparse k-WTA regime, matching the P1 prediction. The convergent three-arc hypothesis -- the substrate needs DG pattern-separation -- is confirmed at the cheap-first gate. Honest caveat: separation is sparsity-dependent (holds at sparse <=~0.05, biological; degrades if DG is driven dense), so the full build must drive DG into the sparse regime via wiring, not hand-tuning.

**Date:** 2026-05-31
**Status:** Cheap-first GATE PASS for the DG-pattern-separation arc (the convergent prescription of three arcs: integrated-loop, D-arc, denoiser). Multi-seed (42/43/44), genuine trained-substrate activity, positive control, isolation verified. Gates the build. The dose-response caveat is load-bearing and carried forward.

## The test + result

The denoiser arc concluded the activity-grounded composition symbol is SEPARABILITY-limited: different concepts' pool activity overlaps at between-concept cosine ~0.82 (within-concept ~0.90), and neither temporal integration (variance) nor the attractor cleanup (needs separable patterns) fixes it. The convergent prescription (three arcs) was a DG pattern-separation stage; the premise matched P1's DG-input regime (0.800 -> DG 0.218). This gate tested whether the hippocampal DG actually orthogonalizes the concept activity.

Probe (`research/findings/raw/_dg_separation_gate.py`, throwaway): a hippocampus-enabled bridge built via the byte-unchanged builder at P1's DG scale (dg=800, dg_pv_basket=240, ec=200), driven by the GENUINE trained-substrate concept activity (the 64-observation-averaged vectors from `activity_level_integration_cache/denoise64_seed{42,43,44}.npz` whose between-concept cosine is the cited ~0.82), via a fixed sparse random concept->ec/dg afferent projection, letting the bridge's real `dg_pv_basket->dg` feed-forward inhibition (the k-WTA) sparsify the DG output. NO lang_input drive anywhere (ec held at the ~0.003 noise floor) -> DG is fed ONLY the concept activity, not the orthogonal lang_input codes.

| regime | POOL between-concept (baseline) | DG between-concept (test) | DG sparsity |
|---|---|---|---|
| headline (sparsity ~0.05) | 0.806 (per-seed 0.806/0.809/0.800) | **0.296** (0.296/0.267/0.258) | 0.044 |
| anchor (sparsity ~0.02, P1-faithful) | 0.806 | **0.169** (0.169/0.160/0.120) | 0.018 |

- Pool baseline 0.806 reproduces the cited ~0.82 (the real problem regime). dg_max 0.596 (headline) -- even the worst-separated concept pair drops from pool ~0.86 to DG ~0.60 (no degenerate near-identical pairs).
- DG between-concept 0.30 / 0.17 brackets P1's validated 0.218. A ~0.5-0.65 cosine drop -- DG genuinely orthogonalizes the overlapping concept activity.
- Positive control: the unmodified P1 validator reproduces in current code (input 0.800 -> DG 0.218). The DG circuit is sound.

## Scrutiny (a PASS scrutinized harder than a FAIL)

An ABANDONED first attempt (the controller caught it independently from an intermediate JSON) had two disqualifying flaws: it fired UNTRAINED pools (between-concept 0.24, not the real 0.82) and DG fired degenerately (silent for ~7 of 12 words; the active pairs at cosine 0.95-0.98; the low mean was a silence artifact). BOTH were fixed in the final probe: (1) it loads the genuine TRAINED-substrate activity (baseline 0.806 reproduced); (2) the drive/FFi were tuned to bring spiking DG into P1's sparse operating band (sparsity 0.018-0.044), DG non-silent (only 1 word at 0), dg_max 0.59 (no degenerate pairs). Isolation verified (ec at noise floor; no lang_input). So the final PASS is on the real regime with a sound, active DG.

## The load-bearing honest caveat (carried to the build)

Separation is SPARSITY-DEPENDENT (expected k-WTA behavior): as DG is driven denser, DG between-concept cosine climbs back toward the input (reported dose-response: sparsity 0.16 -> DG 0.54; 0.81 -> 0.81; 0.95 -> 0.94). Separation holds ONLY in the sparse k-WTA regime (<=~0.05), which is exactly where biological DG operates (P1: 0.007-0.014). The gate reached that regime by tuning drive_scale/ffi_scale. So the honest claim is "DG separates the concept activity WHEN operated sparsely," and the FULL BUILD must ensure DG reaches that sparse regime via proper wiring/learning (the concept->DG afferent + the native FFi balance), not hand-tuning. That is the build's first real risk.

## Disposition + next (the DG-composition build, gated-in)

The convergent hypothesis is confirmed at the gate: DG orthogonalizes the overlapping concept activity to P1's separation regime. The DG arc is gated-in. The build: route concept activity -> DG (sparse k-WTA) -> derive the composition symbol from the DG-SEPARATED activity (instead of the overlapping pool activity) -> re-test whether the DG-grounded symbols clear the 0.80 composition bar at loads {2,3,5} (the bar the raw-activity symbols failed). If yes -> the oracle lookup is biologized via DG pattern-separation (the activity-grounded composable symbol the denoiser arc could not reach), and all three engineering shortcuts are removable -- the artificial-life milestone. If no -> DG separation is necessary but not sufficient (an honest, narrower boundary). The build must (a) drive DG into the sparse regime via wiring, (b) preserve the sparse DG code as the symbol, (c) keep the validated FHRR composition + moat byte-unchanged.

## Discipline

Throwaway probe only (`_dg_separation_gate.py` + 2 result JSONs); no tracked .py modified (builder, validator, sim/, concept_pool_demo, activity_level_integration reused by import byte-unchanged). No bars moved. No autograd. The controller's initial scrutiny correctly flagged the abandoned attempt's flaws; the final result was verified against its JSONs (not taken on the subagent's word); the dose-response caveat is honestly carried forward (the PASS is conditional on the biological sparse regime). The convergent prediction was pre-stated (0.82 -> ~0.22) and met.
