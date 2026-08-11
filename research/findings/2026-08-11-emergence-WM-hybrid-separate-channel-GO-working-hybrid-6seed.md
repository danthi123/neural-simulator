---
type: finding
status: go
date: 2026-08-11
mechanism: SEPARATE-CHANNEL WM+HTM HYBRID — the variable-binding WM's held-subject and the HTM emergence engine's local-class are decoded on DISTINCT neural channels (the WM owns a subject-memory column the HTM never writes; the HTM predicts class on its own column), then combined by a LEARNED DENDRITIC CONJUNCTION bridge (each verb(s,c) column potentiates coincidence synapses from BOTH channels). Closes the rung-3 partial's subject-corruption.
lane: emergence engine / working memory (rung 3b — a WORKING WM+HTM neural hybrid)
verdict: 6-SEED GO — the separate-channel fusion WORKS and closes the rung-3 partial. Hybrid held-out exact 0.974 [min 0.938] (chance 0.125) BEATS both HTM-alone 0.224 and WM-alone 0.516 by +0.46, clearing the strict max+0.20 bar (0.716). Critically it PRESERVES THE SUBJECT under fusion — hybrid subject 1.000 [min 1.000] (the rung-3 coincidence-AND fusion, re-run here as `old_fusion`, corrupted it to 0.667) — while keeping the class clean (0.974). All teeth bite: lesion-WM-channel → 0.237 (≈ HTM-alone), lesion-HTM-channel → 0.479 (≈ WM-alone), lesion-the-hold → 0.266 (collapses; the fusion reads SPIKES), the UNTRAINED conjunction bridge → 0.096 (< chance; the neural bind is LOAD-BEARING, not a host lookup), subject-shuffle → 0.000 (no leakage). The WM is now a WORKING, load-bearing faculty combined into the emergence engine by a genuine neural conjunction.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_emerge_wm_hybrid_sepchan_derisk.py
artifacts:
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_42.json
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_43.json
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_44.json
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_100.json
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_101.json
  - research/findings/raw/_emerge_wm_hybrid_sepchan/seed_102.json
instrument: reuse-by-import of the emergence-engine stream + HTM-TM + the variable-binding WM slot (as `_emerge_wm_hybrid_derisk`). Change ONLY the fusion (coincidence-AND → separate-channel + a learned neural conjunction). The COMBINATION is neural: `train_conjunction` potentiates each verb column's coincidence synapses from the WM-subject and HTM-class channel populations; `conj_read` primes the bridge from the decoded channel populations and reads the verb column's spikes. The per-channel subject/class IDENTIFICATION is a host `np.argmax` read instrument over each channel's spikes (subject_hat = the slot's `cp_firing_states` argmax; class_hat = argmax over the HTM's apical drive) — a read-out INSTRUMENT, not neural WTA (same status as the lane-b/c reads). Coordinator-recovered from a deferred agent (agent built the runner; coordinator ran the 6-seed fan + wrote this finding). SIM_BACKEND=numpy; NO sim/ edit.
---
<!--derived-->

# Separate-channel WM+HTM hybrid — a WORKING neural fusion: hybrid 0.974, subject PRESERVED (1.000, up from the coincidence-AND fusion's 0.667), 6-seed GO

The rung-3 hybrid (`2026-08-11-emergence-engine-plus-WM-afferent-hybrid-PARTIAL-...`) showed a neural WM→HTM fusion GENUINELY
COMBINES the two faculties (hybrid 0.641 beats both single systems, both lesions load-bearing) but MISSED the strict bar
for one precise reason: the naive coincidence-AND fusion is LOSSY on the SUBJECT — the HTM afferent corrupts the WM's
held-subject latch (subj 1.000 → 0.667) while the class combines cleanly. This de-risk fixes that by SEPARATING the
channels, and it clears the bar.

## The fix — separate channels + a learned neural conjunction (not a host ensemble)

<!--derived-->
The WM's held-subject drives a DEDICATED subject-memory column the HTM never writes to (so the HTM's class prediction
cannot overwrite it); the HTM predicts the local class on its OWN column. The verb read-out is a LEARNED DENDRITIC
CONJUNCTION: a conjunction bridge in which each `verb(s,c)` column potentiates incoming coincidence synapses from BOTH
the WM-subject column `wm[s]` AND the HTM-class column `clsrd[c]`; at decision time the bridge is primed from the two
decoded channel populations and the verb column's SPIKES are read. This is NOT a host argmax/ensemble over two
predictions — the `conj_untrained` control (an UNTRAINED conjunction bridge) collapses to 0.096 < chance, proving the
learned neural bind carries the combination.

## Result — 6-seed (`research/findings/raw/_emerge_wm_hybrid_sepchan/seed_*.json`; chance 0.125)

<!--derived-->
Cross-seed mean [min], held-out branch(verb) EXACT + the subject/class decomposition:

| arm | exact [min] | subject | class |
|---|---|---|---|
| HTM-alone | 0.224 | 0.203 | 0.896 |
| WM-alone | 0.516 | 1.000 | 0.531 |
| old coincidence-AND fusion (rung-3, re-run) | 0.641 [0.562] | 0.667 | 0.974 |
| **HYBRID separate-channel** | **0.974 [0.938]** | **1.000 [1.000]** | 0.974 [0.938] |
| lesion-WM-channel | 0.237 [0.156] | — | — |
| lesion-HTM-channel | 0.479 [0.453] | — | — |
| lesion-the-hold | 0.266 [0.156] | — | — |
| conj-untrained (bind lesion) | 0.096 [0.047] | — | — |
| subject-shuffle (leak control) | 0.000 [0.000] | — | — |

**GO on every criterion:** hybrid 0.974 ≥ max(HTM 0.224, WM 0.516) + 0.20 = 0.716; the SUBJECT is fully preserved under
fusion (1.000, vs the coincidence-AND fusion's 0.667 — the exact defect the separation targets); the class stays clean
(0.974). Both channel lesions are load-bearing (each collapses the hybrid to the corresponding single-system baseline),
lesion-the-hold collapses it (the fusion reads the slot's SPIKES; the external input was asserted zero across the span),
the untrained conjunction bridge collapses below chance (the neural bind is load-bearing), and subject-shuffle is 0.000
(no topic→answer leakage).

## Scope / honesty (brain-based-only)

<!--derived-->
- **Established (the north-star integration, working):** the variable-binding WM is now combined INTO the emergence
  engine by a genuine LEARNED NEURAL CONJUNCTION to solve a compositional task NEITHER piece can alone (HTM 0.224,
  WM 0.516 → hybrid 0.974), with the subject preserved and every faculty lesion load-bearing. The WM is not decorative:
  lesioning its channel drops the hybrid to the HTM-alone baseline.
- **The host residual (named, same status as lane-b/c):** the per-channel subject/class IDENTIFICATION is a host
  `np.argmax` read instrument over each channel's spikes (subject_hat = the slot's `cp_firing_states` argmax; class_hat =
  argmax over the HTM's apical drive). The COMBINATION is neural; the per-channel winner READ is a host argmax read-out.
  Replacing those two reads with a neural WTA (the lever-3 stabilizer / rung-2 emergent-WTA machinery) is the burndown.
- **Named next:** wire the LEARNED role-gate (once transport-free reliability lands, or with the working aligned/host
  role-gate as scaffold) so the subject latch is itself emergent; and replace the channel-read argmax with neural WTA.
  Reuse-by-import; NO `sim/` edit.
