---
type: finding
status: contributing
date: 2026-08-11
mechanism: EMERGENCE-ENGINE + WORKING-MEMORY HYBRID — the variable-binding WM slot's held-subject population wired as a NEURAL AFFERENT (apical_drive, the substrate's own prediction primitive) into the on-bridge HTM Temporal-Memory emergence engine; the engine's OWN branch prediction reads out. Tests whether the combined spiking system gains a faculty NEITHER piece has alone.
lane: emergence engine / working memory (rung 3 — wire the WM faculty into the emergence stream)
verdict: 6-SEED PARTIAL — the neural fusion GENUINELY COMBINES the two faculties (hybrid held-out 0.641 [min 0.562] BEATS both HTM-alone 0.224 and WM-alone 0.516; BOTH afferent lesions are load-bearing — remove the WM afferent → 0.229 ≈ HTM-alone, remove the HTM afferent → 0.479 ≈ WM-alone; lesion-the-hold collapses to 0.247, so the fusion reads SPIKES not a host store) — real evidence the WM is a LOAD-BEARING faculty inside the emergence engine. BUT it MISSES the strict GO bar (hybrid ≥ max(HTM,WM)+0.20 = 0.716; got 0.641) because the fusion is LOSSY on the SUBJECT dimension: the WM alone latches the subject perfectly (subj 1.000) but under fusion the HTM afferent corrupts it (hybrid subj 0.667), while the class combines cleanly (hybrid cls 0.974 ≥ HTM 0.896). Precisely-named next lever: SEPARATE-CHANNEL fusion (hold subject + class on distinct channels so the HTM's class prediction cannot override the WM's subject latch).
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_emerge_wm_hybrid_derisk.py
artifacts:
  - research/findings/raw/_emerge_wm_hybrid/seed_42.json
  - research/findings/raw/_emerge_wm_hybrid/seed_43.json
  - research/findings/raw/_emerge_wm_hybrid/seed_44.json
  - research/findings/raw/_emerge_wm_hybrid/seed_100.json
  - research/findings/raw/_emerge_wm_hybrid/seed_101.json
  - research/findings/raw/_emerge_wm_hybrid/seed_102.json
instrument: reuse-by-import of the emergence-engine stream + HTM-TM (`_emerge_stream_language_derisk`: vocab/stream/heldout/ngram + build_pool_bridge/OnBridgeLearner) and the variable-binding WM slot (D3 `build_persistent_slot`, RUNG6c `HebbianBinder`). A COMPOSITIONAL stream: verb = combine(subject feature [LONG-RANGE, L+1 back — only the WM can carry it across NOVEL fillers] , class of the last filler [LOCAL, computable on novel fillers — the HTM's job]). Fusion is NEURAL: the WM slot's held-subject population projects an extra apical drive into the HTM engine's context (the same `apical_drive` primitive `coincidence_predict` uses), NOT a host argmax/ensemble over the two predictions. Coordinator-recovered from a deferred agent: agent built the runner + ran the 1-seed smoke, coordinator ran the 6-seed fan + wrote this finding. SIM_BACKEND=numpy; NO sim/ edit.
---
<!--derived-->

# Emergence engine + WM afferent — the neural fusion COMBINES the two faculties (hybrid beats both, both lesions load-bearing) but is LOSSY on the subject latch, missing the strict bar (6-seed PARTIAL)

The variable-binding WM (a gated slow-NMDA slot + Hebbian bind) latches a long-range latent variable and carries it across
novel fillers (held-out 1.000), exactly the thing the HTM emergence engine cannot do (memorises, held-out 0.000). This
de-risk asks the north-star question: does WIRING the WM slot INTO the emergence engine — as a genuine NEURAL afferent —
give the combined spiking system a faculty neither has alone, on a stream that needs BOTH local sequence structure (the
HTM's strength) AND a long-range latent variable (the WM's strength)?

## The compositional task (neither piece alone suffices, by construction)

<!--derived-->
The verb depends on BOTH the subject (long-range, L+1 tokens back — the WM's job) AND the CLASS of the last filler (a
local property computable on NOVEL fillers — the HTM's job). So WM-alone knows the subject but not the class (caps ~1/n_cls);
HTM-alone recovers the local class on held-out but loses the subject across varying fillers. Only a system that carries
BOTH can reach the ceiling. Held-out uses DISJOINT novel filler tuples; a subject-shuffle control checks for leakage.

## Result — 6-seed (`research/findings/raw/_emerge_wm_hybrid/seed_*.json`; chance 0.125, oracle 1.000)

<!--derived-->
Cross-seed mean [min over seeds], held-out branch(verb) EXACT (both subject + class correct), plus the subject-only and
class-only decompositions:

| arm | exact mean [min] | subject | class |
|---|---|---|---|
| HTM-alone | 0.224 [0.188] | 0.203 | **0.896** |
| WM-alone | 0.516 [0.469] | **1.000** | 0.531 |
| **HYBRID** (WM afferent → HTM engine) | **0.641 [0.562]** | 0.667 | **0.974** |
| lesion-WM-afferent | 0.229 [0.141] | — | — |
| lesion-HTM-afferent | 0.479 [0.453] | 1.000 | — |
| lesion-the-hold (recur=0) | 0.247 [0.141] | — | — |
| subject-shuffle (leakage control) | 0.130 [0.062] | — | — |
| n-gram floor ~0.30; chance 0.125 | | | |

**The fusion genuinely COMBINES the faculties** (a real positive, not a null): the hybrid (0.641) beats BOTH HTM-alone
(0.224, +0.417) and WM-alone (0.516, +0.125); and **both afferent lesions are load-bearing** — removing the WM afferent
drops the hybrid to 0.229 (≈ HTM-alone), removing the HTM afferent drops it to 0.479 (≈ WM-alone). **Lesion-the-hold
collapses it to 0.247**, so the afferent is a genuine spiking read (the D3 slow-NMDA sustain is load-bearing), not a host
store — and the external input was ASSERTED zero across the hold+read span. Subject-shuffle 0.130 ≈ chance: no leakage.

**But it MISSES the strict GO bar** (hybrid ≥ max(HTM,WM)+0.20 = 0.716; got 0.641) — and the decomposition says exactly
why: the WM alone latches the subject perfectly (subj 1.000), but **under fusion the HTM afferent CORRUPTS the subject
latch** (hybrid subj drops to 0.667), while the class combines cleanly (hybrid cls 0.974, even above HTM's 0.896). The
loss is entirely on the SUBJECT dimension — the two afferents interfere where they should be independent.

## Scope / honesty + the named next lever (brain-based-only)

<!--derived-->
NO-EXTERNAL-NEEDED: grounded in our own verified components (the emergence-engine stream + the variable-binding WM GO).
A method-positive-with-a-named-residual, not a wall.

- **Genuinely established:** the WM slot CAN be wired into the emergence engine as a NEURAL afferent (apical_drive, not a
  host ensemble), and it is a LOAD-BEARING faculty there (both lesions bite; the hybrid beats both single systems; the
  read is spiking). This is direct evidence toward "the WM faculty is load-bearing on the emergence engine," the north-star
  integration test.
- **The residual, precisely:** the naive coincidence-AND fusion is LOSSY on the SUBJECT dimension — the HTM's local-class
  prediction interferes with the WM's held-subject latch (subj 1.000 → 0.667), so the composition misses the strict +0.20
  margin. Class combines cleanly; subject does not.
- **Named next lever:** SEPARATE-CHANNEL fusion — route the WM's held-subject and the HTM's local-class on DISTINCT
  channels (e.g. the WM afferent gates a subject-specific subpopulation the HTM's class prediction cannot overwrite), so
  the two faculties compose without cross-interference. Then (b) wire the LEARNED role-gate (once the transport-free
  reliability rung lands) so the subject latch is itself emergent. Reuse-by-import; NO sim/ edit.
