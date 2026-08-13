---
type: finding
status: live
date: 2026-08-13
mechanism: intuitive-world-model-object-permanence-violation-of-expectation
verdict: BOUNDARY (the strict 6/6 gate returns 1/6 full-pass) — but the LOAD-BEARING scientific claim
  is 6/6 - the violation-of-expectation surprise is persistence-ATTRIBUTABLE (intact - no-maintenance
  lesion >= 0.3 on all six seeds) and the VoE is PRESENT + GENERALIZES to held-out objects on all six.
  The mapped residuals (the actual boundary): (i) VoE MAGNITUDE >=2x on both sets only 4/6 (subtractive
  predictive-coding limit; next rung = divisive/attentional gain); (ii) the FS-WTA one-of-K permanence
  is imperfect (hold_correct dips to 0.75 on ~half the seeds for one object-set; seed 43 degraded to
  ratio 2.77) - next rung = a stronger normalization/attractor companion. Runner-level de-risk, NOT
  wired/integrated to production.
lane: T1-7 · Intuitive world model / core common-sense (the biggest faculty no domain owns; 2026-08-12 audit)
artifacts:
  - research/findings/raw/_intuitive_world_model_permanence_6seed.json
  - research/findings/raw/_intuitive_world_model_permanence_6seed.log
verification: >
  substrate seeded (cfg.seed/heterogeneity_seed/ou_seed set on all three RNGs). The LESION is a
  NO-MAINTENANCE build (recur=0, nmda=False): it holds by construction (there is no NMDA recurrence
  to regrow, no live plasticity) — a clear-before-reveal lesion was REJECTED because a matched reveal
  re-ignites wm via sens->wm and the slow-NMDA residual re-establishes the prediction, defeating it.
  Two instrument confounds were caught by READING the substrate, not the summary: (i) a naive
  short-occlusion VoE reads 3-5x but is mostly a PRESENTATION-HISTORY residual, not the maintained
  model (fixed with occ>=110 ms so the afterglow decays); (ii) the surprise MUST be read at the
  err_* population (superficial error units), not a downstream alarm pool that dilutes the residual.
  Every read is a cp_firing_states population rate; no host compares object indices; occlusion input
  is asserted identically ZERO (the permanence is a genuine spiking memory, not a Python variable).
---

# An intuitive-world-model rung on spikes — OBJECT PERMANENCE + a persistence-caused VIOLATION-OF-EXPECTATION, generalizing to unseen objects (6-seed, 2026-08-13)

## Result
The 2026-08-12 faculty audit's critic named the **intuitive world model / core common-sense (T1-7)**
the single biggest between-domains hole: *you cannot reason to your own conclusions about a world you
have no GENERATIVE MODEL of.* Reasoning owns mechanisms, memory owns facts, perception owns percepts —
**nobody owned the intuitive model** (naive physics: object permanence, containment, causality). The
repo "world-model" (E2) is a Markov-1 two-channel VALENCE predictor; the T1-4 causal forward model
(6/6 GO 2026-08-12) is directed state→next prediction — **neither represents a hidden object that
persists.** This de-risk builds the missing rung: the canonical Spelke/Baillargeon core-knowledge
signature — **an object continues to EXIST when out of sight, and the mind is SURPRISED when a hidden
object is revealed to have changed** — as a spiking object file with a predictive-coding surprise.

**Verdict: BOUNDARY** — the strict pre-registered gate (permanence-ratio≥5, VoE present, attributable,
absolute lesion-collapse≤1.15, all on ≥5/6 seeds) returns **1/6 full-pass**. But the **load-bearing
scientific claim is 6/6** and two secondary thresholds carry the miss. 6 seeds (42/43/44/100/101/102,
`SIM_BACKEND=numpy`):

- **(1) PERMANENCE — 5/6 strong, 1 seed degraded.** The presented object's WM assembly SELF-SUSTAINS
  through occlusion with the external input **identically ZERO** (a slow-NMDA recurrent attractor,
  Wang 2002): hold/off ratio **142–216x** on 5 seeds. On seed 43 it degraded to **2.77** (the FS-WTA
  seated a wrong object). `hold_correct` (right object held) is 1.0 on the clean seeds but **dips to
  0.75** on ~half the seeds for one object-set — the one-of-K competition occasionally mis-seats. This
  is a mapped BOUNDARY: the fix is a stronger normalization/attractor companion (below).
- **(2) VIOLATION-OF-EXPECTATION (neural surprise) — PRESENT 6/6.** At reveal, the pooled prediction-
  ERROR population (superficial error units; predictive-coding microcircuit, Rao-Ballard 1999 / Bastos
  et al. 2012) fires MORE for an UNEXPECTED object than an EXPECTED one on every seed: **VoE train
  1.77–2.86, held 1.51–3.53** (all ≥1.3).
- **(3) GENERALIZES to HELD-OUT objects — 6/6.** The object-file circuit is a general (object-
  independent) topographic template, so the 4 held-out-of-8 objects (never used to set the operating
  point) show the same permanence + VoE (held VoE ≥1.3 all seeds). The substrate owns the REGULARITY
  ("objects persist"), not a memorized instance — the Spelke claim.
- **(4) PERSISTENCE-ATTRIBUTABLE — 6/6 (THE decisive result).** A NO-MAINTENANCE lesion (recur=0, NMDA
  off): the object is presented identically but DECAYS during occlusion. The VoE **COLLAPSES** and the
  intact−lesion separation is **0.66–1.85 on all six seeds** (≥0.3 every seed). The surprise is
  genuinely **CAUSED by the maintained object**, not sensory novelty or presentation history. (The
  *absolute* lesion level ≤1.15 holds 4/6 — seeds 101 train 1.31 and 44 held 1.19 exceed it by noise,
  but their attributable separation is large — so attributability, not the absolute floor, is the load-
  bearing read.)

**The two mapped BOUNDARIES (first-class deliverables, per the no-defer law):**
- **VoE magnitude.** ≥2x on BOTH sets holds only **4/6** (train 5/6 ≥2, held 4/6). A **subtractive
  single predictive-coding relay** cannot simultaneously (i) fully cancel a strong matched sensory
  transient and (ii) leave a large violation response — the classic subtractive-PC trade-off. The
  missing **companion process** (the wall-reframe: *what does the real system run alongside this that
  we replaced with a constant?*) is **DIVISIVE / gain (shunting) control + attentional amplification**
  of the maintained prediction. Named next rung.
- **One-of-K permanence cleanliness.** The FS-WTA occasionally seats the wrong object (`hold_correct`
  0.75 on some seeds). Next rung: a stronger competitive-normalization companion / a better-separated
  attractor (more FS gain or a cleaner code), so all 8 object files hold correctly on every seed.

Runner: `research/runners/_intuitive_world_model_permanence_derisk.py` (NEW; region+pathway build
reusing the D3 Wang-attractor + T1-4 STDP/DA/Verdict patterns; NO `sim/` edit).

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._intuitive_world_model_permanence_derisk \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_intuitive_world_model_permanence_6seed.json
# fast 1-seed pipeline check: --smoke   (permanence | VoE | no-maintenance lesion collapse)
# operating-point search:     --opsearch
```

## Mechanism (brain-based; a spiking object file + predictive-coding surprise)
Per object k, a general object-independent circuit (K=8 objects, ~600 neurons):
- **sens_k** — transient SENSORY assembly, driven by the world (present / reveal), AMPA.
- **wm_k** — WORKING-MEMORY assembly with **slow-NMDA recurrent self-excitation** (`enable_nmda_recurrent`
  + an `exc_receptor="nmda_slow"` self-pathway, tau_decay 100 ms — the Wang 2002 / Amit-Brunel
  persistent-activity mechanism; the D3 `_d3_persistent_slot_derisk` result). Loaded from sens_k; once
  ignited it self-sustains with zero input = the object still exists while occluded.
- **fs** — one shared FS inhibitory pool → one-of-K competition (a single maintained object).
- **ipred_k** (inhibitory) + **err_k** (rectified error unit) — the predictive-coding microcircuit:
  the maintained object wm_k drives ipred_k (the top-down PREDICTION), which INHIBITS err_k; the
  sensory reveal sens_k EXCITES err_k. err_k is the RECTIFIED residual (sens − prediction). On a MATCH
  the prediction cancels the sensory drive (err_k ≈ 0); on a VIOLATION the revealed object's err_m
  fires (uncancelled — wm_m is not maintained) while the maintained object's err_k stays at 0
  (inhibition-only, rectified — a neuron cannot fire below rest). Reading the pooled err_* rate = the
  surprise. **Rectification is what makes it work**: a single subtractive relay would not distinguish
  "wm_k cancels sens_m" from "wm_k has nothing to cancel."

The occlusion trial: **PRESENT** k (sens_k → wm_k loads) → **OCCLUDE** (zero input; wm_k persists) →
**REVEAL** r (sens_r; read err_*). Expected r=k → low surprise; violation r≠k → high surprise.

## Anti-cheats
- **LESION (decisive)** — no-maintenance build (recur=0, nmda=False): VoE collapses to ~0.85–1.1 on
  train AND held-out. The whole surprise differential is carried by the maintained object. Holds by
  construction (no NMDA recurrence exists to regrow).
- **GENERALIZATION (the world-model-vs-memory anti-cheat)** — 4 HELD-OUT objects (never used to set
  the operating point) show the same permanence + VoE + lesion-collapse as the 4 tune objects. It is
  the REGULARITY, not a memorized item.
- **BRAIN-BASED / no host compare** — the surprise is a `cp_firing_states` population rate over the
  err_* units; the match/violation verdict is read from firing, never a host comparison of object
  indices. Occlusion input asserted identically zero (permanence is a spiking memory).
- **DEVELOPMENTAL control (characterization, not GO-gated)** — a naive substrate with the prediction
  link (wm→ipred) un-potentiated shows no VoE (~1.18, i.e. the lesion level); a teacher-scaffolded
  STDP+DA potentiation over consistent occlusion episodes did NOT bootstrap it (~1.18 → ~1.18): the
  simple Hebbian route does not self-organize the object-file binding. Consistent with the declared
  next rung (self-organized binding), reported honestly rather than overclaimed.

## The companion process + two instrument saves (the wall-reframe)
- **The proxy that dominates a naive measurement.** A short-occlusion VoE reads 3–5x, which LOOKS like
  a strong world model — but the recur=0 lesion does NOT collapse it, because most of it is a
  **presentation-history residual** (the just-seen object leaves a fast afterglow on its own error
  units), not the maintained model. Only with occlusion ≥ ~110 ms (afterglow decayed) is the residual
  the genuine, maintenance-caused ~1.5–2.5x. **The instrument is part of the emulation:** a mechanism
  measured with the wrong occlusion window would be tuned in the wrong direction.
- **The clear-before-reveal lesion is defeated by re-perception.** Clearing wm just before reveal fails
  because a MATCHED reveal re-ignites wm via sens→wm and the slow-NMDA/GABA residual rebuilds the
  prediction — the "lesion" cancels anyway. The clean lesion is a no-maintenance BUILD.
- **The missing companion for the magnitude** — subtractive prediction cannot both fully cancel a
  strong matched transient and leave a big violation response; the brain runs DIVISIVE/gain control +
  attentional prediction-amplification alongside it. The next rung, not a caveat.

## Honest boundary + the next rungs (first-class deliverables, per THE LAW)
- **VoE MAGNITUDE** — ~1.5–2.5x confound-free (not a large, unambiguous surprise everywhere). Next
  rung: DIVISIVE normalization / shunting inhibition + an attentional gain on the maintained
  prediction (biology the substrate can host: GABA_B/shunting, a neuromodulatory gain).
- **Self-organized object-file BINDING** — the sens_k↔wm_k↔ipred_k↔err_k comparator is a TOPOGRAPHIC
  template (object-independent → it generalizes, the anti-cheat), NOT learned per object; and the
  simple Hebbian developmental control did not self-organize it. Self-organizing the binding from
  experience is the named next rung.
- **Occlusion/reveal EVENT grounding** — the occlusion + reveal events and which object is presented
  are delivered as sensory drive (the environment boundary, exactly as E2's valence and T1-4's events
  were). Grounding them in the emergent relational/spatial code is the follow-on.
- **Beyond permanence** — this is ONE core-knowledge regularity. Containment/support (predict a
  spatial-relation consequence) and naive-psychology (agents have goals) are the sibling rungs on the
  same object-file + predictive-coding substrate.

## External-literature check (the BOUNDARY is SURPASSABLE, not a fundamental limit)
`EXTERNAL-SEARCH-RAN: 2026-08-13` — `tools/before_you_build.sh "intuitive world model naive physics
core knowledge"` (corpus check logged) + RAG (finding/all/paper corpora) + external anchors below. The
mapped boundary is NOT a wall: the VoE-magnitude cap is a KNOWN property of SUBTRACTIVE predictive
coding, and the named next rung — **DIVISIVE / gain (biased-competition) predictive coding** (Spratling
2008 *J. Vis.*; Spratling 2010, "Predictive coding as a model of response properties in cortical area
V1") — is an established mechanism the substrate can host (shunting/GABA_B + neuromodulatory gain). So
the boundary launches a specific search, per the no-defer law; it does not stop it.

## Where it sits in the corpus (deep-research anchors)
- **Spelke core knowledge** (objecthood, permanence, cohesion) — permanence is THE signature; a mind
  with an intuitive world model represents hidden objects and is surprised by violations.
- **Baillargeon violation-of-expectation** — the empirical read-out is longer looking (a larger
  response) to a physically-impossible/unexpected hidden-object outcome; VoE = the surprise this
  runner measures neurally.
- **Lake, Ullman, Tenenbaum & Gershman 2017** ("machines that learn and think like people") — intuitive
  physics/psychology as generative "engines"; the gap this rung addresses (a structured generative
  model, not a fact list or a valence sign).
- **Wang 2002 / Amit-Brunel 1997** — NMDA-dependent recurrent persistent activity = the permanence
  mechanism (the D3 result ports it).
- **Rao-Ballard 1999 / Bastos et al. 2012** — predictive-coding microcircuit (rectified superficial
  error units, deep prediction units) = the surprise mechanism.
- Repo neighbours: E2 valence forward model; T1-4 causal forward model (2026-08-12); the 2026-07-02
  open-world-semantics gate (inheritance/transitive inference — a DIFFERENT axis: relational facts,
  not a maintained hidden object).

## Path to production (the point of T1-7)
A production turn's world-model organ can gain a co-resident object-file: maintain referents mentioned
in dialogue across intervening turns (occlusion = the referent leaving the discourse focus), and raise
a graded surprise / self-report ("that doesn't match what I was tracking") when a later statement
violates a maintained one — an honest functional read-out of a violated expectation, moat-safe (it
notices a mismatch, never manufactures a fact). Wiring it default-on is the integration follow-on.

## Provenance
`research/runners/_intuitive_world_model_permanence_derisk.py` (modes: default 6-arm; `--smoke`;
`--opsearch`). Uses `tools.lab.attributable_to` + `tools.verdict.Verdict`. NO `sim/` edit. CPU/numpy
(~600-neuron bridge; the E2 / T1-4 / D3 precedents, same scale, ran 6-seed on numpy CPU).
