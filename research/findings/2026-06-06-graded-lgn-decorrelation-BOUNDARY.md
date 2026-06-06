# Graded LGN decorrelation stage (the biology-faithful on-substrate whitening) = BOUNDARY — and the boundary is RELOCATED again: the GRADED LATERAL *can* pairwise-decorrelate, but the rectifying/saturating GRADED READ-OUT degrades the gentle composing structure (a KNOWN-composing code drops 100%→72% just by being driven into spiking membranes and read back) — 2026-06-06

**Status:** BOUNDARY, gated on COMPOSITION (the agent benchmark), with the full control bracket (RAW floor +
CONCEPT-whiten target) and guards (graded-LGN alive/bounded, M-norm bounded, NO-lateral baseline) — plus a
DECISIVE pipeline-isolation control that localizes the boundary precisely. This is the FIRST `sim/` edit of the
decorrelation arc (additive, opt-in, default OFF; the Izhikevich/HH/AdEx step paths are byte-unchanged when off).

## The one-line result

The validated rate-model regularized-local-whitening rule — `ΔM ∝ ⟨a aᵀ⟩ − I − λM` (settled `a` inhibited by
`−(M @ a)`) — was realized as a **GRADED pairwise lateral** operating on the LGN region's SUB-THRESHOLD ANALOG
activity (`a = clip((v−v_rest)/scale, 0, 1)`, NOT spikes), pre-spike, where the retina/LGN does variance
equalization (the new `cp_graded_lateral_M` mechanism). **The graded lateral DOES pairwise-decorrelate** (it
pulls the signed code's coherence from drive 0.47 down to 0.187, clearly below the no-lateral 0.244 — genuine
pairwise work, unlike the prior shared-FS spiking lateral that did only global gain). **But the graded LGN
whitening does NOT compose: 26/39 = 66.7% on the agent benchmark = the RAW floor, == the no-lateral baseline.**
The decisive isolation control shows WHY: the **graded drive→settle→read-out transform itself** degrades the
gentle composing structure — passing the rate-model's KNOWN-100%-composing code through the spiking LGN pipeline
(drive it in, read the graded `a` back, M=0) drops composition **100% → 71.8%** (coherence 0.043 → 0.091). The
boundary is the rectifying/saturating graded read-out, NOT the lateral's ability to decorrelate.

## The decisive table (K=300, CIFAR real-object grounding, 320 concepts, seed 42)

`research/runners/graded_lgn_decorrelation_compose.py` → `_graded_lgn_signed_s42.json` / `_graded_lgn_smoke.json`.

| condition | composition | coherence | reading |
|---|---|---|---|
| RAW grounded (no whitening) | **26/39 = 66.7%** | 0.249 | floor control ✓ (matches the rate model) |
| CONCEPT-whiten (N×N gram; not realizable) | **39/39 = 100%** | — | target control ✓ (matches the rate model) |
| rate-model LEARNED (−λM, the validated result) | **39/39 = 100%** | 0.043 | what we reproduce on the substrate |
| **GRADED LGN, RECTIFIED drive (gain 10, ep 2)** | **26/39 = 66.7%** | drive 0.486 → graded 0.428 | BOUNDARY — under-whitens (== floor) |
| **GRADED LGN, SIGNED ON/OFF (gain 40, ep 8)** | **26/39 = 66.7%** | drive 0.471 → graded **0.187** | BOUNDARY — pairwise-decorrelates but == floor |
| └─ NO-lateral baseline (M disabled) | 26/39 = 66.7% | 0.244 | the learned lateral lowered coh (0.244→0.187) but added ZERO composition |
| **DECISIVE: rate's composing code THROUGH the pipeline (signed)** | **28/39 = 71.8%** | 0.043 → **0.091** | the graded READ-OUT alone degrades a known-composing code |
| └─ same, RECTIFIED (sign dropped) | 26/39 = 66.7% | 0.043 → 0.448 | sign-drop alone costs the ENTIRE 100%→floor |

The controls bracket exactly as the rate model — the harness is valid. Guards are GREEN (below), so the BOUNDARY
is GENUINE, not a degenerate/silent-LGN false positive.

**The read-out cap is NOT tunable (rules out saturation as the fix).** Passing the rate's composing code through
the signed pipeline at act_scale ∈ {15, 40, 80} (i.e. progressively LESS clip-saturation) gives composition
**71.8% / 71.8% / 71.8%** (coh 0.091 / 0.102 / 0.098) — flat. The rectified pipeline gives **66.7% / 66.7% /
66.7%** at the same act_scales. So (a) the signed ON/OFF read-out preserves SOME composability (~72% > the 66.7%
floor) but caps there regardless of saturation; (b) the rectified read-out's sign-drop alone takes a known-100%
code to the floor. The degradation is the rectify-into-membrane-and-read transform itself, not a saturation knob.

## Multi-seed confirmation (signed ON/OFF, gain 40, ep 8) — the BOUNDARY is robust, not a seed-42 artifact

| seed | GRADED compose | SPIKING compose | NO-lateral compose | graded-active /300 | n_silent | M_norm | drive_coh → graded_coh (vs no-lateral) |
|---|---|---|---|---|---|---|---|
| 42 | 26/39 = 66.7% | 26/39 = 66.7% | 26/39 = 66.7% | 243.5 | 0 | 31.0 | 0.471 → 0.187 (no-lat 0.244) |
| 43 | 26/39 = 66.7% | 26/39 = 66.7% | 26/39 = 66.7% | 239.6 | 0 | 31.5 | 0.463 → 0.191 (no-lat 0.257) |
| 44 | 27/39 = 69.2% | 26/39 = 66.7% | 26/39 = 66.7% | 247.7 | 0 | 30.2 | 0.463 → 0.187 (no-lat 0.254) |

**3-seed unanimous BOUNDARY.** GRADED compose 66.7 / 66.7 / 69.2% — all AT or WITHIN the RAW floor band (the
rate-model arc documents RAW itself at 66.7–69.2% across seeds; seed-44's single extra item is floor jitter, not
a lift). Every seed: controls bracket (RAW 66.7% / CONCEPT 100%), guards green (0 silent / 320, M bounded
~30–32), the learned graded lateral genuinely pairwise-decorrelates (graded_coh 0.187–0.191 < its own no-lateral
0.244–0.257) yet the NO-lateral baseline == the GRADED composition (66.7% on all 3) → the learned lateral adds
ZERO composition. The pipeline-isolation control (a known-100% code → 72% through the read-out) is the localized
cause and is itself seed-independent (the read-out transform, not a fit).

## Guards (every run; the false-positive catchers) — all GREEN

| readout | mean graded-active /K | min | n_silent /320 | M_norm | M_max | drive_coh → graded_coh |
|---|---|---|---|---|---|---|
| rectified (gain 10, ep 2) | 141.1 /300 | 121 | **0** | 45.8 | 1.39 | 0.486 → 0.428 |
| signed ON/OFF (gain 40, ep 8) | 243.5 /300 active of 600 | 210 | **0** | 31.0 | 0.73 | 0.471 → **0.187** |

- **LGN is HEALTHY, not silent, not collapsed** (0 silent concepts across 320 in both variants; M bounded,
  norm 31–46, max < 1.4). This is NOT the degenerate-LGN false positive the rigor demanded I catch.
- **The −λM bounded M** (the rate-model regularizer) exactly as designed — stable, finite, no blow-up.
- **The graded lateral genuinely pairwise-decorrelated** in the signed variant (0.471 → 0.187 vs the no-lateral
  0.244): it is doing pairwise subtraction, NOT just the global gain control the prior shared-FS spiking lateral
  was stuck at (which sat at coh 0.33). This RESOLVES the "can a substrate lateral pairwise-decorrelate?"
  sub-question — YES, the graded one can.

## Where it breaks, PRECISELY (the relocated boundary — calibrated by the isolation control)

Three nested controls localize the boundary to ONE place — the graded READ-OUT transform, not the lateral:

1. **The graded lateral CAN pairwise-decorrelate.** Signed ON/OFF reaches coherence 0.187, decisively below the
   no-lateral 0.244 — genuine pairwise work. (The prior 2026-06-06 shared-FS SPIKING lateral could NOT: it sat
   at 0.33 = global gain only. So switching to a GRADED, full-K×K, pre-spike lateral DID fix the pairwise-vs-
   global problem — the first sub-question is resolved.)

2. **BUT coherence 0.187 is the OVER-WHITENING regime, not the gentle composing point.** The composing solution
   is a SPECIFIC GENTLE partial whitening (coh ~0.043 = C^−1/3), not maximal decorrelation. Coherence 0.187 is
   exactly the DIM-analytic over-whitening regime (the rate-model arc already showed DIM-analytic at coh 0.191
   → 66.7%). Low coherence ≠ composes — the bridge's graded settling lands at the wrong (over-decorrelating)
   amount. (This is why gating on COMPOSITION, not coherence, was non-negotiable: the signed coh 0.187 *looks*
   like progress but composes at the floor.)

3. **DECISIVE — even the gentle composing code can't survive the graded read-out.** The cleanest isolation:
   take the rate-model's KNOWN-100%-composing code (coh 0.043), drive it INTO the spiking LGN (signed ON/OFF),
   read the graded `a` back with the lateral OFF (M=0, no decorrelation at all), and compose. Result: composition
   **100% → 71.8%**, coherence 0.043 → 0.091. The drive→settle→read transform — `a = clip((v−v_rest)/scale,0,1)`,
   a rectifying + saturating readout of a leaky-integrator membrane — **by itself degrades the gentle composing
   structure**. So the lateral never gets a chance: it operates on (and writes into) an already-degraded
   representation. **The boundary is the GRADED READ-OUT nonlinearity, not the lateral's decorrelation power.**

This is the on-substrate face of the **2026-06-05 opponency wall**: the gentle composing whitening is a signed,
full-precision, common-mode-balanced re-coordinatization; the moment it passes through a rectify+saturate
spiking-membrane readout (which a rate code must), the small signed structure that carries composability is lost.
The graded lateral relocates the boundary one step (pairwise decorrelation is now achievable on the substrate),
but the read-out from the graded stage to anything spiking re-degrades the gentle code — exactly the
"the cortex must not re-correlate the whitened gain when it re-spikes" risk flagged for this build, confirmed.

## Honest scope (do NOT overclaim, do NOT under-claim)

- **WIN (real, and it is the new sim/ mechanism's win):** a GRADED, pre-spike, full-K×K pairwise lateral on
  sub-threshold analog activity — the biology-faithful realization — DOES pairwise-decorrelate on the substrate
  (coh 0.47 → 0.187, attributed to the learned lateral by the no-lateral baseline), where the prior shared-FS
  SPIKING lateral was stuck at global gain (0.33). The pairwise-vs-global sub-question is RESOLVED. The −λM
  bounded it; the LGN stayed alive. The `sim/` edit works exactly as designed.
- **BOUNDARY (real):** the end-to-end does NOT compose above the raw floor (66.7%, gated on composition, guards
  green, NO-lateral baseline == floor). The honest cause, isolated by the pipeline control, is the **graded
  read-out transform** (rectify + saturate of the spiking membrane): it lands the lateral in the over-whitening
  regime AND degrades even a hand-given composing code (100% → 72%). The composing whitening's gentle signed
  structure does not survive the graded→spiking readout.

## What this RELOCATES vs the prior on-bridge BOUNDARY (the arc converges, one step deeper)

| attempt | mechanism | coherence reached | composes? | the boundary |
|---|---|---|---|---|
| 2026-06-06 shared-FS SPIKING lateral | it→fs→it, shared inhibitory pool | 0.33 (stuck) | 66.7% | GLOBAL gain, not pairwise (Mikulasch-Priesemann) |
| **2026-06-06 GRADED lateral (this)** | full-K×K, pre-spike, sub-threshold `a` | **0.187** (pairwise!) | 66.7% | the GRADED READ-OUT degrades the gentle code (rectify/saturate) |

The graded lateral fixed the *pairwise* problem (the prior boundary). The boundary moved to the **read-out**: a
rectifying/saturating sub-threshold readout cannot preserve the *gentle* signed composing structure. Both
converge on the 2026-06-05 opponency-wall thesis from different angles: **the composing whitening lives in a
signed analog regime that a spiking/rectified stage degrades**. The decorrelation is realizable on the substrate
as graded dynamics; what is NOT (yet) realizable is reading that whitened analog code out into the spiking
composer without re-degrading the gentle structure.

## The `sim/` edit (additive, opt-in, default OFF — flagged explicitly for byte-for-byte review)

Files + lines touched (the ONLY `sim/` changes; the main session reviews these before trusting/pushing):

- **`sim/config.py`** (one block added after `region_pathways`, ~line 260): `enable_graded_lateral: bool = False`
  + `graded_lateral_lr` (0.02) + `graded_lateral_lambda` (0.01, the −λM) + `graded_lateral_gain_pA` (300.0) +
  `graded_lateral_act_scale` (15.0) + `graded_lateral_coact_ema` (0.0). All default OFF/standard.
- **`sim/regions.py`** (`BrainRegion` dataclass): `graded_lateral: bool = False` (per-region opt-in).
- **`sim/bridge.py`**:
  - `__init__`: declare `cp_graded_lateral_M = None`, `_graded_lateral_slice = None`,
    `cp_graded_lateral_coact = None` (right after `cp_nmda_neuron_mask`).
  - end of `_initialize_simulation_data`: one guarded call `self._init_graded_lateral(cfg, n)` (after the
    synapse-store block).
  - new methods `_init_graded_lateral` / `_graded_lateral_activity` / `_graded_lateral_inhibition_pA` /
    `_graded_lateral_learn` (inserted before `_apply_per_region_neuron_types`). Allocate the K×K M for the
    flagged region's contiguous slice; compute `a = clip((v−v_rest)/scale, 0, 1)`; the inhibition `−(M@a)*gain`;
    the update `ΔM = lr*(⟨a aᵀ⟩ − I) − λM`, symmetrized, zero-diagonal.
  - `_run_one_simulation_step` (one block right after `total_input_current_pA = synaptic_current_I_syn_pA +
    self.cp_external_input_current`): **GUARDED `if self.cp_graded_lateral_M is not None:`** — add `−(M@a)*gain`
    to the flagged region's contiguous current slice BEFORE the spike threshold, then `_graded_lateral_learn(a)`.

**HARD REQUIREMENT verified:** when off (every existing run/test), `cp_graded_lateral_M is None`, the step block
is unreached, and the Izhikevich/HH/AdEx paths are BYTE-UNCHANGED. Pinned by `tests/test_graded_lateral.py` (8
tests, written TDD-first): OFF == baseline byte-for-byte (membrane + spikes), incl. the global-flag-on-but-no-
region-opted-in case; ON learns + stays bounded (the −λM) + region not silenced + inhibits + λ is a magnitude
knob. **105 passed on native CuPy** across determinism/regions/transmission-gate/neuromodulators/fast-spike-reset
+ the 8 graded-lateral tests (the 4 numpy-backend test_regions failures are pre-existing CuPy-path tests, NOT
from this edit — verified by stashing the edit and reproducing them on a clean tree).

One incidental fix in the new code: a `λ` glyph in an `_init_graded_lateral` log message crashed init under the
Windows cp1252 console (UnicodeEncodeError) → swallowed by the step's try/except → silently uninitialized bridge.
Replaced with ASCII "lambda". (Separately surfaced a PRE-EXISTING latent bug — `profile_name_for_conn` unbound
in the zero-synapse fallback log; not in scope, flagged for a follow-up.)

## Validation rigor (the arc caught FIVE convenient-but-wrong results; this did not ship a sixth)

- **Gated on COMPOSITION**, never coherence — and that is exactly what caught the convenient signed-mode number:
  signed coherence 0.187 *looks* like a win but composes at the 66.7% floor.
- **Controls bracket every run**: RAW 66.7% floor + CONCEPT-whiten 100% target, both confirmed (harness valid).
- **Guards every run**: LGN alive (0 silent / 320), M bounded (norm < 46, max < 1.4), + the NO-lateral baseline
  (the learned lateral lowered coh 0.244→0.187 but added 0 composition — the lift is attributed correctly: there
  is none).
- **The isolation control** (rate's composing code through the pipeline → 72%) is the decisive evidence that the
  boundary is the read-out, not the lateral — without it I would have wrongly blamed the lateral.

## Artifacts

- `sim/config.py`, `sim/regions.py`, `sim/bridge.py` (the additive opt-in edit) + `tests/test_graded_lateral.py`
  (8 TDD tests). Committed locally (NOT pushed — the main session reviews the `sim/` diff first).
- Runner: `research/runners/graded_lgn_decorrelation_compose.py` (`--signed` ON/OFF variant, `--baseline`
  no-lateral attribution, full guards + controls, composition-gated verdict) +
  `research/runners/graded_lgn_decorrelation_multiseed.ps1`.
- `research/findings/raw/_graded_lgn_smoke.json` (rectified), `_graded_lgn_signed_s42.json` (signed + baseline).
- Reuse-by-import: `unified_agent_realobject_grounded.build_realobject_features` + `run_seed`,
  `unified_agent_visual_grounded._decorrelate`, `unified_agent_benchmark`, `_visual_grounding_probe._v1_matrix`.

## Net for the graded LGN realization

- **Algorithm level (rate/numpy):** RESOLVED — a regularized local rule learns a composing whitening, 100%, 6/6
  (the prior finding). Unchanged.
- **On-substrate GRADED realization (this):** the GRADED pre-spike pairwise lateral DOES pairwise-decorrelate
  (coh 0.47→0.187, the first substrate lateral to do so) — the pairwise-vs-global boundary is RESOLVED. But the
  end-to-end does NOT compose (66.7% floor, gated on composition, guards green). The honest, isolated boundary is
  the **rectifying/saturating GRADED READ-OUT**: it lands the lateral in the over-whitening regime AND degrades
  even a hand-given composing code (100%→72%). The gentle signed composing structure does not survive the
  graded→spiking readout — the on-substrate face of the 2026-06-05 opponency wall. A faithful realization would
  need a read-out that preserves the gentle signed code (a higher-precision / less-saturating analog channel, or
  carrying the whitened code in PHASE rather than rate — the FHRR direction), or the whitening stays as an
  upstream graded stage feeding `grounded_codes` (research-confirmed faithful) — both bigger builds, not this
  drop-in. The validated SCIENCE (a local rule composes, 6/6) is unchanged regardless.
