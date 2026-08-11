---
type: finding
status: contributing
date: 2026-08-11
mechanism: spiking FS-WTA (lateral-inhibition) selection replacing the host `if/elif` threshold-binning that renders the SEAM-C affect tone token in the mouth
lane: E-language / INTEGRATION (brain-based-only mouth burn-down)
seeds: [42, 43]
seed-waiver: labelled 2-seed SMOKE; the decisive 6-seed command is returned but NOT run this pass (build + smoke only)
artifacts:
  - research/findings/raw/mouth_tone_wta/tone_wta_smoke_2seed.json
runner: research/runners/_mouth_tone_wta_readout_derisk.py
instrument: a stand-alone read-out de-risk that projects the affect ladder's spike-rate differential onto K=7 tone-level pools via a labelled-line afferent place-code (derived from the host render's OWN bands), then resolves the winning tone level on SPIKES with the validated FS-WTA (`build_fswta_score_bridge`/`fswta_drive` from `_d3_spiking_attractor_derisk`); parity measured vs the deployed host `_graded_tone_level`; SIM_BACKEND=numpy; cfg.seed-controlled; NO `sim/` edit.
---

# Mouth burn-down (smoke) — the SEAM-C tone token the mouth SPEAKS is now decided on SPIKES

The live chat's affect coloring (SEAM-C in `_stageA_full_integration_derisk`) already produces the tone SIGNAL neurally
— the spike-rate differential `rate(aff_pos_readout) - rate(aff_neg_readout)` off the Koulakov graded-affect ladder. But
the step that turns that differential into the DISCRETE tone the mouth articulates ("warmly, gladly …" / "curtly …" /
"coldly, reluctantly …") was a HOST `if/elif` threshold-binning: `_graded_tone_level(differential)`. Per the
BRAIN-BASED-ONLY standard a host threshold/argmax over a neural signal is a shortcut — the SELECTION of the tone the
mouth speaks was being made by Python, not by the brain.

This routes that selection through a SPIKING FS-WTA read-out: the differential is encoded onto K=7 tone-level pools by a
labelled-line afferent place-code (a legitimate host INPUT, same status as a reservoir's `W_in` / the retinal render —
it faithfully reproduces the host render's OWN band boundaries, read off `_graded_tone_level` itself), and a shared
inhibitory FS pool RESOLVES the winner on spikes (winner fires first → recruits FS → suppresses the runners-up → a clean
one-of-K). The tone for the turn is thus decided by the NETWORK's lateral inhibition, not a host threshold. This is the
same standing FS-WTA selector the patient-word burn-down used (2026-08-11, 6/6 GO) — the sibling render on the same day.

## Result — `research/findings/raw/mouth_tone_wta/tone_wta_smoke_2seed.json`

<!--derived-->

A 2-seed SMOKE (42, 43), both individually GO on the runner's per-seed gate:

- `parity_live = 1.000` — on the REAL ladder differentials (the exact signal the live SEAM-C render consumes:
  positive-appraisal held reads at 6 appraisal levels 0.0→1.0, spanning host tone levels {0, +2, +3}), the spiking-WTA
  token equals the host token on every one.
- `parity_synth = 0.992` — on a dense 121-point band sweep over the differential operating range, one single near-a-band-
  boundary tie is the only mismatch.
- `clean_frac = 0.975` — the FS-WTA produced a clean one-of-K winner (>20% margin over the runner-up) on 97.5% of drives.
- `shuffle_parity = 0.145` vs `chance = 0.200` — permuting the place-code collapses agreement with the host to chance:
  the WTA reads the ACTUAL differential, not a host leak (85.4% of the agreement attributable to the true place-code).
- `lexicon_confined = True`, `fm4_saturates = True` — an extreme differential (±10) only SATURATES the tone at ±3;
  every spiking output is a member of the tone lexicon.

The artifact's own top-level `verdict` is `UNDEFINED` by design — the generalisation gate requires ≥6 seeds, so a 2-seed
smoke asserts nothing to earn. This finding is therefore a build + smoke; the DECISIVE 6-seed run is the pending step
(command below).

## Scope / honesty

<!--derived-->

- Additive / default-OFF / NO `sim/` edit — reuse-by-import. The host render `_graded_tone_level` stays the deployed
  default; the spiking render is the opt-in. This burns down the SELECTION shortcut only.
- Faithful to the host, quirks and all: `tol == step == 0.03` in the deployed `_graded_tone_level` makes tone levels ±1
  UNREACHABLE (it jumps 0 → ±2). The place-code is derived from the host function, so the spiking render inherits
  EXACTLY the host's reachable set {0, ±2, ±3} — parity is against the real production behaviour, not an idealisation.
- Moat/honesty (FM4) intact by construction: the tone render is invoked ONLY on an already-matched, already-honest answer
  (`raw is not None` in `_colored_answer_graded`); the WTA's output is confined to the 7-tone lexicon, so it colors
  WITHIN the decided band and can never flip abstain→assert or touch the answer content or the cue-match moat. Verified in
  the smoke (lexicon-confinement + saturation-not-leak).
- RESIDUAL host pieces, DECLARED (the named next mechanisms, honest-negative surface): (i) the afferent place-code is
  host-DESIGNED, not learned/self-organized — a spiking-structure shortcut (`feedback_spiking_structure_must_self_organize`);
  (ii) the level→word lexicon lookup (`GRADED_TONE_LEVELS`) is a fixed host table — the word forms themselves are not yet
  the brain's own production. This closes the tone-SELECTION shortcut; the tone-ENCODING and tone-LEXICALISATION remain.

## What this burns down, and what remains

- BURNED DOWN: the mouth's tone-token SELECTION — previously a host `if/elif` over a neural differential — is now a
  spiking one-of-K WTA. Together with the same-day patient-word read-out, both host renders that decided WHICH discrete
  symbol the mouth emits from a neural signal are now on spikes.
- REMAINS (next mechanisms, in order of tractability): (a) SELF-ORGANIZE the afferent place-code (let the tone bands
  emerge from the ladder statistics rather than be host-derived); (b) the fixed level→word lexicon → a learned/emergent
  articulation of the tone; (c) the generator mouth itself (the Broca-like scaffold) — the standing north-star burn-down.
