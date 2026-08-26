---
type: finding
status: live
date: 2026-08-26
mechanism: affective-theory-of-mind
---

# W5 affective theory of mind — PRODUCTION WIRE-IN (empathy in the live chat turn), GO (2026-08-26)

## Result
The 6/6-seed-GO W5 faculty (affective theory of mind / empathy — infer ANOTHER agent's emotion from THEIR witnessed
situation, in an OTHER-tagged affect region dissociable from the system's OWN affect;
[`2026-08-01-W5-affective-theory-of-mind-6seed-GO.md`](2026-08-01-W5-affective-theory-of-mind-6seed-GO.md)) is now
WIRED into the live production conversational turn (`webapp/server.py::brain_chat`). On a turn about another agent's
affectively-charged situation ("Maria is devastated", "Sam's team lost", "my friend won the award"), the reply LEADS
with an EMPATHIC expression whose tone is read NEURALLY off the OTHER-tagged region's `affect_out`-gated recall
differential. The wiring is ADDITIVE, guarded behind a NEW env flag `BRAIN_AFFECTIVE_TOM` (server anchor
`_AFFECTIVE_TOM_DEFAULT_ON = False` — default-OFF for now; the parent flips it default-ON after the pool soak), and
NO `sim/` edit (reuse-by-import). FLAG-OFF is byte-identical to pre-wiring; the ON path is verified LOAD-BEARING (the
empathic lead flips with the OTHER's situation and VANISHES under the OTHER-region lesion).

## Files
- Organ (NEW, reuse-by-import, NO `sim/` edit): `research/runners/affective_tom_production_organ.py`.
- Wiring (guarded, additive, BEGIN/END-marked blocks for mergeability): `webapp/server.py` — the flag anchor +
  `_affective_tom_on()`, the observe block in `brain_chat`, and the empathic-lead prepend at THREE response sites:
  the rich-path and single-fact-path assembly sites (as #84/#85 use), PLUS the comprehension-repair early-return
  payload (so an OOV/unresolved abstain turn — "Sam's team lost" — that returns before the assembly sites still
  leads with empathy; the observe block runs earlier, so `tom_lead` is available there).
- Organ-level verify (NEW, the direct load-bearing proof): `research/runners/_affective_tom_production_organ_verify.py`.
- Handler no-regression soak (NEW, the pool gate): `research/runners/_affective_tom_flip_soak.py`.
- Artifacts: `research/findings/raw/_affective_tom_prodflip/organ_verify.json` (6-seed organ verify, below) and
  `research/findings/raw/_affective_tom_prodflip/soak_seed42.json` (the handler no-regression soak).
- Reuse-by-import (unchanged): `_affect_state_region_derisk.AffectStateBrain` (the OTHER-tagged P0.3 opponent
  slow-NMDA region), `_affective_tom_derisk.read_tone` (the exact tone read), `affect_production_organ.appraise_text`
  (the Gate-B DR-2 learned-valence appraisal boundary).

## Mechanism (production wire-in; brain-based read, host comprehension + articulation boundaries declared)
On each turn: (1) a host regex boundary detects whether the turn is about ANOTHER agent and its display name
(`detect_other_agent`; first/second person excluded — that is the self / the system); (2) the OTHER agent's situation
valence is appraised via the SAME Gate-B DR-2 boundary the wired affect organs use; (3) the OTHER-tagged
`AffectStateBrain` is driven on that valence sign and the synaptic recall differential `rate(recall_pos) -
rate(recall_neg)` is read through the ONE `affect_out` transmission gate (the de-risk's exact `read_tone`);
(4) `tone_sign = 0` if `|differential| < 0.020` else `sign(differential)` → an empathic lead
(`-1 → "That sounds really hard for {agent} -- "`, `+1 → "That's wonderful for {agent} -- "`, `0 → ""`). The lead is
prepended to the answer surface; the moat/recall/abstain verdict and the content fields
(`abstained`/`recalled_svo`/`verified`) are untouched. The OTHER-model build + read run on a PRIVATE RNG timeline
(host process-global RNG snapshotted + restored — the #77 footgun), so a triggered turn cannot perturb the downstream
RNG-dependent organs.

## Load-bearing (organ-level, 6/6-seed data-verified) — the coupling CHANGES an output and the change RIDES the read
Measured directly on the production organ (numpy CPU, seeds 42/43/44/100/101/102, 6/6 GO; `read_tone` re-seeds
`cfg.seed` per read so each read is deterministic; artifact
`research/findings/raw/_affective_tom_prodflip/organ_verify.json`):
<!--derived-->
(The differential values below are rounded re-quotes / ranges of the full-precision per-seed values in the cited
`organ_verify.json` — bad_intact −0.0656 to −0.0691, good_intact +0.0653 to +0.0688, lesion 0.0 to +0.0006.)
- **VARY** (the faculty changes the output): a bad-other situation drives the empathic lead
  `"That sounds really hard for {agent} -- "` (tone_sign −1, differential **−0.066 to −0.069** across the 6 seeds); a
  good-other situation drives `"That's wonderful for {agent} -- "` (tone_sign +1, differential ≈ **+0.067**). The lead
  and the neural tone_sign flip with the OTHER's situation (every seed).
- **LESION** (the finding's other-output oracle, `BRAIN_AFFECTIVE_TOM_LESION=1`): cutting the OTHER region's
  `affect_out` gate collapses the recall differential to **≈0.0000** (pos ≈ neg; +0.0000 to +0.0006 across the 6
  seeds, all below the 0.020 neutral tolerance) → tone_sign 0 → the empathic lead VANISHES (`""`) on the SAME
  bad-other input whose appraised valence is unchanged, on every seed. This is the finding's
  egocentric|incongruent=0.000 vs
  other|incongruent=1.000 dissociation in production form: the empathic tone rides the OTHER-region SPIKING read, not a
  host `if valence<0` — kill the neural OTHER read and the empathic tone disappears though the appraisal is identical.
  (The lesion holds at measurement: plasticity is off, the gate stays 0 through the read.)
- **Trigger specificity / flag-off inertness**: an ordinary turn ("what does the cat eat", "I am sad", "you are
  great") detects no OTHER agent → `acted=False` → no lead, no `affective_tom` key, and the OTHER-model bridge is
  NEVER built (byte-identical + no RNG perturbation; verified in the artifact: `all_inert=True`, `organ_built=False`).

## FLAG-OFF byte-identical (verified)
- Importing the organ is side-effect-free (no bridge built at import); the flag defaults are all OFF
  (`affective_tom_enabled()`/`affective_tom_off()`/`affective_tom_lesioned()` all False with the env unset); an
  ordinary turn with the flag ON returns `acted=False` and never builds/reads the bridge.
- The server edits are PURELY ADDITIVE inside guards: with the flag off, `_affective_tom_on()` is False → the observe
  block is skipped (`tom_info=None`, `tom_lead=""`) → neither prepend fires and no key attaches → the flag-off path
  executes the identical statements as pre-wiring (byte-identical to HEAD by construction; the additive block adds
  only guarded-false branches).
- The no-regression SOAK (`_affective_tom_flip_soak.py`) asserts this IN THE DATA through the REAL `brain_chat`
  handler (stub renderer, other default-ON drive faculties silenced so the ONLY OFF→ON delta is the W5 lead;
  artifact `research/findings/raw/_affective_tom_prodflip/soak_seed42.json`).

## Handler no-regression (through the real `brain_chat`, seed-42 GO; parent runs the 6-seed pool gate)
A 6-turn conversation (2 ordinary recalls/abstains + 3 other-agent triggers) run OFF vs ON through the real handler:
- `ordinary_identical = True` — every ORDINARY turn's stable surface (answer/abstained/recalled_svo/verified) is
  exact-equal OFF vs ON, and no `affective_tom` key is attached (byte-identical).
- `triggered_content_identical = True`, `lead_present_on = True` — on each of the 3 triggers the empathic lead is
  prepended and `answer == lead + off_answer`, with the content fields byte-identical: "Maria is devastated" →
  "That sounds really hard for Maria -- " (main assembly path); "Tom is delighted" → "That's wonderful for Tom -- ";
  "Sam's team lost" → "That sounds really hard for Sam -- " (this turn is an OOV/unresolved abstain that returns via
  the comprehension-repair early-return — the third prepend site carries the lead there too).
- `vary_ok = True`, `sign_ok = True` — the bad-other and good-other leads DIFFER and the neural tone_sign is −1 (bad)
  vs +1 (good). `LESION collapsed = True` — with `BRAIN_AFFECTIVE_TOM_LESION=1` the bad-other lead vanishes and the
  answer reverts byte-identically to the OFF surface. Seed-42 handler soak: **GO**. The parent runs the 6-seed pool
  gate (`--seeds 42 43 44 100 101 102`) before flipping the anchor default-ON.

## Honest residuals (named, ride existing burn-down items)
- The message→OTHER-situation valence APPRAISAL is host (a language-comprehension boundary, the P0.3 interface /
  DR-2 learned-tag precedent). The tone READ (the OTHER-tagged affect state → the synaptic recall differential) and
  its `affect_out` dependence ARE the neural W5 mechanism (lesion-proven).
- The other-agent DETECTION + display name is a host regex boundary (a comprehension boundary, like curiosity's
  wh-frame or the prospective-memory cue text).
- The tone_sign→EXPRESSION-MARKER string is a host conditioned-articulation scaffold (the "mouth"): the tone that
  DRIVES it is neural (lesion-proven), the surface string for a given sign is a host template — the sanctioned
  articulation-crutch pattern. A brain-native empathic mouth is the named next rung.
- Scoped to VALENCE (good/bad) — matches the P0.3 bistable good/bad latch (QUALIFIED-GO/BOUNDARY). A FUNCTIONAL
  affective-mentalizing correlate, NOT a claim of access to another mind. Fine discrete emotions need the same
  graded-circumplex surpass P0.3 already named, NOT a new wall.
- The OTHER-tagged region runs on its OWN co-resident bridge alongside the recall composer, not merged onto the one
  recall bridge (the one-brain consolidation step, shared with the #84 / Gate-B affect burn-down).

## Roadmap / next
Wired default-OFF. The parent runs the 6-seed pool soak
(`SIM_BACKEND=numpy python -m research.runners._affective_tom_flip_soak --seeds 42 43 44 100 101 102`); on a 6/6 GO
it flips `_AFFECTIVE_TOM_DEFAULT_ON = True`. The faculty is then integrated (wired + on-by-default); scaffold-retirement
(a brain-native empathic mouth + the one-brain merge) remains the burn-down.
