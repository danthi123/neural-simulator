---
type: finding
status: live
date: 2026-09-01
mechanism: affective-theory-of-mind
artifacts:
  - research/findings/raw/_affective_tom_graded/graded_other_6seed.json
  - research/findings/raw/_affective_tom_graded/graded_other_6seed_smoke.json
  - research/findings/raw/_affective_tom_graded/realword_endtoend_demo.json
runner: research/runners/_affective_tom_graded_derisk.py
---

# W5 affective Theory-of-Mind — the OTHER-model read is now GRADED, not bistable-sign (6/6-seed GO)

## The named residual, closed

`2026-08-26-W5-affective-tom-production-wirein-GO.md` (the live production-integration finding) explicitly named
its own next rung: *"Scoped to VALENCE (good/bad) ... Fine discrete emotions need the SAME graded-circumplex
surpass P0.3 already named, NOT a new wall."* Its OTHER-model read (`affective_tom_production_organ.observe_turn`)
collapsed the Gate-B DR-2 appraised valence to a bare `sign()` and drove the P0.3 BISTABLE `AffectStateBrain`
(`_affect_state_region_derisk.py`) via `_affective_tom_derisk.read_tone`, so the empathic lead was a 3-state switch
(comfort / neutral / share-joy) regardless of how mild or extreme the OTHER's situation was — "Maria is lonely"
and "Maria is heartbroken" produced the IDENTICAL lead. Meanwhile the SELF's OWN affect had ALREADY been upgraded
past this exact wall on 2026-08-19 (`2026-08-19-graded-affect-attractor-GO.md`: valence Pearson +0.97, arousal
+0.95, 6/6 seeds, production-wired via `webapp/affect_drives_chat.py`). This finding applies that SAME upgrade to
the OTHER-model read, closing the residual the wire-in itself flagged.

**Verdict: BRAIN-BASED GO, 6/6 seeds** (numpy-CPU, NO `sim/` edit, reuse-by-import). Additive, default-OFF.

## Mechanism (a SEPARATE OTHER-tagged instance of the SAME #81 ladder the SELF read uses; reuse-by-import)

`research/runners/_affective_tom_graded_derisk.py` builds a dedicated `OtherGradedAffectOrgan` around a SECOND,
independent `GradedAffectBrain` instance (verbatim import of `_graded_affect_attractor_derisk.GradedAffectBrain` /
`read_body` — no new brain-building code). This preserves the "separate slot per agent" motif W3/W5 already use for
dissociability: the OTHER's affect state never shares a bridge with the SELF's #84 ladder or with the bistable W5's
own `AffectStateBrain`.

The OTHER's DR-2 appraised valence (`affect_production_organ.appraise_text`, unchanged — still the host
language-comprehension boundary) maps onto the SAME comfort/discomfort opponent body-state #81 reads
(`h=(valence+1)/2`, so comfort=h / discomfort=1-h are anti-correlated exactly as the SELF path), and the appraised
arousal maps directly onto the arousal channel. `read_body(brain, h, a, ...)` — the VERBATIM #81 read — drives the 3
interoceptive relays into the 3 ladders (vplus/vminus/arousal; N_L=6 independently-latching self-recurrent NMDA
sub-pools each, no intra-sign lateral inhibition, the load-bearing Koulakov rule) and reads the population
differential `mood = rate(V+ ladder) - rate(V- ladder)`, `felt_arousal = rate(arousal ladder)`, off
`cp_firing_states` — never a host formula. `graded_tone_level()` quantizes the continuous `mood` into a 7-level
(-3..+3) staircase for the empathic-expression map (`empathic_lead_graded`), replacing the old `sign()`-only read.

## 6-seed evidence (seeds 42/43/44/100/101/102; `research/findings/raw/_affective_tom_graded/graded_other_6seed.json`)

| gate | result |
| --- | --- |
| VALENCE graded (Pearson(valence,mood) >= 0.8, > 2 resolvable levels) | **6/6 seeds**, mean corr **+0.96**, mean **7.0** distinct levels on a 15-point sweep (mean 11.7 SNR-resolvable levels) |
| AROUSAL graded (Pearson(arousal,felt) >= 0.8) | **6/6 seeds**, mean corr **+0.95** |
| MORE distinct levels than the old bistable's fixed 2 nonzero states | **6/6 seeds** |
| LESION collapses (the #81 embodiment lesion, `intero_out=0`, reused) | **6/6 seeds**: mood/felt-arousal range -> **0.000** exactly on every seed |
| Empathic lead > 3 distinct tiers (old bistable ships exactly 3: `''`/comfort/share-joy) | **6/6 seeds**, mean **7.0** distinct lead strings |
| Lead vanishes under lesion | **6/6 seeds** |
| Substrate determinism (two builds at one seed -> identical thresholds) | **PASS** |

`intero_owns_valence_frac` / `intero_owns_arousal_frac` = **100% / 100%** on every seed (`tools.lab.attributable_to`)
— the coupling range collapses entirely under the interoceptive-gate lesion, none of it is "also present in the
control." Full per-seed curves, leads, and the lesion arrays are in the cited artifact.

## Load-bearing, end-to-end through the REAL production entry point (not just the isolated organ sweep)

`affective_tom_production_organ.observe_turn` is the exact function `webapp/server.py`'s `brain_chat` calls. With
`BRAIN_AFFECTIVE_TOM_GRADED=1` and real message text (not a synthetic appraisal dict), well-separated real words
produce genuinely different, appropriately-ordered empathic tiers:

message + DR-2 valence + mood (graded read) + level + lead, from
`research/findings/raw/_affective_tom_graded/realword_endtoend_demo.json` (seed 42):

| message | DR-2 valence | mood (graded read) | level | lead |
| --- | --- | --- | --- | --- |
| "Maria is lonely" | -0.25245 | -0.0492 | -2 | "That sounds really hard for Maria -- " |
| "Maria feels lost" | -0.299175 | -0.060167 | -2 | "That sounds really hard for Maria -- " |
| "Maria is heartbroken" | -0.825 | -0.075033 | -3 | "That sounds devastating for Maria -- " |
| "Maria was hurt" | -0.828075 | -0.075167 | -3 | "That sounds devastating for Maria -- " |
| "Maria is cheerful" | +0.725 | +0.066433 | +3 | "That's absolutely thrilling for Maria -- " |

With the SAME messages run through the flag-OFF (default) path, "lonely" and "heartbroken" produce the IDENTICAL
lead (`"That sounds really hard for Maria -- "`, `tone_sign=-1` both) — this is the frontier residual, reproduced
live and then closed. `BRAIN_AFFECTIVE_TOM_LESION=1` collapses the graded lead to `''` on every trigger regardless
of magnitude, reverting to the bare surface (same lesion contract as the bistable path).

## Byte-identical-off (verified, in the cited artifact's `byte_off` block)

With `BRAIN_AFFECTIVE_TOM_GRADED` unset (the default), `affective_tom_production_organ.observe_turn` is completely
unperturbed: bad-other lead `"That sounds really hard for Maria -- "`, good-other lead `"That's wonderful for Tom
-- "` — the SAME strings the shipped 6/6-seed-GO bistable path has always produced — with no `tone_level` key
attached. The graded module (`_affective_tom_graded_derisk.py`) is imported ONLY inside the `if
affective_tom_graded_enabled():` branch, so with the flag off it is never even loaded; `_affect_state_region_derisk.py`,
`_affective_tom_derisk.py`, and the bistable `AffectiveToMOrgan` class are untouched by this change.

## Contract (additive, reversible)

- **NEW flag**: `BRAIN_AFFECTIVE_TOM_GRADED` (default-OFF), read entirely inside
  `research/runners/affective_tom_production_organ.py::affective_tom_graded_enabled()`. No `webapp/server.py` edit
  was needed — the server only reads `tom_info.get("lead")` / `tom_info.get("acted")`, both of which the graded
  branch populates identically to the bistable branch (`git diff webapp/server.py` is empty).
- **NO `sim/` edit.** Reuse-by-import of `GradedAffectBrain` / `read_body` (`_graded_affect_attractor_derisk.py`,
  itself unmodified) plus the existing `affect_production_organ.appraise_text` appraisal boundary.
- The base W5 faculty (`BRAIN_AFFECTIVE_TOM`, on-by-default in production since 2026-08-26) is completely
  unaffected when the new flag is off; the two flags compose (`BRAIN_AFFECTIVE_TOM=1 BRAIN_AFFECTIVE_TOM_GRADED=1`
  is the only combination that exercises the new path; the graded flag is a no-op if the base faculty is off).

## Honest residuals (named, ride existing burn-down items — inherited from the #81 GO and the bistable W5 GO)

- **Gradedness is QUANTIZED** (a 7-level Koulakov staircase), not a smooth continuum — the #81 honest boundary,
  inherited verbatim (more resolution is more sub-pools, a linear cost, not a wall).
- **On REALISTIC single-word appraisals the achieved granularity is 2-3 meaningfully distinct tiers per side**, not
  the full 7: the Warriner strongly-affective salience gate only lets through words with `|v9-5|>=2`
  (empirically `|valence|` ranges roughly 0.19 to 0.97 across the ~118 gated words, most mass around 0.5-0.83
  <!--derived-->), and near-synonymous words ("sad" ~-0.68 vs "devastated" ~-0.83 <!--derived-->) can still land
  in the same tier. The underlying neural representation itself resolves the full 7 levels across the theoretical
  sweep (measured above); the calibration honestly reports fewer DISTINCT tiers hit by the specific vocabulary in
  practice. This is a genuine, substantive surpass over the bistable's fixed 1 nonzero tier per sign regardless of
  ANY magnitude — not a claim of full-continuum resolution.
- **The message -> OTHER-situation valence/arousal APPRAISAL stays host** (the same Gate-B DR-2 boundary the
  bistable path already used) — unchanged by this upgrade; only the READ of that appraisal into a neural
  OTHER-tagged state is upgraded from bistable to graded.
- **The level -> EXPRESSION-MARKER string is a host conditioned-articulation scaffold** ("the mouth"): the tone
  that DRIVES the tier is the neural graded ladder read (lesion-provable), the surface STRING per tier is a
  template (the same sanctioned articulation-crutch pattern as the bistable path's `empathic_lead`).
- **A third co-resident affect bridge** now exists in the process (SELF's #84 ladder + the bistable W5's OTHER
  `AffectStateBrain` + this graded W5's OTHER `GradedAffectBrain`) — the one-brain consolidation step (merging
  affect bridges) remains a follow-on, shared with the existing affect burn-down.
- **NOT YET FLIPPED default-ON.** The organ-level 6-seed GO above and the direct end-to-end `observe_turn` checks
  are complete and clean, but a HANDLER-level no-regression soak through the REAL `webapp.server.brain_chat`
  (mirroring `_affective_tom_flip_soak.py`, the gate the original bistable wire-in ran before its own flip) needs
  the full ~47k-neuron production one-brain build, which this worktree could not complete in-session (a missing
  local `.venv` blocked pool-queueing the 6-seed handler soak; `research/runners/_affective_tom_graded_flip_soak.py`
  is written and confirmed to import/parse cleanly — `SIM_BACKEND=numpy python -m
  research.runners._affective_tom_graded_flip_soak --seeds 42 43 44 100 101 102` — and is the next action before an
  auto-flip). Per `docs/TERMS.md`, this faculty is honestly `de_risked: YES`, `wired: YES` (reachable from
  `/api/brain-chat` on `BRAIN_AFFECTIVE_TOM_GRADED=1`), `on_by_default: NO` (pending that soak).
- **Honesty boundary.** A functional graded affective-mentalizing correlate with an honest functional read-out; no
  claim of access to another mind's feelings.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affective_tom_graded_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affective_tom_graded_derisk --seeds 42 43 44 100 101 102
SIM_BACKEND=numpy python -u -m research.runners._affective_tom_graded_flip_soak --seeds 42 43 44 100 101 102   # NEXT: the handler-level pre-flip soak
```

## Roadmap / next

Wired, default-OFF (`BRAIN_AFFECTIVE_TOM_GRADED`). The immediate next action is the 6-seed handler-level
no-regression soak above; on a 6/6 GO, flip `affective_tom_graded_enabled()`'s default (mirroring
`_AFFECTIVE_TOM_DEFAULT_ON` in `webapp/server.py`) to make the graded OTHER-model read the production default,
per the standing auto-flip policy (clean GO + load-bearing + moat-safe + byte-identical-off + no-regression, no
owner-gate). This closes the graded-circumplex residual the 2026-08-26 wire-in named for the OTHER-model read; the
SELF-model side of the same wall closed on 2026-08-19.
