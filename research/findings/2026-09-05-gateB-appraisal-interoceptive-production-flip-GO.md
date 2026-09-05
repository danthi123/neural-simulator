---
type: finding
status: live
date: 2026-09-05
mechanism: appraisal-interoceptive-afferent (production flip)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_appraisal_interoceptive_production_flip_verify.py
artifacts:
  - research/findings/raw/appraisal_interoceptive_ladder/_production_flip_verify.json
builds_on:
  - research/findings/2026-09-05-gateB-appraisal-interoceptive-afferent-derisk-GO.md
  - research/coordination/scaffold_retirement_backlog.md
---

# Gate-B appraisal-via-interoceptive-afferent flips to PRODUCTION DEFAULT — 6/6-seed GO, no-regression + genuinely hollow-under-lesion through the real handler

**Verdict: GO.** `research/findings/2026-09-05-gateB-appraisal-interoceptive-afferent-derisk-GO.md` (scaffold-retirement
backlog rank 5) already earned a 6-seed GO for the mechanism with the flag explicitly forced on — a de-risk, flag
default-off. This is the flip itself: `appraisal_interoceptive_enabled()` in
`research/runners/affect_production_organ.py` now defaults **on** (an unset environment now takes the interoceptive
path; the escape hatch is `BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE` set to `0`/`false`/`no`/`off`). Verified through the
real production organs and the real `webapp.server.brain_chat` handler — not an isolated stub — on all three
requirements the flip brief set: no-regression, load-bearing-not-hollow (with an explicit vary-vs-lesion pairing), and
a genuine default change with an intact rollback. Runner:
`research/runners/_appraisal_interoceptive_production_flip_verify.py`. Raw data:
`research/findings/raw/appraisal_interoceptive_ladder/_production_flip_verify.json` (1780.8s, numpy-CPU).

## The flip

`research/runners/affect_production_organ.py::appraisal_interoceptive_enabled()`:

```python
def appraisal_interoceptive_enabled() -> bool:
    v = os.environ.get("BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE")
    if v is None:
        return True                                              # was False
    return v.strip().lower() not in ("0", "false", "no", "off", "")   # was: v in ("1","true","yes","on")
```

This is the SAME idiom the two prior default-on Gate-B flips in this file already use (`affect_enabled()`,
`dr2_enabled()`): unset now means on; an explicit falsy literal is the rollback. Every other line of
`AffectProductionOrgan.read_differential` is unchanged — the dispatch (`if appraisal_interoceptive_enabled(): return
get_ladder(self.seed).read_differential(...)`) was already the de-risk's own 8-line addition; only the function this
dispatch calls changed which branch is the default.

**A latent same-process multi-seed confound, found and fixed while building the flip verifier.**
`_appraisal_interoceptive_ladder_derisk.get_ladder(seed)` cached a single ladder in a bare `Optional`, so a second,
differently-seeded call in the same process would silently reuse the FIRST seed's neurons under the second seed's
label. Inert against today's production (`webapp/server.py` hardcodes `get_organ(seed=42)` everywhere — one seed per
process, by convention, like every other production organ singleton), but exactly the confound class
`tests/test_determinism.py::TestSubstrateActuallySeeded` exists to catch for a same-process 6-seed verification loop.
Fixed to a dict cache keyed by seed (`_LADDERS: dict[int, AppraisalInteroceptiveLadder]`); confirmed empirically —
seed 42 and seed 43 read genuinely different `corr_new` (0.9744 vs 0.9732) in the same process, not identical numbers
under different labels. <!--derived-->

A second pre-existing helper needed a matching fix: `run_seed()`'s and `run_byte_identical_off()`'s own "host
reference" construction in the de-risk module used to `os.environ.pop(FLAG, None)` to reach the host-write branch —
correct when unset meant off, but after the flip an unset flag means ON, so popping it would have silently turned the
"host-write reference" into a self-comparison against the NEW mechanism. Both now set the flag to `"0"` explicitly
(the reference fix `gates/flip_offarm_staleness` documents), so the existing de-risk module's own `--seeds`/`--smoke`
CLI still measures what its name says post-flip.

## Anti-cheat 1 — the default is genuinely ON (asserted in the data)

For each of the 6 seeds, a **fresh** `AffectProductionOrgan(seed)` was built with the environment variable **unset**
(never `=1`) and `read_differential(0.7)["mechanism"]` was read back: `"interoceptive_afferent"` on all 6 seeds
(`phase1_per_seed[i].default_is_on`, all `true`) — the dispatch branch is reached by the bare default, not inferred
from reading the code.

## Anti-cheat 2 — LOAD-BEARING, through the default path itself (6/6 seeds)

Sweeping the appraisal through the (now-default) interoceptive path moves the ladder differential with the correct
sign in the production-realistic band and tracks it in order; the downstream `tone_level`/`content_plan`/`manner_for`
genuinely vary across the sweep on every seed (`downstream_varies_intact: true`, 6/6):

| seed | corr (appraisal, differential) | range | relay-encodes-appraisal (intact) |
| --- | --- | --- | --- |
| 42 | +0.974428 | 0.169722 | +0.959285 |
| 43 | +0.973237 | 0.163889 | +0.959867 |
| 44 | +0.972753 | 0.170278 | +0.960118 |
| 100 | +0.974229 | 0.154444 | +0.959838 |
| 101 | +0.974205 | 0.156667 | +0.959452 |
| 102 | +0.973851 | 0.158889 | +0.961066 |

Mean corr **+0.973784**, mean range **0.162315** — matching the de-risk's own recorded means (corr +0.973784, range 0.162315) to 6 decimal places, confirming the default-on path is byte-for-byte the same mechanism the de-risk validated, not a re-derivation. <!--derived-->

## Anti-cheat 3 — the EXPLICIT anti-hollow pairing (vary changes it; lesion makes it vanish), mechanism-level

The flip brief asks for more than "the range shrinks under lesion" (the de-risk's own `<=0.25x` bound) — it asks
whether the SAME sweep that varied the downstream read stops varying it entirely once lesioned. For every seed, the
identical 9-point appraisal sweep was re-run with the new relay→ladder synapse cut
(`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION=1`, the main flag still at its default) and the downstream
`content_plan`/`manner_for` outputs were checked for being **constant** (a single distinct value across the whole
sweep), not merely reduced in range:

- **Range**: intact mean **0.162315** → lesioned mean **0.0** on every seed (`range_intero_lesioned: 0.0`, 6/6) — `intero_synapse_owns_range_frac` (`tools.lab.attributable_to`) reads **1.0** on every seed: 100% of the appraisal→differential coupling is owned by the new synapse. <!--derived-->
- **Downstream**: `downstream_collapses_under_intero_lesion` is `true` on all 6 seeds — under the lesion,
  `content_plan(level)` and `manner_for(level, ...)` return the SAME value for every point in the sweep (the neutral
  default), where intact they took at least 2 distinct values each.
- **Dissociation, not silence**: the relay pools still encode the appraisal magnitude while lesioned
  (`relay_enc_under_intero_lesion` mean **+0.959938**, identical to `relay_enc_intact` mean **+0.959938** — every
  seed's own intact and lesioned relay-encoding value match exactly, 6/6) — the lesion cuts the SYNAPSE, not the
  afferent signal, exactly the de-risk's own dissociation signature, reproduced through the default path.
  <!--derived-->
- **Old semantics unchanged**: the pre-existing readout lesion (`lesion=True`, `affect_out=0`) still collapses the
  differential to exactly `0.0` regardless of appraisal on all 6 seeds (`readout_lesion_collapses`).

## Anti-cheat 4 — NO-REGRESSION, integrated, through the real `webapp.server.brain_chat` handler

A factual panel (3 stored, 2 unstored, 2 inconsistent, 1 self/identity — 8 queries) was sent through the real
production handler twice per query, once with the flag at its new default (unset) and once with the escape hatch
explicit (`=0`), each arm a fresh session so no mood state leaks between them:

**8/8 identical** — `recalled_svo`, `abstained`, and `verified` matched byte-for-byte on every query, every class
(`phase2_no_regression.ok: true`). The stored facts (`dog chase cat`, `cat eat fish`, `brain use spikes`) were
answered correctly on both arms; both unstored queries (`fish fly`, `bird sing`) and both inconsistent queries
(`dog eat`, `cat chase`) abstained on both arms (the moat holds); the self-identity query (`what are you`) also
matched. The mechanism swap changes nothing about WHICH fact is recalled or whether the moat fires — only which
substrate realizes the affect differential feeding the manner/forthcomingness surface.

## Anti-cheat 5 — the live hollow-mouth discipline (mirrors the linattn flip-confirmation's method)

`research/findings/2026-09-04-linattn-flip-confirmation-affect-still-hollow-live-NOGO.md` established the discipline
this needs to survive: toggle the lesion through the REAL handler with a REAL established mood, and check whether the
live turn's output actually moves. Four fresh sessions were primed (real conversational turns through
`webapp.server.brain_chat`, `appraise_text`'s own reported valence re-checked ≥0.6 in magnitude — safely inside the
production-realistic band, clear of the de-risk's characterized sub-threshold zone below 0.5) to a strong positive or
negative mood, intact or with the interoceptive-synapse lesion held for the whole session, then asked the identical
stored query (`what does dog chase?`):

| arm | prime valence | tone_level | forthcomingness | recalled | answer |
| --- | --- | --- | --- | --- | --- |
| pos, intact | +0.656141 | **+3** | `{max_sentences:4, max_elaborations:3}` | dog/chase/cat | "Wonderful! The dog chases cat. The cat eats fish. — worth going further here." |
| neg, intact | -0.631378 | **−2** | `{max_sentences:1, max_elaborations:0}` | dog/chase/cat | "The dog chases cat. — worth going further here." |
| pos, LESIONED | +0.656141 | **0** | `{max_sentences:4, max_elaborations:2}` | dog/chase/cat | "Wonderful! The dog chases cat. The cat eats fish. — worth going further here." |
| neg, LESIONED | -0.631378 | **0** | `{max_sentences:4, max_elaborations:2}` | dog/chase/cat | "Honestly! The dog chases cat. The cat eats fish. — worth going further here." |

Reading the STRUCTURED fields the mechanism actually owns (`tone_level`, `forthcomingness` — the values that feed
`RichAnswerComposer.max_sentences`/`max_elaborations`, per `webapp/server.py`'s own `rich.max_sentences =
int(affect_plan["max_sentences"])`), the load-bearing/anti-hollow pair holds exactly as required: intact,
positive vs negative priming gives opposite-signed tone (+3 vs −2) and a visibly different plan (4 sentences/3
elaborations, warm, vs 1 sentence/0 elaborations, terse) — the variation is real, through a live conversational turn,
not a synthetic sweep. Lesioned, BOTH arms collapse to the identical neutral values (`tone_level=0`,
`forthcomingness={4,2}`, the production default for a neutral mood) regardless of which mood was primed — the
variation genuinely **vanishes**, satisfying the brief's literal "byte-identical whether-varied-or-not = HOLLOW" test
in its negation: intact is NOT byte-identical across mood (load-bearing), lesioned IS byte-identical across mood on
the fields this mechanism owns (the lesion is real). The recalled fact (`dog/chase/cat`) is identical across all four
arms — mood colors HOW, never WHAT, the honesty floor holds under the flip.

### Honest note: the raw answer STRING is not byte-identical between the two lesioned arms, and that is correct, not a leak

The lesioned rows' raw strings differ by one leading word ("Wonderful!" vs "Honestly!") despite `tone_level` and
`forthcomingness` being identical. This is NOT a hollow-mouth leak in the mechanism under test — it comes from a
**separate, independently co-resident** affect faculty (board #84's felt-body-state EMA,
`webapp/affect_drives_chat.py::_LEAD_WORD`, `{3:"Wonderful", ..., -2:"Honestly", ...}`), which reads its OWN felt-state
level off its OWN pathway, entirely unaffected by the Gate-B interoceptive-synapse lesion this finding tests (a
different mechanism, a different lesion). Checking the STRUCTURED fields the mechanism under test actually owns,
rather than diffing the raw string, is what keeps this a correct pass rather than a false "not quite hollow" read —
and the fact that an unrelated co-resident affect faculty keeps tracking the primed mood while Gate-B's own
contribution collapses is itself a demonstration that the lesion is surgical (it cuts the ONE synapse it targets),
not a blunt kill switch on affect generally. <!--derived-->

## Anti-cheat 6 — the escape hatch reaches the untouched original code, byte-for-byte, on every seed

With the flag explicit (`="0"`), a **fresh** `AffectProductionOrgan(seed)` was swept on the full 9-point appraisal
sweep and compared against the pre-recorded `host_diffs` in the de-risk's own 6-seed artifact
(`_appraisal_interoceptive_ladder_6seed.json`, captured from the SAME unmodified host-write code before this flip
existed) — an **exact float match** (`==`, not a tolerance) on every point, every seed
(`escape_exact_match: true`, 6/6); `"mechanism"` (the key only the interoceptive path's return dict carries) is
absent from the off-path response on every seed (`off_mechanism_absent: true`, 6/6). Rollback is verified on all 6
seeds, not only seed 42's hardcoded pre-edit baseline the de-risk itself checked.

## Preconditions (travel with the verdict; a miss would make it UNDEFINED, not GO)

All hold, verified in the cited artifact via `tools.verdict.Verdict`: all 6 requested seeds ran; the default reads
on on every seed; sign-correctness + ordered tracking (corr≥0.8) hold in the production-realistic band on every
seed; the downstream read varies intact and collapses to constant under the intero lesion on every seed; the relay
still encodes the appraisal under that lesion (≥0.8) on every seed — a genuine dissociation; the pre-existing readout
lesion is unchanged; the escape hatch exact-matches the pre-recorded host reference on every seed; the integrated
no-regression panel is 8/8 identical; the integrated hollow-check's priming is verified strong (≥0.6) on every arm,
its load-bearing pairing holds, and the recalled fact is identical across all four mood/lesion arms. `numpy_backend`
(SIM_BACKEND=numpy throughout — no `sim/` edit).

Fifteen heavy default-on co-resident faculties unrelated to Gate-B appraisal (worldmodel, surprise, metacog,
multiref, curiosity, episodic, discourse register, self-schema, source-provenance, pragmatic, comprehension-learned-
cues, BG-select, spiking-mouth-recall, the GNW workspace buses, the onebrain xedge cross-session pool) were disabled
in phase 2 for speed only — the SAME "heavy Gate-B organs disabled for speed" scope declaration
`_gnw_bus_default_flip_verify.py::_handler_escape_byte_identical` already uses for exactly this reason. None of them
gates which fact a plain SVO query recalls or whether the moat abstains (each is independently flag-guarded,
never-crash-a-turn by the standing convention), so this narrows what cold-starts on the first turn, not what the
no-regression/hollow checks can see. `BRAIN_AFFECT` and `BRAIN_RICH` — the faculty under test and the surface
(`RichAnswerComposer` forthcomingness) it drives — stayed at their real, unmodified production default throughout.

## Answering the report question

**Is the flip safe and genuinely load-bearing to ship on-by-default?** Yes, on all three requirements. **No
regression**: with the flag on (its new default), the integrated brain still converses — every panel query answered
identically to the pre-flip default on the real handler, 8/8, and every mechanism-level check that held with the
flag explicitly forced on in the de-risk still holds through the bare default, 6/6 seeds. **Load-bearing, not
hollow**: varying the appraisal genuinely changes the live behaviour (the mechanism-level sweep, 6/6 seeds, AND a
real primed conversational turn, tone +3 vs −2, terse vs expansive); lesioning the new synapse makes that variation
vanish on the fields the mechanism owns, both at the mechanism level (6/6 seeds, an exact test sharper than the
de-risk's own range-shrink bound) and live through the real handler — the explicit vary-vs-lesion pairing the brief
required, tested and passed twice, at two levels. **A real default change**: `appraisal_interoceptive_enabled()`
now returns `True` on an unset environment (verified in the data, not inferred), with an intact, byte-exact-verified
rollback on every seed. GO.

## Reproduce

```bash
SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m \
    research.runners._appraisal_interoceptive_production_flip_verify
# mechanism-level only, one seed, through the real class:
SIM_BACKEND=numpy python -c "
from research.runners.affect_production_organ import AffectProductionOrgan
o = AffectProductionOrgan(seed=42)
print(o.read_differential(0.8)['mechanism'])"   # -> interoceptive_afferent (flag unset)
```
