---
status: live
type: finding
lane: integration
date: 2026-09-05
mechanism: production-flip-rank16-rank20-rank10
integration_faculty: da-write-gain-spiking
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_da_write_gain_spiking_hook_verify.py
artifacts:
  - research/findings/raw/_da_write_gain_spiking/hook_verify.json
  - research/findings/raw/_da_write_gain_spiking/6seed.json
  - research/findings/raw/_curiosity_graded_novelty_derisk.json
external: NO-EXTERNAL-NEEDED -- this is a production-default FLIP of two already-committed, already-6/6-seed-GO
  de-risks; no new mechanism, no new biology claim.
---

# Production-flip campaign: rank-16 (DA write-gain) FLIPPED, rank-10 (curiosity graded novelty) FLIPPED, rank-20 (value-choice reward-context) already flipped by rank-4 -- SKIPPED

**Verdict: GO for 2 of 3.** Pure-CPU (`SIM_BACKEND=numpy`) session flipping three already-6/6-seed-GO'd,
committed de-risks to production default-ON, following the exact recipe this session already executed four times
(ranks 4, 5, 8, 12). Per-faculty verdict:

| rank | faculty | pre-flip state | action | verdict |
|---|---|---|---|---|
| 16 | DA write-gain spiking leaf | de-risked 6/6, wired default-OFF | FLIPPED default-ON | GO |
| 20 | value-choice reward-context | de-risked 6/6 (extends rank-4's own verification) | none -- already default-ON via rank-4 | SKIPPED (already done) |
| 10 | curiosity graded novelty | de-risked 6/6, wired default-OFF | FLIPPED default-ON | GO |

## Step 1 -- verify-first (per docs/TERMS.md / drift-#12: the plan doc + backlog are POINTERS, not the truth)

**Rank-16** (`research/findings/2026-09-05-da-write-gain-spiking-derisk-GO.md`): confirmed genuinely GO 6/6
(load-bearing span 1.35-1.52 every seed, lesion collapses to 0.0000 every seed, parity corr 0.9958-0.9999,
deterministic). Confirmed against CURRENT code: `webapp/da_encoding_drives_chat.py::da_encoding_spiking_gain_enabled()`
read `os.environ.get("BRAIN_DA_ENCODING_SPIKING_GAIN", "0")` -- default OFF, unflipped. Flip point identified:
that one function's unset-branch.

**Rank-20** (`research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md`):
re-read in full. **This finding introduces NO independent flag.** It extends the verification depth (6 seeds, the
REAL trained `striosome_value` critic, 4 candidate scenarios) of `value_choice_production_organ.py::default_context_fn`'s
THIRD consumer site of the shared spiking salience afferent (`research/runners/shared_salience_afferent.py`),
governed entirely by `BRAIN_SHARED_SALIENCE`. Checked against CURRENT code:
`shared_salience_afferent.py::shared_salience_enabled()` reads:
```python
v = os.environ.get("BRAIN_SHARED_SALIENCE")
if v is None:
    return True   # DEFAULT-ON anchor (wave-4 flip, 2026-09-05)
```
**Already default-ON** -- flipped earlier this same session by the rank-4 shared-salience-afferent production-flip
(commit `04cbd1bec`, `research/findings/2026-09-05-shared-spiking-salience-afferent-production-default-flip-GO.md`
lineage). Per the task's own instruction ("If any is already flipped ... SKIP it and report; do NOT force a
flip"), rank-20 is **SKIPPED** -- there is no separate flag left to flip, and forcing a duplicate "flip" would be
either a no-op or would require inventing a new, narrower flag this de-risk never proposed. `value_choice_production_organ.py`'s
own inline comments (lines ~62, ~317) still read "default-OFF (`BRAIN_SHARED_SALIENCE`)" -- this is a **stale
comment**, not a code defect (the actual reader function is correct); flagged here, not fixed in this session
(out of the narrow scope this task specifies), so a future doc-sync pass can correct it.

**Rank-10** (`research/findings/2026-09-05-rank10-curiosity-graded-novelty-familiarity-scaffold-derisk-GO.md`):
confirmed genuinely GO 6/6 across 3 independent runs (graded order strict every seed, lesion collapses to <1e-3,
permuted-vocabulary control confirms the imprint<->query correspondence, never crashes a turn). Confirmed against
CURRENT code: `research/runners/curiosity_production_organ.py::graded_novelty_enabled()` read
`os.environ.get("BRAIN_CURIOSITY_GRADED_NOVELTY")` returning `False` on unset -- default OFF, unflipped.

## Step 2 -- the flip (the `enabled()`-unset-returns-True pattern this session used)

**Rank-16** (`webapp/da_encoding_drives_chat.py`): added `_DA_ENCODING_SPIKING_GAIN_DEFAULT_ON = True`;
`da_encoding_spiking_gain_enabled()` now returns that constant when `BRAIN_DA_ENCODING_SPIKING_GAIN` is unset,
falling through to the pre-existing truthy-check only when the var is explicitly set. `=0`/false/off/no is the
byte-identical escape.

**Rank-10** (`research/runners/curiosity_production_organ.py`): added `_GRADED_NOVELTY_DEFAULT_ON = True`;
`graded_novelty_enabled()` mirrors the same pattern for `BRAIN_CURIOSITY_GRADED_NOVELTY`.

Both mirror the exact anchor pattern `da_encoding_enabled()` (the sibling coupling in the same file, flipped
2026-08-25) and `shared_salience_enabled()` (rank-4, this session) already use.

## Step 3 -- verify each flip (SIM_BACKEND=numpy)

### Rank-16: `research/runners/_da_write_gain_spiking_hook_verify.py` (re-run against the new default)

This file was originally the default-OFF de-risk's hook-verify; its OFF arm relied on the env var being UNSET,
which the flip_offarm_staleness class (`gates/flip_offarm_staleness`) would now silently read as ON. Fixed: the
OFF arm is pinned to the EXPLICIT `"0"` escape (never popped), and a new arm (A2) proves the flip itself.

```
SIM_BACKEND=numpy python -m research.runners._da_write_gain_spiking_hook_verify
```

Result (`research/findings/raw/_da_write_gain_spiking/hook_verify.json`), all gates GO:

| gate | result |
|---|---|
| (A) explicit-"0" escape byte-identical to pre-flip `_gain_map()`, both leaf branches | True / True |
| (A2) THE FLIP: unset dispatches the IDENTICAL boolean branch as explicit "1" (decisive, exact) | True |
| (A2) sanity: unset's 3-rep-averaged live read within noise tolerance (<0.3) of explicit-1's | True (max_abs_diff=0.0564 <!--derived--> field `A2_flip_correctness_unset_eq_explicit1.max_abs_diff`) |
| (B) ON load-bearing (span 1.3872 <!--derived--> across the DA sweep, threshold 0.3) | True |
| (B) ON parity with the host formula (corr 0.9988 <!--derived-->, threshold 0.9) | True |
| (C) this mechanism's OWN inner lesion collapses the span to 0.0000 at the floor | True |
| (D) the PRE-EXISTING outer `da_encoding_lesioned()` gate still pins g=1.0 regardless | True |
| (E) the spiking module is never imported on the explicit-off escape | True |

**A2 methodology note**: the decisive check is the BOOLEAN dispatch (`da_encoding_spiking_gain_enabled()` returns
identically `True` whether unset or explicit `"1"` -- exact, not stochastic). An EARLIER attempt at A2 compared the
downstream SUBSTRATE READS for exact float equality (a single un-averaged read per arm) and failed -- NOT a flip
defect: two SEPARATE reads of a genuinely stochastic OU-noise-driven spiking population are not expected to
bit-match, the same documented noise floor the original de-risk's "Instrument notes" section names. Corrected to
the 3-rep-averaged tolerance check reported in the table above (field `A2_flip_correctness_unset_eq_explicit1` in
the cited artifact carries both the per-DA averaged arrays and `max_abs_diff`), per `docs/TERMS.md`'s discipline
of not conflating "the mechanism is stochastic" with "the flip is wrong." That earlier attempt's own raw numbers
were not saved to any artifact (overwritten by this corrected re-run) and are not reproduced here.

The full `brain_chat` handler round trip (`research/runners/_da_encoding_wired_verify.py`) was **not** separately
re-run: it is unmodified by this lever, was already confirmed byte-for-byte unchanged earlier this session at the
flag's pre-flip (unset) state, and the hook-level dispatch check above isolates the ONE call site the flip
changes -- matching the original de-risk's own scoping (`v.disabled(...)`, same file).

**Load-bearing-not-hollow**: (B) is the "the reply actually changes under the faculty" proof (span 1.3872 <!--derived-->,
correlated 0.9988 <!--derived--> with the pre-existing host formula it replaces) and (C) is the "the change
vanishes on lesion" proof (span collapses to 0.0000 at the floor) -- both at the NEW default. Not hollow.

### Rank-10: unit tests + real `/api/brain-chat` handler tests (SIM_BACKEND=numpy)

Fixed the pre-existing tests that assumed the OLD default (OFF): `tests/test_curiosity_graded_novelty.py`
(`test_graded_novelty_enabled_defaults_off` -> `test_graded_novelty_enabled_defaults_on`, asserting `True`; added
`test_graded_novelty_enabled_explicit_off_is_the_byte_identical_escape`) and `tests/test_webapp_server.py`
(`test_brain_chat_curiosity_graded_novelty_default_off_is_byte_identical` -> `..._explicit_off_is_byte_identical`,
pinning `BRAIN_CURIOSITY_GRADED_NOVELTY=0` explicitly rather than relying on unset; added
`test_brain_chat_curiosity_graded_novelty_default_unset_matches_explicit_on`, the flip-correctness proof through
the REAL handler).

```
SIM_BACKEND=numpy python -m pytest tests/test_curiosity_graded_novelty.py -q
  -> 20 passed
SIM_BACKEND=numpy python -m pytest tests/test_webapp_server.py -q -k curiosity_graded_novelty
  -> 3 passed (229s; each builds a real tiny-demo brain through the FastAPI test client)
```

The three handler-level tests, through the REAL `/api/brain-chat` endpoint:
1. **Explicit-off byte-identical**: `BRAIN_CURIOSITY_GRADED_NOVELTY=0` -> `curiosity.novelty == NOVEL_SIGNAL`
   exactly, no `graded_novelty` trace key -- the escape reproduces pre-flip HEAD.
2. **THE FLIP**: fully-unset (the shipped default) attaches the SAME `graded_novelty` trace shape (`on: True`, a
   real `[0,1]` value, `lesioned: False`) that explicit `=1` produces, and `curiosity.novelty` equals that value
   exactly -- unset and explicit-ON take the identical branch, proving the flip is safe.
3. **Load-bearing + lesion-reverting** (pre-existing test, still passes post-flip): the ON arm attaches a real
   graded value and `BRAIN_CURIOSITY_GRADED_NOVELTY_LESION=1` flips `graded_novelty.lesioned` to `True` live,
   per-turn.

**Load-bearing-not-hollow**: test 3 (and the de-risk's own 6/6-seed lesion-collapse gate, mechanism-level,
unaffected by the flag flip since the 6-seed derisk gate calls `TopicNoveltyGate`/`topic_novelty` directly, never
`graded_novelty_enabled()`) is the "actually changes + vanishes on lesion" proof: the graded novelty value differs
by topic (known vs novel), and lesioning the projector collapses every topic to the SAME ceiling, reverting the
crave decision to the old constant's always-curious signature. Not hollow.

**6-seed scoping note** (mirrors the exact precedent set by ranks 4/12's own PART-C scoping this session): the
underlying mechanism's 6-seed GO (`_da_write_gain_spiking_derisk.py` for rank-16, `_curiosity_graded_novelty_derisk.py`
for rank-10) is INDEPENDENT of the flag -- neither derisk gate calls the `enabled()` function the flip touches
(confirmed by grep: `graded_novelty_enabled` appears zero times in `_curiosity_graded_novelty_derisk.py`;
`spiking_write_gain` never reads any env var). The flag only gates a single dispatch `if` in each production hook,
which is seed-independent (a boolean read) and runs at production's fixed single-process seed (matching how this
codebase's other flip-verifies scope their handler-level arm, e.g. rank-4/12's PART C: "production runs ONE
process at ONE seed"). So the 6-seed load-bearing+lesion GO already committed transfers verbatim to the new
default (same argument the rank-4 commit message uses), and the flip-specific verification above (which the
6-seed derisk cannot itself exercise, since it never reads the flag) is what this finding adds: the DISPATCH is
correct at production's actual seed, through the actual production entry point.

## Step 4 -- ledger + board

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`: two new rows added (`da-write-gain-spiking`, `curiosity-graded-novelty`),
both `on_by_default: YES`, `default_anchor` pointing at the new named constants above.
`default_on_spiking_faculties` bumped 29->30 (rank-16 builds a genuinely NEW dedicated population, `write_gain`,
40-80 IZH2007_HIPPO_PYRAMIDAL neurons on its own bridge; rank-10 does NOT bump the count -- it reuses the
already-counted DR-1 curiosity ASK-pool substrate + the v320 gate's `phase_sum_neuron` cue-bind, per its own row's
honest-scope note that the novelty readout itself is a host-rate-form energy on that bind, the same declared
"composer-as-idealization" boundary the v320 gate and INTEGRATION #7 already carry). `total_faculties` bumped
64->66. Vikunja board #209 (the scaffold-retirement wave's tracking task) appended with both flips and the
rank-20 skip reasoning; `docs/.vikunja_sync` stamped.

## Honesty boundary

Both flipped mechanisms are FUNCTIONAL correlates (a write-gain population's firing rate; a familiarity-projector's
mismatch energy), never phenomenal claims. Neither flip touches what the brain asserts as fact (rank-16 changes
HOW STRONGLY a fact is written; rank-10 changes HOW NOVEL a topic reads for the curiosity follow-up decision) --
both are moat-safe by construction (rank-16 cannot invent or suppress a fact, only scale its write magnitude;
rank-10's `topic_novelty()` degrades to the pre-existing constant on any error, never crashing a turn).

## Files

- `webapp/da_encoding_drives_chat.py` (EDIT, additive) -- `_DA_ENCODING_SPIKING_GAIN_DEFAULT_ON`, flip.
- `research/runners/curiosity_production_organ.py` (EDIT, additive) -- `_GRADED_NOVELTY_DEFAULT_ON`, flip.
- `research/runners/_da_write_gain_spiking_hook_verify.py` (EDIT) -- OFF-arm fixed to explicit `"0"`, new A2 arm,
  (E) probe fixed to test the explicit-off escape's laziness.
- `tests/test_curiosity_graded_novelty.py` (EDIT) -- default-on rename, new explicit-off-escape test.
- `tests/test_webapp_server.py` (EDIT) -- explicit-off rename, new default-unset-matches-explicit-on test.
- `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (EDIT) -- two new rows, headline counts bumped.
- `docs/.vikunja_sync` (EDIT, append-only) -- stamped.

## Citations

- Rank-16 de-risk (reused verbatim): `research/findings/2026-09-05-da-write-gain-spiking-derisk-GO.md`.
- Rank-20 de-risk (reused verbatim, no code change): `research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md`.
- Rank-10 de-risk (reused verbatim): `research/findings/2026-09-05-rank10-curiosity-graded-novelty-familiarity-scaffold-derisk-GO.md`.
- The flip recipe precedent (this session): rank-4 shared-salience-afferent flip (`04cbd1bec`), rank-12 GNW-stop-trigger flip (`653c62d17`).
- Attribution / lesion discipline: `docs/TERMS.md`, `tools/lab.py`.
