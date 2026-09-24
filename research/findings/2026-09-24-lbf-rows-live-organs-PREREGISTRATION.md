---
type: preregistration
status: preregistered
date: 2026-09-24
mechanism: ten candidate load-bearing-fraction ROWS for PRODUCTION-WIRED organs that already ship a dedicated
  `BRAIN_<X>_LESION` knob but have no row in `research/runners/load_bearing_fraction.py::FACULTY_LESIONS`/
  `FACULTY_PROBES` yet -- self-schema, affective-tom, causal-whatif, spiking-anaphor, spiking-qroute,
  learned-referent, onebrain-xedge, gnw-bus, multiref-competition, affect-appraisal-interoceptive. Delivered via
  the `research/runners/lbf_rows/` row interface (EXTRA_LESIONS/EXTRA_PROBES/EXTRA_TURNS), not by editing
  FACULTY_LESIONS/FACULTY_PROBES directly.
lane: load-bearing (2026-09-24 midnight plan, lane A1, step S12)
seeds: [7]
verdict: PRE-REGISTRATION only, filed before any lesion-arm smoke has run. No 6-seed capability claim is made
  here -- that is explicitly B2b's job, not this lane's. An amendment follows in its own commit once the row
  module + a seed-7 smoke attempt are in.
artifacts: []
external: none new -- this is a wiring/instrumentation exercise over already-banked production mechanisms; no new
  biological claim.
---

# Ten new load-bearing-fraction rows for already-production-wired organs: pre-registration

**Filed 2026-09-24, branch `research/lbf-rows-live-organs`, cut from origin/main `f554056eb`.** Lane A1 of the
midnight plan, step S12 (depends on S07 dispatch + S08's AG-REG import-hook prep, both running concurrently in
sibling worktrees at filing time).

## Why these ten, and why now

`research/runners/load_bearing_fraction.py::FACULTY_LESIONS`/`FACULTY_PROBES` is the #1 project metric's
instrument (owner-ratified 2026-09-19: the fraction of production faculties where lesioning the brain's own
contribution provably changes the reply). Ten shipped organs with a real dedicated `BRAIN_<X>_LESION` knob,
confirmed present in `webapp/` or `research/runners/` source, currently have NO row in that instrument at all --
they are invisible to the metric, neither counted load-bearing nor honestly excluded. S12's job is to add rows
for exactly these ten, following the existing FACULTY_LESIONS/FACULTY_PROBES pattern.

## Row interface (no edit to FACULTY_LESIONS/FACULTY_PROBES; AG-REG owns the merge)

`research/runners/lbf_rows/live_organs.py` exposes three module-level names (the third, `EXTRA_TURNS`, mirrors
the sibling A2/S13 lane's own `research/runners/lbf_rows/learning.py`, which independently arrived at the same
need -- see its PREREGISTRATION, "the two turn groups ... added to `_EXTRA_TURNS`, NOT to `PROBE_TURNS`"):

```python
EXTRA_LESIONS: dict[str, dict]   # SAME shape as FACULTY_LESIONS: {key: dict(flag=..., value=..., kind=..., note=...)}
EXTRA_PROBES:  list[tuple]       # SAME shape as FACULTY_PROBES: [(key, turn_label, [field_paths], thin_bool), ...]
EXTRA_TURNS:   list[tuple]       # SAME shape as PROBE_TURNS/_EXTRA_TURNS: (label, message, session, reset, percept, rich)
```

AG-REG's import hook (S08/S26) merges `EXTRA_LESIONS` into `FACULTY_LESIONS`, extends `FACULTY_PROBES` with
`EXTRA_PROBES`, and folds `EXTRA_TURNS` into `onebrain_regression_battery._EXTRA_TURNS` (label-only, so
`_TURN_BY_LABEL` resolves them without growing the default `PROBE_TURNS` roster any existing runner iterates by
default). Until that hook lands, this lane's own smoke (`research/runners/_lbf_rows_live_organs_smoke.py`) reads
`EXTRA_LESIONS`/`EXTRA_PROBES`/`EXTRA_TURNS` directly and drives the real handler without touching either literal
registry -- the same in-process approach the sibling A2/`learning.py` smoke uses.

## The ten rows, their kind, and why

Full detail (source citations, exact field paths, the reasoning per row) lives in `EXTRA_LESIONS`'s own `note`
field in `research/runners/lbf_rows/live_organs.py` -- this section summarizes the verdicts.

**Six register `kind="neural-lesion"`** (a genuine dedicated cut, default-ON, a driving turn identified from
source): self-schema (turn `rich_open`, field `authorship.is_self`/`label`), affective-tom (new turn "Maria is
devastated", field `affective_tom.reason`/`tone_level` -- NOTE the graded-circumplex branch is default-ON, so the
compared field is `reason`/`tone_level`, NOT the bistable `tone_sign`), causal-whatif (a new taught-then-asked
session, field `abstained`/`causal.confirmed`), spiking-anaphor (reuses the existing `bc_a`->`bc_b` pair, field
`abstained`/`recalled_svo`), gnw-bus (reuses the existing `question` turn, field `abstained` -- backed directly by
the 2026-08-13 flip finding's own production-verify claim that a bus lesion collapses the ANSWER, not just the
opt-in debug trace), multiref-competition (reuses the existing `hold` turn, field `multiref.all_recovered`), and
affect-appraisal-interoceptive (reuses the existing `emo` turn, field `affect.valence_sign`, dissociated from the
existing affect-coloring row's DIFFERENT lesion via a distinct `intero_lesion` parameter on the same read).

**Four register `kind="thin"`** after two probe designs each, per the plan's own explicit fallback ("a row still
unexercisable after 2 probe designs is registered kind='thin' with its reason") -- never silently folded into a
false "not load-bearing":
- **spiking-qroute**: the route-dispatch WTA's winning route is never attached to any response field, and the
  two KB-grounded candidate routes (RELFRONT/KBREL) are structurally unreachable (the tiny-demo brain_chat build
  carries no Wikidata-style KB).
- **learned-referent**: needs a companion enable flag (`BRAIN_LEARNED_REFERENT_LEXICON`, default-OFF) the
  single flag/value row shape has no slot for; measuring it via this flag ALONE would silently read "pass" on a
  no-op (the false-negative the instrument's own docstring warns against). The concurrent, already-preregistered
  `research/findings/2026-09-24-d6-multiref-wm-learned-referent-env-flag-route-PREREGISTERED.md` (a sibling
  lane) already measures this correctly with both flags set.
- **onebrain-xedge**: the credit-trace field available on the existing hold->held turns is insensitive to this
  particular lesion by construction (it reads SVO structure, not the cross-edge); the cross-edge's real target
  (a near-threshold comprehension-margin nudge) needs a probe tuned to an operating point this lane's compute
  could not confirm in the time available (see the Resource residual below).

## Honesty boundary / declared residuals

- **No felt-experience claim.** Every compared field is a functional decision read-out (abstained/label/
  reason/valence_sign/...), never a claim of subjective experience.
- **cfg.seed.** Every arm this smoke builds threads the seed through `BRAIN_CHAT_SEED`, matching
  `load_bearing_fraction.main()`'s existing convention (never `actual_seed_used`).
- **Brain-based-only.** Every row lesions a `BRAIN_<X>_LESION` flag that cuts a specific synaptic/spiking
  pathway inside an already brain-based organ; nothing here adds host computation between sensation and action.
  Two of the six neural-lesion rows reuse an EXISTING host regex/pattern-match SURFACE gate that was already
  declared elsewhere as the organ's own host boundary (affective-tom's `detect_other_agent`, spiking-anaphor's
  token normalization before the substrate lookup) -- not a new residual, inherited unchanged from those organs'
  own documentation.
- **RESOURCE RESIDUAL (declared up front; see the Amendment for what it turned into).** At filing time this box
  was running ~13 concurrent agent lanes (the midnight plan's own dispatch: 10 sonnet + 3 opus builds, per S07)
  on one 46 GB / 20-core machine; `bash tools/mem_ok.sh` refused or barely cleared repeatedly and `uptime` read
  load average 22-35 on 20 cores. `ssh pool2` did not resolve from this worktree (the designated overflow lane
  per the plan's COMPUTE rules). This preregistration is filed BEFORE attempting the seed-7 smoke for exactly
  this reason -- so the design is locked in independent of whether the smoke completes, stalls, or must be
  deferred to a less-contended window.

## What would fail this design, and the fallback

If a `neural-lesion` row's compared field turns out NOT to move under lesion once actually run (e.g.
causal-whatif's chat-taught chain fails to ground for a reason the direct-`composer.store()` verify script never
hit), that row is downgraded to `thin` with the measured reason, in an amendment to this document -- never
silently reported as "pass" (not load-bearing) when the true cause is an unconfirmed construction.

## Commands (staged; ready to run once RAM/pool2 clears -- see the Resource residual)

Seed-7 smoke, one row at a time (lightest first) or all seven `neural-lesion` rows in one sequential pass:
```
bash tools/mem_ok.sh 6 4 && OMP_NUM_THREADS=1 bash tools/memcap.sh 6 -- \
    .venv/bin/python -u -m research.runners._lbf_rows_live_organs_smoke --seed 7 \
    --out-dir research/findings/raw/_lbf_rows_live_organs --only self-schema
# repeat --only for: affective-tom, causal-whatif, spiking-anaphor, gnw-bus, multiref-competition,
# affect-appraisal-interoceptive -- or omit --only to run all seven sequentially in one invocation.
```
On pool2 (once reachable), per the plan's own recipe:
```
POOL_PROVISION_ALLOW_STALE=1 bash tools/pool_provision.sh --isolated --revision <this-branch-head> pool2
ssh pool2 'cd ~/derisk-pool/revisions/<sha> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m \
    research.runners._lbf_rows_live_organs_smoke --seed 7 --out-dir research/findings/raw/_lbf_rows_live_organs'
```

Staged 6-seed jobs for the orchestrator's B2b pool battery (NOT run by this lane), once AG-REG's import hook
merges `research/runners/lbf_rows/*.py` into `FACULTY_LESIONS`/`FACULTY_PROBES`:
```
cd ~/derisk-pool/revisions/<M1-or-F-sha> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m \
    research.runners.load_bearing_fraction --only <row-key> --repeats 2 --seed <s> \
    --out research/findings/raw/_load_bearing/lbf_rows_live_organs/<row-key>/s<s>/lb.json   # mem_gb=<measured>
```
for `<row-key>` in self-schema, affective-tom, causal-whatif, spiking-anaphor, gnw-bus, multiref-competition,
affect-appraisal-interoceptive, and `<s>` in 42 43 44 100 101 102.
