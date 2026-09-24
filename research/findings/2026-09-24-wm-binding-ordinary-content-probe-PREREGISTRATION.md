---
type: finding
status: live
date: 2026-09-24
lane: load-bearing
mechanism: PRE-REGISTRATION of an ordinary-content load-bearing probe for wm-binding-advanced (the D6 multi-referent working-memory organ) that CAN fail -- two content-swapped sessions (fox/wolf, cat/dog) that load two referents and then ask an ordinary transitive, with an EDGE-CONFINED lesion (only the organ's w_k->w_k slow-NMDA synapses zeroed on the shared one-brain slice); opt-in flags LB_WMB_CONTENT_PROBE and BRAIN_MULTIREF_LESION_SCOPE=recur (both default OFF)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No result is claimed here. Supersedes the hold-query probe as the load-bearing measurement for this faculty; the hold-query probe is relabelled an INTEGRITY SMOKE.
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_load_bearing/wmb_holdquery/s42/lbf.json
  - research/findings/raw/_load_bearing/wmb_holdquery/s42/intact_a_wmb_intro_wmb_ask.json
---

# wm-binding-advanced: an ordinary-content probe that can fail, PRE-REGISTRATION (2026-09-24)

Branch `fix/wm-binding-adequate-probe-r2` (a fix round of `research/wm-binding-adequate-probe`), merged with
origin/main `36a175534` (then `0c265b93d`, see the AMENDMENT LOG), the pinned pre-change SHA for the flag-OFF check.

## Why the earlier probe is not evidence

The adversarial review (key `v2:7a3b94367`) found the hold-query probe
(`2026-09-23-wm-binding-holdquery-adequate-probe-PREREGISTRATION.md`) to be PASS-BY-CONSTRUCTION. The reply
to "who are we talking about" is `hold_readout(recovered)`, a template whose only input is the buffer the
lesion disables. Once the route is reached, a reply change is guaranteed. The review also found three more
problems:

- The cross-turn referents come from the host dict `_slot_of_ref`, not from spiking activity.
- The default lesion was not confined to the claimed edge. It moved the buffer to a private bridge, skipped
  `read_isolation`, and withheld the xedge focus.
- The route check ran only on the intact arm.

The s42 smoke (`research/findings/raw/_load_bearing/wmb_holdquery/s42/lbf.json`) is therefore an integrity
smoke. The code now records it as `verdict: integrity-smoke`, `load_bearing: null`, `integrity_smoke: true`.
The fraction excludes it from both the numerator and the denominator. The 6-seed hold-query runs dispatched
from revision `624e90674` are superseded and carry no weight.

## The edge-confined lesion (new, default OFF)

`BRAIN_MULTIREF_LESION_SCOPE=recur`, together with `BRAIN_MULTIREF_LESION=1`, applies to a shared one-brain
slice (production default: the xedge pool). It zeroes in place exactly the synapses whose pre- and
post-synaptic neurons lie in the same organ register pool `w_k`. These are the slow-NMDA self-recurrence the
organ's hold is claimed to rest on. The code is
`MultiReferentWMOrgan._set_recur_lesion`, and it uses the same idiom as the xedge `lesion_cross`. Everything
else is unchanged and runs the intact code path:

- the buffer object (`shared=self._shared`)
- the `read_isolation` guard
- the `_own_focus` xedge focus
- the semantic-drop drive
- the host parse and binder

When the knob is unset, behaviour is byte-identical to before.

Unit check (exploratory, seed 42, `organ_check.py` in the session scratchpad):

| arm | recovered | hold_alive_min | focus |
|---|---|---|---|
| intact | fox, wolf | 0.0611 <!--derived--> | w0 |
| confined lesion | null, null | 0.0 | w0 (unchanged) |

The confined lesion zeroes 10,819 synapses (≈ 30 pools × 20 × 20 × 0.9). Restoring them gives read-back
identical to intact. The pre-existing organ-scope lesion also reads null/null, but its focus is None: that is
the confound the review named.

## The probe (flag `LB_WMB_CONTENT_PROBE=1`, label-only turns, NOT in the default roster)

| session | turn | message |
|---|---|---|
| `wmc` | `wmc_intro` | the fox and the wolf walked in |
| `wmc` | `wmc_drive` | the wolf watches the owl |
| `wmcx` (content-swapped control) | `wmcx_intro` | the cat and the dog walked in |
| `wmcx` (content-swapped control) | `wmcx_drive` | the dog watches the owl |

Why this drive turn: in the integrated brain the organ's held state reaches an ordinary reply by one path
only. The session's `_own_focus` is passed as `wm_focus` into the comprehension organ. There the held WM pool
is co-driven through the one-brain d6->comprehension cross-edge (`_xedge_codrive`, `_wm_resolved_role`), and
the cross-edge learns per turn (`credit_live_turn_from_comprehension`). An ordinary transitive reaches that
path. The hold-query does not; it short-circuits. The organ can influence this reply but does not decide it
alone: the comprehension content cues, recall and the moat all contribute. So the lesion can leave the reply
unchanged, and the probe can fail.

Arms per seed, for EACH content session: intact `a`, intact rebuild `b`, lesion, and lesion rebuild. Every
arm sets `BRAIN_MULTIREF_LESION_SCOPE=recur`, which is a no-op without the lesion flag, so the only
difference between the intact and lesion arms is `BRAIN_MULTIREF_LESION=1`. That makes 8 brain builds per
seed. The runs use the numpy backend and set the seed with `--seed` (`BRAIN_CHAT_SEED`).

## Pre-registered decision rule (per seed; `_wmc_gate`, checked in both directions by `--selftest`)

The conditions below are evaluated for each content session c in {A = fox/wolf, B = cat/dog}. They are checked
in the order listed, and the first one that fails names the UNDEFINED verdict. An UNDEFINED verdict is never
a pass and never a negative.

1. **build**: every arm built, with no per-turn `_error`. Otherwise `arm-build-failed`.
2. **R1**: the organ is in scope on the intro on EVERY arm (`multiref.kind == "maintain"`,
   `n_referents == 2`). The lesion arms must also carry `multiref.lesion_scope == "recur"`. Otherwise
   `probe-inadequate:route`.
3. **R2**: the drive reply is ORDINARY on EVERY arm: no `inner_state_readout` and no multiref hold-query.
   Otherwise `probe-inadequate:not-ordinary`.
4. **L**: on both lesion arms, the intro's `multiref.hold_alive_min == 0.0`, i.e. the confined lesion killed
   the hold. Otherwise `lesion-not-effective`.
5. **N**: intact `a` == intact `b` on `answer`. Otherwise `noisy-null-control`.
6. **C**: the intact reply follows the input. It names at least one of c's own referents and none of the
   other session's. Otherwise `probe-inadequate:content`.
7. **R**: lesion == lesion rebuild on `answer`. Otherwise `noisy`.

**T_c** = intact `a` `answer` != lesion `answer`.

| outcome | load_bearing | verdict |
|---|---|---|
| T_A and T_B both hold | true | regressed |
| neither holds | false | pass (a real NEGATIVE, counted in the fraction) |
| exactly one holds | null | content-dependent-effect (reported, not counted) |

The decision field is the reply (`answer`) only. The following are **report-only** (`wmc_mechanism`), never
gated: per arm and per turn, `multiref.recovered` / `hold_alive_min` / `lesion_scope`, `comprehension.margin`
/ `comprehended` / `repair` / `xedge_live_learn.focus`, and `abstained`. The per-faculty record carries
`wmb_probe_flags` and `lesion_env`, and the provenance sidecar now records every `LB_*` and
`BRAIN_MULTIREF_*` env var.

## Headline rule (6 seeds: 42 43 44 100 101 102)

- **GO** (wm-binding-advanced is load-bearing on an ordinary reply under this probe with an opt-in flag):
  `load_bearing = true` on 6 of 6 seeds.
- **PARTIAL**: 4-5 of 6 true, with the rest UNDEFINED or negative. Reported per seed; not a GO.
- **NO-GO (not load-bearing on an ordinary reply)**: 3 or fewer true. If 4 or more seeds read `pass`,
  the honest verdict is: *the organ's held state does not reach an ordinary reply; only its introspective
  read-out does.*
- UNDEFINED on 3 or more seeds means the probe is inadequate. The reasons are reported, and no load-bearing
  claim either way is made.

Per `docs/TERMS.md` and `docs/BUILD_LANE_CHECKLIST.md`, a GO here does NOT grow the robust core.

## Host shortcuts on this path (declared)

- Referent extraction is the host lexicon.
- The referent->slot bind is the host RUNG6c binder, and the cross-turn referent identities live in the host
  `_slot_of_ref` (review finding 2, unchanged by this probe).
- The register read is a host argmax.
- The WM focus passed to comprehension is POSITIONAL (`CAND_POOLS[0]`, the same pool whatever referent is
  held). The WHICH-referent content therefore cannot reach the reply through this path by construction.
  Only the held bump's persistence can. This is why condition C tests that the reply follows the input and
  T tests the lesion, but no condition claims the reply follows the HELD content through the organ. That
  claim is not made.

## Next mechanism if the verdict is NO-GO (NO-DEFER)

A spiking referent -> focus binding would replace the positional `CAND_POOLS[0]`: the focus becomes the
register whose held bump is live, read off `cp_firing_states`. A spiking pronoun-resolution read-out would go
with it: an anaphor turn retrieves the held register by cue-driven competition and substitutes the resolved
referent into the parsed SVO. That makes an ordinary answer ("who was tired" after "it was tired") depend on
WHICH referent the buffer holds. It also retires the host `_slot_of_ref` cross-turn carry. The lesion would
then remove the resolution, not a template input.

## Seen before this was written (declared)

- The s42 hold-query smoke (all conditions passed; now an integrity smoke).
- The organ unit check above.
- Exploratory, un-gated s42 runs of candidate drive turns (session scratchpad, not committed; memory-throttled
  locally, so only partly complete):
  - The referent-question candidate ('the fox and the wolf walked in' -> 'the fox was tired' -> 'who was tired')
    was not usable. The intact reply was "I don't know about that. ... what can you tell me about tired?", with
    the same shape under the confined lesion. The tiny-demo brain cannot answer it at all, so condition C would
    fail. It was dropped.
  - For the chosen drive ('the fox and the wolf walked in' -> 'the wolf watches the owl'), the confined-lesion
    reply was the comprehension-repair clarification "I caught the verb 'watch' with the wolf and the owl, but
    my role-binding didn't resolve the PATIENT ...".
- The same two-turn sequence is the DEFAULT battery row (`hold` -> `held`). In the committed 6-seed all-fixes
  shard `research/findings/raw/_load_bearing/_shards/allfixes2/s42/wm-binding-advanced/` (pre-merge main), the
  intact reply is that same sentence, and the organ-scope lesion reply is identical to it. The intro turn's
  comprehension margin (0.3243) is also identical intact vs confined lesion.

**Expected outcome, declared before the run: T false (a real NEGATIVE, `pass`) on this drive.** Staging it
is not "a run the pre-registration predicts will fail". The design can fail and is expected to read
not-load-bearing. The run establishes that in data, with the edge-confined lesion, on 6 seeds and on the
content-swapped control. Seed 42 is NOT held out, so the headline is also reported on the 5 seeds
{43, 44, 100, 101, 102} alone.

## Byte-identity (flags OFF)

The check is asserted in data against the pinned SHA `36a175534` (amended to `0c265b93d`, see the AMENDMENT LOG):
`load_bearing_fraction --only wm-binding-advanced --repeats 2` at seed 42, with `LB_WMB_HOLDQUERY_PROBE`,
`LB_WMB_CONTENT_PROBE` and `BRAIN_MULTIREF_LESION_SCOPE` all unset. It is run from an extracted tree at
`36a175534` and from this branch's revision. The sha256 of every arm file and of the per-faculty record must
match exactly, and `PROBE_TURNS` and `FACULTY_PROBES` must hash identically
(`research.runners._wmb_offflag_byte_identity`, which must report `byte_identical_off: true`). No merge
happens before this is in data.

## Compute

The runs go to the mini-PC pool from an isolated revision provisioned on pool41 and pool42: one line per
seed, `mem_gb=6`, each with its own `--out` directory under
`research/findings/raw/_load_bearing/wmb_content/s<seed>/`. The byte-identity job runs on one node from the
same revision.

## AMENDMENT LOG

(empty at filing; entries below)

- **2026-09-24, before ANY governed run (none had been built).** Pinned byte-identity SHA changed from `36a175534`
  to `0c265b93d`: origin/main advanced while this round was open, the branch merged it (two-parent merge
  `0e37c637a`), and `tools/pool_provision.sh` refuses a revision that does not contain origin/main, so the old pin
  could not be provisioned on the pool. `git diff --stat 36a175534 0c265b93d -- webapp sim
  research/runners/load_bearing_fraction.py research/runners/onebrain_regression_battery.py
  research/runners/d6_multiref_wm_production_organ.py` is empty, so the pin's brain and battery are unchanged.
  Nothing else changes.

- **2026-09-24, AMENDMENT B, filed before any of the 6 content-probe seeds staged on the pool at `c5c0f67ba`
  (42/43/44/100/101/102) were pulled or read.** Adversarial re-review (journal key
  `v2:7512414c9a65ead57a205e7a2e166bcf476472d9dfbc61180205447d18d460df`, label `rereview:wm-binding`) found that
  the "Why this drive turn" section above misstates the mechanism and would let a GO over-credit
  wm-binding-advanced. The corrected reading, verified directly against the code paths named:

  1. **The D6 organ does NOT run on the drive turn.** `_wmc_drive` ("the wolf watches the owl" / "the dog watches
     the owl") names exactly one referent the organ's hand lexicon admits (`wolf`/`dog`); `owl` is not in
     `_REFERENT_NOUNS` (`research/runners/d6_multiref_wm_production_organ.py`). The MAINTAIN branch that calls
     `d6org.judge(msg, lesion=d6les)` (`webapp/server.py`, the `else:` under the D6 hold-query check) returns
     `None` on any turn with fewer than 2 referents — `load()` is never invoked on the drive turn, and
     `multiref_info` stays `None`. So there is no d6-organ processing, and no fresh read of the held bumps, on the
     turn the T gate scores.
  2. **`_own_focus` is a positional constant set once, on the intro, independent of held content.** `load()`
     (called on `wmc_intro`/`wmcx_intro`) sets `self._own_focus = CAND_POOLS[0]` whenever the intro names >=1
     referent and the xedge pool is live, UNCONDITIONALLY on which referent is held or whether the hold
     survives — this branch's `lesion` variable has already been forced to `False` for the confined-lesion arms
     (`confined: ... lesion = False # every other branch below runs exactly as the intact arm's`), so `_own_focus`
     is set to the SAME value (`w0`) on the intact and confined-lesion arms alike. It is never read from, or
     conditioned on, `hold_alive_min`, `recovered`, or any other content-carrying field.
  3. **What actually reaches the drive-turn reply is a host-templated re-drive of that fixed pool, not a fresh
     read of the organ's held content.** `comprehension_production_organ.py`'s `_read`/`_read_per_noun` call
     `_hard_reset(comp)` (zeroing firing/conductances on the whole shared bridge, including the D6 slice) and then
     `_xedge_codrive(comp, wm_focus=focus)`, which injects a HOST-CHOSEN external current (`load_pa=400.0` pA for
     `load_steps=30` steps, then `hold_steps=6` steps of quiet) directly onto the neurons of `wm_focus` (`w0`)
     BEFORE the cue settle — the same pool, the same current, on every arm, whether or not the intro's hold
     content (which referent, or whether it decayed) differs.

  **What T can and cannot show.** The confined lesion (`BRAIN_MULTIREF_LESION_SCOPE=recur`) zeros exactly `w0`'s
  own `w_k->w_k` slow-NMDA self-recurrence, so `w0`'s own post-drive dynamics CAN legitimately differ between the
  intact and lesion arms after the same host re-injection — the probe is not fail-by-construction; T can be true
  or false in data. But a `T=true` (`regressed`) means only: *`w0`'s own recurrent self-synapses causally shape
  comprehension's read of a host-reinjected, host-timed, host-targeted current pulse into that fixed pool.* It
  does NOT mean, and this pre-registration's headline rule must not be read to credit, that *"wm-binding-advanced
  is load-bearing on an ordinary reply"* in the sense of the organ's HELD REFERENT CONTENT reaching that reply —
  no field carrying WHICH referent is held, or whether the hold survived, is read anywhere on this path. The
  "Host shortcuts on this path" section already declared the positional `CAND_POOLS[0]` bind and that "no
  condition claims the reply follows the HELD content through the organ" — this amendment makes that limit
  apply to the T gate itself, not only to condition C.

  **Rescoped headline rule (replaces §"Headline rule" above for the counted verdict; the seed table format is
  unchanged).** A `GO` (T true on 6/6 seeds) is reported and MAY still count in `load_bearing_fraction`'s
  numerator under the label `wm-binding-advanced` (per `FACULTY_LESIONS`'s existing `kind="neural-lesion"`
  wiring — this amendment does not change the code), but any prose reporting it MUST read: *"the confined
  recurrence lesion of the organ's register pool changes comprehension's read of a host-reinjected drive into
  that pool on an ordinary reply; this does not show the organ's held referent CONTENT reaches an ordinary
  reply — the WM focus routed into comprehension is positional (`CAND_POOLS[0]`), not content-addressed."* A
  `NO-GO`/`pass` (T false, 4+ of 6 seeds) is unaffected by this correction and reads as originally registered:
  *"the organ's held state does not reach an ordinary reply; only its introspective (hold-query) read-out
  does."* The next mechanism named in "Next mechanism if the verdict is NO-GO" (a spiking referent->focus bind
  replacing the positional `CAND_POOLS[0]`) is unchanged and is now also the prerequisite for a future GO on this
  probe to mean content-binding rather than recurrence-shapes-a-host-drive.

  No seed under `LB_WMB_CONTENT_PROBE` had been pulled from the pool or read by anyone in this round before this
  amendment was filed; this amendment governs how those results, once pulled, must be reported.
