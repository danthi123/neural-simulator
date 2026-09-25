---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: wm-referent-focus-bind anaphor probe (LB_WMB_FOCUS_PROBE; BRAIN_MULTIREF_FOCUS_BIND, default OFF -- a
  spiking referent->focus binding in the D6 multi-referent WM organ + a 5-way lateral-inhibition WTA anaphor
  resolution, reused from `affective-marker-lateral-inhibition-wta.md`); counted under `wm-binding-referent-focus`,
  kind `neural-lesion-opt-in`, excluded from the production load-bearing fraction per the prereg
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_load_bearing/wmb_focus/s42/lbf.json
  - research/findings/raw/_load_bearing/wmb_focus/s43/lbf.json
  - research/findings/raw/_load_bearing/wmb_focus/s44/lbf.json
  - research/findings/raw/_load_bearing/wmb_focus/s100/lbf.json
  - research/findings/raw/_load_bearing/wmb_focus/s101/lbf.json
  - research/findings/raw/_load_bearing/wmb_focus/s102/lbf.json
verdict: GO on the pre-registered probe, 6 of 6 seeds. Every seed reads `load_bearing: true` / `verdict: regressed`
  on the ON gate (both pairs' intact replies differ by held referent, and the confined hold lesion erases the
  difference, verified to still hold at the ask), and `verdict: pass` on the OFF gate on all 6 (the positional
  route does not pass the probe; `failing_direction_ok: true` throughout). This is a GO on an isolated,
  default-OFF diagnostic probe, not flip candidacy -- see "Flip candidacy" below, which is explicitly NOT met.
---

# wm-referent-focus-bind anaphor probe, 6 seeds: GO on the pre-registered gate

Per-seed records: `research/findings/raw/_load_bearing/wmb_focus/s42/lbf.json` and the same file under `s43/`,
`s44/`, `s100/`, `s101/`, `s102/` (frontmatter `artifacts:` lists all six paths in full).

## Liveness and completeness, checked before scoring

`ssh -n -F research/queue/.pool_ssh_config pool1|pool2 'ps -eo etimes,args | grep load_bearing_fraction'` (run
2026-09-25 ~09:03 UTC) shows no `wmb_focus` / `LB_WMB_FOCUS_PROBE` process on either pool host -- both hosts are
running unrelated jobs (a `b2b0924-base` battery and `_da_tag_capture_chat_probe`/`d6_capacity_curve` runs). The
six `s{42,43,44,100,101,102}/lbf.json.prov.json` sidecars record `started` between `2026-09-25T01:22:54` and
`2026-09-25T01:24:37` (matches the task's "dispatched 2026-09-24 21:22-21:24"), ~7h40m before this check -- the
jobs are not merely absent from `ps`, they finished long enough ago that a still-running job would have shown.

Completeness: every one of the 6 seed directories holds exactly 32 per-arm files (8 turns x 4 arm-kinds: intact_a,
intact_b, lesion, lesion_rep) plus their `.prov.json` sidecars, plus `lbf.json`/`lbf.json.prov.json` -- 198 result files (192 per-arm + 6 lbf.json), each with its own
.prov.json sidecar, 396 files in all (verified against every seed directory listing). Every one of the 192 per-arm JSON files was scanned
recursively for an `_error` key at any depth: 0 found. No arm silently truncated mid-run.

## Provenance

All 198 `.prov.json` sidecars (6 seeds x 33 files) were checked programmatically: `git_sha ==
4da72fd2316902697972917a72536d9e9def95ef` (the task's registered revision), `source_kind: git_archive`,
`source_manifest_verified_at_start` and `_at_exit` both `true` with no start/exit error, `git_dirty: false` -- on
every single file, 0 exceptions. `env.LB_WMB_FOCUS_PROBE == "1"`, `SIM_BACKEND == "numpy"` on every seed's
`lbf.json.prov.json`. This is stronger provenance coverage than the prereg's own compute recipe required (it
names one pool line per seed; the actual run additionally provenanced every one of the 32 per-arm files that
line produced).

## Result: per-seed table (pre-registered gate `_wmf_gate`, `LB_WMB_FOCUS_PROBE=1`, `--only wm-binding-advanced`)

| seed | ON: load_bearing | ON: verdict | pair A (T, X) | pair B (T, X) | OFF: load_bearing | OFF: verdict | failing_direction_ok |
|---|---|---|---|---|---|---|---|
| 42  | true | regressed | (true, true) | (true, true) | false | pass | true |
| 43  | true | regressed | (true, true) | (true, true) | false | pass | true |
| 44  | true | regressed | (true, true) | (true, true) | false | pass | true |
| 100 | true | regressed | (true, true) | (true, true) | false | pass | true |
| 101 | true | regressed | (true, true) | (true, true) | false | pass | true |
| 102 | true | regressed | (true, true) | (true, true) | false | pass | true |

T = the pair's two intact replies differ by held referent (`recalled_svo`/`abstained`). X = the pair's two
LESION replies are identical. By the prereg's outcome table, "every pair T and X" reads `load_bearing: true`,
`verdict: regressed` -- read on 6 of 6 seeds, meeting the prereg's headline **GO** rule (`load_bearing = true` on
6 of 6) with the failing-direction check also clean on all 6 (the OFF arms, run through the same gate without RES
and C, read `pass` rather than falsely reading `load_bearing: true`; a positional route does not pass this probe).

Per-pair validity conditions (build, R1 intros-in-scope, R2 asks-ordinary, RES resolved-to-held-and-differs, L
lesion-hold-dead-and-holds-at-ask, N null-clean, C reply-follows-resolution, R lesion-reproduced) all read `true`
on every pair, every seed, both gates where applicable -- no seed went UNDEFINED on any `probe-inadequate:*`
reason. The resolved referents differ by mention order as designed: pair A resolves `[dog, cat]` on seeds
42/43/44/101 and `[cat, dog]` on seeds 100/102 (both intro orders occur across the 6-seed set, as expected since
which referent lands in which register is a race the organ's intrinsic excitability decides, not a fixed rule);
pair B resolves symmetrically (`[cat, bird]` / `[bird, cat]`). The ON intact arm's WTA margin at the ask ranged
0.066111-0.166944 across the 6 seeds; the ON lesion arm's margin collapsed to 0.000278-0.001111, consistent with
the registered lesion (`BRAIN_MULTIREF_LESION=1`, `BRAIN_MULTIREF_LESION_SCOPE=recur`) killing the hold before the
ask (`hold_alive_min: 0.0` on every lesion arm, every seed) while the un-lesioned hold read `hold_alive_min`
0.058333-0.063889.

## What it does not show

This is a diagnostic probe of an isolated, default-OFF mechanism, run with `--only wm-binding-advanced` on the
numpy backend, two label-only sessions per pair, not the production chat path or the combined battery. It does
not show:

- **That the faculty is on by default, wired, or scaffold-retired** (per `docs/TERMS.md`). `BRAIN_MULTIREF_FOCUS_BIND`
  stays default OFF; nothing here touches `webapp/server.py`'s default config.
- **Referent identity living in the substrate.** The prereg's declared residual shortcuts still hold verbatim:
  the winning register's word comes from the host RUNG6c binder codebook (`_ref_of_slot`), not from anything the
  spiking WTA computes -- the organ's spiking contribution is only WHICH slot is live and which register wins.
- **A linguistically correct pronoun preference.** Which of two held referents wins is set by the pools'
  intrinsic excitability, not discourse salience (subjecthood/recency/topic); the Lewis & Vasishth (2005)
  interference-of-cue-based-retrieval framing the prereg cites for this residual is unchanged by this run.
- **A fully-spiking path.** The register-to-assembly projection is a host rate relay; the WTA read-out is an
  argmax over settled rates; referent extraction and pronoun substitution are host lexicon/string operations.
- **No regression in the combined battery with the flag ON**, or a production-default validation run -- this
  probe was not run alongside the rest of the load-bearing battery, and no `aggregate.json`/combined-battery
  artifact exists for this flag.
- **A SOUND adversarial review.** This document reports the pre-registered gate's own output; it is not itself
  the independent verify-go pass the owner's flip bar requires.

## Flip candidacy: NOT MET

The owner's bar for flipping any validated fix/faculty default-on is: 6-seed GO + SOUND review + no regression in
the combined battery with it ON + production-default validation (`feedback_flip_validated_fixes_without_waiting`).
This run satisfies only the first clause -- a clean 6-seed GO on the pre-registered isolated probe. It does not
satisfy the other three: no independent adversarial (verify-go) review has been run on this result; no combined-
battery run with `BRAIN_MULTIREF_FOCUS_BIND=1` exists to check for regressions elsewhere; and no production-default
validation (the flag flipped in the real `/api/brain-chat` config, not just the runner) has been attempted. No
default is flipped here. The prereg's own headline rule says the same thing independently: a GO here "does not
mean the faculty is on by default... or that the path is fully spiking."

## Honesty

Functional read-outs only. "Holds", "retrieves", "resolves" and "regressed" (the schema's own verdict label for
"the reply follows the held referent; killing the hold removes it") name the organ's measured firing and the
chat output. No felt state is asserted.
