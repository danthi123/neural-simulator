---
type: finding
status: live
date: 2026-09-23
lane: load-bearing
mechanism: PRE-REGISTRATION of an adequate load-bearing probe for wm-binding-advanced (the D6 multi-referent working-memory organ, lesion BRAIN_MULTIREF_LESION = slow-NMDA recurrence 0) -- a two-turn exchange whose second reply is the organ's hold-query read-out, gated by two intact-arm adequacy conditions and a one-referent specificity control; opt-in flag LB_WMB_HOLDQUERY_PROBE (default OFF)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any LB_WMB_HOLDQUERY_PROBE arm was built). No result is claimed here.
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json
  - research/findings/raw/_load_bearing/_shards/allfixes2/s42/wm-binding-advanced/intact_a_hold_held.json
  - research/findings/raw/_load_bearing/_shards/allfixes2/s42/wm-binding-advanced/lb.json
---

# wm-binding-advanced: an adequate hold-query probe, PRE-REGISTRATION (2026-09-23)

**Filed in its own commit before any arm with `LB_WMB_HOLDQUERY_PROBE=1` was built.** Branch
`research/wm-binding-adequate-probe`, from main `f35196e66` (the pinned pre-change SHA for the byte-identity check).

## What was SEEN when this was written

- In the 6-seed all-fixes battery, `wm-binding-advanced` reads NOT-EXERCISED on every seed
  (`research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json`). Its probe turn `held` ("the wolf watches
  the owl") carries no `multiref` key. The reason is two-fold: "owl" is not on the organ's hand-typed
  `_REFERENT_NOUNS`, so `judge()` sees one referent and returns None; and the turn exits through the
  comprehension-repair clarification.
- In the same s42 shard, the PRECEDING turn `hold` ("the fox and the wolf walked in") does carry the organ's maintain
  record: `n_referents` 2, `recovered` fox and wolf, `all_recovered` true
  (`research/findings/raw/_load_bearing/_shards/allfixes2/s42/wm-binding-advanced/intact_a_hold_held.json`). The
  organ loads two referents in the integrated brain. Nothing reads them back into a reply.
- The language lane (branch `research/language-lane-next`, commit 078bd16b3, NOT on main) exercised the `held` turn
  with its learned referent detector (`BRAIN_LEARNED_REFERENT_LEXICON`) plus `BRAIN_MULTIREF_ON_ABSTAIN`. There
  the battery field `multiref.n_referents` did not change under the lesion (it counts extracted referents), while
  the organ's own `recovered` went from wolf/owl to null/null. That lexicon is not on main, so this probe does not
  use it.
- Organ level (2026-08-12 finding, 6 seeds): the lesion collapses k=2 read-back from 1.000 to 0.000.

So the outcome on the reply is predictable IF the integrated brain routes the ask to the organ. This is written with
all of the above in view.

## What the probe measures, and what it does not

The organ's only path to the reply is its hold-query read-out (`webapp/server.py`, the `_D6.is_hold_query(msg)`
branch): "who are we talking about" is answered by reading every held referent back off the spiking buffer. The
probe (flag `LB_WMB_HOLDQUERY_PROBE=1`, label-only turns in `_EXTRA_TURNS`, NOT in the default roster):

| session | turn | message |
|---|---|---|
| `wmb` | `wmb_intro` | the fox and the wolf walked in |
| `wmb` | `wmb_ask` | who are we talking about |
| `wmb1` (control) | `wmb1_intro` | the fox walked in |
| `wmb1` (control) | `wmb1_ask` | who are we talking about |

Both referents are on the organ's hand lexicon; no new noun is added to it.

**Evidential value (declared up front).** Given the organ-level result, a reply change under the lesion is expected
once the ask reaches the organ. The measurement's new information is integration and specificity: that the reply
to this ask comes from the organ in the full brain on each seed, that the intact brain reads back BOTH referents,
and that the lesion changes nothing when the organ is out of scope. It is not new evidence that the hold works.

**Not claimed.** It does not show that the organ drives ordinary content replies (pronoun resolution, answer
choice). The hold-query is an introspective read-out route. It does not show a spiking hold ACROSS turns: the
referent identities carried from `wmb_intro` to `wmb_ask` live in the organ's host codebook (`_slot_of_ref`); each
load resets the buffer and re-writes it, so the spiking hold carries the referents over the within-load
write->hold->hold span only.

**Host shortcuts on this path (declared).** Referent extraction is a host lexicon; the register read is a host argmax
over per-pool firing rates; the read-out sentence is a host template; the referent->slot bind is the host RUNG6c
binder. The lesion cuts the spiking hold only; all of these are identical in both arms.

## Pre-registered decision rule (per seed)

Arms per seed: intact `a`, intact rebuild `b` (null control), lesion `BRAIN_MULTIREF_LESION=1`, lesion rebuild
(`--repeats 2`), plus the control group's intact `a`, intact `b` and lesion. numpy backend, seed via `--seed`.

- **T (treatment):** `answer` on `wmb_ask` differs intact `a` vs lesion. The decision field is the REPLY only.
- **N (null):** `answer` on `wmb_ask` identical intact `a` vs intact `b`.
- **R (reproduce):** the lesion rebuild gives the same verdict.
- **A1 (route):** intact `a` `wmb_ask` carries `multiref.kind == "query"` and `multiref.is_hold_query == true`.
- **A2 (two referents):** intact `a` `multiref.n_referents >= 2` AND its `answer` names both "fox" and "wolf".
- **S1 (specificity):** on `wmb1_ask`, intact `a` is NOT a multiref query read-out, intact `a` == intact `b` on
  `answer`, and intact `a` == lesion on `answer`.

`load_bearing = true` iff T, N, R, A1, A2 and S1 all hold. T false with the rest holding reads `pass` (NOT
load-bearing) and counts as a real negative. Any failed A1/A2/S1 makes the seed UNDEFINED (`probe-inadequate:*`,
`control-inadequate`, `noisy-null-control`, `off-target-lesion`), never a pass and never a negative. The gate logic is
exercised in both directions by `load_bearing_fraction --selftest` (the `wmb gate:` checks).

**Report-only (not gated):** per arm, `multiref.recovered`, `all_recovered`, `hold_alive_min`. The lesion is recorded
as holding at the moment of measurement iff the lesion arm's `hold_alive_min` on `wmb_ask` is 0.0.

## Headline rule (6 seeds: 42 43 44 100 101 102)

- **GO** (wm-binding-advanced is load-bearing under this adequate probe with an opt-in flag): `load_bearing = true`
  on 6 of 6 seeds.
- **PARTIAL**: 4-5 of 6 true with the rest UNDEFINED or negative; reported per seed, not a GO.
- **NO-GO**: 3 or fewer true.
- Any seed UNDEFINED means the probe is not adequate on that seed; the reason is reported.

Per `docs/TERMS.md` and `docs/BUILD_LANE_CHECKLIST.md`: a GO here does NOT grow the robust core. It is "load-bearing
under the adequate probe with an opt-in flag".

## Byte-identity (flag OFF)

Asserted in data against the pinned SHA `f35196e66`: `load_bearing_fraction --only wm-binding-advanced --repeats 2`
at seed 42 with the flag unset, run from an extracted `f35196e66` tree and from this branch; sha256 of every arm
file and of the per-faculty record must match exactly. Also: `PROBE_TURNS` and `FACULTY_PROBES` hash identically.

## Compute

1-seed local smoke (s42) under `tools/mem_ok.sh 12` and `tools/memcap.sh 12`, one brain build at a time. The 6-seed
run is staged on the mini-PC pool (`tools/pool_provision.sh --isolated --revision <sha> pool41 pool42`), one
invocation per seed, each with its own `--out` directory under
`research/findings/raw/_load_bearing/wmb_holdquery/s<seed>/`.
