---
type: finding
status: no-go
claim_check: measured
date: 2026-09-24
lane: load-bearing
mechanism: wm-binding ordinary-content probe (LB_WMB_CONTENT_PROBE; edge-confined recurrence lesion
  BRAIN_MULTIREF_LESION_SCOPE=recur of the multi-referent WM organ; counted under wm-binding-recurrence-drive per
  AMENDMENT C)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-wm-binding-ordinary-content-probe-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_load_bearing/wmb_content/s42/lbf.json
  - research/findings/raw/_load_bearing/wmb_holdquery/offflag_byte_identity.json
verdict: NO-GO, 6 of 6 seeds read `pass` (a real negative). Every validity condition held on every seed and both
  content sessions, the lesion killed the organ's hold, and the ordinary reply did not change. As registered - the
  organ's held state does not reach an ordinary reply; only its introspective (hold-query) read-out does.
---

# wm-binding ordinary-content probe, 6 seeds: the held state does not reach an ordinary reply

## Result

Per-seed records: `research/findings/raw/_load_bearing/wmb_content/s42/lbf.json` and the same file under `s43/`, `s44/`,
`s100/`, `s101/`, `s102/`. Runs at pool revision `c5c0f67ba`, numpy backend.

On every seed and in both content sessions (A = fox/wolf, B = cat/dog) every pre-registered condition passed in order:
build, R1 (organ in scope on the intro on every arm), R2 (the drive reply is ordinary on every arm), L (the confined
lesion killed the hold), N (clean null control), C (the intact reply follows the input), R (the lesion reproduced).
The test T (intact reply differs from lesion reply) was false in both sessions on all six seeds, so each seed reads
`load_bearing: false`, `verdict: pass`.

By the headline rule, 4 or more `pass` seeds give the registered sentence: *the organ's held state does not reach an
ordinary reply; only its introspective (hold-query) read-out does.* Per AMENDMENT C the measurement is reported under
the key `wm-binding-recurrence-drive`, not `wm-binding-advanced`. The code at `c5c0f67ba` still wrote
`wm-binding-advanced` into these records; the amendment's key applies to how they are counted and reported.

**Merge gate, in data:** `research/findings/raw/_load_bearing/wmb_holdquery/offflag_byte_identity.json` reports
`byte_identical_off: true`. With every probe flag unset, all four arm files, the per-faculty record and the probe
roster hash identically between the pinned pre-change revision (`0c265b93d`) and the branch revision (`c5c0f67ba`).
The branch's final head adds only the AMENDMENT C key relabel on the content-probe path, which is inactive with the
flags off.

## What it means

The ordinary reply path reads the WM focus through a positional pool (`CAND_POOLS[0]`), not through which referent
the organ holds, so killing the hold leaves that reply unchanged. This is the probe doing its job: a hollow coupling
reads as not load-bearing.

## Next (named in the prereg, NO-DEFER)

A spiking referent -> focus binding replaces the positional `CAND_POOLS[0]`: the focus becomes the register whose held
bump is live, read off the organ's firing state, with a spiking pronoun-resolution read-out so an ordinary answer
("who was tired" after "it was tired") depends on which referent the buffer holds. It also retires the host
`_slot_of_ref` cross-turn carry.

## Honesty

Functional read-outs only. "Holds" and "reply" name the organ's measured firing and the chat output; no felt state is
asserted.
