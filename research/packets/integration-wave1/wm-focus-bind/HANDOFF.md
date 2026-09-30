# HANDOFF packet: wm-focus-bind (integration wave 1)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-29-iw1-wm-focus-bind-flip-PREREGISTRATION.md`.

**What flips.** The anchor `_MULTIREF_FOCUS_BIND_DEFAULT_ON` in `research/runners/d6_multiref_wm_production_organ.py`
turns on `BRAIN_MULTIREF_FOCUS_BIND`: after a turn that introduces two referents, an anaphor ("what does it chase")
resolves to the referent a spiking 5-way lateral-inhibition WTA retrieves from the held registers, instead of the
positional first candidate. Ledger row `wm-binding-advanced`. Escape after the flip: `BRAIN_MULTIREF_FOCUS_BIND=0`.

**Why it was chosen.** Its registered probe is reply content (`recalled_svo`, `abstained`) and read GO 6/6
(`2026-09-25-wm-referent-focus-bind-GO-6seed.md`, independently re-derived); the finding names the missing legs
(combined battery, production default). The reply-level re-score lists `wm-binding-advanced` as trace-only; this is
its route to a reply. Phase: a host position decision becomes a spiking competition's outcome.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_wmf_<P7>.txt` (6 lines; each line is 32 fresh builds, 2-3 h, one per node), tag `iw1-R-wmf-<P7>`,
shard dir `wm-binding-advanced`, record `wm-binding-referent-focus`. The probe sets the flag itself (ON `=1`,
OFF `=0`). Score: `tools/iw1_score.py R --packet wm-focus-bind ...`.

| R verdict | per-seed detail | do |
|---|---|---|
| PASS | 6/6: `verdict regressed`, ON gate `regressed`, OFF gate not load-bearing, `failing_direction_ok` | into the flip list (if D0 and D_wm-focus-bind pass) |
| FAIL | a seed's ON gate reads `pass` | not flipped. Draft `research/findings/<date>-iw1-wm-focus-bind-reply-level-NO-GO-<n>of6.md` (provisional) from `R_wmf.json` |
| UNDEFINED | `probe-inadequate:*`, `lesion-not-effective`, `noisy*`, `off-organ-route`, `content-dependent-effect`, provenance, missing | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw1 wmf R UNDEFINED: <seed>: <verdict>" |

## Production default (shared steps B and F)

No new battery row (the paired-session probe is excluded from the headline by design). Its production-default check is
F2: `default@F` equals `on:<flipped>@F` through the real handler on the identity script, which runs "the dog and the cat walked in" -> "what does it chase" and the swapped order in its own session
(the probe's pair A). B1 still applies to every other faculty, including
`wm-binding-advanced`'s own battery row.

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw1-wm-focus-bind-default-on-flip-validated-PROVISIONAL.md`, frontmatter
`integration_faculty: wm-binding-advanced`, `status: provisional`; sections as in the d6 packet's template, with
the R table from `R_wmf.json` (ON/OFF gate verdicts per seed) and the F2 line. The declared residuals from the
prereg are copied verbatim (winner set by intrinsic excitability, not discourse salience).

## Dev seed 7 (Claude, 2026-09-29)

The full 32-build probe was not re-run locally (2-3 h per seed; it is step R on the pool, and its 6-seed GO at
4da72fd23 used the same organ code, which differs at this packet only by the anchor, pinned by the truth-table
tests). Dev evidence here is the real-handler identity script, seed 7, commit 6f4c02368
(`research/findings/raw/_iw1/dev_s7/identity/`, verdicts in `../verdicts/`): `default` equals `off` on all 13
turns (byte-identical-off), and `on:wm-focus-bind` differs from `default`. The reply that changes is the swapped-order
session: after "the cat and the dog walked in", "what does it chase" answers "The dog chases cat." by the positional
route and "I don't know about that." with the flag on (the WTA retrieved "cat", and no stored fact has the cat
chasing). The first ordering answers "the dog chases the cat" either way.

## NEEDS CLAUDE (packet-specific)

1. Once default-on, should `LB_WMB_FOCUS_PROBE` count in the headline fraction (kind `neural-lesion-opt-in` today)?
   A harness decision; until then the faculty's reply-level evidence lives in this packet's R record.
2. Product behavior after a flip: an anaphor now lands on whichever held referent wins the WTA (set by intrinsic
   excitability, not salience), so some "it" questions that the positional route answered will abstain (dev seed 7:
   "what does it chase" after "the cat and the dog walked in"). Rule whether that is acceptable before the merge.
