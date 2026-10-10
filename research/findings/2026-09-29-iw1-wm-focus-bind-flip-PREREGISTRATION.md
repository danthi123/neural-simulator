---
type: preregistration
status: preregistered
date: 2026-09-29
lane: integration (wave 1, packet wm-focus-bind; ledger row wm-binding-advanced)
mechanism: WM referent->focus binding (BRAIN_MULTIREF_FOCUS_BIND, research/runners/d6_multiref_wm_production_organ.py)
  -- an anaphor after a two-referent intro resolves to the referent a spiking 5-way lateral-inhibition WTA retrieves
  from the held registers, instead of the positional CAND_POOLS[0]; now governed by the anchor
  `_MULTIREF_FOCUS_BIND_DEFAULT_ON`; measured at reply content by the registered LB_WMB_FOCUS_PROBE
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed with the anchor, before any gate-seed job exists. No result is claimed here.
---

# Packet wm-focus-bind: flip the WM referent->focus binding default-on (pre-registration)

Shared steps (D, B, F, decision table, UNDEFINED rule): `2026-09-29-iw1-wave-battery-PREREGISTRATION.md`.

## Why this faculty (selection)

- **Reply impact.** Its registered probe is a reply-content gate: two order-swapped sessions per pair introduce the
  same two referents, then ask the same anaphor question; the replies' `recalled_svo`/`abstained` must follow the held
  referent, the confined hold lesion must erase the difference, and the positional route (flag `=0`) must NOT pass.
  GO 6/6 (`2026-09-25-wm-referent-focus-bind-GO-6seed.md`, independently re-derived, merge 0e7e8bce8). The
  2026-09-24 ordinary-content probe had found the held state never reached an ordinary reply
  (`2026-09-24-wm-binding-ordinary-content-probe-6seed-NOGO-held-state-does-not-reach-an-ordinary-reply.md`);
  the reply-level re-score lists `wm-binding-advanced` as trace-only
  (`2026-09-29-load-bearing-fraction-reply-level-rescore.md`). This flip is the named route to a reply.
- **Integration risk.** The flag has one reader; this packet adds its anchor (truth table pinned by
  tests/test_integration_wave1_anchors.py). The mechanism merged after a SOUND re-review (c3c2800a3). Its own finding
  names the missing legs: "no combined-battery run ... no production-default validation".
- **Phase and stages.** Which referent the reply commits to becomes a spiking competition's outcome instead of a
  host position: a host decision moved into the brain (the 2026-09-25 host-decision share). Stage 2 exit:
  grammatical grounded conversation needs pronouns to land on what the brain holds.
- **Checked.** Not retracted (docs/RETRACTED.md, `git log --all --grep=focus-bind`). Not on by default (reader
  default OFF at P).

## Step R: reply content, capability, at P

`tools/lb_shard.py jobs --no-fixes --faculties wm-binding-advanced --extra-env LB_WMB_FOCUS_PROBE=1`, tag
`iw1-R-wmf-<P7>`, six gate seeds. The probe sets the flag itself in both modes (ON `=1`, OFF `=0`), so step R is
independent of the anchor. 32 fresh builds per seed (one per session and arm kind); one job per node.

- **R PASS:** `python tools/iw1_score.py R --packet wm-focus-bind ...` reads, on all six seeds, the record's
  provenance pinned to P and `load_bearing: true`, `verdict: regressed`, ON gate `regressed`, OFF gate not
  load-bearing, `failing_direction_ok: true` (exactly the 2026-09-25 gate).
- **R FAIL:** a seed whose ON gate reads `pass`. **UNDEFINED:** any other verdict (`probe-inadequate:*`,
  `lesion-not-effective`, `noisy*`, `off-organ-route`, `content-dependent-effect`, missing) after one re-run.
- Prediction: PASS (the 2026-09-25 result at 4da72fd23; the code between that revision and P differs in this organ
  only by the anchor).

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

There is no new battery row: the probe is a paired-session gate, kind `neural-lesion-opt-in`, which the headline
fraction excludes by design. F2 (`default@F` equals `on:...wm-focus-bind...@F` through the real handler) is this
packet's production-default check. Whether the probe should count in the headline once the flag is default-on is a
harness decision, listed as NEEDS-CLAUDE in the HANDOFF.

## Declared residuals (unchanged by a flip)

Which of two held referents wins is set by the pools' intrinsic excitability, not discourse salience; the winning
register's word comes from the host binder codebook; the WTA read-out is an argmax over settled rates; referent
extraction and pronoun substitution are host string operations; the register-to-assembly projection is a host rate
relay. Nothing here is "fully spiking" (docs/TERMS.md).
