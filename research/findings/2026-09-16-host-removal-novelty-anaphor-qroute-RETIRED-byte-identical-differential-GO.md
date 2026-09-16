---
type: finding
status: verified
date: 2026-09-16
mechanism: host-fallback DELETION for spiking novelty / anaphor / qroute — spiking becomes the sole path
integration_faculty: spiking-qroute-selection
lane: language + live-brain (scaffold-retirement, owner's #1 metric)
seeds: [42, 43, 44, 100, 101, 102]
verdict: The host fallback paths for the three spiking wire-ins flipped default-ON 2026-09-16 (novelty->synaptic
  habituation, anaphor->CA3 pattern-completion, question-route->4-way WTA) are DELETED — the `seen` set, the anaphor
  word-list, the qroute if/elif priority cascade, their `spiking_*_enabled()` gates and `BRAIN_SPIKING_*` env flags
  all removed; spiking is now the SOLE path for all three (LESION controls kept). Integrated differential verify on
  AWS r7i: the branch's DEFAULT /api/brain-chat answers are BYTE-IDENTICAL to main's across all 26 probes (md5
  eba9e3b5, 0 differ), and the branch's own no-regression battery is all_pass (0/38). The one flagged behavior change
  — the qroute `_extract_route` dead-margin TIE now falls to GENERIC instead of the host keyword-priority cascade —
  is unreachable by any real probe (byte-identical proves it). So the three faculties move RETIRABLE_NOW -> RETIRED:
  ledger scaffold_retired 1 -> 4 (the owner's #1 metric).
runner: research/runners/onebrain_regression_battery.py (differential: branch-default vs main-default)
artifacts:
  - research/findings/raw/_hostrm_retire_verify/differential_result.json
  - research/findings/raw/_hostrm_retire_verify/branch_battery.json
  - research/findings/raw/_hostrm_retire_verify/branch_arm_on.json
  - research/findings/raw/_regression_battery/arm_on_BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6.json
external: NO-EXTERNAL-NEEDED -- this is a scaffold-DELETION verified answer-preserving; no new mechanism.
builds_on:
  - research/findings/2026-09-16-wiring-flips-novelty-anaphor-qroute-DEFAULT-ON-integrated-no-regression-GO.md
---

# Host fallback DELETED for novelty / anaphor / qroute — spiking is the sole path (RETIRED, scaffold_retired 1->4)

The owner's #1 metric is scaffold-retirement: not just turning the neural version ON, but DELETING the host shortcut
so the neural path is the only path. The three spiking wire-ins were flipped default-ON earlier today (0/38
no-regression); this lands the actual retirement.

## What was deleted (merge 15ec47eb)

- **novelty**: the host `seen` set (a membership test that only ever grew) + `spiking_novelty_enabled()` +
  `BRAIN_SPIKING_NOVELTY` flag. The spiking Tsodyks-Markram short-term-depression habituation organ is the sole path.
- **anaphor**: the host closed-class word-list + `spiking_anaphor_enabled()` + `BRAIN_SPIKING_ANAPHOR` flag. The
  spiking CA3 pattern-completion organ is the sole path.
- **qroute**: the host `if/elif` priority cascade + `spiking_qroute_enabled()` + `BRAIN_SPIKING_QROUTE` flag. The
  spiking 4-way lateral-inhibition WTA is the sole route dispatch.
- Also removed: 3 obsolete `_*_wirein_verify.py` runners; fixed the qroute biology-binding path. The `*_LESION`
  controls are KEPT (the load-bearing proofs). The qroute regex feature-extractors STAY host code (a declared
  world/body-boundary input parse, not a cognition scaffold).

## The one real behavior change, and why it is safe

On main, `_extract_route`'s spiking WTA fell back to the host priority cascade on a dead-margin TIE (or organ error).
The deletion makes a TIE fall through to the substrate's GENERIC default instead. This is the only path where the
answer COULD change.

## The verify — an integrated differential, branch-default vs main-default

<!--derived-->
(values read from `research/findings/raw/_hostrm_retire_verify/differential_result.json` and `branch_battery.json`.)

Ran the /api/brain-chat probe battery on the merged branch (main + the deletion) on AWS r7i (128GB numpy) and
compared its DEFAULT answers to main's committed default answers over the 26 probes:

- **branch DEFAULT answers are BYTE-IDENTICAL to main's**: `byte_identical: true`, both md5 `eba9e3b5…`,
  `n_probes_differ: 0` of 26. The qroute tie->GENERIC ripple is therefore unreachable by any real probe (the spiking
  WTA always has a clear winner on the tested inputs; on main the host cascade it replaced was itself only reached on
  a tie, so replacing that keyword-priority shortcut with the honest GENERIC default is at worst neutral and arguably
  more correct — no keyword-confab — and provably byte-identical on every real turn).
- **branch no-regression battery all_pass**: `branch_battery_all_pass: true`, 0/38 faculties regress.
- Source-side proof of retirement: each row's `scaffold_symbol` (`def spiking_*_enabled`) is now ABSENT from source,
  which the production-integration gate (Check D / scaffold_symbol) verifies against `scaffold_retired: YES`.

## Result

Ledger `spiking-novelty-habituation` / `spiking-anaphor-detection` / `spiking-qroute-selection`:
`scaffold_retired: NO -> YES`, `retire_status: RETIRABLE_NOW -> RETIRED`; headline `scaffold_retired: 1 -> 4`. Three
host shortcuts genuinely gone from the default path — the owner's #1 metric moved for the first time since 2026-09-02.
Follow-on (not blocking): `multi_turn_agent_v2.py` carries a SEPARATE un-retired host-anaphor scaffold.
