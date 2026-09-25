---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: BRAIN_COMPOSER_KIND=slotbinder through the REAL production chat path (webapp.server._build_chat_brain ->
  developed_brain_io.load_developed_brain -> MultiTurnAgent -> BrainConversationalAgent -> SlotBinderComposer)
seeds: [42, 43, 44, 100, 101, 102]
instrument: control comparison on every seed -- the host FHRR composer as the reference arm (recall_ge_fhrr,
  parity_1_0), the never-taught-pair moat probe and the cross-fact mismatch probe as correctness controls, and the
  zeroed-synapse ablation as the falsifiability control (recall must COLLAPSE, not merely change)
prereg: research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md (AMENDMENT 1 + AMENDMENT 2,
  which registered N=32 as the sizing choice BEFORE this battery was queued)
artifacts:
  - research/findings/raw/_slotbinder_production_gate/n32/seed42.json
  - research/findings/raw/_slotbinder_production_gate/n32/seed43.json
  - research/findings/raw/_slotbinder_production_gate/n32/seed44.json
  - research/findings/raw/_slotbinder_production_gate/n32/seed100.json
  - research/findings/raw/_slotbinder_production_gate/n32/seed101.json
  - research/findings/raw/_slotbinder_production_gate/n32/seed102.json
  - research/findings/raw/_slotbinder_production_gate/sizing/seed7_n128.json
verdict: SlotBinder production composer gate GO 6/6, SCOPED TO N=32 -- every one of the prereg's seven
  `verdict_criteria` reads `true` on every one of the six evaluation seeds, through the real `/api/brain-chat`
  entry point, with a genuine (not merely flagged) ablation collapse on every seed. This is NOT a validation of
  the full 404-fact production corpus: N=32 is the largest size the prereg's own AMENDMENT 2 found BOTH inside
  its 10-min/seed wall-clock budget AND a full dev-seed GO; N=128 (dev-seed only, not a 6-seed claim) already
  reads NOT-YET on `parity_1_0`; N=404 was measured impractical (AMENDMENT 1: 6h08m GPU time, no artifact, killed
  to free the shared queue). Flip candidacy against the owner's bar (6-seed GO + review + no-regression battery
  before flipping a default myself): NOT YET -- the tested scope (32 of 404 facts) does not license flipping
  `BRAIN_COMPOSER_KIND`'s production default away from `onebrain`.
---

# SlotBinder PRODUCTION composer gate: GO 6/6, but only at N=32 of the real 404-fact corpus

## Provenance and job confirmation

All six jobs ran on the GPU queue (`tools/gpu_queue.sh`) from detached worktrees pinned to
`835fc252e` (the prereg's AMENDMENT 1 commit) and completed `DONE(rc=0)` in
`research/queue/gpu_queue.log` (lines 13700283-13701728, 2026-09-25 06:07:52-06:35:02). Each artifact's
`.prov.json` sidecar carries `"git_sha": "835fc252e"`, `"git_dirty": false`, and `"sim_backend": "cupy"` --
confirmed by reading all six sidecars directly, not inferred. No progress sidecar or partial file was left behind
in any of the six seed worktrees; each directory holds exactly `seed<N>.json` + `seed<N>.json.prov.json`. The six
result JSONs and their `.prov.json` files were copied byte-for-byte from
`.claude/worktrees/slotbinder-sizing-seed{42,43,44,100,101,102}/research/findings/raw/_slotbinder_production_gate/n32/`
into this worktree at the paths listed in this document's `artifacts:` frontmatter; nothing in those source
worktrees was modified or deleted.

## Per-seed result (all fields read directly from the cited artifacts)

| seed | recall (SlotBinder) | recall (FHRR) | parity_rate | moat (SB/FHRR) | mismatch (SB/FHRR) | ablation: intact -> zeroed | all 7 `verdict_criteria` | verdict |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.90625 | 0.90625 | 1.0 | pass/pass | pass/pass | 0.90625 -> 0.0 | true | GO |
| 43 | 0.90625 | 0.90625 | 1.0 | pass/pass | pass/pass | 0.90625 -> 0.0 | true | GO |
| 44 | 0.875 | 0.875 | 1.0 | pass/pass | pass/pass | 0.875 -> 0.0 | true | GO |
| 100 | 0.78125 | 0.78125 | 1.0 | pass/pass | pass/pass | 0.78125 -> 0.0 | true | GO |
| 101 | 0.875 | 0.875 | 1.0 | pass/pass | pass/pass | 0.875 -> 0.0 | true | GO |
| 102 | 0.875 | 0.875 | 1.0 | pass/pass | pass/pass | 0.875 -> 0.0 | true | GO |

Every one of the seven `verdict_criteria` (`recall_ge_fhrr`, `parity_1_0`, `slotbinder_moat_pass`,
`slotbinder_mismatch_pass`, `fhrr_moat_pass`, `fhrr_mismatch_pass`, `ablation_falsifies_intact_pass`) reads `true`
on every one of the six seeds, and each artifact's own `"verdict"` field reads `"GO"`. Per the prereg's registered
multi-seed rule ("requires all of the above `true` on every one of the six evaluation seeds"), the headline is
**SlotBinder production composer gate GO 6/6**.

**`recall_ge_fhrr` held by exact TIE, not by SlotBinder exceeding the reference**, on all six seeds
(`arms.slotbinder.recall_accuracy == arms.rf.recall_accuracy` in every artifact) -- the criterion is `>=`, and
equality satisfies it, but this is not a case of SlotBinder outperforming FHRR at this N. **The ablation is a
genuine collapse, not a flagged pass**: `arms.slotbinder.ablation_zeroed_synapses.recall_accuracy` reads `0.0` on
all six seeds against an intact recall well above zero (see table above), and every one of the 32 per-fact
ablation queries per seed returned `got_patient: null` (read directly from `seed42.json`'s `ablation_zeroed_synapses.per_fact`) -- the
zeroed-synapse manipulation removes recall entirely, satisfying the falsifiability requirement in fact, not just
in the criterion's boolean. (Intact recall ranged 0.78125-0.90625 across the six seeds, table above.)

`flagoff_check` is `null` in all six artifacts: `--check-flagoff` was correctly NOT passed for the evaluation
battery (only for the dev seed, per the prereg's "Run (fixed)" section), so this document makes no claim about
flag-off byte-identity.

## Wall-clock (measured, against the registered <=600 s/seed budget)

<!--derived-->
| seed | bundle_staging_s | slotbinder wall_s | rf wall_s | TOTAL (3 arms) | mean per-query latency (s) |
|---|---|---|---|---|---|
| 42 | 124.550 | 106.864 | 11.507 | 242.920 | 1.090 |
| 43 | 58.368 | 100.640 | 29.702 | 188.709 | 0.844 |
| 44 | 73.532 | 82.317 | 11.560 | 167.409 | 0.765 |
| 100 | 105.304 | 85.767 | 10.167 | 201.238 | 0.855 |
| 101 | 63.359 | 109.262 | 9.518 | 182.139 | 1.363 |
| 102 | 54.875 | 88.919 | 10.904 | 154.699 | 0.886 |

<!--derived-->Every seed finished well inside the registered 600 s budget (max 242.920 s, seed 42); the 6-seed
total (1137.114 s <!--derived-->, ~18.95 min <!--derived-->) came in under the prereg's own projection
(~1303.26 s) because per-seed sample variance (a different random 32-fact sub-corpus per seed) ran faster on
average than the single seed-7 sizing run the projection was based on.

## Scope: this is N=32, not the production corpus, and that gap is the headline honestly stated

The real production bundle (`day_33`) carries the **full 404-fact corpus**. This battery ran against a **32-fact
random sub-sample** of it, per seed, because AMENDMENT 1 measured N=404 as impractical on this path (6h08m of GPU
time with no completed arm, killed to free the shared queue -- no artifact exists for N=404) and AMENDMENT 2's
N=8/32/128 sizing runs (dev seed 7 only) found: N=32 is the largest size that is BOTH within a 600 s/seed budget
AND a full dev-seed GO; **N=128 already reads NOT-YET**, failing `parity_1_0` (`parity_rate: 0.9765625`, read
directly from `research/findings/raw/_slotbinder_production_gate/sizing/seed7_n128.json`) even though SlotBinder's
own three mismatching answers were the CORRECT ones and the FHRR reference arm was wrong on all three -- the
criterion is written as exact agreement with the reference, so a reference-arm degradation at larger scale still
fails it. This document's GO 6/6 is a genuine result at N=32 and says nothing, one way or the other, about
behavior at N=128 or the full N=404 -- no 6-seed battery exists at either of those sizes, and building one at
N=404 is currently blocked on the same wall-clock cost this document's own table above measures growing
super-linearly with N (AMENDMENT 2's local exponent estimate, not re-derived here).

## Flip candidacy against the owner's bar

The owner's standing bar for flipping a validated fix/faculty default-on without waiting for approval is a 6-seed
GO plus review plus a no-regression battery. This gate clears the 6-seed-GO half of that bar, but **only at
N=32**, while the production default this would replace (`BRAIN_COMPOSER_KIND` unset -> `onebrain`, per
`webapp/server.py`'s own `_COMPOSER_KIND_DEFAULT` comment) serves the real, full-corpus bundle. Recommending a
default flip on the strength of a 32-of-404-fact result would extrapolate past the one boundary this arc has
already found (N=128 NOT-YET, N=404 impractical) -- exactly the kind of extrapolation AMENDMENT 2 warns against
for the wall-clock curve, and the same caution applies to the correctness criteria. **Verdict: NOT a flip
candidate yet.** What would change that: either (a) a mechanism that brings per-query cost down enough to run a
6-seed battery at N=404 (or at least N=128, re-run across seeds rather than dev-seed-only), or (b) a prereg
amendment to `parity_1_0` that is decided BEFORE seeing more N=128+ data, not fitted to it. Absent either, this
finding's GO stands as a real result at its stated scope and is not evidence for a production-default change.

## Known residuals (carried from the prereg, unchanged by this battery)

The SlotBinderComposer's recall composer still uses exact-inverse FHRR bind/unbind arithmetic -- host arithmetic,
open since the L1-L3 findings, unchanged here. This gate exercises the real production entry point and a real
sub-corpus, which is what it was designed to add over the L3 latency de-risk; it does not close the host-arithmetic
item.

## Honesty

Functional read-outs only. No claim of phenomenal experience. "GO" here means the gate's own seven criteria read
`true` on all six seeds at N=32 -- not that SlotBinder is ready to replace the production composer default, and
not that recall improves on FHRR (it ties).
