---
type: finding
status: qualified
claim_check: measured
date: 2026-09-24
lane: load-bearing
mechanism: the v3 DA-gated synaptic tag-and-capture ledger (webapp/da_tag_capture.py SynapticTagCaptureLedger)
  wired into the live /api/brain-chat store path and the continuous engine's idle/sleep tick behind
  BRAIN_DA_TAG_CAPTURE (default OFF, webapp/da_tag_capture_chat.py); scored here is the buffer-only (LTM OFF,
  BRAIN_LTM_SHIP_DEFAULT=0) arm ONLY, per the pre-registration's declared deviation
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md (commit 6da12e683; Amendment 1
  daa4b382d/5d3810f2d; Amendment 2 6ee934b0a/32250d3ce)
artifacts:
  - research/findings/raw/_da_tag_capture_chat/seed42.json
  - research/findings/raw/_da_tag_capture_chat/seed43.json
  - research/findings/raw/_da_tag_capture_chat/seed44.json
  - research/findings/raw/_da_tag_capture_chat/seed100.json
  - research/findings/raw/_da_tag_capture_chat/seed101.json
  - research/findings/raw/_da_tag_capture_chat/seed102.json
verdict: GO 6/6 seeds under the pre-registered gates (G0, P1, G1-G6, G_isolation_gamma_consistent), LTM-OFF
  (buffer-only) arm ONLY -- sign-flip p=1/64 both for intact-vs-lesion and salient-vs-neutral. The production
  default (LTM ON) arm is NOT run by this family and remains the flip-deciding read (prereg's declared
  deviation, restated in Amendment 2); this GO does not license flipping BRAIN_DA_TAG_CAPTURE's default nor
  reading da-gated-encoding as load-bearing on the production battery. Byte-identical-OFF at the corrected pin
  (36a175534) is still PENDING (Amendment 2 point 3); the on-disk offcheck.json is at the stale pin f35196e66
  and must not be read as verifying this branch's OFF path.
---

# DA tag-and-capture wired into chat: 6/6 seeds GO at runner level, LTM-off arm only (2026-09-24)

Pre-registration: `2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md` (commit `6da12e683`, before any
gate-seed run), amended twice on 2026-09-23/24. `grade_seed`/`aggregate` in
`research/runners/_da_tag_capture_chat_probe.py` implement the gates verbatim; this finding reports their
verdict on the 6 seeds the pre-registration named (42, 43, 44, 100, 101, 102), scored by the runner's own
`--aggregate` combine rule (GO iff 6/6 seeds GO). Terms follow `docs/TERMS.md`: "GO" here is the gate's own
verdict, nothing lifted from a run it disagreed with; the faculty is `wired` (reachable from
`webapp/server.py` `/api/brain-chat`) but NOT `on-by-default` (`BRAIN_DA_TAG_CAPTURE` defaults OFF), so it is
not `closed`/`integrated` by that file's definition.

## What had to happen before this could be scored: harvesting the post-fix seed 42

Amendment 1 (2026-09-23) found the ledger's D1 reader shared a process-level RNG cache with production's own
spiking write-gain reader, so a companion-ON intact arm and a companion-ON lesion arm could read DIFFERENT
`gamma`/`d1_a_go` calibrations purely from build order -- a confound, not evidence about the lesion. The fix
(`daa4b382d`) isolates the reader; `grade_seed` gained `G_isolation_gamma_consistent` to catch a recurrence.
Amendment 2 found `aggregate()` did not re-grade stored rows (so a stale pre-fix `GO` could survive under the
current gates) and that the production `seed42.json` at the time was still the CONFOUNDED pre-fix run
(`gamma` in the mid-40s on intact arms vs the low-30s on lesion arms <!--derived--> -- VOID per Amendment 1, together with the ten
per-arm files under `_da_tag_capture_chat/seed42/` they were computed from. It named the fix's own re-run --
at that point still in flight in a separate supplementary output directory (`_da_tag_capture_chat_verify`) --
as the one to harvest as the production `seed42.json` "before the 6-seed `--aggregate` is read as a verdict."

That harvest had not happened before this scoring pass: `research/findings/raw/_da_tag_capture_chat/seed42.json`
on disk was still the confounded run committed at `3fe72c9dd`, sitting next to five freshly-landed pool seeds
(43, 44, 100, 101, 102, all built post-fix at revision `5d3810f2d`, `gamma=32.774` consistent on every
companion-ON arm). This pass copies that supplementary run's `seed42.json` and its `seed42/` per-arm
directory over the confounded ones (superseding them, per Amendment 2), then re-runs the runner's own
`--aggregate` over the resulting six-seed directory. `bash tools/pool_sync.sh` was run once first; it pulled 0
new files (pool41/pool42/pool1 reachable and current, pool40 unreachable) -- nothing was stranded on a node.

## Result (runner's own `grade_seed` / `--aggregate`, re-graded under the CURRENT code on every row)

| seed | seed_verdict | gamma (all companion-ON arms) | p_max sal_night_intact | p_max sal_night_lesion | p_max neu_night_intact | lesion/intact ratio (G6 < 0.25) |
|---|---|---|---|---|---|---|
| 42 | GO | 32.7735 | 0.02396 | 0.000791 | 0.000887 | 0.033 <!--derived--> |
| 43 | GO | 32.7735 | 0.01845 | 0.0 | 0.0 | 0.0 <!--derived--> |
| 44 | GO | 32.7735 | 0.01962 | 0.00225 | 0.0 | 0.1147 <!--derived--> |
| 100 | GO | 32.7735 | 0.02106 | 0.000201 | 0.0 | 0.0095 <!--derived--> |
| 101 | GO | 32.7735 | 0.01841 | 0.000164 | 0.0 | 0.0089 <!--derived--> |
| 102 | GO | 32.7735 | 0.01359 | 0.002192 | 0.0 | 0.1613 <!--derived--> |

(the last column is `p_max sal_night_lesion / p_max sal_night_intact`, computed from the two preceding columns, both read directly off each cited `seed<N>.json`'s `tag_capture_at_recall.p_max`.)

`gamma`/`d1_a_go` are identical across every companion-ON arm at every seed (`G_isolation_gamma_consistent`
holds on all 6) -- the Amendment-1 confound is absent from the record this verdict rests on. Every seed passes
G0 (null-control rebuild reproduces recalled_svo/abstained/ledger state exactly), P1 (both tellings immediately
recallable), G1 (salient intact recalls, salient lesion abstains), G2 (lesion spares immediate recall), G3
(neutral telling not kept overnight), G4 (companion-OFF: lesion does not change the reply), G5 (no
confabulation anywhere), and G6 (lesion PRP held below 25% of intact, actual range 0.9%-16.1%).

`--aggregate` (runner's own combine rule, re-grading every row with the current `grade_seed` per Amendment 2's
fix, not trusting any stored `seed_verdict`) reports `n_go: 6`, `verdict: GO`,
`diffs_intact_minus_lesion: [1, 1, 1, 1, 1, 1]` and `diffs_salient_minus_neutral: [1, 1, 1, 1, 1, 1]` (each diff
read off the cited `seed<N>.json` files' `gates.outcomes`, +1 per seed where the intact/salient arm recalls
correctly and its lesion/neutral counterpart does not). `seed_signflip_p` over those all-+1 diff vectors gives
`signflip_p_intact_vs_lesion = signflip_p_salient_vs_neutral = 1/64 = 0.015625` <!--derived--> (one-sided exact
sign-flip p over 6 seeds, `aggregate.json` is reproducible from the cited `seed*.json` files via `--aggregate`
but is not itself committed here, see "Compute" below).

`grade_seed`'s reported (not gated) `attributable_to` on every seed: 100% of the lesion's next-day-reply change
is attributable to the manipulation versus the null-rebuild control (0% also present in the intact-vs-intact
replicate), and 100% of the lesion's effect needs the companion armed (0% also present with the companion OFF)
-- consistent with G0 and G4 holding on every seed rather than an independent measurement.

## What this GO does NOT show (honest, per the prereg's own "what each gate can and cannot show" + the
## amendments' open items)

- **LTM-on (the production default) is not measured.** Every arm here ran with `BRAIN_LTM_SHIP_DEFAULT=0`
  (buffer-only; the default tiered `wikidata_100k` LTM exceeded the local and pool-node RAM caps at build time,
  per the prereg's design-time note). `LB_DA_TAG_CAPTURE_PROBE`'s battery row under the production LTM tier is
  a SEPARATE, still-unread configuration -- it is the flip-deciding arm, and this finding does not speak to it.
  Do not read this GO as license to flip `BRAIN_DA_TAG_CAPTURE`'s default or to count `da-gated-encoding` as
  load-bearing on the production (LTM-on) battery.
- **Byte-identical-OFF at the corrected pin is still PENDING.** Amendment 2 fixed `PINNED_SHA` to `36a175534`
  (the true branch/`origin/main` merge-base) after finding the old pin (`f35196e66`) predated two of this
  branch's own merges. The on-disk `research/findings/raw/_da_tag_capture_chat/offcheck.json` (committed
  `9a774a2bb`) is still at the STALE pin `f35196e66` and reports `byte_identical_off: true` there -- that result
  cannot be attributed to this branch's change (the pin difference alone touches `webapp/server.py`, +55/-1) and
  is not superseded by a run at the correct pin. A `--offcheck --pinned-sha 36a175534` run is still needed before
  the OFF path can be called verified; it was blocked by local RAM contention as of Amendment 2 and is not
  attempted in this scoring pass (no full-brain build was run here; this pass only re-grades existing JSON).
- **G1's lesion half is an integrity check of the wiring, not evidence** (pinning DA to tonic mechanically
  starves the D1 pool of drive); the evidence is the intact arm's own brain-driven DA crossing the capture
  threshold, together with G2-G4 -- see the prereg's "what each gate can and cannot show" section, unchanged by
  this pass.
- **This is a synaptic-state GO, not a consolidation claim** in the `docs/TERMS.md` sense: the late-phase
  variable is a per-synapse state, not a replay path, and no replay code executes here.
- **`da-gated-encoding` is `wired` but not `on-by-default`**: `BRAIN_DA_TAG_CAPTURE` defaults OFF, so this GO
  grows the de-risked, opt-in-flag evidence base, not the production battery's load-bearing fraction.

## Compute

Re-grading only: no new brain build was run in this scoring pass. `research/findings/raw/_da_tag_capture_chat/`
now holds all 6 seeds at the current (post-Amendment-1-fix) code: seed 42 harvested from the supplementary
verification run named in Amendment 2 (built at `daa4b382d`+, superseding the confounded `3fe72c9dd` run per
Amendment 1/2); seeds 43/44/100/101/102 as landed from the pool at revision `5d3810f2d` (already past the
isolation fix). `bash tools/pool_sync.sh` run once beforehand (0 new files pulled). `--aggregate` was run
against this six-seed directory to obtain the verdict above; its output `aggregate.json` is reproducible via
`.venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --aggregate
research/findings/raw/_da_tag_capture_chat` and is not itself committed here (its bare top-level `verdict`
string carries no `preconditions` block, the shape `tools/gates/verdict_preconditions.py` requires of a newly
added artifact; the six `seed*.json` files each carry a top-level `seed` key and pass that gate as-is).
