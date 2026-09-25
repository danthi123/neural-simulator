---
type: finding
status: partial
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: the pair BRAIN_DA_TAG_CAPTURE + BRAIN_SLEEP_REPLAY_CAPTURE (both default OFF) on its production path -- the
  ledger's default wall clock fed by the environment's virtual wall clock, idle ticks at the server loop's cadence, any
  idle of 5 min or more counted as sleep (several SWR epochs a day), three facts sharing one PRP pool; plus the
  one-family salient/neutral long-delay contrast with a waking-only DA lesion, and the weak telling read at once
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendment 7, committed at ccb38be4b before this run)
seeds: [7]
verdict: DEV-SEED SMOKE (seed 7, not a gate seed, not a gate row). Plumbing ran end-to-end on both families with
  no error; qualitative signs point the expected direction. PARKED 2026-09-25 (owner ruling, see status note below)
  before either family's gate row is queued.
artifacts:
  - research/findings/raw/_pair_production_path_smoke/pp_seed7/wd_a.json
  - research/findings/raw/_pair_production_path_smoke/pp_seed7/wd_b.json
  - research/findings/raw/_pair_production_path_smoke/sn_seed7/*.json
  - research/findings/raw/_pair_production_path/design_fake_substrate.json
  - research/findings/raw/_pair_production_path/envcheck_seed7.json
---

# The pair on its production path: seed-7 dev smoke of the pp and sn families

seed-waiver: this is the dev-seed smoke that Amendment 7 schedules after its own commit. Seed 7 is not one of the six
gate seeds, no gate row ran, and nothing here is a 6-seed claim. The pool and GPU lines for the gate rows are at the
end, prepared and NOT queued.

> **STATUS (2026-09-25, owner ruling): PARKED.** The DA-tag-capture + sleep-replay-capture memory pair's move
> toward production waits for the prioritized-memory design (retention that keeps important facts and lets minor
> detail fade); nothing on this branch runs further, and neither family's six-seed gate row below is queued, until
> that design lands.

## What ran

- **pp family** (`--family pp`), the wall-clock "wd" day schedule: `wd_a`
  (`research/findings/raw/_pair_production_path_smoke/pp_seed7/wd_a.json`) and `wd_b` (the same directory) -- two
  independent builds of the SAME registered null-control config (`{WALL_ON, RC}`, Amendment 7's own "G0
  null-control rebuild" pair) -- both ran the full schedule to completion (`wd_vclock` through `wd_rC_d5`, 238
  keys each). The two builds produced IDENTICAL abstain/recall outcomes at every checked probe (immediate,
  evening, morning, day-5) for all three registered facts A/B/C: no rebuild-difference surfaced on this one seed.
- **sn family** (`--family sn`), the salient/neutral long-delay contrast: 10 arms ran to completion (`lneu_rc`,
  `lsal_ledger_off`, `lsal_rc_a`, `lsal_rc_b`, `lsal_rc_dalesion`, `lsal_rc_wakelesion`, `neu_imm_rc`, `sal_imm_rc`,
  `sal_rc_wakelesion`, `wk_imm_rc`). At each arm's own long-delay recall probe (its final `*_recall` key), the one
  told fact ("the cat chases the ball") was recalled (not abstained) on 7 of 10 arms; 3 of 10 abstained at that
  probe, and the three split into two categorically different groups. `lneu_rc` (the neutral-telling control)
  abstaining is the pre-registered EXPECTED outcome for that arm -- `research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md`'s
  own Amendment 7 GO condition SN1 reads `lsal_rc_a` correct AND `lneu_rc` not recalled, so this abstain is the
  salient-vs-neutral contrast already pointing its registered direction on this one seed, not an anomaly to
  explain. The other two abstains are the two arms whose own label names a lesion -- `lsal_rc_dalesion` (a
  DA-encoding lesion) and `lsal_rc_wakelesion` (a waking-only lesion) -- pointing the direction each is meant to
  test. One seed, one fact, no scrambled control, no statistics: a plumbing sanity signal, not a scored lesion
  result.
- **A separate, uncommitted dev script** (its full text and output are `envcheck_seed7.json`, run at the branch base
  `381f608e3` before the Amendment-7 code, and already committed with the amendment): 32 real `brain_chat` turns
  plus 4 idle/sleep ticks under numpy/tiny-demo, both flags on turn-clocked, completed with no error. Per-turn
  latency on numpy ranged 5.29 s to 549.81 s <!--derived--> in this one run, with no pattern tied to arm complexity
  visible here -- an operational-cost observation for the wall-clock production path (the pair verify-go review's
  B3/D2), not diagnosed further in this smoke.
- `design_fake_substrate.json` exercises the Amendment 7 scoring functions against the tests' fake composer (no
  brain, not a gate row) -- a scorer smoke, already committed with the amendment.

## What did not run (or did not finish)

- **`wd_epi`** -- the pp family's `BRAIN_EPISODIC_STORE=1` (numpy) arm, exactly the arm the pair verify-go review's
  B3/D1 call for (episodic-vs-composer next-day agreement) -- was launched TWICE, per the provenance log's run ids
  `1790360882-3655262` and `1790361950-3690699`, and never produced a `wd_epi.json` inside `pp_seed7`; both
  attempts were still mid model-build (`laneB_pp.log`, `laneB2_epi.log`) when the session was interrupted, with no
  traceback in either log. This is corroborating, not diagnostic: it is consistent with the review's own finding
  that the numpy D5 BTSP write costs ~510 s/topic and can itself run long against a wall clock, but a build that
  never reaches its first turn does not confirm that mechanism specifically.
- **`wk_imm_off`** -- the sn family's ledger-off immediate-recall control (`SN_REPORTED_ARMS`) -- was launched once
  (run id `1790362276-3700542`) and never produced a `wk_imm_off.json` inside `sn_seed7`; also mid model-build in
  `laneB2_epi.log` when interrupted.
- The pp family's replay-lesion and ledger-off arms (`wd_replaylesion`, `wd_ledger_off`) were queued in an
  uncommitted local launcher that waited on the sn family finishing before starting; the session was interrupted
  before that launcher ran (it is removed, not part of this branch).
- No traceback or exception appears in ANY captured build log, including the two incomplete ones above -- every
  stop is a session interruption mid-build, not a crash.
- Neither family's `--aggregate` step ran (no `aggregate_pp.json` / `aggregate_sn.json` exists). No gate row for
  any family ran; the six gate seeds (42/43/44/100/101/102) were not touched by this smoke. The `cu` (cupy/3090)
  family has a prepared, NOT-queued gate row script (`research/queue/_pair_cu_gate_row.sh`) and no seed-7 dev
  smoke of its own.

## Prepared, NOT queued (per Amendment 7's "Compute for this amendment")

The six gate rows are pool runs (`pp`, `sn`) and a local 3090 `gpu_queue.sh` job (`cu`) at a full-SHA-pinned
revision containing Amendment 7. None are queued by this smoke or by this parking commit. Per the STATUS note
above, they stay unqueued until the prioritized-memory design lands and the owner lifts the park.
