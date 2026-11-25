# Pre-registration: cascade screen, decaying own-write (`Brain(own_decay)`) on K-P's per-night homes, judged on replies

Written 2026-11-25 before any run. Capability: dialogue (in-conversation loop onset, D2 of
`brain/research/2026-11-24-routing-cascade-research.md`). Offline replay on the pool, CPU, 3 seen seeds: a lead at most.

## 1. Record brief
- **The defect.** In K-P, whole battery nights answer told questions in the user's voice ("My cat is called Pixel.") or
  as a refusal loop. P(first person | previous row first person) is 0.83 against 0.17. On pool40 the M1 s320 intact
  night-15 home was a CASCADE: banked order 4/8, each question from a fresh state 0/8. M1 s319 was WEIGHTS
  (`results/2026-11-24-flip-night-diagnosis.md`). K-P itself was never tested, because no per-night homes existed. The
  KR snapshots (`lifeKR_s319-321`, `--night-snapshots`) now supply them.
- **Levers already tried on this loop.** All are default OFF, and the verdicts are on the method:
  - `habituation`: an "I don't know." loop is left.
  - `reply_monitor`: no worse only; the redraws were copies too.
  - flat `own_gain`: at 0.5 the echo stays 14/14; at 0.25 or below the brain asks user-style questions, because it
    loses whose turn it is.
  - `own_gain_copies`: question loops in 2/6.
  - `carry_penalty` and `fake_think`: no lead.
  
  The own-gain probe located the amplifier in the write of the brain's own reply (`results/2026-11-16-own-gain-probe.md`).
- **This method** is the research note's D2 method 1. Older own replies fade by beta per later turn, and the reply just
  spoken stays at full weight, so whose turn it is stays.
  - **How the fade is computed.** The faded part is what the earlier replies left in the state now: U - N on each
    matrix-memory slot. U is the conversation state's read of the heard turn and "Assistant:". N is the same read from
    a second track of the conversation in which no reply was ever written.
  - **What it is not.** It is not a stale subtraction: the RWKV state's own propagation carries the earlier replies.
  - **Code.** `brain/brain.py` `_commit_decay`; tests in `brain/tests/test_own_decay.py`. beta 1 equals the full-gain
    split read, and a state reset restarts the track.
- **Prerequisites.** A self/other tag on the state write is "missing" under Self and identity (`brain/PREREQUISITES.md`);
  this method stands in for it.
  - **Expected risks.**
    - The just-spoken reply is still at full weight, so an echo that repeats the latest reply is not reached. Expected
      to FAIL the echo guard K5, where flat own_gain 0.5 echoed 14/14.
    - Flip nights that are in the weights (B >= 4) are not reached.
- **Host decision.** The no-reply track is a declared scaffold, entered in `brain/SCAFFOLDS.md` with this commit. Its
  cap is one extra read of each heard turn. beta is fixed a priori at 0.5 and 0.25: both are reported and neither is
  selected. Nothing is fitted.

## 2. Population and arms
- **Homes.** KR night snapshots s319 / s320 / s321, nights 2-15. Each home is opened exactly as the routing replay opens
  one (`routing_replay.open_snapshot`, with deltas from the final trunk), with the probe clock
  `life.day_start(n+1, weekend)` and the probe reseed `life._reseed(seed, 500+n)`. The battery is the banked one in
  banked order (`routing_replay._qs`).
- **Lesion arms.** intact (use boost ON, as banked) and hippocampus-off.
- **Conditions per lesion arm.**
  - A: as is.
  - B: each question from a fresh state. Diagnostic only.
  - D50: `own_decay=0.5`.
  - D25: `own_decay=0.25`.
- **Where it runs.** Pool40/41/42, CPU fp32, one seed per node, under `tools/memcap.sh`, from a pushed commit. One sampled
  reply per row.
- **Validity, per night and arm.** The intact A run must fetch the banked episode on every row (recalled text equals the
  banked intact row's), or that night's intact arm is VOID. The hippocampus-off arm has no fetch, so its validity is
  report-only. A seed is VOID if more than 3 of its 14 nights are void. The reply match of A against the banked rows is
  reported.
- **Off-voice row (V).** A told row (H, rotating, or told neardup) whose reply meets either test below, counted by
  `probe3.py` (`own2` / `REF`, copied verbatim into the harness):
  - **User voice:** a first-person mine or I-opening reply that is not a refusal and not "my friend".
  - **Refusal:** a refusal.

  **Flip night:** V >= 4 in A on CPU, decided per lesion arm. The B diagnosis per flip night is CASCADE if B <= 1,
  WEIGHTS if B >= 4, else BETWEEN.

## 3. Bars (KILL first; for each beta separately; pooled over seeds and both lesion arms unless stated)
- **K1 cascade cut.**
  - **Pass condition:** on A's flip nights, V under D is at most 0.5 times A's.
  - **Kill:** V under D is above 0.5 times A's, or the cut (D below A) is missing on any seed that has 2 or more
    flip-night arms.
- **K2 told recall.** Kill if told right (`ok`) on the intact arm, pooled over all valid nights, falls by 2 or more
  below A on any seed.
- **K3 no new flips.** On A's non-flip nights, kill if V under D rises above A's by more than 2 per seed (n about 23
  arm-nights).
- **K4 turn role.** Kill if told replies that end in "?" or open with "What"/"Do you"/"Can you" under D exceed A's by
  more than 2 per seed. This is the own_gain failure.
- **K5 echo and stored-line guard.** Run `brain/experiments/own_gain_probe.py` with an own_decay arm at each beta, on
  the 11-16 inputs: E with 6 samples and B with 4, script seed 1121. Kill if E echoes on 10 or more of 14 free turns in
  any sample, or if 2 or more samples show user-style question loops (at least 4 of 14 free replies ending "?").
  - If the 11-16 inputs are missing, K5 is UNDEFINED and the verdict is at most HOLD.
- **Decision.**
  - LEAD: no kill and K5 defined. It earns one live DEV K-P life with `own_decay` on s319-321, with a reply-level prereg.
  - HOLD: no kill, K5 undefined.
  - KILLED otherwise.
- **Six seeds before any claim.**
- **Reported, not barred.**
  - Per night and arm: V, told ok, the B diagnosis, and refusal and user-voice rows separately.
  - Open-prompt intrusion and never-told false recall: these come from fresh states, so they should be unchanged
    except through the use boost.
  - A whole transcript of two flip nights and two non-flip nights, in each condition, read before the verdict.
- A negative is a verdict on this method, not on the capability.

## 4. Cost and timebox
- **Compute.** About 3 min per battery on pool CPU: 14 nights x 2 arms x 4 conditions per seed, about 6 h per seed and
  node. Plus K5, about 1 h on one node. $0, and no workstation compute.
- **Timebox.** Read by end of 11-26.

## Review (sonnet, 11-25, before data): OK_WITH_FIXES, applied
- **A2 control (regression to the mean).** Flip nights are chosen by A's own sample, so condition A2 is added: A with
  the probe reseed `500+n` replaced by `900+n`, and nothing else changed.
  - K1 now needs V_D <= 0.5 x V_A and V_D <= 0.5 x V_A2, pooled.
  - Per seed with 2 or more flip-arm nights, it needs V_D below both A and A2.
  - K3 compares D with A2 (non-flip nights selected on A drift up under any resample).
  - Cost: 5 conditions, about 7.5 h per seed and node.
- **No flip nights, or every seed void:** UNDEFINED / VOID, not KILLED.
- **K5 is fed by `cascade_screen k5`** from the own_gain_probe JSONs: E on pool40 (`homes_d_s1121/hippocampus_off`,
  `echo_onset.json` index 4, hippocampus and night off) and B on pool41 (`home_m1121`, `ocm_s1121.rows.json` index 0).
  The runs are labelled `d0.5` / `d0.25`.
  - **Loop count:** free replies ending in "?", the 11-16 probe's count. K4 keeps `user_q`.
  - **Thresholds:** 10/14 and 4/14, scaled to `n_free` if that is not 14.
- **K5 is reported apart from K1-K4.** `core` is PASS or KILLED, `k5` is PASS, FAIL or UNDEFINED. A K5-only kill reads
  "the cascade cut works on the battery, the echo guard fails". It is still KILLED as registered.
- **Fetch drift.** Each condition reports the rows whose fetched episode differs from A's.
- **Wording.**
  - Cost per turn: four reads (two for the heard turn, the reply, and the no-reply track's turn end).
  - U - N is a nonlinear estimate of what earlier replies left.
  - At turn t+1 the reply just spoken is at 1 and the one before at beta.
- **Frozen-file note.** `brain/brain.py` is in `life.FROZEN_FILES` (hashed into each life's provenance). The switch is
  default OFF, and the default path is unchanged.
- **Episode files are not written through.** The snapshot's episode files are symlinks to the final trunk. `save`
  replaces a file with `os.replace` only when its key object changed, which never happens in a battery, so nothing is
  written through a link.
