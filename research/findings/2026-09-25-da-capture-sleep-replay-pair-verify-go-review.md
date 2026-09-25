---
type: finding
status: review
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: adversarial verify-go review (flip-rule leg b) of the PAIR BRAIN_DA_TAG_CAPTURE (webapp/da_tag_capture.py
  + webapp/da_tag_capture_chat.py) + BRAIN_SLEEP_REPLAY_CAPTURE (webapp/sleep_replay_capture.py), both default OFF;
  re-derives the three GO families and reads the r2 context family's raw arms; no brain was built, no default flipped
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_tag_capture_chat/seed*.json
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed*.json
  - research/findings/raw/_sleep_replay_capture/seed*/*.json
  - research/findings/raw/_sleep_replay_capture/aggregate.json
  - research/findings/raw/_sleep_replay_capture_r2/seed*/*.json
  - research/findings/raw/_sleep_replay_capture_r2/aggregate.json
verdict: leg (b) SOUND-WITH-ISSUES. The three registered GOs (DA tag-capture LTM-off 6/6, LTM-on 6/6, sleep-replay
  rc 6/6) re-grade exactly and their instruments hold within their registered scope. The PAIR's flip rationale does
  NOT hold as stated -- "with both flags on, an ordinary fact told once is kept overnight" fails on 2 of 6 seeds for
  a second ordinary telling and on 6 of 6 seeds for a fact told 4 h before sleep -- and the production path (wall
  clock, a 5-minute pause counts as sleep, several SWR epochs per day, cupy, LTM on with the route armed, threads)
  was never exercised. Leg (b) is therefore NOT met until items B1-B4 below are done; legs (c)/(d) need C1-C3/D1-D6.
---

# Verify-go review of the DA tag-capture + sleep-replay capture pair (flip-rule leg b)

Running verify-go: adversarially probing this result before it lands. Five skeptic angles, each told to REFUTE, all
run by one reviewer on the committed artifacts (no new brain build, no pool or GPU job, no default touched). Terms
follow `docs/TERMS.md`. Everything below that is a number was read off the cited raw JSON in this worktree.

## Verdict: SOUND-WITH-ISSUES (leg b not yet met)

What survives: every registered verdict re-derives, the UNDEFINED rules are honest, the lesions held on the record,
the brain substrate is seeded, and no finding uses "consolidation" for this mechanism.

What does not: the claim the flip rests on (board: the DA flag "must ship WITH BRAIN_SLEEP_REPLAY_CAPTURE (alone it
loses an ordinary fact overnight)") reads as if the pair keeps ordinary facts. The project's own r2 arms show it
does not in general. The whole evidence base is one conversation per family, the scripted turn clock, numpy, and
no LTM with the route armed; production runs a different clock, a different sleep trigger and a different backend.

## 1. What was re-derived (cheap, from the committed JSON)

- The registered graders (`grade_seed`, `grade_seed_rc`, `grade_seed_r2` in
  `research/runners/_da_tag_capture_chat_probe.py`), re-run on the stored arms: base LTM-off GO on all 6 seeds, base
  LTM-on GO on all 6, rc GO on all 6, r2 item 2 NO-GO on 42/44/100/102 and UNDEFINED on 43/101. All match the findings.
- The p_max tables of both base findings reproduce seed for seed (gamma 32.7735 on every arm of every seed).
- LTM-on vs LTM-off, all 6 seeds: identical outcomes on all 10 shared arms; the LTM-on arms really carried the LTM
  (every reply's `source` reads "tiny-demo +LTM", the LTM-off ones read "tiny-demo").
- Sign-flip p = 1/64 for six all-+1 diffs, as reported.
- Provenance: rc arms at 269ae8f76 and LTM-on arms at cce3c1dbd both descend from the flip merge 60f81f68f, so
  BRAIN_EPISODIC_STORE_VERIFY, BRAIN_PMEM_FACILITATION and BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE were ON in those arms;
  the LTM-off base pin 5d3810f2d predates that merge.
- Code drift since the rc pin (269ae8f76 to HEAD) in the flag-ON path: the r2 change runs one epoch per 24 h of
  CONTINUED idle (was one per idle stretch), plus default-OFF sub-flags (downscaling, load renorm, awake replay). For
  any one-night protocol the path is unchanged; for idle over 24 h HEAD runs extra epochs the rc runs never did.

## 2. Skeptic 1 -- the instrument

**Holds.** G0 compares the ledger state and the sleep record of two rebuilds, not just the reply; I1 asserts the
replay branch executed and I2 reads each lesion off the epoch record (replay lesion: R_eff all 0 and SWR DA at
tonic; capture lesion: D1 drive 0; DA lesion: DA seen by D1 at tonic). UNDEFINED is never scored as a pass or a 0.
G1's and RC5's lesion halves are integrity checks, and both preregs say so.

**Finding I-1 (HIGH): with both flags on, an ordinary fact's overnight fate is a step function of one read, and the
rc GO sits near the step.** Across 24 seed x telling cells with both flags on (rc arms plus the r2 arms that run the
same pair), the next-day outcome is decided by R, the store's cleanup margin at the one SWR epoch:

| seed | telling | R at the epoch | SWR DA | z at recall | next day |
|---|---|---|---|---|---|
| 100 | datl, told 4 h before sleep | 0.004059292 | 0.503003876 | 0 | abstain |
| 101 | d10w weak telling, night 1 | 0.035599825 | 0.526343871 | 0 | abstain |
| 43 | d10w weak telling, night 1 | 0.168587758 | 0.624754941 | 0 | abstain |
| 102 | datl, told 4 h before sleep | 0.184615442 | 0.636615427 | 0 | abstain |
| 101 | datn (rc RC1 arm) | 0.209145737 | 0.654767845 | 1 | correct |
| 43 | datn (rc RC1 arm) | 0.283229339 | 0.709589711 | 1 | correct |
| 42 | datn (rc RC1 arm) | 0.426299712 | 0.815461787 | 1 | correct |
| 102 | datc (rc salient arm) | 0.548083297 | 0.905581640 | 1 | correct |

(Eight of the 24 cells shown; the other 16 follow the same rule: all 8 cells with R at or below the fourth row are
lost, all 16 at or above the fifth are kept. Sources: research/findings/raw/_sleep_replay_capture/seed*/neu_night_rc_a.json
and the `sal_night_rc` arms beside them; research/findings/raw/_sleep_replay_capture_r2/seed*/d10w_rc.json and the
`ld_rc` arms beside them; each arm's recall-turn `da_tag_capture.sleep_replay_capture.epochs[0]` and
`blocks[0].z_mean`.) The rescue edge lies between R 0.185 and 0.209; the rc GO's seed 101 sits about 0.024 above it. <!--derived-->

Consequences for the pair:
- The weak telling (`d10w`/`d3w`: the same fact, said last after its words habituated the DA) is LOST after the first
  night on seeds 43 and 101 with both flags on (`d10w_rc`: abstain on all ten mornings; `d3w_rc` abstains too). The
  still-running fi family's `fiv_lr` arm (same telling, untracked, not a gate row) shows the same two seeds lost
  from night 1.
- A fact told 4 h before sleep is lost on 6/6 seeds with both flags on (r2 item 1), while today's default keeps it
  (`ld_ledger_off` correct 6/6). In production that is any fact told early in a conversation that runs on for hours
  with no 5-minute pause.
- So flipping the pair still makes some ordinary facts forgettable relative to today's default. That may be the
  right behaviour (weakly encoded and old facts are forgotten in real brains), but it is a decision for the owner,
  and the board's wording hides it.

**Finding I-2 (HIGH, honesty; a wrong-quantity comparison in a merged finding).** The r2 finding explains the 43/101
P2 failure as "a readout miss on the composer's cleanup margin" with "the fact's synaptic trace ... stronger than
baseline at recall", citing increment-to-baseline ratios above 1 <!--derived--> (1.0842 and 1.1488). Those ratios are the
ledger's STORED increment `inc_mag` over `base_mag`, a bookkeeping quantity that decay does not change. The EXPRESSED
weight is b + (e + z(1 - e)) inc; at recall z is 0 on both seeds and the early-phase factor e is about 1e-7 <!--derived-->
(exp(-24/1.5)), so the synapses sit at baseline: the trace was never captured and it decayed. The r2 verdict stands
(P2 failing makes those seeds UNDEFINED either way), but the stated cause is wrong, and it is the cause that bears on
the pair. Logged in `research/FAILURE_LOG.md`.

**Finding I-3 (MEDIUM): the DA-encoding lesion cannot separate waking salience from the sleep route.** Under
`BRAIN_DA_ENCODING_LESION` one flag moves three things: the write gain (seed 42 salient R falls from 0.466426613 to
0.306307325), the waking PRP drive, and the DA the D1 pool sees during the SWR bout (the route computes 0.72666742
but the pool sees tonic). At seed 42 the lesioned R is above the edge in the table, so that pattern predicts the
route alone would have kept the fact (an inference, not a run): RC5's abstain there is then carried by the sleep
route's own DA edge. RC5 shows "no capture without the D1 DA input". It does not show that waking salience stays
load-bearing once the route is on.

**Finding I-4 (MEDIUM, seeding): six seeds vary the brain, not the D1 pool.** BRAIN_CHAT_SEED reaches the tiny-demo
build and the ledger baseline (G0 rebuilds are identical; R and the turn DA differ by seed). The D1 reader is always
built with `cfg.seed = 42` (`D1_READER_SEED`, the production reader seed), so gamma and d1_a_go are identical at
every seed. The six seeds never sample a different D1 population, and with a sharp capture edge that population
sets where the edge falls. On a GPU host the same `cfg.seed = 42` draws from cupy's RNG, so production most likely
builds a different pool from the one tested (not checked here).

## 3. Skeptic 2 -- is the runner the production path?

Shared with production: the real `webapp.server.brain_chat` handler, the same three hooks (`observe_chat_turn`,
`after_store_chat`, `tick_chat`) and the real `continuous_engine.tick_idle_sessions` at the night step. Not shared:

| aspect | gate runs | production with both flags on |
|---|---|---|
| world clock | `BRAIN_DA_TAG_CAPTURE_CLOCK=turn`: 30 s per turn, one scripted 24 h jump | `wall` (the default): real time since the ledger was built; the jump is never called |
| what makes the brain "sleep" | the one scripted night; epoch 5 min after the last turn | any idle of 5 min or more after an observed turn (`SLEEP_IDLE_SEC`, a host timer); the live server has no sleep signal |
| SWR epochs per day | exactly 1 | one per pause of 5 min or more (each new turn opens a new episode), plus one per further 24 h of idle |
| idle ticks | one call at +24 h | every 20 s once idle (`BRAIN_CONTINUOUS` default on); the Turrigiano pass fires ~20 s after each new fact |
| backend | numpy | cupy on a GPU host (`server.py` sets it when CUDA is present) |
| renderer | stub, LLM disabled | Qwen (recall is read brain-side, so this matters least) |
| LTM | off in the rc family | on (TieredFactStore) |
| concurrency | one sequential process | idle tick in an executor thread beside request threads; no lock on the ledger |
| facts per session | 1 managed block (at most 3 in r2) | unbounded |

The row that matters most is the sleep trigger. In production a normal day with a few 5-minute breaks runs several
SWR epochs, each re-tagging every managed block and releasing DA in proportion to the summed R. The module's own
design note records that a five-epoch night, on the fake substrate, "RESURRECTED noise-level traces", which is why
one epoch per night was chosen. Production would enter that regime through ordinary pauses, and it has never been
run on a real brain. No test and no gate row exercises the wall clock at all (every probe, the
`LB_DA_TAG_CAPTURE_PROBE` row included, forces `turn`).

Smaller production-path items: `_private_rng` restores numpy and python RNG state but on cupy only reseeds, so with
the pair on, every ledger read leaves other organs on a reseeded cupy stream (declared "flag-ON only"); the same
save/restore run from two threads can leave the global numpy state seeded; the ledger's drive list and blocks grow
without bound and `_integrate` walks both in Python at 15 s steps inside the request path after any long gap; and
the ledger lives only in memory, so a "next day" in production needs the process to stay up.

## 4. Skeptic 3 -- interactions

- **Was the pair ever run together?** Yes, but narrowly: every rc arm and the r2 `*_rc` arms set both flags. All of
  them are LTM off, turn clock, numpy, one fact. The pair was never run with LTM on, never on cupy, never in the
  combined battery, and never with two or more facts in a gated family.
- **Default-on memory fixes.** The three 2026-09-23 fixes were on at the rc and LTM-on pins (section 1), and the
  episodic organ did not answer the recall when the store abstained. That is one conversation, not a battery.
- **Every stored fact changes the moment the flag is on.** `on_store` rewrites each new block as its seeded baseline
  plus the increment (BETA_BASELINE 1, same magnitude) at store time, with no night involved. Every faculty that
  reads the composer store (multi-fact recall, episodic, provenance, WM binding) sees that added baseline, so leg (c)
  is essential and cannot be argued from these families.
- **Reconsolidation.** `sync_from_store` treats any non-multiplicative rewrite of a managed block as a fresh early
  increment with z reset to 0, so a captured fact that is rewritten loses its late phase. In all 258 arm files of
  the four families `external_rewrites` is 0: this path has never run.
- **Salience under the pair.** With both flags on, the one-fact next-day contrast between salient and neutral is
  gone (both kept, 6/6). The only evidence that waking salience still decides anything is across two families
  (awake-replay ARC6: the salient fact told 4 h before sleep is kept on waking capture, 6/6; r2 `ld_rc`: the neutral
  one is lost, 6/6), not one registered contrast. On the battery, da-gated-encoding's lesion row under the pair
  would measure "no DA, no capture", not salience gating (I-3).
- **Downscaling and awake replay** are separate default-OFF flags with their own NO-GO results and are not part of
  the pair; with both off, nothing brakes repeated epochs (above).

## 5. Skeptic 4 -- brain-based only

Neurons and synapses: the waking DA level (spiking SNc driven by the novelty/habituation organ); the D1 population
(an Izhikevich pool whose spikes are counted); the SWR read-back (each managed trigger driven on the composer's
resonate-and-fire substrate and read by the store's own cleanup); the store synapses the recall reads.

Host shortcuts, each still in the path the flip would ship:
1. the tag / PRP / bistable late-phase equations, integrated by host Euler steps (the PRP pool is one scalar);
2. the rate-to-activation normalisation and the a-priori gamma bisection;
3. the ledger REWRITES the composer's synapses (w = b + f inc) on every turn and tick, and holds the increment,
   the baseline draw and the tag that the synapse itself does not carry;
4. `TURN_DRIVE_H` (a turn drives the D1 pool for exactly 30 s);
5. sleep onset as a 5-minute idle timer, and one SWR epoch per night;
6. R as host arithmetic on the cleanup scores, min over roles;
7. the replay tag R x |inc|, from the ledger's stored increment, and max(write tag, replay tag);
8. the SWR DA as a formula of summed R; the sleep DA does not come from the spiking SNc;
9. `sync_from_store`'s rescale-versus-rewrite classification;
10. unmanaged build-time blocks (`block_offset`).
The world clock is environment and legitimate. Items 1, 3 and 8 carry the decision; the spiking parts supply
inputs to host dynamics. The findings declare all of this; the flip would still put host-ODE-controlled forgetting
on the default path, which the owner should see stated as such.

## 6. Skeptic 5 -- honesty (docs/TERMS.md)

- **consolidation:** avoided correctly in all three GO findings (no source lesion, same store).
- **GO:** each is its gate's own verdict; nothing lifted from a negative run.
- **wired / on-by-default:** stated correctly (wired, default OFF, runner-level).
- **byte-identical:** the r2 counterfactual offcheck read IDENTICAL at 24380cbe4 in the data; not re-run since the
  awake-replay and load-renorm code landed.
- **Overclaims found:** the board line quoted above (implies the pair keeps ordinary facts; I-1); the r2 finding's
  P2 cause (I-2); the sleep-replay prereg's mapping that RC5 shows "flipping both flags does not make
  da-gated-encoding lose its next-day load-bearing read" (true only as "no DA, no capture", I-3); and the rc
  finding's "Salient-vs-neutral separation ... is kept", which holds only with the replay edge cut.

## 7. What must happen, in order

Before leg (b) counts as met:
- **B1** Correct the record: a correction note on
  `research/findings/2026-09-25-sleep-replay-capture-r2-NO-GO-6seed.md` (the 43/101 failure is a capture failure,
  z 0, expressed increment near 0, not a readout miss), and scope notes on the rc finding and the board line: the
  pair keeps the datn telling 6/6, the weak telling 4/6, a fact told 4 h before sleep 0/6.
- **B2** An owner decision, written down, on whether that forgetting is acceptable at flip time or blocks the flip
  until a further mechanism lands (the fi family is the running candidate).
- **B3** A registered production-path arm set on the wall clock: a realistic day (several turns, two or more facts
  told at different times, three or more pauses of 5 min or more, then a night and a multi-night idle), with gates
  for no resurrection of a decayed fact, no confabulation, and the ordinary/salient outcomes. Until this runs, the
  multi-epoch regime is unmeasured.
- **B4** A registered salient-vs-neutral contrast inside ONE family with both flags intact (long-delay salient vs
  neutral), plus a waking-only DA lesion that leaves the SWR DA edge intact, so da-gated-encoding's role under the
  pair is measured rather than inferred across families.

Leg (c), the combined no-regression battery:
- **C1** Pin a revision that contains 3bdf8b619 and the current flag-ON code; B2b's F (a308f1e09) does not.
- **C2** A base arm and a flipcand arm differing only by `--extra-env BRAIN_DA_TAG_CAPTURE=1
  BRAIN_SLEEP_REPLAY_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn`. The clock token is needed because a wall-clock
  numpy shard is not deterministic (decay and a possible mid-conversation "sleep" would depend on host speed); say so
  in the prereg, and let `tools/assert_flipped_defaults.py` and the LBP per-cell rule admit exactly those tokens.
- **C3** Add `LB_DA_TAG_CAPTURE_PROBE=1` to both arms (it is not in the adequate probe set), so da-gated-encoding is
  measured on its next-day probe; R1 per faculty with no tolerance, as in B2b.

Leg (d), production-default validation, on a branch with both defaults flipped and NO flag in the env:
- **D1** Re-run the base, rc and r2 `ld`/`d10w` arms with LTM on (the pair with LTM on has never run).
- **D2** The B3 wall-clock day, at production defaults.
- **D3** A cupy/GPU arm on the reference 3090: gamma and d1_a_go, the capture edge, and no drift in other organs'
  replies from the unrestored cupy reseed (or fix `_private_rng` to restore the cupy state first).
- **D4** An idle tick overlapping a chat turn (or a per-chat lock added first).
- **D5** Re-run the counterfactual offcheck at the flip revision.
- **D6** A session with many managed facts, timing the first turn after a long gap (unbounded drive list).

## What this review did not do

It built no brain and ran no battery; every statement rests on committed arm JSON and a code read at this
worktree's HEAD (fd29040db). The fi family was read only as an untracked, unscored corroboration. It did not flip,
queue or edit any default or any other finding.
