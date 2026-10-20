# CS-race: retire the content-selection seam (Phase B, first seam): build packet

Claude, 2026-10-20. Phase B of `docs/plans/2026-09-29-roadmap-to-the-primary-goal.md` starts here: "spiking
competition in the store replaces the host first-match". This packet is the design; the prereg comes after step 2,
filed alone, with `flip: no` (the retirement runs behind a default-off flag; turning it on is a separate flip packet).

Evidence base: three read-only readers (`/home/dant123/Projects/sim-rev/cs-packet/readers.json`) and a design panel
of three designers and two judges (`/home/dant123/Projects/sim-rev/cs-packet/panel.json`, workflow
`wf_f5eb8593-f52`). The panel's falsifiability-first design is adopted, with the grafts both judges asked for.

## 1. The seam: `cs-compete`

Which of two or more stored facts that share one (agent, action) cue reaches the reply. Today three host steps decide
it, and they act as ONE decision because each masks the others:

- **H1, k-select first-match:** `_fact_shard_first_match` takes the first decoded block in ascending order
  (`research/runners/one_brain_composer.py:1447-1461`; the same loop in `_seq_block`, `:1655-1666`).
- **H2, organ B's last-write expectation:** `webapp/gnw_two_organ_bus.py:236-237`.
- **H3, value choice:** a host candidate list plus a host recency+referent engagement scalar
  (`research/runners/value_choice_production_organ.py:354-361, 364-393`).

Retiring H1 alone is unobservable: in the host-share baseline 10 of 12 k-select "order" events were UNDEFINED because
value choice re-commits downstream (`research/findings/2026-10-02-host-share-baselines-RESULT.md`).

**Scope:** the buffer store (`composer.kb`), (agent, action) cues, single-match and compete. **Out of scope, measured
before-only and named as the next seams:** the gather stage's first-unsaid-fact (`rich_answer_composer.py:552-557`),
the LTM topic fallback `facts[0]` (`:503`), the LTM tier's `idx[0]`, tier priority, and candidate generation.

## 2. The mechanism (flag `BRAIN_CS_RACE`, default OFF)

Generalise the idle-replay engram race (`idle_replay_organ.py:209-321`) into a cue-addressed, tag-free competition,
reusing the `spiking_commit` stage (gates, cross-inhibition, bistable latches, occupancy; read by persistence,
`spiking_commit.py:157-171`). Only the drive changes; **no host-computed per-candidate number reaches the race.**

- **Engrams:** one assembly E_k per buffer fact, each with an inhibitory pool that inhibits the other engrams.
- **Cue match is a synaptic sum.** Role-bound cue assemblies (agent word, action word) project to every engram whose
  fact holds that word in that role. The host delivers only the parsed question words as sensory current; it never
  evaluates a match, never pre-filters candidates (no DG shard, no K_MAX list).
- **Discourse context:** role-free mention assemblies are driven by every content word heard on any turn and project
  weakly onto each engram containing the word: enough to prime, never enough to ignite.
- **Recency is substrate state:** Tsodyks-Markram facilitation on each engram's own recurrence only (Mongillo, Barak
  and Tsodyks 2008; `enable_short_term_plasticity`, every other pathway `stp_disabled`, `sim/regions.py:572`). The
  engram that fired most recently (its write, or a mention of one of its words) re-ignites fastest. This retires both
  host tags: value choice's recency scalar and idle replay's `x_k`.
- **Encoding strength is a synaptic weight:** the DA write gain g scales the cue->E_k weight at write time; no bias
  current (judge graft). A mass/strength argmax then becomes a real control (N8).
- **The moat:** a full two-role match ignites; one role plus maximal priming must stay below threshold. Exactly one
  latch ON names the winner; none ON, or two ON, abstains with no host fallback. The winner's block is decoded by the
  existing spiking RF decode (`_read_one_block`), as idle replay already does.
- **Allocate once, then change weights in place.** Preallocate engram slots to `K_CAP` and every cue/mention->E edge
  at zero weight. Never re-inject wiring after learning: `inject_explicit_wiring` re-initialises STP state and the
  eligibility traces (`sim/bridge.py:4782-4790`), which would erase every engram's recency.

**Declared host residuals** (each named, none of them chooses among candidates): the cue parse (the comprehension
seam); engram-slot bookkeeping and the engram<->block map; the write-time wiring of cue/mention->E_k from the written
triple (a host-transcribed copy of the store index: one-brain and host-list debt; the next rung is a Hebbian write
gated open only during encoding via `set_plasticity_gate`, `sim/bridge.py:5264`); the turn clock (a fixed simulated gap
per turn boundary) and the turn-end reset pulse; the verify filter (it can only abstain); the private bridge (step 3b
tests co-residence); under the flag, organ B's expectation becomes set membership and value choice passes the inner
result through when the race engaged.

## 3. Build order, each step with its STOP line

**Step 0, the default-regime probe (one dev seed, flag off, no build).** At the production default
(`BRAIN_RECONSOLIDATION` on) does a second patient taught for one cue APPEND or REWRITE in place? The record
conflicts (`research/runners/host_share.py:36` and `load_bearing_fraction.py:1392-1396` against
`2026-10-12-validation-battery-prereg.md:394-397`). The answer decides whether the compete reversal counts at the
default or only under the declared `BRAIN_RECONSOLIDATION=0` departure. Report it; it changes no design choice below.

**Step 1, F1 feasibility (organ only, CPU, dev seeds 1-8, at most 2 days, a `tools.lab` probe).** Extend
`ReplayCircuit._build` with the cue/mention layers and E->E facilitation. Write the kill criteria into the probe
BEFORE its first run. Every clause must hold on 8/8 dev seeds x 10 noise draws at one grid row (grid over U, tau_f,
assembly size); no such row is a NO-GO for the method, banked, and the named next carrier is a recurrent persistent
mention trace. Clauses:
- (a) A old, B new, same cue: B wins >= 8/10. (b) A mention of A's patient after both writes: A wins >= 8/10.
- (c) One-role match with maximal priming: 0 latched draws. (d) No matching engram: 0 commits.
- (e) Steer into A's engram over 0, 2, 4, 6, 8, 10, 12, 16 pA: a monotone win curve with STEER* defined; the same
  steer delivered after the natural winner's first gate step + 20 ms moves <= 1/10; a non-selective steer (all
  engrams) moves <= 1/10.
- (f) Reversed slot order leaves the content winner unchanged in >= 9/10 non-tie draws.
- (g) No latch ON in the 30 ms before cue onset across 5 consecutive turns, AND no facilitation bleed: the turn after
  a win, the winner does not win an unrelated or half-matching cue (judge graft); STP state is recorded per turn.
- (h) Near ties: P(winner = lowest slot) and P(winner = alphabetically first word) inside the chance band, with
  per-position heterogeneity reported (the idle-replay row-2 seed-6 effect, `idle_replay_organ.py:116-121`).
- (i) Which lesion is deterministic: closing the mention edge on the context turn, and closing exclusion. Only a
  deterministic lesion may later be a counted bar (the coin-flip lesion sank BG G3).

STOP after step 1 and post `[cs-race-f1] RESULT` with the clause table; Claude rules before step 2.

**Step 2, the default-off organ and wiring.** `research/runners/cs_race_organ.py`; three flag-gated routing edits
(the `query_patient` buffer branch, organ B membership, the value-choice pass-through); the steer, late-steer, lesion
and shadow hooks; a cs-race intervention wrapper in `host_share.py`. Tests: flag-off byte identity against a pinned
40-character SHA; the steer variable (`BRAIN_CS_RACE_STEER`) read in exactly one function whose only effect is a write
to the input-current array (AST test); slot-permutation equivariance. The steer hook added to `ReplayCircuit._run`
must be byte-identical with `steer=None` (pinned test, because the default-on C5 idle organ shares the code) and is
kept OUT of the Stage-2 pin; it doubles as C5 Amendment C later.

**Step 3a, the dev operating point and the smoke.** The grid, with the first-all-pass-row rule written before the
sweep; STEER* frozen on dev seeds 1-8 as the smallest grid value winning >= 8/10 per seed; `tools/answer_rate.py
paired` flag on vs off on the packet probes plus the iw1/iw2 scripts at dev seeds 7 and 8 (records under
`research/findings/smoke/`); a `tools/bar_sanity.py` record from `docs/drafts/cs-race/bar_sanity_spec.py`, which calls
the registered scorer's `decide()`. **Step 3b, in parallel and off the critical path:** co-residence on the composer's
own bridge (precedent: the parser steps on that bridge, `brain_conversational_agent.py:132-157`; wire the race before
parser training with `defer_train=True`; preallocate the cue vocabulary). Its STOP line: if the race and the RF store
cannot interleave without RF phasor damage (`_zero_rf_v_u` per op) by 11-02, the gate runs on the private bridge with
the debt declared.

## 4. The prereg skeleton (registered after step 3a, before any gate seed)

**Arms.** BEFORE (flag off, same probes and revision): `prod_a`, `prod_b` (sham), `prod_int_vchoice`,
`prod_int_kselect_vcoff`, `prod_int_eb` (new: organ B's expectation set to the other patient). AFTER (flag on): shadow
interventions on the retired rules (`BRAIN_CS_SHADOW=1`, instrument-only, `applied=True` required), `slot_reverse`,
`distractor_add`, `loser_remove`, the targeted synapse-weakening arm (the winner's cue->E weights x 0.5 must hand the
win to the runner-up, not to abstain), the half-cue moat probe, a hub control (an agent present in many facts must not
win a half-matched cue by mass), a permuted-cue control. State arms over one cue with A older and B newer: Z (neutral
turn), X (a turn mentioning A's patient), Y (mentioning B's), X2 (two neutral turns of lag, reported dose); both
teaching orders, counterbalanced; word sets counterbalanced so the older fact is alphabetically first in half.

**Bars.** Preconditions (UNDEFINED, never NO-GO, when unmet): sham equality; an engagement record on 100% of flag-on
events; `organ_seed == BRAIN_CHAT_SEED`; membrane in bounds; >= 6 defined events per gated measure; every probe cue
absent from the LTM tier at build. Then:
- **A, before:** pooled F >= 0.8 on compete events, HOST-OWNED.
- **B, answer preservation:** single-match recall gives the same patient flag on and off, 6/6 per item; moat and
  half-cue probes abstain, 6/6; arm Z voices the newer fact on >= 5/6.
- **C, state-dependent reversal (capability only):** arm X voices the older fact and arm Z the newer, >= 5/6, both
  orders. It is not an ownership clause: a host last-mention argmax (N2) passes it by construction.
- **D, ownership by steer (decisive):** D-early at STEER* voices the older fact >= 5/6; D-late leaves the newer fact
  >= 5/6; D-nonsel leaves the newer fact >= 5/6.
- **E, ownership by intervention:** pooled HS <= 0.2 and NS >= 0.8 (NEURAL-OWNED) over the AFTER arms; slot-reverse
  content invariance >= 5/6. This is the Phase B exit quantity on this seam.
- **F, reply lesion:** the deterministic lesion from step 1 (i) turns arm X's reply A->B on >= 5/6, reproduced in the
  repeat; arm Z and single-match replies unchanged 6/6. In the flip packet, the commit or cue-gate lesion on
  SINGLE-MATCH recall is the counted default-regime reply lesion, since single-match turns exist at the default.
- **G, read invariance:** 0 events whose winner changes at READ_FRAC x 0.5 or x 2.
- **H, no bleed:** 0 latches ON before cue onset; 0 non-cue turns holding a latch; facilitation bleed as in F1 (g).

**Nulls, killed before any bar is written.** A clause x null kill matrix run through `tools/bar_sanity.py`: host
recency argmax (N1), last-mention argmax (N2, killed only by D), LIFO and lowest-index (N3, N4), alphabetical (N5),
steer-inert race (N6), non-selective steer (N7), mass/strength argmax (N8), random (N9), timing-blind steer-flag reader
(N10, killed by D-late), fitted host softmax (N11), lesion-flag reader (N12), heterogeneity slot-picker (N13), bleeding
brain (N14). A realistic brain (2.5% misses, a 60/40 near-tie, about 1/80 silent draws) must pass >= 0.95 on the
overall verdict; every null and the random brain <= 0.05.

**Engagement record per flag-on event:** route `cs_race`, cue, K, slot-map hash, facilitation-state hash at cue
onset, per-engram synaptic input integrals (raw masses beside every ratio), first-gate steps, slot counts, the read
at x0.5 and x2, gate values read from the bridge, steer record, noise seed, decode route, `decode_matches_winner`. A
reply whose recalled patient differs from the race output is **MASKED**, never a pass.

**Outcome sentences, fixed in advance.** A GO means "the race, not the host, chooses among buffer facts for one cue,
behind a default-off flag". It does not mean learned wiring, on by default, or the one-brain substrate (unless 3b
landed). If D fails, ownership reads UNDETERMINED whatever C shows.

## 5. Calendar (honest)

- 10-21..10-23: step 0, step 1 (CPU), and the bar-sanity spec and kill matrix (no data needed).
- 10-24..10-30: step 2. 10-31..11-02: step 3a; 3b in parallel.
- Prereg filed alone after 3a. Step O (organ battery, cheap, CPU) first. Step R (reply level) gate lines run either
  before 11-03 or after the Stage-2 lines drain (Stage 2 has the pool 11-03..11-10), or on AWS within the cap,
  started by Claude.
- Realistic target by 11-21: a PROVISIONAL six-seed verdict behind the flag. The flip packet is the stretch. A miss
  rolls the flip to the Stage-3 attempt (12-19); the capability is never dropped (no-defer).

## 6. Who does what

Hermes builds steps 0-3 on a `research/cs-race` branch, posting `[cs-race-f0]`, `[cs-race-f1]` and `[cs-race-op]`
RESULT lines; Claude rules at each STOP, writes the prereg, reviews and merges. The capability stays open whatever F1
shows; a failed carrier is banked and the next one is named in the same ruling.
