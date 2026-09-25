---
type: preregistration
status: preregistered
date: 2026-09-24
lane: A · Affect (affect-marker SETTLE default-flip preconditions)
mechanism: BRAIN_AFFECT_MARKER_SETTLE (default-OFF) -- the three flip preconditions the owner set (6-seed multi-turn contrast, abstention-congruence rule, affective GPU timing), and the instruments that decide them
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION AMENDMENT only. No amended score has been computed on any seed when this is committed.
runner: research/runners/_affect_marker_settle_multiturn_derisk.py
---

# SETTLE flip criteria: amendment after the independent review (A1 multi-turn, A2 congruence, A3 GPU timing)

An independent review of branch `research/settle-multiturn-contrast` at 79900af1a found that the instruments deciding the
three SETTLE flip preconditions could not be trusted as written. This amendment fixes each instrument and states the rule it
now applies. It is committed in the same commit as the runner code it governs and BEFORE any amended score or new run.

Nothing here flips a default. SETTLE stays opt-in (`BRAIN_AFFECT_MARKER_SETTLE`, default OFF). The owner rule stands: SETTLE
flips only after all three preconditions pass.

## What was already seen when this was written (disclosure)

- The multi-turn arms of s42, s43 and s44 were seen by the lane and by the reviewer. The review's numbers are known to the
  author: on s42-s44 every affective turn abstained, OFF emitted a marker on every warm read, and the Gate-B sign never read
  '-'. Shipping SETTLE without a congruence rule raised markers attached to abstentions from 10 to 12 on those seeds.
- The s100 arms and the three OFF arms of s101 had been harvested into the primary checkout. The author listed their file
  names and did not open them. s101's ON arms and all of s102 are still running on pool2 (pinned revision 49a089d8d).
- The replay lever (A1.3) was checked only on the DEV seed 7, synthetic moods, no brain build: the WTA reader is seeded by
  cfg.seed and gives identical reads with and without process-global RNG draws in between (settle OFF and ON).
- The GPU timing (A3) has never been run in its amended form. The pre-A3 s42 artifacts stay on the branch as the record of
  the old runner and support no verdict.

## A1 — multi-turn contrast (runner `research/runners/_affect_marker_settle_multiturn_derisk.py`)

The worker is unchanged from 49a089d8d, so every arm on pool2 is valid input. Only the scorer changes.

1. **H4 reads the lead wherever it sits.** A later stage prepends a clause ("That's absolutely thrilling for This -- Wonderful!
   I don't know..."). H4 now holds iff removing each arm's own lead at one of its verbatim occurrences leaves the two replies
   equal. A lead the reply does not contain fails H4. The old prefix-only strip scored a mid-reply rescue as NO-GO.
2. **Validity is checked turn by turn.** Each of the six arm files must name the right seed and the right SETTLE/LESION env and
   carry all five turns. Each turn must be a real reply: no `_error`, an `answer` string, and an `affect_drives` record with
   `acted` true and a `reason` that is not `error:*`. Any failure makes the seed invalid. An errored lesion turn no longer counts
   as silenced. `off_lesion` is validated, and H3 on the OFF arms is reported, not gated.
3. **Lever: SETTLE must reach the reader.** Production `expression_lead` returns '' on a reader exception, so a crashed WTA and a
   real cut look the same in a reply. The scorer replays each arm's own recorded (level, high_arousal, mood, felt_arousal)
   sequence through the production `expression_lead`, with a fresh reader built from the arm's env, and records every reader
   exception. Precondition on every valid seed: the env builds the expected reader (SETTLE 500/1000 ms, OFF 60/40 ms), the replay
   raises nothing, and it reproduces every recorded lead of all six arms. Reported: the turns on which the SETTLE setting alone,
   or the lesion alone, decides the lead (counterfactual replays).
4. **Non-vacuous.** ON must differ from OFF on at least one (seed, turn). A SETTLE that changes nothing reads UNDEFINED.
5. **D1 is split, still reported and never gated.** `D1_first_read` is mt_emo1 (the single-turn defect already shown on
   2026-09-23). `D1_warm_read` is mt_emo2, mt_neg1 and mt_emo3 (the warm-reader defect this probe was built for). A pass with
   `D1_warm_read` false means the flip is safe on multi-turn sessions. It does not mean the flip fixes a multi-turn defect,
   and the verdict's `interpretation` field says so.
6. **Rates per affective turn** (4 per seed) are reported beside the per-turn rates.

**Rule.** GO iff every precondition holds (6/6 valid seeds, H5 determinism, the A1.3 lever, A1.4 non-vacuous) and H1-H4 hold
on every (seed, turn). NO-GO iff the preconditions hold and any H1-H4 cell fails. Otherwise UNDEFINED.

## A2 — abstention-congruence (runner `research/runners/_affect_marker_settle_congruence.py`)

1. **The 8d4605b74 "rule holds" status is withdrawn as a flip criterion.** It was written after s42-s44 were seen and carries
   no pre-registered weight. Its numbers are descriptive only.
2. **Not a production property.** `apply_policy` edits replies after they were recorded. No module in `webapp/` reads
   `BRAIN_AFFECT_MARKER_CONGRUENCE`, so the policy is not wired (docs/TERMS.md). The rule's status now requires the policy to have
   been applied by the production path. Every current mode is post-hoc, so the status reads UNDEFINED until a production wiring
   exists and is scored as shipped.
3. **Both halves must be exercised.** Condition (2) fires only when the Gate-B sign is opposite to the marker. The status
   requires Gate-B to read '-' on at least one scored turn and '+' on at least one. Otherwise the valence half is UNTESTED and
   the status reads UNDEFINED.
4. **Named host shortcut S6.** The policy is a host string edit: the reply abstained, so the marker word is deleted. It is
   declared as a host shortcut. A production congruence mechanism is a separate design decision for the owner (for example,
   gating the marker with the same spiking speak/abstain race that decides abstention). This amendment wires nothing.
5. **Descriptive SETTLE contrast** (`--score-settle-contrast`, no threshold, no verdict). For off_a and on_a as shipped, per
   affective turn: markers, markers attached to an abstention, and markers on an opposite-sign Gate-B read. It also reports
   the markers that would survive the policy in each arm, which is SETTLE's visible effect if the policy shipped.

Consequence, stated now: criterion (2) cannot pass on this branch. What it should mean is an owner decision (which congruence
mechanism ships, and on what probe it is tested). The probe needs affective turns that do not abstain, and turns where
Gate-B reads '-'.

## A3 — affective GPU timing (runner `research/runners/_affect_marker_settle_gpu_timing.py`, rewritten)

- **Quantity:** criterion L (228ba16f0): the warm-turn total wall time of `webapp.server.brain_chat` on the default production
  path (cupy, tiny-demo, the Qwen renderer with the LLM enabled, `rich` at the production default), ON minus OFF, bound +0.3 s.
- **One change to criterion L, and why:** its default message pair is neutral, so the affect-marker WTA never runs and the
  SETTLE delta is zero by construction. Every scored warm turn here is affective.
- **Protocol:** seed 42 (the production default; a latency de-risk, not a capability claim). Four fresh sequential processes in
  the counterbalanced order off, on, on, off. Each process runs a build turn, a first affective turn (not scored), then 4 warm
  affective turns alternating EMO2_TEXT and EMO_TEXT (scored). `AffectMarkerWTA._select` is wrapped by a timer inside each
  process, so each warm turn also reports the time spent in the spiking WTA, the reader config it ran with, and its read count.
- **Metrics:** M1 = arm(ON) - arm(OFF) of warm-turn total time. M2 = the same for WTA time. An arm's value is the median of its
  processes' warm-turn medians. NOISE = the largest spread of process medians within one arm. M3 (total minus WTA) is reported.
- **Preconditions (UNDEFINED if any fails):** every process completed; at least 2 processes per arm; balanced order; one code
  revision with no modified tracked file except the provenance log; backend cupy; the reply names the Qwen renderer; every
  warm turn is affective and made at least one WTA read; every read and every cached reader used the arm's expected config.
- **Rule:** PASS region: M2 <= 0.3 and M1 + NOISE <= 0.3. FAIL region: M2 > 0.3 or M1 - NOISE > 0.3. GO in the PASS region,
  NO-GO in the FAIL region, UNDEFINED when the result falls in neither (the whole-turn delta cannot be told from the bound).

## Commands (verbatim)

Output paths not yet written are named by placeholders (`<multiturn_verdict_json>` is the multi-turn
raw dir's `verdict.json`; `<a3_dir>` is the `a3` subdirectory of the GPU timing raw dir), the convention
`2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md` uses, so no future artifact is cited as evidence.

```
# A1: score whatever is harvested (UNDEFINED until all six seeds are present); replay runs automatically
.venv/bin/python -m research.runners._affect_marker_settle_multiturn_derisk --score \
  --raw-dir research/findings/raw/_affect_marker_settle_multiturn --seeds "42 43 44 100 101 102" \
  --out <multiturn_verdict_json>

# A2: descriptive SETTLE contrast, and the rule status (UNDEFINED while post-hoc)
.venv/bin/python -m research.runners._affect_marker_settle_congruence --score-settle-contrast \
  --raw-dir research/findings/raw/_affect_marker_settle_multiturn --seeds "42 43 44 100 101 102"
.venv/bin/python -m research.runners._affect_marker_settle_congruence --score-multiturn --arm on_a \
  --raw-dir research/findings/raw/_affect_marker_settle_multiturn --seeds "42 43 44 100 101 102"

# A3: queued through tools/gpu_queue.sh from a clean checkout pinned at this commit (see the runner docstring)
SIM_BACKEND=cupy OMP_NUM_THREADS=1 bash tools/memcap.sh 20 -- .venv/bin/python -u -m \
  research.runners._affect_marker_settle_gpu_timing --run --seeds 42 --order off,on,on,off \
  --out-dir <a3_dir> --out <a3_dir>/verdict.json

# selftests (no brain build): each drives its verdict through every failing direction
.venv/bin/python -m research.runners._affect_marker_settle_multiturn_derisk --selftest
.venv/bin/python -m research.runners._affect_marker_settle_congruence --selftest
.venv/bin/python -m research.runners._affect_marker_settle_gpu_timing --selftest
```

## Amendment 2 — A2 production wiring + its production-path measurement (2026-09-25)

The A2 section above withdrew `apply_policy`'s "rule holds" status and named the open design question: "a
production congruence mechanism is a separate design decision for the owner (for example, gating the marker with
the same spiking speak/abstain race that decides abstention)." This amendment makes that decision, wires it, and
registers how it is measured on the production path, BEFORE any run it governs.

**The chosen mechanism** (bound in `research/biology/affective-marker-abstention-congruence-gate.md`): a
conflict-monitoring GATE, `webapp/affect_drives_chat.congruence_gate`, called at the SAME two production sites
that already prepend the affect-marker lead (`webapp/server.py`'s rich path and single-fact path), BEFORE the
lead is prepended — not a post-hoc string edit on an already-composed reply. It reads two signals the brain has
ALREADY computed this turn: `resp["abstained"]` (the moat/BG speak-vs-abstain decision) and
`resp["affect"]["valence_sign"]` (the Gate-B spiking affect organ's independent valence read). When the marker's
register disagrees with either (abstention conflict, or a register/Gate-B sign mismatch), the marker is withheld
before it ever reaches the surface. Flag: `BRAIN_AFFECT_MARKER_CONGRUENCE` (default OFF; unchanged name from the
research runner, so `congruence_wired_in_webapp()`'s static check now reads True). This is NOT the literal
same-circuit spiking veto the parenthetical evoked (a projection from the abstain/moat organ onto the marker
WTA's own assemblies, gating its selection before it fires) — that is named as the next rung, not claimed here.

**Declared host step.** The register-word→sign lookup (the existing `_LEAD_WORD`-derived table) and the
withhold/surface branch are host control flow over two neural booleans/signs — the same pattern already used for
every other Gate-B-driven coupling in `webapp/server.py` (metacog hedge, curiosity follow-up, surprise/
reconsolidation prefixes each gate a string operation on a spiking read's boolean). Nothing computes abstain or
valence sign; the gate only reads them.

**A2's measurement on the production path** (the battery/probe hook this amendment adds,
`research/runners/_affect_marker_settle_congruence.py --run-wiring` / `--score-wiring`): for each of the 6
pre-registered seeds, spawn TWO fresh production `webapp.server.brain_chat` processes (congruence OFF, congruence
ON; `BRAIN_AFFECT_MARKER_SETTLE` held at its production default, OFF) over the SAME 5-turn multi-turn sequence
`_affect_marker_settle_multiturn_derisk` already uses (`mt_neutral, mt_emo1, mt_emo2, mt_neg1, mt_emo3`), which is
known (this document's disclosure above) to contain both an abstention-conflict candidate and, via `mt_neg1`, a
candidate for a Gate-B negative read — so the SAME turns already used for A1 exercise A2's preconditions instead
of a new hand-built battery.

  1. **Byte-identical-OFF** (checked in data, not inferred): with the flag unset, every field of every turn's
     response — including `answer`, `abstained`, `affect_drives`, `affect` — is compared exact-equal against a
     process run with `BRAIN_AFFECT_MARKER_CONGRUENCE=0` and against the pre-existing (pre-amendment) production
     response with the env var absent entirely. A single differing byte on any turn is a hard FAIL of this
     precondition (not UNDEFINED — the additive/byte-identical-off contract is unconditional).
  2. **Both conflict halves exercised (A2.2, carried over).** The status is UNDEFINED unless `resp["affect"]
     ["valence_sign"]` reads `"-"` on >= 1 scored turn and `"+"` on >= 1 scored turn across the 6-seed set.
  3. **Non-vacuous.** UNDEFINED unless the ON arm withholds a marker relative to OFF on >= 1 (seed, turn) that
     the OFF arm actually emitted one on — the gate must be shown to DO something on a live production turn, not
     merely fail to crash.
  4. **Register/attribution.** Every ON-arm withheld turn must trace, via `resp["affect_marker_congruence"]`
     (the additive trace this amendment's wiring attaches only when the gate ran and had a lead to check), to
     EXACTLY the abstention-conflict or valence-conflict condition claimed — a withheld marker with no matching
     condition recorded is a FAIL, not a silent pass.
  5. **Rule.** GO iff (1) holds on all 6 seeds AND (2) and (3) hold across the set. NO-GO iff (1) holds but (2) or
     (3) fails. UNDEFINED if (1) fails, or a process errors, or fewer than 6 seeds are valid.

Command (verbatim; `<out>` is `research/findings/raw/_affect_marker_settle_congruence/wiring`):
```
.venv/bin/python -m research.runners._affect_marker_settle_congruence --run-wiring \
  --seeds "42 43 44 100 101 102" --out-dir <out>
.venv/bin/python -m research.runners._affect_marker_settle_congruence --score-wiring \
  --raw-dir <out> --seeds "42 43 44 100 101 102" --out <out>/verdict.json
```

**Committed same-commit as the wiring it governs** (`webapp/affect_drives_chat.py`, `webapp/server.py`,
`research/runners/_affect_marker_settle_congruence.py`, `tests/test_affect_marker_congruence_gate.py`), before any
run against it.

## Amendment 3 — A3: a within-process crossover that can resolve the 0.3 s bound (2026-09-25)

**Disclosure.** This section is written AFTER the amended A3 run was seen and BEFORE any new data. That run (gpu_queue
03:21-04:18 2026-09-25, clean pinned checkout d71c1a2c9, seed 42, order off,on,on,off) read UNDEFINED with every
other precondition met: M2 (WTA-time delta) +0.133 s, M1 (total delta) -3.46 s, NOISE 4.72 s <!--derived-->
(`research/findings/raw/_affect_marker_settle_gpu_timing/a3/verdict.json`). The author has read all four per-process
records and decomposed their noise (finding `2026-09-25-affect-marker-settle-a3-gpu-timing-UNDEFINED-noise.md`,
committed right after this section). The A3 rule above is not changed retroactively: the A3 run stays UNDEFINED.
This amendment replaces the INSTRUMENT that decides precondition 3. The quantity and the bound do not change.
**Stated plainly (review LOW, 2026-09-25): the DECISION STATISTIC changes too, and it is a less strict standard of
evidence, adopted only after A3 read UNDEFINED.** A3 compared a worst-case RANGE of process medians against the
bound (PASS needs M1 + NOISE <= bound, i.e. the observed delta plus the full spread of what was seen). This
amendment instead uses one-sided 95% confidence bounds from an OLS fit (PASS needs the upper 95% bound <= bound).
A calibration check (`_xo_go_rate` in the runner, added in the 2026-09-25 fix round below and tightened to 1000
independent synthetic reps in the SECOND 2026-09-25 fix round after review) reads a GO rate of 5.1% at a true
whole-turn cost exactly at the bound -- i.e. this instrument reads GO on roughly 1 run in 20 at the bound itself,
matching its nominal one-sided alpha=0.05, where A3's worst-case range would essentially never resolve GO by
chance. (An earlier, 100-rep draw of the same check read 9% -- within that small sample's own binomial noise, not
a second measured rate; see Addendum item 2 below.) This is the SAME kind of change reviewers reject when made
after a negative result without disclosure; it
is disclosed here, in this amendment, before any Amendment 3 gate data exists (the smoke below is a pre-flight
check, not the gate), and the reason is stated in the candidates weighed above: a worst-case range provably cannot
resolve 0.3 s at any affordable process count, so a probabilistic bound is the only design that can decide this
question at all.

**Quantity (unchanged).** Criterion L (228ba16f0) as amended in A3: the warm affective turn total wall time of
`webapp.server.brain_chat` on the default production path (cupy, tiny-demo, the Qwen renderer with the LLM on, `rich`
at the production default), SETTLE ON minus OFF, bound +0.3 s. In words: does SETTLE add more than 0.3 s to a
production affective warm turn?

**What made A3 unresolvable (from the decomposition).**

- The noise is between processes. The two OFF processes had warm medians 35.31 s and 30.59 s. <!--derived-->
  The SETTLE-free build turn was slow in the same processes: 706 s and 668 s (first two) vs 605 s and 600 s. <!--derived-->
- A3's NOISE is a RANGE of process medians. The expected range of n draws grows with n (about 1.13, 2.06 and 3.08
  standard deviations at n = 2, 4, 10). No number of A3-style processes resolves the bound. <!--derived-->
- Read as a standard error instead, the arm difference needs about 1168 processes per arm to put its one-sided 95%
  upper bound under 0.3 s at the observed M2, and about 2670 per arm for 80% power (pooled process SD 2.457 s). <!--derived-->

**Candidates weighed.**

1. *More A3 processes.* Rejected: above. At about 14 minutes per process, 1168 per arm is about 545 hours. <!--derived-->
2. *Deterministic renderer settings or a fixed token budget.* Rejected. The Qwen mouth already decodes greedily
   (`do_sample=False`), reseeds before every call and caps each call at 24 new tokens
   (`research/runners/_grounded_lang_integration_derisk.py::SpikingQwenFaculty._generate`). Its output for a given
   prompt and state is already fixed. Forcing a fixed budget (for example `min_new_tokens`) would change the production
   reply, so the run would no longer measure the production path.
3. *Bound = M2 + (render-length difference SETTLE causes) x (measured per-token time).* Not the gate. At the default
   flags the lead SETTLE selects is prepended after the reply is rendered, and nothing downstream reads it, so the
   render-length channel is expected to be zero. A bound built from counted work cannot see a cost SETTLE adds to
   work it does not change (GPU clock state, allocator, cache). It would also measure a model of the wall time, not the
   wall time. The same counts are recorded below as a diagnostic that explains any non-WTA delta. They are not gated.
4. *Within-process paired ON/OFF turns on identical input.* Chosen, with three additions that A3's data showed are
   needed: runs of same-arm turns with a washout, mirrored orientation across processes, and run-level fixed effects.
   Reasons: (a) it removes the between-process noise that made A3 UNDEFINED, because both arms share each process;
   (b) the A3 turn-index means ranged 28.75-33.81 s and all four processes shared this pattern. Mirrored orientations
   put two ON and two OFF processes at every run slot, so a slot fixed effect removes it. (c) It measures the quantity
   itself: the wall time of production turns. <!--derived-->

**Protocol.** Runner `research/runners/_affect_marker_settle_gpu_timing.py`, `--xo-run` (the A3 code path is kept
and unchanged).

- Four fresh sequential processes at seed 42 (the production default). Orientations `off,on,on,off` give each
  process's first run arm. This keeps A3's 4-process counterbalance as mirrored orientations.
- `BRAIN_AFFECT_MARKER_SETTLE` is set explicitly to "1" or "0" before every turn. `get_reader` keys its cache by the
  flag, so each arm keeps its own process-warm reader. Nothing is rebuilt between turns. A rebuild would add the reader
  build cost to every turn, and a production warm turn never pays that cost.
- Per process: the warm Qwen renderer is built first (the server's own startup warm). Then a build turn (NEU_TEXT,
  reset). Then two unscored warm-up affective turns, one per arm, so both readers exist before scoring. Then 48 runs of
  4 same-arm turns. Run arms are ABBA repeated (A = the orientation), so linear drift cancels within each process.
- The first turn of every run is a washout and is not scored. Every scored turn therefore follows a turn of its own
  arm. A lag-1 carry-over of SETTLE into the next turn is charged to the arm that causes it, as in an all-ON session.
- Turn k gets `(EMO_TEXT, EMO2_TEXT)[k % 2]` in every process, so every index has identical input in both arms.

**Differences from a production session, and why none of them touches a scored turn's work.** (i) Both readers are
cached. The second one is a separate tiny private bridge that is never stepped on the other arm's turns. (ii) The
OFF reader is read only on OFF turns. Its 40 ms washout makes its winner history-dependent, so its read count per turn
(1 or 2) can differ from an all-OFF session. The difference is at most one read, about 0.011 s per turn. (iii) The <!--derived-->
renderer is pre-warmed, which is the server's own startup path. <!--derived-->

**Instrument (wrappers only time and count; no production code changes).** Per turn: wall time; process CPU time;
1-minute load average at turn start; the spiking WTA `_select` time, config, read count and exceptions; every
`generate` call on the warm Qwen model (time, batch, new tokens, prompt tokens); the reply with this turn's lead
removed (sha1) and its length; CuPy pool bytes in use; max RSS.

**Estimate.** A run's value is the mean over its kept scored turns. OLS on run values with process and run-slot fixed
effects and an ON indicator. delta = the ON coefficient; one-sided 95% bounds delta +/- t(0.95, df) x SE.
M1 = delta for the total wall time; M2 = delta for the WTA time. Reported, not gated: delta for the Qwen render time
and for the rest; work identity per kept index (render calls, tokens, reply without the lead, ON vs OFF); a lag-1
carry-over estimate over all affective run turns; a process-demeaned per-slot median; the load average by arm.

**Preconditions (UNDEFINED if any is unmet).**

- Every process completed and validated: cupy; the Qwen renderer on every turn; HTTP 200; a clean tree; the planned
  arm on every turn; one cached reader per arm with its config; every affective run turn read the WTA with its arm's
  config (ON 500/1000 ms, OFF 60/40 ms); no reader exception. This is the lever check.
- At least 2 processes per orientation, with equal counts. One code revision. One plan. The same message at every
  index in every process.
- Kept scored indices are at least 80% of those planned. An index is dropped for ALL processes if any process's turn
  there is neutral. The level is set before the WTA reads, so the drop cannot depend on the arm.
- Resolvable (the rule below).

**Rule.** bound = 0.3 s. PASS region: U(M2) <= 0.3 AND U(M1) <= 0.3 -> **GO**. FAIL region: L(M2) > 0.3 OR
L(M1) > 0.3 -> **NO-GO**. Anything else is **UNDEFINED**. The rule can fail in both of its measured directions:
the WTA's own cost (M2), or the whole turn including every non-WTA and lag-1 carry-over effect (M1). Selftest
(no brain build) drives each direction: a 0.6 s WTA -> NO-GO; +1.0 s outside the WTA -> NO-GO; +0.8 s charged to the
next turn -> NO-GO; a true 0.30 s cost with 0.3 s noise -> UNDEFINED; 5 s noise -> UNDEFINED; each lever, validity and
balance failure -> UNDEFINED.

**Size (derived from the A3 records; the arithmetic is in the finding).** The scored-turn count needed for 80% power
to read GO when the true cost equals the observed M2 is N = 4 sigma^2 ((1.645 + 0.842) / (0.3 - 0.133))^2. <!--derived-->
With sigma = 0.80 s (the two-way process x turn-index residual without the contended process) N = 571.
With sigma = 1.72 s (all four processes) N = 2626. <!--derived-->
Chosen: 4 processes x 48 runs x 3 scored turns = 576 scored turns (768 run turns plus builds and warm-ups). This
gives an expected half-width of about 0.11 s at sigma = 0.80 s and about 0.24 s at sigma = 1.72 s. The second case
would read UNDEFINED again. <!--derived-->
Projected GPU-queue time is about 7.3 h (A3: build about 700 s, warm turn about 30 s). The per-process timeout is 4 h.
A process that crashes keeps its partial record; `--xo-run` reruns only processes whose complete record at the same
HEAD is missing. <!--derived-->

**What each outcome would mean.** GO: at seed 42 on this machine, SETTLE adds at most 0.3 s to a production
affective warm turn, counting the WTA and every other effect SETTLE has on that turn and on the next turn.
NO-GO: it adds more. UNDEFINED: the run could not tell. If turn noise is the reason, the next rung is variance
control (a quiet-machine window or CPU isolation), not a smaller bound or a looser rule. The verdict is a 1-seed
latency de-risk, as criterion L scopes it. It is not a capability claim.

**HELD (owner, 2026-09-25): do not queue the run below.** The owner is reconsidering the prepended affect-marker
design itself (it may be retired from replies) independent of what this instrument would read.

Commands (verbatim). The pin is a **SHA, not a branch name** (Addendum below): a branch name resolves relative to
whichever checkout reads it, and a stale worktree elsewhere can hold it pointed at an old commit. **A literal SHA
written here drifted stale across three straight fix rounds (8dd9c1ed0, then ce4afac2b, then the b21140758 smoke
worktree) before this round replaced it with `<pin>`/`<that SHA>` symbols (review LOW 2026-09-25) -- resolve the
SHA at QUEUE TIME, never from a value written in this document:** `git fetch origin research/settle-a3-amendment3
gitea && git log -1 --format=%H origin/research/settle-a3-amendment3` (verify
`gitea/research/settle-a3-amendment3` reads the SAME SHA -- `push_both.sh` keeps both remotes identical). `<pin>`
is `/home/dant123/Projects/sim/.claude/worktrees/settle-a3x-run-<that SHA>` (detached-HEAD worktree at it);
`<a3x>` is `<pin>/research/findings/raw/_affect_marker_settle_gpu_timing/a3x`:
```
# selftest (no brain build): both A3 and Amendment 3 verdicts through every failing direction
.venv/bin/python -m research.runners._affect_marker_settle_gpu_timing --selftest

# 1. pinned worktree + corpus symlink (data/ is gitignored; the Qwen renderer reads data/corpus/tinystories.txt)
cd /home/dant123/Projects/sim && git fetch origin research/settle-a3-amendment3 && \
  git worktree add --detach <pin> <that SHA> && \
  mkdir -p <pin>/data && ln -s /home/dant123/Projects/sim/data/corpus <pin>/data/corpus

# 2. the run (GPU queue, one job): mem_ok wait, before_you_build (corpus-check gate), then the memcap-bounded run
bash tools/gpu_queue.sh add 'cd <pin> && until bash tools/mem_ok.sh 16 4 >/dev/null 2>&1; do sleep 60; done; \
  bash tools/before_you_build.sh "affect-marker SETTLE A3 whole-turn GPU timing at the 0.3s bound (Amendment 3 within-process crossover)" >/dev/null 2>&1; \
  SIM_BACKEND=cupy OMP_NUM_THREADS=1 bash tools/memcap.sh 20 -- \
  /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._affect_marker_settle_gpu_timing --xo-run --seeds 42 \
  --orient off,on,on,off --runs 48 --run-len 4 --out-dir <a3x> --out <a3x>/verdict.json'

# score an existing raw dir again (no brain build)
.venv/bin/python -m research.runners._affect_marker_settle_gpu_timing --xo-score --raw-dir <a3x> \
  --seeds 42 --orient off,on,on,off --runs 48 --run-len 4 --out <a3x>/verdict.json
```

**Addendum to Amendment 3 (2026-09-25, fix round after an independent review of this document and the runner; no
Amendment 3 gate data exists yet -- the short smoke described below is a pre-flight check of the worker code
path, never the decisive 576-scored-turn run -- so this instrument change is made before any data it would
decide, per the standing rule that an instrument/rule change after review is allowed only in that window.** Every
change is in `research/runners/_affect_marker_settle_gpu_timing.py`; none touches `sim/` or `webapp/`.

1. **New precondition: both warm-up turns must each commit exactly 2 WTA reads.** `check_process_xo` previously
   checked only which arm ran during warm-up, not whether it fully ran. If a warm-up turn's valence axis does not
   select a word, `expression_lead` returns '' before the arousal axis is read (1 read, not 2) -- observed in A3
   itself at one OFF warm index. A reader left with its arousal bridge unbuilt then builds it lazily inside the
   first SCORED turn of that arm, adding one-time build latency to that arm's early scored turns only, which
   would bias M1 toward GO without anything in the record flagging it. Unmet -> UNDEFINED, not a silent pass.
2. **Calibration case (`_xo_go_rate`).** Every existing selftest case checks one synthetic draw's sign (does this
   scenario read NO-GO/GO/UNDEFINED), which cannot pin the one-sided 95% bound's WIDTH -- halving
   `t = float(stats.t.ppf(1.0 - alpha, df))` in `_fe_fit` passed every one of them. The case originally ran 100
   independent synthetic reps at a true whole-turn cost exactly at the 0.3 s bound (runs=8) and asserted the GO
   rate stays <= 10%; it read 9% on the correct code, and the halved-`t` mutant was hand-verified (mutate, rerun,
   revert, diff back to clean) to push it to 24%. **Revised in the SECOND 2026-09-25 fix round (review LOW): a
   ceiling alone lets an OVER-conservative mutant (`t` inflated, e.g. `t * 1.5`) pass unnoticed while still
   silently widening every CI (wastes GPU time, never corrupts a verdict, so nothing else here catches it) --
   and a fixed 100-rep draw's own binomial noise (95% CI roughly 4-16% around a true ~5%) meant the "9%" was
   never a precise measurement of the rate in the first place.** `n_reps` raised to 1000 (~4 s) and the assertion
   changed to a two-sided `0.025 <= rate <= 0.08` band (a CHOSEN threshold, not a measurement) around the nominal <!--derived-->
   one-sided alpha=0.05: the correct code now reads 5.1%; the halved-`t` mutant reads ~9.2%, and a lighter
   `t * 0.8` mutant (which the old <=10% ceiling alone would have passed) reads ~8.9% -- both hand-verified
   (mutate, rerun, revert, diff back to clean) to clear the new 8% ceiling.
3. **Carry case near the bound.** The existing carry=0.8 selftest case reads NO-GO whether or not washout turns
   are correctly excluded from scoring (a true M1 of +0.93 s clears 0.3 s either way), so it does not test that
   the washout's carry-over is charged to the arm that CAUSES it. A new case (carry=0.25, noise=0.05) asserts
   both the NO-GO verdict and that M1's estimate lands near its true value (~0.38 s), which a washout-scoring
   regression would move.
4. **The crossover worker's first execution risk (`_worker_xo`, never run before this fix round) is addressed
   procedurally, not by a code change:** a short `--xo-run --orient off,on --runs 4 --run-len 2` smoke against
   the real Qwen renderer, queued separately and read before the full run (see the finding
   `2026-09-25-affect-marker-settle-a3-gpu-timing-UNDEFINED-noise.md` for its result once available).
5. **The pin is now a SHA, not a branch name** (both command blocks above), for the reason stated where they
   appear: a branch name resolves relative to whichever checkout reads it, and a stale worktree elsewhere left it
   pointed at an old commit.

**SECOND fix round (2026-09-25, after review of the round above), still before any Amendment 3 gate data:**

6. **`_worker_xo` now fails fast on an under-read warm-up**, instead of only rejecting the finished record after
   burning the rest of the process's GPU time on a run already doomed to UNDEFINED (review LOW: this is
   procedurally the same failure (1) above catches after the fact, addressed here at the SOURCE). Extracted the
   condition into a pure `_bad_warmup_reads(warmup_turns)` helper (unit-tested in
   `tests/test_affect_marker_settle_gpu_timing_xo.py` without a GPU) so the worker itself needs no test harness
   change to stay covered.
7. **New precondition: at least one real `model.generate()` call somewhere in the process when the Qwen renderer
   is required.** Root cause of the orchestrator's "M4_render reads exactly 0.0 (se 0, resid 0) on every turn"
   finding against the b21140758 smoke: every one of its turns read `abstained: True` (the affect-exclamation
   messages match no stored fact in the tiny-demo brain), and an abstain's reply is the HOST-composed curiosity
   follow-up (`curiosity_production_organ.followup_question`) -- `MoodConditionedRenderer.render_svo`, and
   therefore `model.generate()` and the `timed_generate` wrapper around it, is only ever reached for a
   GATE-MATCHED fact. The wrapper is not broken; `check_process_xo`'s existing `renderer` check is a static
   per-response identity label, not evidence Qwen actually ran, and had no precondition catching this. Unmet
   (zero `n_gen_calls` across build + warm-up + every scored turn) -> UNDEFINED.
8. **The queued-recipe pin is now resolved AT QUEUE TIME, never written into this document as a literal SHA**
   (review LOW 2026-09-25: a literal SHA drifted stale across three straight fix rounds -- 8dd9c1ed0, then
   ce4afac2b, then the b21140758 smoke worktree -- and the paired finding's fully-expanded literal path made
   `tools/claim_check.py` treat a not-yet-existing future artifact as a cited one, 1 UNSUPPORTED). Both documents'
   command blocks now use `<pin>`/`<that SHA>`/`<a3x>` symbols exclusively (the prereg's own pre-existing style);
   the resolving command (`git log -1 --format=%H origin/research/settle-a3-amendment3`, cross-checked against
   `gitea`) is stated in prose, outside any fenced code block.

None of (1)-(3), (6), (7) or (8) touches the RULE (`decide_xo`'s GO/NO-GO/UNDEFINED regions) or the QUANTITY/bound
stated at the top of this amendment -- they tighten what counts as a VALID process ((1), (6), (7)), add tests that
pin the instrument's already-stated behavior ((2), (3), and the mutation tests in
`tests/test_affect_marker_settle_gpu_timing_xo.py`) rather than changing it, and fix a documentation-only
citation/pin-drift regression ((8)). `--selftest` (no brain build) passes 12 + 31 = 43 cases after both fix rounds
(26 -> 30 -> 31 Amendment 3 cases).

## Amendment log

- A1, A2, A3 (this document, 2026-09-24): first amendment of the three preregistrations committed in 205604a80, b2e50bd37
  and 49a089d8d.
- Amendment 2 (this document, 2026-09-25): A2 production wiring decided + wired (`BRAIN_AFFECT_MARKER_CONGRUENCE`,
  default OFF, `webapp/affect_drives_chat.congruence_gate`) + its production-path measurement registered, ahead of
  any run.
- Amendment 3 (this document, 2026-09-25): A3's instrument replaced by a within-process crossover (`--xo-run`), written
  after the A3 run read UNDEFINED (M2 +0.133 s inside the bound, M1 not resolvable at NOISE 4.72 s) and before any new <!--derived-->
  data. Quantity and bound unchanged. <!--derived-->
- Addendum to Amendment 3 (this document, 2026-09-25, fix round after independent review): a new warm-up
  read-count precondition, a CI-width calibration selftest, a carry-near-bound selftest, and a SHA (not branch
  name) pin -- before any Amendment 3 gate data. Quantity, bound and rule unchanged; see the addendum above for
  each change and why.
- Second addendum to Amendment 3 (this document, 2026-09-25, fix round after review of the addendum above): the
  crossover worker now fails fast on an under-read warm-up instead of only rejecting the finished record; a new
  precondition rejects a process that never actually invoked `model.generate()` when the Qwen renderer is
  required (the b21140758 smoke's M4_render finding); the calibration case widened from a ceiling-only 100-rep
  check to a two-sided 1000-rep floor+ceiling; the false-GO-rate prose above corrected to the 1000-rep 5.1%
  measurement; and both documents' queued-recipe pin switched from a literal SHA (stale twice already) to
  resolve-at-queue-time symbols, fixing a `claim_check` false-citation regression in the paired finding.
  Quantity, bound and rule unchanged; still before any Amendment 3 gate data; the 7.3-7.5 h run itself stays
  HELD pending the owner's affect-marker-design reconsideration. See addendum items (6)-(8) and the revised (2)
  above.
