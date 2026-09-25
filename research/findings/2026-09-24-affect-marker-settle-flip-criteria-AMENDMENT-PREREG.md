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

## Amendment 3 — A3's GPU timing run CANCELLED by owner decision (2026-09-25; branch research/retire-affect-marker-word)

The owner, verbatim: "It would be weird for the brain's replies to just be adding 'wonderful!' randomly. Its speech
should be influenced by its feelings, not just have a feeling-related word thrown in randomly." Approved option A:
the affective-marker word (`affect_drives.lead`) STOPS being prepended to the answer surface by default (new flag
`BRAIN_AFFECT_MARKER_SURFACE`, default OFF; `=1` restores the pre-2026-09-25 prepend byte-identically) — it is still
COMPUTED and RECORDED every turn (the #81 felt-state read, the #86 spiking WTA selection, and the A2 congruence gate
above are ALL unchanged code paths).

**A3's premise no longer holds.** A3 exists to decide whether `BRAIN_AFFECT_MARKER_SETTLE`'s wall-clock overhead is
small enough to flip its OWN default — a decision that only mattered because SETTLE's product payoff was a better
WORD SHOWN to the user. With the word retired from the surface by owner decision, that payoff is gone: no product
decision now depends on SETTLE's timing, so the queued A3 run (~7.5 GPU-hours, `tools/gpu_queue.sh`, 4 sequential
cupy processes) is **CANCELLED** — not run, not deferred, not silently dropped: named here as withdrawn-by-decision.
`BRAIN_AFFECT_MARKER_SETTLE` itself is untouched by this branch (still default OFF, still selectable for a future
measurement-only comparison of INTERNAL marker quality if one is ever wanted).

**A1 and A2 stand as measurement-only records.** Nothing in this amendment retracts A1's multi-turn-contrast result
or A2's congruence-gate wiring/measurement — both describe the quality/congruence of the INTERNAL marker selection
(`affect_drives.lead`, `affect_marker_congruence`), which is exactly what stays live and load-bearing in production
(docs/PRODUCTION_INTEGRATION_LEDGER.yaml `affect-drives-response` / `affect-marker-spiking-wta` rows; the
FACULTY_PROBES fields `load_bearing_fraction.py` reads for both rows were already the recorded fields, not
`resp['answer']`, so the #1 load-bearing metric is UNCHANGED by this decision). They simply no longer feed a
surface-default decision, since there is no surface default left to flip.

## Amendment log

- A1, A2, A3 (this document, 2026-09-24): first amendment of the three preregistrations committed in 205604a80, b2e50bd37
  and 49a089d8d.
- Amendment 2 (this document, 2026-09-25): A2 production wiring decided + wired (`BRAIN_AFFECT_MARKER_CONGRUENCE`,
  default OFF, `webapp/affect_drives_chat.congruence_gate`) + its production-path measurement registered, ahead of
  any run.
- Amendment 3 (this document, 2026-09-25, branch research/retire-affect-marker-word): owner decision retires the
  affect-marker word from the answer surface by default (`BRAIN_AFFECT_MARKER_SURFACE`, default OFF) — A3's queued
  GPU timing run is CANCELLED (its premise, deciding a surface-default flip, no longer applies); A1/A2 stand as
  measurement-only records of the internal marker selection/congruence, unaffected in status.
