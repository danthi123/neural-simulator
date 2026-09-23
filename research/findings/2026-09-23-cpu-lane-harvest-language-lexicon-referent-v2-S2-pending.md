---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-23
mechanism: Hebbian(Oja)-learned feedforward frame->category synapses (all-to-all FR->CN referent / FR->CX
  non-referent) drive two spiking pools with reciprocal FSI cross-inhibition; the DECISION is a host
  comparison of the two pools' rates (rn>rx with a dead-margin abstain), not the inhibitory competition --
  lateral inhibition exists and has a measurable wiring effect but does not carry the decision (see below).
  Teacher-supervised curriculum (host-selected 38+37 word seed set at deployment scale) -- "self-organized"
  does not apply; "teacher-driven" per docs/TERMS.md. Localist frame input code. Harvest of
  research/language-lane-next (HEAD 87bf35d1abf725340cc6c9d19ef7afa502e62010, NOT merged here) -- no new
  mechanism, runner, or sim/ edit in this finding.
lane: language (D6 referent-lexicon competition, v2 spiking-Hebbian-frames)
seeds: [42, 43, 44, 100, 101, 102]
verdict: INCOMPLETE -- S2 (>=1000-seed-label permutation null, the last required v2 evidence gate) has NOT
  landed. It is actively running on pool42 as of 2026-09-23 15:34 EDT (all 6 seeds in flight, started
  12:03-13:37 EDT, 2-3.5h elapsed with no ETA available from the runner). No v2 GO/NO-GO verdict is given.
  The other 7 gates (S1, S3a, S3b, S4, S5, S7, S8) HAVE landed and are read here directly from the committed
  per-seed artifacts, with the re-review's corrections applied to how each is characterized.
artifacts:
  - research/findings/raw/_lexicon_spiking_referent/main_s42.json
  - research/findings/raw/_lexicon_spiking_referent/main_s43.json
  - research/findings/raw/_lexicon_spiking_referent/main_s44.json
  - research/findings/raw/_lexicon_spiking_referent/main_s100.json
  - research/findings/raw/_lexicon_spiking_referent/main_s101.json
  - research/findings/raw/_lexicon_spiking_referent/main_s102.json
  - (+ matching .prov.json sidecars) -- copied via `git show` from research/language-lane-next HEAD 87bf35d1a
    (NOT merged) for this harvest; these were already committed there and are unmodified here
external: NO-EXTERNAL-NEEDED -- reads already-run, already-committed gate arms; no new claim.
builds_on:
  - research/runners/_lexicon_spiking_referent_derisk.py (research/language-lane-next branch, NOT merged)
  - research/findings/2026-09-23-lexicon-referent-v1-host-label-spreading-6seed-NO-GO-and-relabel.md (same
    branch)
  - research/findings/2026-09-23-lexicon-referent-v2-spiking-wta-hebbian-frames-main-gates-pass-null-pending.md
    (same branch)
  - re-review of that branch (verdict: fix-required; prior_issues_resolved: false -- 5 issues, honored below)
next_command: "bash tools/pool_sync.sh   # then, once all 6 null_s*.json exist under
  research/findings/raw/_lexicon_spiking_referent/: .venv/bin/python -m
  research.runners._lexicon_spiking_referent_derisk --score research/findings/raw/_lexicon_spiking_referent"
---

# Language CPU-lane harvest: v2 main gates read, S2 null still in flight

**Artifacts read (copied from `research/language-lane-next`, HEAD `87bf35d1a`, NOT merged here):**
`research/findings/raw/_lexicon_spiking_referent/main_s42.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s43.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s44.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s100.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s101.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s102.json`.

## Exact status of the pending gate (S2)

`ssh pool42 ps aux` (this session, 2026-09-23 15:34 EDT) shows all 6 seed-label permutation-null jobs still
running as live processes, none finished:

| seed | PID | started | CPU time so far |
|---|---|---|---|
| 42  | 3345947 | 12:03 | 129m43s |
| 43  | 3655934 | 13:25 | 91m14s |
| 44  | 3656058 | 13:25 | 91m56s |
| 100 | 3656623 | 13:25 | 91m46s |
| 101 | 3656235 | 13:25 | 91m51s |
| 102 | 3668159 | 13:37 | 82m07s |

`pool41` has none of these processes; all 6 are on `pool42`. The live queue
(`/home/dant123/Projects/sim/research/queue/pool.queue`) is empty (already dispatched). No `null_s*.json`
artifact exists anywhere in this worktree, on the lane branch, or (checked via the process list) has been
written yet by these processes. **Per this harvest task's own instruction, no verdict is given for v2 as a
whole.** Next action: `bash tools/pool_sync.sh` on a later pass to retrieve the six `null_s*.json` files once
they land, then run the runner's own `--score` entrypoint (command above) — no re-run needed.

## The 7 landed gates, read directly from `main_s{42,43,44,100,101,102}.json`

| seed | S1 bacc (intact) | S3a: bacc under learned-edge lesion | S3b: decided-only bacc under lesion | S4: loser/winner rise | S5: Spearman(drive,rate) | bacc under competition lesion | decided-bacc under competition lesion |
|---|---|---|---|---|---|---|---|
| 42  | 0.7876 | 0.3316 | 0.4960 | 0.2827 | 0.9825 | 0.7799 | 0.8463 |
| 43  | 0.8244 | 0.1365 | 0.4468 | 0.3160 | 0.9753 | 0.7920 | 0.8678 |
| 44  | 0.7566 | 0.2427 | 0.4962 | 0.2594 | 0.9803 | 0.7403 | 0.7981 |
| 100 | 0.7665 | 0.1024 | 0.3879 | 0.3028 | 0.9826 | 0.7383 | 0.8072 |
| 101 | 0.8133 | 0.1184 | 0.5664 | 0.2657 | 0.9731 | 0.7966 | 0.8341 |
| 102 | 0.7939 | 0.1619 | 0.5254 | 0.3232 | 0.9745 | 0.7839 | 0.8610 |
| **mean** | **0.7904** | **0.1822** | **0.4865** | min **0.2594** | min **0.9731** | **0.7718** | **0.8358** |
| gate | mean≥0.75, min≥0.70 | mean≤0.55, drop≥0.20 | ≤0.60 | every seed ≥0.25 | every seed ≥0.50 | (not gated) | (not gated) |
| result | **PASS** (min 0.7566) | **PASS** (drop 0.6081) | **PASS** | **PASS** (narrowly, 0.2594) | **PASS** | see below | see below |

S1, S3a, S3b, S5 pass cleanly on independent re-computation from the raw per-seed fields. S4 clears its
threshold narrowly (min rise 0.2594 against a 0.25 bar, calibrated on a dev rise of 0.313) — every seed
passes, but with little margin.

## Applying the re-review's corrections (all 5, honored)

**1. The competition does NOT carry the decision — credit the learned synapses only.** This is the central
correction and it is directly visible in the table above: zeroing the FSI cross-inhibition (the
"competition lesion" columns) leaves intact-vs-lesioned balanced accuracy **almost unchanged**
(mean 0.7904 → 0.7718, Δ = **−0.0186**), and accuracy on the words the circuit still decides on actually
**increases** under the lesion (0.8358 vs. 0.7904 intact). If the reciprocal inhibition were deciding the
category, removing it should collapse accuracy toward chance; instead it barely moves and, if anything, the
circuit decides *more* words correctly when the competition is gone. **The decision is carried by the
Hebbian-learned feedforward frame→category synapses and a host comparison of the resulting pool rates,
not by the lateral-inhibition WTA.** Any description of this mechanism as "a coupled spiking WTA decides
the category" is an overclaim; the corrected description is: *Hebbian-learned feedforward drive into two
spiking pools, host rate-comparator decision; lateral inhibition present, wired, and measurably effective on
the loser pool's own rate (S4), but not decision-bearing.*

**2. S4 is reclassified accordingly.** S4 (loser/winner ratio rise under the competition lesion) shows only
that the inhibitory synapse exists and does something to the *loser's* rate — it says nothing about whether
that competition decides anything, and per point 1 it demonstrably does not. S4 is reported here as a wiring
/ integrity check (the inhibition is real and has a measurable effect), not as evidence that competition is
the decision mechanism. The gate's evidence for the actual decision mechanism is S1 (accuracy), S3a/S3b
(learned-edge lesion), and S5 (drive-to-rate correlation) — all of which point to the feedforward synapses.

**3. S3b's UNDEFINED-as-pass bug did not trigger in this data, but remains unfixed on the lane branch.** The
scorer (`_lexicon_spiking_referent_derisk.py:313`) computes `(mean(dd) <= threshold) if dd else True` —
if the lesioned circuit decides *zero* words of one class on a given seed, `dd` is empty and the check passes
by default, silently, even though `tools.lab.undefined_if_empty` is imported and never used. Checking
directly: every one of the 6 seeds has a genuine, non-empty `decided_bacc_learned_edge` value (range
0.388–0.566, table above) — none hit the empty-`dd` path, so S3b's PASS here is a real result, not the bug.
Per this harvest's instruction ("S3b-on-UNDEFINED is a fail"), this is recorded as: **the bug is real and
must be fixed on the lane branch before its scorer is trusted on any other data, but it did not corrupt this
particular 6-seed reading.**

**4. v2 is less accurate than v1 — an honest regression, not a hidden one.** v1's host label-spreading scorer
reached roughly 0.93 mean balanced accuracy (though v1 itself was already banked NO-GO, failing gate G7 at
0.7778 against a 0.80 bar). v2's spiking, Hebbian-learned replacement reaches **0.7904 mean** — a ~14-point
drop in raw accuracy for removing the host label-spreading shortcut. This is the expected and declared cost of
replacing a host computation with a brain-based one (per `feedback_brain_based_only_standard`), not a defect
to paper over.

**5. Capability-level residual, read directly from the `capability` field:** on "the wolf watches the owl",
the intact circuit extracts exactly `[wolf, owl]` on 6/6 seeds (`held_refs`, `held_n_referents=2`) — S8 passes
cleanly. Under the learned-edge lesion, the organ loses "owl" entirely on 5 of 6 seeds (`held_refs=['wolf']`,
`held_n_referents=None`). On **seed 101 specifically**, the lesioned circuit does not just fail silently — it
recovers `['wolf', 'watches']`, i.e. it admits the verb "watches" as a second referent, a genuine wrong
answer rather than an abstain. This is a real, if minor (1/6 seeds), failure mode of the lesioned pathway
worth tracking if this organ is revisited.

## Bottom line

Seven of eight v2 evidence gates pass on independent re-computation from the committed per-seed artifacts,
**once the mechanism is correctly described** (feedforward-Hebbian decision, host comparator read-out,
non-decision-bearing lateral inhibition — not "a coupled spiking WTA decides"). The eighth (S2, the
permutation null) is still running on the pool and has not landed; per `docs/TERMS.md`, this is reported as
**PARTIAL / INCOMPLETE**, never GO, until S2 lands and is scored. v2 is honestly less accurate than the v1
host shortcut it replaces (0.79 vs. ~0.93), and one lesion-arm failure mode (a verb admitted as a referent on
1/6 seeds) is recorded as a residual, not silently dropped.
