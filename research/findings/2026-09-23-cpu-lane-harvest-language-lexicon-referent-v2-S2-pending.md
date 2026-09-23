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
  landed, re-confirmed by `bash tools/pool_sync.sh` on this fix round (0 of 6 null_s*.json files pulled). It
  is actively running: seed 42 on pool42, seeds 43/44/100/101/102 on pool41 (corrected node attribution --
  an earlier version of this finding put all 6 on pool42), started 12:03-13:37 EDT, now 2.7-3.3h CPU time each
  with no ETA available from the runner. No v2 GO/NO-GO verdict is given. The other 7 gates (S1, S3a, S3b, S4,
  S5, S7, S8) HAVE landed and are read here directly from the committed per-seed artifacts, with the
  re-review's corrections applied to how each is characterized.
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
  research/findings/raw/_lexicon_spiking_referent/: on research/language-lane-next (HEAD 87bf35d1a or its
  descendant -- the --score entrypoint and null-scoring path do not exist on main), run:
  .venv/bin/python -m research.runners._lexicon_spiking_referent_derisk --score
  research/findings/raw/_lexicon_spiking_referent"
---

# Language CPU-lane harvest: v2 main gates read, S2 null still in flight

**Artifacts read (copied from `research/language-lane-next`, HEAD `87bf35d1a`, NOT merged here):**
`research/findings/raw/_lexicon_spiking_referent/main_s42.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s43.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s44.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s100.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s101.json`,
`research/findings/raw/_lexicon_spiking_referent/main_s102.json`.

## Exact status of the pending gate (S2) -- STILL NOT LANDED (re-checked this fix round)

`bash tools/pool_sync.sh` (this fix round, 2026-09-23) pulled 1101 files pool-wide but **zero** `null_s*.json`
files under `research/findings/raw/_lexicon_spiking_referent/` — the directory still holds only the six
`main_s*.json` files. Per this task's explicit instruction, **S2 stays INCOMPLETE; no v2 GO/NO-GO is given.**

Re-checking the live processes directly (`ssh pool<N> ps aux`, this fix round) also **corrects an error in the
node attribution** from the earlier version of this section, which claimed "all 6 are on `pool42`, none on
`pool41`":

| seed | PID | node | started | CPU time (this fix round's check) |
|---|---|---|---|---|
| 42  | 3345947 | **pool42** | 12:03 | 198m52s |
| 43  | 3655934 | **pool41** (not pool42, per the correction below) | 13:25 | 169m56s |
| 44  | 3656058 | **pool41** | 13:25 | 170m40s |
| 100 | 3656623 | **pool41** | 13:25 | 171m19s |
| 101 | 3656235 | **pool41** | 13:25 | 170m52s |
| 102 | 3668159 | **pool41** | 13:37 | 161m17s |

Only seed 42 is on `pool42`; the other five (43, 44, 100, 101, 102) are on `pool41` — the same PIDs and start
times the earlier version of this section reported, just attributed to the wrong node (`pool41` was checked
and reported as having none of these processes; it does). This is a correction to this finding's own prior
`ps aux` read, not a new dispatch — no job was re-run, and the correction does not change the INCOMPLETE
verdict. All 6 jobs are still live, none finished, 2.7–3.3 hours of CPU time each and climbing. The live queue
(`/home/dant123/Projects/sim/research/queue/pool.queue`) is empty. **Next action unchanged:** `bash
tools/pool_sync.sh` on a later pass to retrieve the six `null_s*.json` files once they land, then run the
runner's own `--score` entrypoint (command in `next_command` above, on the lane branch) — no re-run needed.

## S1, S3a, S3b, S4, S5 and the competition-lesion readouts, read directly from `main_s{...}.json`

S7 has its own table below the S3a/S4 discussion — it was already landed and already counted in this
finding's "seven of eight" bottom line, but the earlier version of this finding never showed its numbers.

| seed | S1 bacc (intact) | S3a: bacc under learned-edge lesion | S3a drop (bacc_spiking − bacc_learned_edge) | S3b: decided-only bacc under lesion | S4: loser/winner rise | S5: Spearman(drive,rate) | bacc under competition lesion | decided-bacc under competition lesion |
|---|---|---|---|---|---|---|---|---|
| 42  | 0.7876 | 0.3316 | 0.4561 | 0.4960 | 0.2827 | 0.9825 | 0.7799 | 0.8463 |
| 43  | 0.8244 | 0.1365 | 0.6880 | 0.4468 | 0.3160 | 0.9753 | 0.7920 | 0.8678 |
| 44  | 0.7566 | 0.2427 | 0.5138 | 0.4962 | 0.2594 | 0.9803 | 0.7403 | 0.7981 |
| 100 | 0.7665 | 0.1024 | 0.6641 | 0.3879 | 0.3028 | 0.9826 | 0.7383 | 0.8072 |
| 101 | 0.8133 | 0.1184 | 0.6948 | 0.5664 | 0.2657 | 0.9731 | 0.7966 | 0.8341 |
| 102 | 0.7939 | 0.1619 | 0.6320 | 0.5254 | 0.3232 | 0.9745 | 0.7839 | 0.8610 |
| **mean** | **0.7904** | **0.1822** | **0.6081** | **0.4865** | min **0.2594** | min **0.9731** | **0.7718** | **0.8358** |
| gate | mean≥0.75, min≥0.70 | mean≤0.55, **min drop≥0.20** | (this is the gated column) | ≤0.60 | every seed ≥0.25 | every seed ≥0.50 | (not gated) | (not gated) |
| result | **PASS** (min 0.7566) | **PASS** (lesioned mean 0.1822 ≤ 0.55) | **PASS on the gated quantity: min drop = 0.4561** (seed 42), well above the 0.20 bar — the mean drop of 0.6081 shown above is NOT what S3a gates on and is reported here only as descriptive context | **PASS** | **PASS** (narrowly, 0.2594) | **PASS** | see below | see below |

S1, S3a, S3b, S5 pass cleanly on independent re-computation from the raw per-seed fields; **the S3a pass is
on the pre-registered gated quantity, `S3a_drop_min = min_seed(bacc_spiking − bacc_learned_edge) = 0.4561`
(seed 42), not the mean drop of 0.6081** (an earlier version of this table reported the mean against the
gate, which is the wrong quantity — the pre-registered scorer (`_lexicon_spiking_referent_derisk.py:309-310`)
gates on the per-seed minimum). S4 clears its threshold narrowly (min rise 0.2594 against a 0.25 bar,
calibrated on a dev rise of 0.313) — every seed passes, but with little margin.

## S7 (organ-level, reply-level capability probe), read directly from `main_s{...}.json` → `per_seed[0].organ`

| seed | learned recover (both referents) | lesion recover | FP in-scope rate | hand-table baseline in-scope |
|---|---|---|---|---|
| 42  | 0.7500 | 0.0 | **0.3333** | 0.0 |
| 43  | 0.8333 | 0.0 | 0.0 | 0.0 |
| 44  | 0.6667 | 0.0 | 0.0 | 0.0 |
| 100 | 0.6667 | 0.0 | 0.1667 | 0.0 |
| 101 | 0.8333 | 0.0 | 0.0833 | 0.0 |
| 102 | 0.8333 | 0.0 | 0.0 | 0.0 |
| **mean** | **0.7639** | **0.0** | **0.0972** | **0.0** |
| gate | mean ≥ 0.60 | mean ≤ 0.20 | mean ≤ 0.20 | mean ≤ 0.10 |
| result | **PASS** | **PASS** | **PASS** (mean) | **PASS** |

S7 landed and passes on every mean-gated quantity, independently recomputed from `organ.{learned_recover_rate,
lesion_recover_rate, fp_in_scope_rate, hand_in_scope_rate}` across all 6 seeds — this table was previously
missing from the finding even though the bottom line already counted S7 among the "seven of eight" landed-or-pending
gates. **Seed 42's own `fp_in_scope_rate` is 0.3333, above the 0.20 per-arm bar** — the gate is defined as a
mean over seeds, so this does not flip the S7 verdict, but it is a real per-seed exceedance that the mean
hides and is recorded here rather than silently averaged away.

## Applying the re-review's corrections (all 5, honored)

**1. The competition does NOT carry the decision — credit the learned synapses only.** This is the central
correction and it is directly visible in the table above: zeroing the FSI cross-inhibition (the
"competition lesion" columns) leaves intact-vs-lesioned balanced accuracy **almost unchanged**
(mean 0.7904 → 0.7718, Δ = **−0.0186**). **Correction (this comparison, not the −0.0186 headline, was wrong
in an earlier version of this finding):** 0.7904 (`bacc_spiking`, intact) and 0.8358 (`decided_bacc_competition`,
under the lesion) are not the same quantity — 0.7904 counts every abstention as wrong (runner docstring,
`_lexicon_spiking_referent_derisk.py`), while 0.8358 is scored only over the words the lesioned circuit
actually decided on (decided-only). The artifacts do not record an intact decided-only balanced accuracy
directly; dividing out each seed's own intact abstain rate (`bacc_spiking / (1 − abstain)`, 3.0–6.75%
abstention) gives an **approximate** intact decided-only accuracy of ≈0.79–0.86 per seed (mean ≈0.828),
essentially the same as the lesion's decided-only mean of 0.8358. **The honest reading is that decided-only
accuracy is approximately unchanged under the competition lesion, not that it "rises" or that the circuit
"decides more words correctly" when the competition is removed** — those claims compared the wrong quantities
and are withdrawn. The quantity that does support the correction, and is computed like-for-like, is the
overall (abstention-counted) balanced accuracy: mean 0.7904 → 0.7718, Δ = −0.0186, i.e. removing the lateral
inhibition costs almost nothing in overall accuracy. **The decision is carried by the Hebbian-learned
feedforward frame→category synapses and a host comparison of the resulting pool rates, not by the
lateral-inhibition WTA.** Any description of this mechanism as "a coupled spiking WTA decides the category"
is an overclaim; the corrected description is: *Hebbian-learned feedforward drive into two spiking pools, host
rate-comparator decision; lateral inhibition present, wired, and measurably effective on the loser pool's own
rate (S4), but not decision-bearing.*

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
