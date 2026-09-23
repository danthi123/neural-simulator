---
type: finding
status: live
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
verdict: GO -- all 8 pre-registered v2 evidence gates (S1, S2, S3a, S3b, S4, S5, S7, S8) and all 4 integrity
  smokes pass on independent re-computation, per the runner's own `--score` entrypoint
  (`_lexicon_spiking_referent_derisk.py score()`, run on this branch, output committed as `verdict.json`).
  S2 (the >=1000-seed-label permutation null, the gate this finding's prior version left INCOMPLETE) has now
  landed for all 6 seeds and scores 6/6 seeds at or above the 99th percentile (>= 5/6 required) -- see the new
  "S2" section below. This is a de-risk/runner-level GO on `docs/TERMS.md`'s own definition ("the gate's OWN
  verdict is positive"), NOT a claim that the mechanism is `wired`, `on-by-default`, or `integrated` in
  production -- it is not; that is a separate, unaddressed step.
artifacts:
  - research/findings/raw/_lexicon_spiking_referent/main_s42.json
  - research/findings/raw/_lexicon_spiking_referent/main_s43.json
  - research/findings/raw/_lexicon_spiking_referent/main_s44.json
  - research/findings/raw/_lexicon_spiking_referent/main_s100.json
  - research/findings/raw/_lexicon_spiking_referent/main_s101.json
  - research/findings/raw/_lexicon_spiking_referent/main_s102.json
  - research/findings/raw/_lexicon_spiking_referent/null_s42.json
  - research/findings/raw/_lexicon_spiking_referent/null_s43.json
  - research/findings/raw/_lexicon_spiking_referent/null_s44.json
  - research/findings/raw/_lexicon_spiking_referent/null_s100.json
  - research/findings/raw/_lexicon_spiking_referent/null_s101.json
  - research/findings/raw/_lexicon_spiking_referent/null_s102.json
  - research/findings/raw/_lexicon_spiking_referent/verdict.json
  - (+ matching .prov.json sidecars for every `main_s*.json`/`null_s*.json` above) -- the `main_s*` artifacts
    were copied via `git show` from research/language-lane-next HEAD 87bf35d1a (NOT merged) for the prior
    harvest and are unmodified here; the 6 `null_s*.json` (+ `.prov.json`) were produced by the pre-registered
    pool run (per-file provenance: runner `_lexicon_spiking_referent_derisk.py`, git_sha `b9d0349...`, part
    `null`, `--seeds <s>`, `SIM_BACKEND=numpy`) and are committed here for the first time; `verdict.json` (+
    its own `.prov.json`) is this scoring step's own output.
external: NO-EXTERNAL-NEEDED -- reads already-run, already-committed gate arms; scores them with the
  pre-registered scorer; no new claim beyond what the pre-registration specified.
builds_on:
  - research/runners/_lexicon_spiking_referent_derisk.py (research/language-lane-next HEAD 87bf35d1a, NOT
    merged and NOT copied onto this branch -- see "Branch note" below for why the code stays off this
    data-only branch)
  - research/findings/2026-09-23-lexicon-referent-v1-host-label-spreading-6seed-NO-GO-and-relabel.md (same
    branch)
  - research/findings/2026-09-23-lexicon-referent-v2-spiking-wta-hebbian-frames-main-gates-pass-null-pending.md
    (same branch)
  - re-review of that branch (verdict: fix-required; prior_issues_resolved: false -- 5 issues, honored below,
    unchanged from the prior version of this finding)
  - research/findings/2026-09-23-cpu-lane-harvest-language-lexicon-referent-v2-S2-pending.md (the prior
    INCOMPLETE finding, now `status: superseded`), which read S1/S3a/S3b/S4/S5/S7/S8 from the committed
    `main_s*.json` artifacts before S2 had landed
next_command: "none -- scored. To re-derive: check out research/language-lane-next HEAD 87bf35d1a into a
  scratch worktree (or point PYTHONPATH at it) and run: SIM_BACKEND=numpy python -m
  research.runners._lexicon_spiking_referent_derisk --score
  <path-to-this-branch>/research/findings/raw/_lexicon_spiking_referent -- the runner code is intentionally
  NOT committed on this data-only branch (see 'Branch note')."
---

# Language CPU-lane harvest: v2 S2 null landed and scored -- 6-seed GO on all 8 gates

**Branch note.** The prior version of this finding was harvested on `main` by reading committed
`main_s*.json` artifacts without the runner itself (main does not carry
`research/runners/_lexicon_spiking_referent_derisk.py` or its dependency chain
`lexicon_spiking_frame_category.py` / `d6_multiref_wm_production_organ.py` /
`_lexicon_learned_referent_derisk.py` / `lexicon_learned_referent.py` /
`_comprehension_learned_animacy_cue_derisk.py` -- all five live only on
`research/language-lane-next` HEAD `87bf35d1abf725340cc6c9d19ef7afa502e62010` (the same HEAD the prior
harvest cited), NOT merged into `main`. Running the pre-registered `--score` entrypoint needs that code, so
it was run from a copy of that HEAD's `research/runners/` tree pointed at this branch's
`research/findings/raw/_lexicon_spiking_referent/` data directory. **That runner code is deliberately NOT
committed onto this branch**: `research/runners/d6_multiref_wm_production_organ.py` already exists on `main`
in an OLDER form (missing the `referent_lexicon` injection parameter the v2 de-risk's organ-level arms
need), so pulling in the `language-lane-next` copy would silently carry an unreviewed, unmerged change to a
production-adjacent file (the multi-referent WM organ `webapp/server.py` calls) into `main` as a side effect
of a scoring branch -- additive and default-off (env-var gated) on that lane branch, but a merge decision
this task did not ask for and does not make here. This branch therefore carries only DATA: the 6 new
`null_s*.json` (+ `.prov.json`) artifacts and this finding's own scored `verdict.json` (+ `.prov.json`); the
`main_s*.json` artifacts were already on `main` from the earlier harvest and are untouched. Diffing
`_lexicon_spiking_referent_derisk.py` itself between the git SHA that produced the null artifacts
(`b9d0349404f3aa4e11172b2df668fa2a4797218b`, an earlier commit on the same lane branch, per each
`null_s*.json.prov.json`) and the pinned HEAD used to score (`87bf35d1a`) shows only two additive,
non-scoring fields added to the JSON report (`sim_backend`, `preconditions`) -- the `GATE` thresholds,
`score()` gating logic, and `run_null()` null-generation logic are byte-identical between the two commits, so
scoring the artifacts with the HEAD version of the runner is scoring them with the exact registered gate. No
`sim/` file was touched on this branch; this remains a harvest + scoring step, not a new mechanism or runner
change.

**Artifacts read:** the same six `main_s{42,43,44,100,101,102}.json` as the prior harvest, plus the six
`null_s{42,43,44,100,101,102}.json` that have now landed (produced by the pool run the prior harvest reported
as in-flight), all under `research/findings/raw/_lexicon_spiking_referent/`.

## S2 (>=1000-seed-label permutation null) -- LANDED, 6/6 seeds pass

All six `null_s*.json` files are present, each with `n_perm=1000`, `replicas=41` (40 permutations + the true-
label replica), matching the pre-registration (`_lexicon_spiking_referent_derisk.py` docstring, S2: "at or
above the 99th percentile of >= 1000 seed-label permutations ... on >= 5 of 6 seeds").

| seed | true-label percentile vs. 1000-permutation null | true-replica stable across batches | gate | result |
|---|---|---|---|---|
| 42  | 99.6  | yes | >= 99.0 | PASS |
| 43  | 100.0 | yes | >= 99.0 | PASS |
| 44  | 99.9  | yes | >= 99.0 | PASS |
| 100 | 100.0 | yes | >= 99.0 | PASS |
| 101 | 100.0 | yes | >= 99.0 | PASS |
| 102 | 100.0 | yes | >= 99.0 | PASS |

6 of 6 seeds clear the 99th-percentile bar (gate requires >= 5 of 6) -- **S2 PASSES**. The true-label
replica's balanced accuracy is identical across every permutation batch on every seed
(`true_bacc_all_batches_identical`), the integrity smoke the scorer checks alongside S2 -- **passes on all
6**. The true-label circuit's balanced accuracy sits above essentially the entire null distribution generated
by retraining the same circuit from `W_INIT` on 1000 independently-permuted label sets per seed (25 batches of
40 permutations; each batch retrains `replicas=41` circuits, the 41st being the true-label replica): the learned
FR->category synapses are not fitting noise.

## S1, S3a, S3b, S4, S5 and the competition-lesion readouts, read directly from `main_s{...}.json`

(Unchanged from the prior harvest -- reproduced here for a single self-contained record.)

| seed | S1 bacc (intact) | S3a: bacc under learned-edge lesion | S3a drop (bacc_spiking − bacc_learned_edge) | S3b: decided-only bacc under lesion | S4: loser/winner rise | S5: Spearman(drive,rate) | bacc under competition lesion | decided-bacc under competition lesion |
|---|---|---|---|---|---|---|---|---|
| 42  | 0.7876 | 0.3316 | 0.4561 | 0.4960 | 0.2827 | 0.9825 | 0.7799 | 0.8463 |
| 43  | 0.8244 | 0.1365 | 0.6880 | 0.4468 | 0.3160 | 0.9753 | 0.7920 | 0.8678 |
| 44  | 0.7566 | 0.2427 | 0.5138 | 0.4962 | 0.2594 | 0.9803 | 0.7403 | 0.7981 |
| 100 | 0.7665 | 0.1024 | 0.6641 | 0.3879 | 0.3028 | 0.9826 | 0.7383 | 0.8072 |
| 101 | 0.8133 | 0.1184 | 0.6948 | 0.5664 | 0.2657 | 0.9731 | 0.7966 | 0.8341 |
| 102 | 0.7939 | 0.1619 | 0.6320 | 0.5254 | 0.3232 | 0.9745 | 0.7839 | 0.8610 |
| **mean** | **0.7904** | **0.1822** | **0.6081** | **0.4865** | min **0.2594** | min **0.9731** | **0.7718** | **0.8358** |
| gate | mean≥0.75, min≥0.70 | mean≤0.55, **min drop≥0.20** | (this is the gated column) | ≤0.60 | every seed ≥0.25 | mean ≥0.50 (pre-registered on the mean; min 0.9731 also clears) | (not gated) | (not gated) |
| result | **PASS** (min 0.7566) | **PASS** (lesioned mean 0.1822 ≤ 0.55) | **PASS on the gated quantity: min drop = 0.4561** (seed 42), well above the 0.20 bar — the mean drop of 0.6081 shown above is NOT what S3a gates on and is reported here only as descriptive context | **PASS** | **PASS** (narrowly, 0.2594) | **PASS** | see below | see below |

S1, S3a, S3b, S5 pass cleanly on independent re-computation from the raw per-seed fields; **the S3a pass is
on the pre-registered gated quantity, `S3a_drop_min = min_seed(bacc_spiking − bacc_learned_edge) = 0.4561`
(seed 42), not the mean drop of 0.6081** (the pre-registered scorer
(`_lexicon_spiking_referent_derisk.py:309-310`) gates on the per-seed minimum, and the runner's own `--score`
output, `verdict.json`, confirms `S3a_drop_min: 0.4561 (PASS)`). S4 clears its threshold narrowly (min rise
0.2594 against a 0.25 bar, calibrated on a dev rise of 0.313) — every seed passes, but with little margin.

## S7 (organ-level, reply-level capability probe), read directly from `main_s{...}.json` → `per_seed[0].organ`

(Unchanged from the prior harvest.)

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

S7 passes on every mean-gated quantity, independently recomputed from `organ.{learned_recover_rate,
lesion_recover_rate, fp_in_scope_rate, hand_in_scope_rate}` across all 6 seeds. **Seed 42's own
`fp_in_scope_rate` is 0.3333, above the 0.20 per-arm bar** — the gate is defined as a mean over seeds
(`S7_fp_in_scope_mean` in `verdict.json`, gated on the mean per the pre-registration), so this does not flip
the S7 verdict, but it is a real per-seed exceedance that the mean hides and is recorded here rather than
silently averaged away.

## S8 (capability), read directly from `main_s{...}.json` → `per_seed[0].capability`

6 of 6 seeds satisfy the pre-registered S8 condition (`held_refs == ["wolf","owl"]` AND `held_in_scope` AND
`held_n_referents == 2`, gate: >= 5 of 6) -- **S8 PASSES** (`S8_exercised_seeds: 6, pass: true` in
`verdict.json`). This is reported (not gated) alongside the lesion-arm capability read: under the
learned-edge lesion, 5 of 6 seeds go out-of-scope entirely on the held turn, and seed 101 recovers
`["wolf","watches"]` instead (see correction 5 below).

## Applying the re-review's corrections (all 5, honored, unchanged from the prior harvest)

**1. The competition does NOT carry the decision — credit the learned synapses only.** This is the central
correction and it is directly visible in the table above: zeroing the FSI cross-inhibition (the
"competition lesion" columns) leaves intact-vs-lesioned balanced accuracy **almost unchanged**
(mean 0.7904 → 0.7718, Δ = **−0.0186**). 0.7904 (`bacc_spiking`, intact) and 0.8358
(`decided_bacc_competition`, under the lesion) are not the same quantity — 0.7904 counts every abstention as
wrong (runner docstring, `_lexicon_spiking_referent_derisk.py`), while 0.8358 is scored only over the words
the lesioned circuit actually decided on (decided-only). The artifacts do not record an intact decided-only
balanced accuracy directly; dividing out each seed's own intact abstain rate
(`bacc_spiking / (1 − abstain)`, 3.0–6.75% abstention) gives an **approximate** intact decided-only accuracy
of ≈0.79–0.86 per seed (mean ≈0.828), essentially the same as the lesion's decided-only mean of 0.8358. **The
honest reading is that decided-only accuracy is approximately unchanged under the competition lesion, not
that it "rises" or that the circuit "decides more words correctly" when the competition is removed.** The
quantity that supports the correction, computed like-for-like, is the overall (abstention-counted) balanced
accuracy: mean 0.7904 → 0.7718, Δ = −0.0186, i.e. removing the lateral inhibition costs almost nothing in
overall accuracy. **The decision is carried by the Hebbian-learned feedforward frame→category synapses and a
host comparison of the resulting pool rates, not by the lateral-inhibition WTA.** The corrected description:
*Hebbian-learned feedforward drive into two spiking pools, host rate-comparator decision; lateral inhibition
present, wired, and measurably effective on the loser pool's own rate (S4), but not decision-bearing.*

**2. S4 is reclassified accordingly.** S4 (loser/winner ratio rise under the competition lesion) shows only
that the inhibitory synapse exists and does something to the *loser's* rate — it says nothing about whether
that competition decides anything, and per point 1 it demonstrably does not. **S4 remains one of the eight
pre-registered scorer gates** (the runner's own `score()` computes and gates on it as evidence, and it passes:
`S4_loser_ratio_rise_min: 0.2594, pass: true`) — the reclassification here is about what S4 is evidence *for*,
not whether it counts toward the GO verdict. S4 is evidence that the inhibition is real and wired with a
measurable effect (an integrity/wiring fact), not evidence that competition is the decision mechanism. The
gate's evidence for the actual decision mechanism is S1 (accuracy), S3a/S3b (learned-edge lesion), and S5
(drive-to-rate correlation) — all of which point to the feedforward synapses.

**3. S3b's UNDEFINED-as-pass bug did not trigger in this data, but remains unfixed on the lane branch.** The
scorer (`_lexicon_spiking_referent_derisk.py:313`) computes `(mean(dd) <= threshold) if dd else True` —
if the lesioned circuit decides *zero* words of one class on a given seed, `dd` is empty and the check passes
by default, silently, even though `tools.lab.undefined_if_empty` is imported and never used. Checking
directly: every one of the 6 seeds has a genuine, non-empty `decided_bacc_learned_edge` value (range
0.388–0.566, table above) — none hit the empty-`dd` path, so S3b's PASS here is a real result, not the bug.
Per the standing instruction ("S3b-on-UNDEFINED is a fail"): **the bug is real and must be fixed on the lane
branch before its scorer is trusted on any other data, but it did not corrupt this particular 6-seed reading,
and it is orthogonal to S2 (which has its own, separately-verified true-replica-stability integrity check).**

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

**All 8 pre-registered v2 evidence gates (S1, S2, S3a, S3b, S4, S5, S7, S8) and all 4 integrity smokes (S6
full-rebuild determinism, afferent-zero abstain, hand-table baseline, null true-replica stability) pass** on
independent re-computation by the runner's own pre-registered `--score` entrypoint, run on this branch and
committed as `research/findings/raw/_lexicon_spiking_referent/verdict.json`
(`{"verdict": "GO", "complete_6seed": true, ...}`). Note: `verdict.json`'s `mechanism` field is `score()`'s fixed,
pre-correction label ("coupled spiking WTA ... decides"); it is NOT this finding's description -- the competition
is not decision-bearing (see the mechanism frontmatter and correction 1). This closes the gap the prior version of this finding left
open: S2 (the permutation null) has landed for all 6 seeds and clears the 99th-percentile bar on 6 of 6
(gate: >= 5 of 6). **Per `docs/TERMS.md` ("GO" = the gate's own verdict is positive), this v2 de-risk is a
GO** — once the mechanism is correctly described (feedforward-Hebbian decision, host comparator read-out,
non-decision-bearing lateral inhibition — not "a coupled spiking WTA decides"). v2 is honestly less accurate
than the v1 host shortcut it replaces (0.79 vs. ~0.93), and one lesion-arm failure mode (a verb admitted as a
referent on 1/6 seeds) is recorded as a residual, not silently dropped. **This GO is a runner-level /
de-risk-level result only** — the mechanism (`lexicon_spiking_frame_category.py`, on
`research/language-lane-next`, NOT merged) is not `wired` into `/api/brain-chat`, not `on-by-default`, and no
scaffold has been retired; production integration is a separate, unaddressed step per `docs/TERMS.md`'s
`wired`/`integrated` distinction.
