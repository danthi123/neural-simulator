---
type: finding
status: live
date: 2026-10-04
lane: integration-first (stage exits on the calendar)
mechanism: Stage-1 (INFANT) exit gate on the integrated production brain, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/findings/raw/_stage1_exit/gate/aggregate.json
---

# Stage-1 (INFANT) exit gate: result (2026-10-04)

Prereg: research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md (gate revision c238ff325a38e40a68b1f9751822630ff729377b). Scored only by `python -m research.runners.stage1_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_exit/gate/aggregate.json.

**Stage-1 exit verdict: FAIL** (primary criteria: C1 FAIL, C2 PASS, C3 FAIL, C4 FAIL)

> **⚠ Claude correction (2026-10-04, verify-go): C4 is UNDEFINED (instrument), not "not exposure-driven".** The
> permuted-map arm deranged **0 of 24** held-out words on every seed (`patch.n_words 0`), and the `dr2_off` arm reads
> the same r as intact. The DR-2 learned map is stored under `research/findings/raw/`, which pool provisioning
> excluded, so on the pool nodes `_get_learned_valence()` returned `{}`. The affect organ then silently read the
> **raw Warriner norm**, the value prereg section 4.4 says the brain never reads at run time. So r = 0.94 is the
> norm read through the ladder, and both C4 controls were inert. Re-scored with an engagement guard (a permuted map
> that does not derange all 24 words makes the seed UNDEFINED):
> `research/findings/raw/_stage1_exit/gate/aggregate-rescore-c4guard-2026-10-04.json`, which reads **C4 INCOMPLETE
> (0 of 6 seeds defined)**, so the registered stage verdict becomes **UNDEFINED (INCOMPLETE)**. C1 and C3 fail on
> defined data either way, so the stage did not pass. The committed `aggregate.json` is left as scored. Fixes:
> `tools/pool_provision.sh` now ships the map as a runtime input, and the scorer carries the guard. The C4 arms
> re-run at the same pin on re-provisioned nodes (RULING `s1-c4-void`).
>
> **⚠ Claude correction 2 (2026-10-04): C3 is UNDEFINED (instrument), not "not load-bearing".** `lesion_recall` set
> `BRAIN_SUBSTRATE_RECALL_LESION`, which cuts `ChatBrain._substrate_recall`. The production default never calls that
> method: `brain_reply` installs the GNW bus gate, and a routable recall goes through `gnw_bus_shadow.gate_via_bus` ->
> `composer.query_patient`. So the lesion never engaged, and the reply was unchanged on every seed for that reason.
> The scorer had assumed every env-flag lesion engaged. Amendment A1 (`docs/drafts/stage1-exit/AMENDMENT-A1.md`)
> replaces the arm with a patch on the read the bus makes, which records its engagement. It also adds the rule that a
> no-effect env lesion with no engagement record is UNDEFINED. With both corrections the attempt reads C1 FAIL, C2
> PASS, C3 and C4 INCOMPLETE. The stage did not pass, because C1 fails on defined data.

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| cat_disc | C1 | primary | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| cat_told | C1 | diagnostic | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| word | C2 | primary | PASS | 6/6 | lesion_recruit 6/6 | lesion_recruit 6/6 | 6/6 | - |
| lived | C3 | primary | FAIL | 6/6 | lesion_recall 0/6, lesion_episodic 6/6 | lesion_recall 6/6, lesion_episodic 6/6 | 6/6 | lesion_recall: reply-level effect on 0 of 6 seeds (not load-bearing) |
| val_heldout | C4 | primary | FAIL | 6/6 | lesion_affect 6/6 | lesion_affect 6/6 | 0/6 | permuted exposure tracked on 0 of 6 seeds (not exposure-driven) |
| val_novel | C4 | diagnostic | FAIL | 0/6 | lesion_affect 0/6 | lesion_affect 6/6 | 0/6 | capability absent on 6 of 6 seeds |

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.938 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.938; defined
- seed 43: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 44: r = 0.951 (n = 24, sign agreement 0.88); lesion r = None; permuted-map r = 0.951; defined
- seed 100: r = 0.947 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.947; defined
- seed 101: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 102: r = 0.944 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.944; defined


## Per-criterion reading (fixed sentences from HANDOFF section 8)

C1 (`cat_disc`): FAIL, capability absent on 6 of 6 seeds. The integrated brain does not show this Stage-1 ability at production defaults; the gap named in prereg section 8 applies.

C2 (`word`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6.

C3 (`lived`): FAIL, not load-bearing: the reply was correct but the lesion_recall arm changed it on only 0 of 6 seeds.

C4 (`val_heldout`): FAIL, not exposure-driven: the permuted exposure was tracked on only 0 of 6 seeds.

`cat_told` (diagnostic, not counted): FAIL, correct 0/6.

`val_novel` (diagnostic, not counted): FAIL, correct 0/6.

## Claude review (2026-10-04, verify-go): the C2 PASS stands, narrowed

Two independent skeptics were told to refute the C2 PASS: one on gate-cheat and instrument trust, one on control
integrity and reproducibility. Neither refuted it. Both found caveats, which are recorded here and travel with every
board or ledger use of the word PASS.

- **Re-score.** The registered scorer, re-run from the committed raw shards, gives a `groups.word` block equal to
  `aggregate.json`.
- **Instrument.** "Correct" is read from the structured `recalled_svo` / `abstained` fields, not by substring
  matching on reply text. The counted replies are genuine: `p_bear_chase` "... the bear chases the otter",
  `p_otter_eat` "... the otter eats the clam". Both moat probes and the pre-exposure probe say "I don't know about
  that", and dog->cat survives.
- **Determinism and null.** `intact_a` and `intact_b` are whole-JSON identical on every turn of every seed, and
  `lesion_recruit` equals its `.rep0`. The lesion effect is therefore not run-to-run noise.
- **Scorer weaker than the prereg prose (caveat 1).** The aggregate's lesion `effect` is `any(differs)` over the two
  counted probes, and it never scores dog->cat survival in the lesion arm. The prereg asks that BOTH answers are
  removed and the built-in fact survives. Checked by hand on all six seeds: the only labels that differ from intact
  are `p_bear_chase` and `p_otter_eat`, both become abstains, and `c_dog_chase` stays `[dog, chase, cat]`. The claim
  holds in the data, but the registered scorer would not have caught a one-answer or a dog-killing lesion. Tighten
  it before the next stage-exit attempt; this result stands as scored.
- **Lesion scope (caveat 2).** `lesion_recruit` replaces `OneBrainComposer._recruit_word` with `return False`
  (21 calls on every seed). The host `kb` still holds the taught fact and the reply still abstains, so the answer
  depends on the cleanup decode and not on the host table. But the lesion removes all three unseen words at once
  through the shared codebook. "Necessary" names the cleanup-slot pool, not one assembly.
- **Seeds (caveat 3).** The six seeds are distinct substrates: the otter-agent margins (rounded from research/findings/raw/_stage1_exit/gate/s*/word__intact_a.json) are 0.643 / 0.578 / 0.716 / <!--derived-->
  0.678 / 0.621 / 0.676. The reply-level outcome is the same on every seed, so "6/6" is one deterministic outcome <!--derived-->
  replicated six times, not six independent draws.
- **Ownership (caveat 4, from research/findings/2026-10-04-stage1-host-share-RESULT.md).** At the same C2 turns,
  every decision seam is HOST-OWNED: k-select, bg-select, organ-c, metacog and ignition. The panel host decision
  share is 1.000 and the neural fraction 0.000 (n 48). The content itself is learned (learned_content_fraction
  1.000). C3's seams read host 0.967 / neural 0.033 (n 30; research/findings/raw/_stage1_host_share/gate/aggregate.json). <!--derived-->

**What C2 PASS shows:** in the integrated production handler, a word heard for the first time this conversation is
recalled in both roles, with the moat intact, and the recruit cleanup-slot pool is necessary for that reply. **What
it does not show:** that neurons made the decision. The SVO acquisition, the first-match selection and the
commitment are host scaffolds, as prereg section 5 predicted. In-session words also never become discourse referents
(anaphora 0/6, reported, not counted).

Ledger wording: `Stage 1 exit attempt 1 (2026-10-04): not passed: C1 FAIL; C3 UNDEFINED (recall lesion inert on the bus
path; re-measured at a new pin, Amendment A1); C4 UNDEFINED (learned map absent on the pool, raw norm read; re-run owed);
C2 word PASS 6/6, reply-level, recruit pool load-bearing, decision host-owned (host share 1.000)`.

## Claude review 2026-10-06 (verify-go): C3 re-measured at P2 PASS, narrowed; C4 re-run FAIL as registered

**C3 (lived-fact recall), Amendment A1, pin 7851dd220, 42 shards
(`research/findings/raw/_stage1_exit/gate_c3rerun/aggregate.json`): PASS 6/6 at reply level, NARROWED.** A skeptic
re-ran the scorer (the `lived` block is JSON-equal to the committed aggregate), checked all 42 provenance sidecars
(git_sha 7851dd220, git_dirty false, PYTHONHASHSEED ok) and diffed every label on every seed. `lesion_recall` (patch
`lesion_lived_recall`, 4 of 30 `query_patient` calls cut) changes only `p_wolf_bite` (apple becomes "I don't know");
`lesion_episodic` changes only `p_wolf_discuss`; the permuted arm names pear; intact_a equals intact_b; both
pre-exposure probes abstain. Caveats that travel with the word PASS:
- The recall lesion is an interface cut. It makes `OneBrainComposer.query_patient` return None for exactly the probed
  cue, so it shows the reply depends on the composer read. It does not show that spiking dynamics carry the recall:
  the read's block selection is the host first-match unless the K-way sequencer is on (off by default), and the
  host-share result reads the C3 seams as host-owned (k-select).
- `lesion_episodic` is an env lesion with no scored engagement record. The shards show `episodic.lesioned` true and
  `in_memory` false in every lesion cell and repeat, so it engaged; the scorer gates on the effect alone.
- The reply-level outcome is one deterministic outcome replicated on six substrates.
- C3 is judged at P2 alone (A1.2); C1, C2 and C4 ran at c238ff325. Write it as "C3 re-measured at P2".

**C4 (valence inheritance), re-run at c238ff325 with the learned map shipped
(`research/findings/raw/_stage1_exit/gate_c4rerun/aggregate.json`): FAIL as registered. The failing clause is a
property of the registered control, not evidence that the reply ignores the map.** The map engaged this time: the
permuted patch deranged 24 of 24 words on every seed, and intact differs from `dr2_off` on every seed. Intact r is
above the 0.55 bar on 6/6 and the ladder lesion makes every read neutral, 6/6, reproduced. The clause "the permuted
map gives r < 0.20 on every seed" holds on 4 of 6 (seeds 44 and 102 exceed it). Two readers recomputed from the
shards:
- the permuted arm's reply follows the permuted map at r 0.93-0.96 on every seed; <!--derived-->
- the seed-44 derangement itself correlates r 0.30 with the norm, so a reply that followed the permuted map exactly
  would still exceed 0.20 there; <!--derived-->
- over random derangements of these 24 learned values, P(r >= 0.20) is about 0.13 per seed, so a perfectly
  map-driven brain passes the six-seed clause only about 44% of the time; <!--derived-->
- the pooled partial r(reply, norm | permuted map) over the 144 permuted reads is about 0.10 (p 0.25): no route to
  the norm beside the map was found, though a leak of 10% or less cannot be excluded at n = 24. <!--derived-->

The verdict stays FAIL (bars are frozen). The aggregate's reason string "not exposure-driven" misdescribes the data.
Further caveats: r rides mostly on the 12/12 positive/negative split (within a sign group, reply-vs-norm r is
0.12-0.51, and 5-8 reads per seed are neutral); `BRAIN_AFFECT_LESION` gates the ladder's `affect_out` read-out, so
it shows the output pathway is necessary, not the ladder's internal dynamics; the per-word value is a host lookup of
the DR-2 learned map. `val_novel` (diagnostic): FAIL 0/6, as the prereg predicted. <!--derived-->

**Next for C4 (no defer).** Attempt 2 needs an amended control, registered before any data, that scores what the
permuted arm is for: the reply following the permuted map (for example r(reply, permuted map) >= 0.55 and r(reply,
norm) no higher than r(permuted map, norm) + 0.20), or a fixed sign-balanced derangement set registered in advance.
It must not choose derangements by their correlation with the norm using what seeds 44 and 102 showed. Report the
within-sign r beside the pooled r. `val_novel` stays open: the learned vocabulary that would carry it is NO-GO on G3
(`research/findings/2026-09-25-affect-learned-vocabulary-eval-NOGO.md`).

**Audit ruling (s1-c4-void):** `docs/drafts/stage1-exit/AUDIT-affect-pool-results.md`, section "Claude ruling
2026-10-06". The only void results are this attempt's `gate/` C4 cells (every val_heldout and val_novel arm),
corrected above. C1 and C2 here stand: their scored fields (`recalled_svo`, `abstained`, the draw) read no valence.

**Ledger wording (replaces the 10-04 line):** `Stage 1 exit attempt 1 (2026-10-04; C3/C4 re-measured 10-05): not
passed. C1 FAIL; C2 word PASS 6/6 (reply level, recruit pool load-bearing, decision host-owned); C3 lived recall PASS
6/6 re-measured at P2 (reply level, composer read load-bearing, selection host-owned); C4 feeling FAIL as registered
(permuted-map clause 4/6; the reply follows the learned map and the control is under-powered by design; an amended
control is owed for attempt 2).`
