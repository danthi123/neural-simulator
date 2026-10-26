---
type: finding
status: live
date: 2026-10-26
lane: prove who owns the computation (Stage-2 exit C2, inference; re-attempt due 11-21)
mechanism: possessive-chain route hop reads (research/runners/compositional_chain_route.py) over OneBrainComposer.kb
seeds: [7, 1]
artifacts:
  - research/findings/raw/_stage2_diag/c2_kb_scramble/kb_scramble_summary_dev7_dev1.json
  - research/runners/stage2_exit_gate.py
  - tests/test_stage2_kb_scramble.py
  - research/findings/2026-10-26-stage2-fail-diagnoses.md
---

# C2 host ownership by intervention: the conclusion follows the Python list, not the bound vectors (dev seed 7)

Status of the evidence: DIAGNOSTIC. Dev seeds only (7 and 1). The measurements are read from the committed summary
`research/findings/raw/_stage2_diag/c2_kb_scramble/kb_scramble_summary_dev7_dev1.json` (written by the runner's `--kb-summary`, with its
provenance sidecar). It aggregates six raw shards (three arms on each seed) that stay on the workstation outside the repo, uncommitted;
their sha256 values are in the summary. Shards were run at branch revision 8491e0df1 with provenance off. The committed evidence will
be a later pool run at a pinned revision (the two new arms are informational and `pool_job_lines` emits them for the next gate run).
No gate seed was run.

## Question

The 2026-10-26 diagnosis (`research/findings/2026-10-26-stage2-fail-diagnoses.md`) read the C2 conclusion ("what does the heron's prey
eat" -> gnat) off code: both hops of the possessive-chain route scan a Python list (`OneBrainComposer.kb`, `_distinct_patients`), and
the spiking `query_patient` is only a fallback the production composer never reaches. The supporting evidence was lesion invariance
(the `qp_lesion` arm leaves answer, derived flag and derived_from unchanged on 7/7 seeds) plus code reading. That shows the spiking
read is not needed. It does not show that the list decides. This test intervenes on the list.

## Prediction (registered in the committed oracle and test code before the runs; this prose was written after)

- Under `kb_scramble`, `derived_svo` and `derived_from` of `p_prey_eat` follow the scrambled list, because the host list owns the
  conclusion: `[frog, eat, wren]` from `[[heron, hunt, frog], [frog, eat, wren]]`, against `[frog, eat, gnat]` for the unscrambled brain.
- Under `kb_scramble_spiking_control` (same scramble, the route forced onto its own spiking fallback) they follow the bound vectors:
  `[frog, eat, gnat]`.
- Direct recalls through the gate (`c_heron_hunt`, `c_frog_eat`) read the bound store and stay unchanged; supporting text built from
  `.kb` by other readers may mention the scrambled facts.

Registration: predictions 1 and 2 are the oracle's `predicted_from_list` and `predicted_from_bound`, and prediction 3 is the
bound-store assertion in `tests/test_stage2_kb_scramble.py` (`c_frog_eat` stays out of `changed_vs_intact`). Both were committed at
8491e0df1 (2026-10-26 14:10) before the runs. The wording above was written after the runs.

Outcome: predictions 1 and 2 held. Prediction 3 was FALSIFIED for `c_frog_eat` and `p_yesno`: both became flagged guesses under the
scrambled list. It held for `c_heron_hunt` and `c_dog_chase`, which are unchanged. See Result.

## Method

Two informational arms of the `infer` group (never counted in any verdict; every pre-existing arm keeps its turn table, env, patch
and kind, pinned by a golden hash in `tests/test_stage2_kb_scramble.py`). The four teach turns run as in `intact_a`. Before the first
probe the worker rewrites `.kb` on the live session's buffer composer (`Pool1BoundOneBrainComposer` behind the `TieredFactStore`, the
object the route receives):

- Scope: the told facts except the hop-1 bridge `heron hunt frog`, which is kept so the scrambled list still resolves to a different
  two-hop conclusion. In kb order that is `frog eat gnat`, `crow chase wren`, `wren chase moth`.
- Derangement: the smallest cyclic shift of their patient strings with no fixed point and no self-referential fact. The patients stay
  distinct and the multiset is preserved, so no (agent, action) becomes multi-valued. Result: `frog eat wren`, `crow chase moth`,
  `wren chase gnat`.
- Only `.kb` is written (a fresh (fact, handle) tuple per changed entry). The bound spiking vectors, the sequencer drives and the
  LTM are not touched. Built-ins (`dog chase cat`, `cat eat fish`, ...) are not in scope.
- Engagement is read back from `.kb` after the write, not taken from the plan: the list must equal the plan, every scoped fact must
  differ, the patients must be distinct, and the conclusion-relevant hop-2 fact must differ from the bound one. An independent
  list-chain oracle (not the route's code) gives both predictions from the before and after lists. If engagement fails the arm
  reports "not engaged" and withholds the follow read.
- `kb_scramble` records every `_distinct_patients` call the route makes (the list read and its result) and every
  `OneBrainComposer.query_patient` call (the spiking read and its result), per turn. Behaviour-preserving wrappers in the arm's own
  worker process; no production file is edited.
- `kb_scramble_spiking_control` makes `_distinct_patients` report that nothing is scannable (return None), which is the route's own
  documented fallback condition. Each hop then becomes a plain `composer.query_patient`. This needed no production edit, only the
  in-process wrapper. A side effect, stated: the multi-valued-hop check is also off in this arm.

## Result, dev seed 7 (numpy CPU, tiny-demo +LTM, corpus present, peak RSS about 6.4 GB per arm)

Engagement, identical in both arms: 9 facts in `.kb`, 3 scoped, 3 changed, read-back equals plan, derangement true, patients distinct,
conclusion fact changed. `.kb` before and after (changed entries marked):

    heron hunt frog          heron hunt frog
    frog eat gnat     ->     frog eat wren     *
    crow chase wren   ->     crow chase moth   *
    wren chase moth   ->     wren chase gnat   *

Probe turn `p_prey_eat` ("what does the heron's prey eat"), fields `derived`, `derived_svo`, `derived_from`:

| arm | derived | derived_svo | derived_from | follows |
|---|---|---|---|---|
| intact_a | true | frog eat gnat | heron hunt frog ; frog eat gnat | (unscrambled) |
| kb_scramble | true | frog eat wren | heron hunt frog ; frog eat wren | the scrambled list |
| kb_scramble_spiking_control | true | frog eat gnat | heron hunt frog ; frog eat gnat | the bound vectors |

Predictions 1 and 2 held on both arms. `abstained` is false on all three. Prediction 3 did not hold for two direct recalls (below).

The route's own reads on `p_prey_eat`:

- `kb_scramble`: `_distinct_patients(heron, hunt)` returned `[frog]` and `_distinct_patients(frog, eat)` returned `[wren]`. The only
  spiking `query_patient` calls on that turn were `(wren, chase) -> moth` and `(moth, chase) -> None`: the supporting chain
  extension ran on the host-chosen intermediate `wren`, and it never read `(frog, eat)`.
- `kb_scramble_spiking_control`: both `_distinct_patients` calls were forced to None; the spiking reads were `(heron, hunt) -> frog`,
  `(frog, eat) -> gnat` (twice, route hop 2 and the supporting chain), `(gnat, eat) -> None`.

Dev seed 1 replicates every row of the above (summary file, `seeds["1"]`): both arms engaged (9 facts, 3 changed), `kb_scramble`
derived `frog eat wren` from `heron hunt frog ; frog eat wren` with route list reads `[frog]` then `[wren]` and spiking reads
`(wren, chase) -> moth`, `(moth, chase) -> None`; the control derived `frog eat gnat` with both list reads forced to None and spiking
reads `(heron, hunt) -> frog`, `(frog, eat) -> gnat` twice, `(gnat, eat) -> None`. `c_frog_eat` and `p_yesno` became flagged guesses
on seed 1 too; `p_prey_eat`, `c_frog_eat` and `p_yesno` are the three turns that differ from `intact_a` on both seeds and both arms.

All turns on seed 7, answer fields summarised (every `abstained` is as shown; "guess" = a reply with no `recalled_svo` carrying the
authorship-monitor marker and the "a guess from what I've learned" suffix):

| turn | intact_a | kb_scramble | kb_scramble_spiking_control |
|---|---|---|---|
| t_heron_hunt, t_frog_eat, t_crow_chase, t_wren_chase | recalled | same as intact | same as intact |
| p_prey_eat | derived frog-eat-gnat | derived frog-eat-wren | derived frog-eat-gnat |
| c_heron_hunt | recalled heron-hunt-frog | same | same |
| c_frog_eat | recalled frog-eat-gnat | guess "perhaps the frog eats the gnat" | guess, same text |
| c_prey_chase | abstained | abstained | abstained |
| p_yesno | recalled crow-chase-wren | guess "perhaps the crow chases the wren" | guess, same text |
| c_dog_chase | recalled dog-chase-cat | same | same |

Two things beyond predictions 1 and 2:

1. The answer TEXT of `p_prey_eat` in both scrambled arms carries sentences read from the deranged `.kb` by other readers ("the frog
   eats the wren the wren chases the gnat the crow chases the moth"), as the task brief anticipated. In the control the structured
   conclusion fields still follow the bound vectors while the text contains the list's sentences.
2. Prediction 3 was falsified: direct recall of a scrambled pair changed framing in both arms. `c_frog_eat` and `p_yesno` lost
   `recalled_svo` and became flagged guesses, although the spiking recall still returned `gnat` for `(frog, eat)` (both arms, `spiking_reads_by_turn.c_frog_eat` in the summary). The content `gnat` came
   from the learned category-inheritance fallback (`flagged_guesses.c_frog_eat.category_inherit_patient`), not from the strict recall. Mechanism
   not isolated here. A candidate consistent with the data: the host router picks a fact from `chat.stored_facts` (rebuilt from
   `.kb` by `ChatBrain._refresh_facts`, `brain_chat_tui.py:985-1012`), then verifies it against the spiking recall
   (`_gate_router_combine`, `brain_chat_tui.py:1063-1082`); with a deranged list the pick (`wren`) differs from the spiking answer
   (`gnat`), the verify fails, and the strict recall abstains. `c_heron_hunt` (bridge kept) and `c_dog_chase` (built-in) are
   unchanged, as that candidate predicts.

## What it means for the host-decision share of C2

On the C2 probe turn the identity of the conclusion is decided by the host list. Deranging the list while leaving the bound spiking
vectors untouched moves `derived_svo` and `derived_from` to exactly the conclusion the independent oracle computes from the list,
with the spiking store unchanged and still holding the unscrambled facts. The control shows the reverse also holds: when the route's
reads are forced onto spiking `query_patient`, the same bound vectors yield the unscrambled conclusion. So in the production-default
path the bound vectors do not decide the answer. The list is decisive when it is present, and it wins when list and vectors
disagree. The spiking read is not needed (the `qp_lesion` arm leaves the answer
unchanged on 7/7 seeds), and the forced spiking fallback also gives the correct conclusion.

Consequences, stated with their limits:

- The intact-arm C2 conclusion is host-owned. The C2 criterion itself FAILED 0/6 at the 10-26 gate, because the permuted teach was
  refused by the D4 comprehension gate before it was stored (`research/findings/2026-10-26-stage2-fail-diagnoses.md`). By this intervention the host-decision share of the C2 conclusion is 1 of 1 probe
  turns, on each of dev seeds 7 and 1. The spiking substrate's contribution to this reply is the content of the bound vectors that
  the list was filled from at teach time, and the strict-recall content of direct recalls.
- The list's authority also reaches direct recall: a list that disagrees with the vectors turns a strict recall into a flagged
  guess (second observation above). So the same host list is plausibly a verification gate over the spiking recall. This candidate
  is not isolated.
- The spiking route is a viable owner: with the hop reads forced onto it, the bound vectors produce the correct conclusion on this
  seed. That is the evidence that moving the hop reads onto `query_patient` would not lose the intact pass. The multi-valued-hop
  abstain (a deterministic distinct-count over `.kb` today) would need its own spiking replacement, as the diagnosis already lists.

## Limits

- One probe turn, one question shape, two dev seeds. The diagnosis found the infer group identical
  across seven seeds in every non-text field, so more dev seeds are expected to replicate the same deterministic measurement, not to
  add independent evidence.
- The scope excludes the hop-1 bridge by design; a derangement that breaks the bridge would make the chain abstain, which is a weaker
  signature than a specific new conclusion.
- The control is a forced fallback via an in-process wrapper, not a production path. It shows what the bound vectors say, not that
  production would ever take that route.
- The oracle scans the buffer list only. The told words (heron, frog, gnat, wren, crow, moth) are absent from the LTM, so this
  matches the route here; it would not on LTM-resident words.
- The prior diagnostic run `s7_infer__permuted_gateoff.json` ran without `data/` in its worktree (`corpus_present` false). These runs
  linked the repo's `data/` into the worktree (`corpus_present` true), so they are not comparable on corpus-learned organs.

## Next

- Pool run of the two arms at a pinned revision on the six gate seeds when the next gate run is emitted (informational, 12 more
  shards; `pool_job_lines` count 198 -> 210).
- Move the hop reads onto the spiking `query_patient` with a spiking multi-valued-hop abstain, behind a default-OFF flag, then re-run
  `kb_scramble` and expect the conclusion to follow the bound vectors in the production path.
- Isolate the direct-recall framing change (which reader consumes the deranged list) before any claim about it.
