# Fable design review of the source-monitor port (11-27)

**Verdict: DO-NOT-RUN.** The outcome is predictable from the record and from the design's own gradient path, and neither likely outcome changes what we do next. Run the forward-only ceiling measurement below instead (about 2-3 h, no sleeps), and build a learned tag only if it passes.

## 1. Is it the right mechanism? No.

- **The wall is not night-made.** The voice-split read shows the raw cortex already answers "What is my cat called?" with "My cat is called Fluffy". Raw user voice is 62/336 against 68/336 for hippocampus-off slept (+0.018, CI [-0.038, 0.078]). A lever that acts through sleep has about 2 points to remove, and GO needs -0.10.
- **The GO bar sits at the in-context ceiling.** A0 is 0.235, so GO needs about 0.135 or lower. The ctx arm, with the telling literally in context, scores 0.132. A one-vector tag would have to match full context.
- **What `c` is trained to do points the wrong way.** This follows from the objective, not from a run.
  - Keep KL on tagged bank tokens pulls the slept cortex back to pretrained wherever the tag is on. At test the tag sits on `Assistant:` and fed-back tokens, so an engaged tag pushes the reply toward the raw cortex, which has knows 0 and the "My X is called" frame on name items.
  - CE on the tagged identity self-statements, if they are replayed, teaches first token " My" / " I'm" after the tagged label. The onset measure is log p(You/Your) - log p(My), so this is the anti-GO direction.
  - So if it engages, expect knows to fall and onset to fall. If it does not, expect NO-ENGAGEMENT: the keep KL is already near zero, so its gradient to `c` is weaker than the killed tag's, which reached norm 0.14-0.26.
- **The contrast is not self versus other.** Bank texts share no content with tellings, and tag-on coincides with "KL-held" while tag-off coincides with "CE-learned". The tag marks the training objective and the content class, not provenance of the same content. In biology the efference copy contrasts the same content, self-produced versus heard, and binds source to the episode at encoding. It does not perform the my-to-your shift, as the builder concedes.
- **It does not meet the reopen criterion.** HANDOFF already killed the voice arc on 11-27 and set the reopen condition as "both speakers' turns in replay", after test2. This port meets neither part.
- **The fetch side is already answered.** M2 speaker contrast cut intact echo 71 -> 48 without raising recall, so a residual direction can move voice at reinstatement. The open question is the weights path, which this design cannot reach.

## 2. Does the answer change what we do next? No.

- Every non-GO branch in prereg section 5 ends at "close the class, go to routing", which is where HANDOFF already is.
- GO is about 10-15% by the builder's estimate and lower by the argument above. Even then it would rest on 3 DEV seeds where A0 minus A0snap re-sleep noise was about 40% of the last screen's shift and s320 carried the pooled number.
- It costs about 11 h on pool40, the only 18 GB node, which is running the 0.4B local-sleep sweep for the 12-31 de-risk. It also cannot finish inside the 11-28 timebox unless started this afternoon.
- The seven code-review findings are real, but fixing them does not rescue the design. Finding 7 (tag trained by CE on own text) is the anti-GO gradient above.

## 3. Best alternative: oracle source-direction ceiling

This is a measurement, like routing-ceiling: forward passes only, nothing trained. It asks whether any one-vector residual signal on the brain's own-production positions can move told-fact voice in the slept weights without losing the fact. It bounds the whole tag class in hours.

- **Data.** Banked n14 adapters (A0snap), seeds 319/320/321, the same 67 told questions, hippocampus off, fresh state per question.
- **Two reads per question, both the cortex's own.**
  - H = `User: q\n\nAssistant:`.
  - C = the voice-split ctx construction (telling in the preceding turn).
  - d_L(q) = residual after block L at the label position, C minus H.
- **Oracle vector.** v_L = mean of d_L over the other items, leaving out the item under test, fitted per seed. Primary L = n_layers // 2 (the builder's layer); L = 6 and 18 are report-only.

| arm | what is applied |
|---|---|
| H | nothing; must reproduce the voice-split hippocampus-off B rows, else UNDEFINED |
| C | reference ceiling |
| ORACLE | v_L added on the builder's footprint (`Assistant:` label plus fed-back tokens; reuse `Provenance.mask` and `tag_mask_fn`) |
| SHAM | norm-matched random vector, same footprint |
| ALL | v_L on every position (no-contrast control) |
| WRONG | v built from C-reads whose preceding turn is an unrelated telling ("something was said", no source relevance) |

The lesion is the H arm itself, since no weights change.

- **Measure.** Onset, generated-reply P(uv|knows), knows and refusals, using `voice_tag` / `voice_split` code. Item-clustered bootstrap, 2000 draws. Report pixel, marguerite and volvo separately, and add a blind 40-row hand-read.
- **Pass rule (CLASS-ALIVE), all of:**
  - onset ORACLE minus H, CI lower bound > 0;
  - ORACLE minus SHAM, ORACLE minus ALL and ORACLE minus WRONG, each CI lower bound > 0;
  - P(uv|knows) at or below H minus 0.10;
  - knows at least 0.9 of H;
  - refusals up by less than 10% of rows;
  - direction holds on at least 2 of 3 seeds, and on pixel and marguerite individually.
- **CLASS-CLOSED** if ORACLE fails the onset or voice clause. An oracle direction is the upper bound on a learned one, so no residual tag is built at 1.5B.
- **MIXED** if voice moves but knows falls: the fact and the frame are entangled in the weights. Write it up and stop.
- **Cost.** About six batteries per seed at 90-130 s each. I did not measure memory; with no backward pass it should fit the 15 GB nodes, so check that first and run the three seeds in parallel on pool41/42 around the six-seed job. Needs one pre-data sonnet review. $0.

**If ALIVE:** the learned-tag question becomes well-posed, with a known target direction, norm and layer. Build it after test2 with real two-speaker replay, per the HANDOFF criterion. The candidate not yet killed is own turns stored only when the user's next turn confirms them, a world signal rather than a host judgment. Unfiltered own-turn storage is what killed `lived`. Measure on unpractised items so it does not train on the scripted probe.

**If CLOSED:** the voice arc stays killed, test2 runs with the defect documented, and effort goes to routing and the 12-31 local-sleep de-risk.

## 4. What to do with the build

- Keep `brain/source_monitor.py` and the `tag_mask_fn` hook unregistered; the ceiling reuses the mask and footprint.
- Mark the prereg DRAFT as SUPERSEDED-BY-CEILING and record in HANDOFF: "11th lever not run; ceiling measurement replaces it."
- Fix findings 1-6 only if a learned-tag screen is ever scheduled after ALIVE.

Files read:
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-27-source-monitor-port-prereg-DRAFT.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/source_monitor.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/source_monitor_screen.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/brain.py` (keep bank, K-P-U mask, `_grow_self`)
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-26-{source-tag,voice-split,voice-ceiling}-read.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-24-recall-voice-dev-read-1.5b.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-25-routing-ceiling-read.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/HANDOFF.md`
