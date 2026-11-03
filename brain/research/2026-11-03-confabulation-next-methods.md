# Confabulation: what the night of 2026-11-03 showed, and the next methods (ranked)

Plan, not a result. Capability: Metacognition (`brain/PREREQUISITES.md`: PARTIAL; informative = abstains on never-told
facts) and Consolidation (sleep). Record check: `tools/before_you_build.sh` reported a STALE corpus (local main two
commits behind published main; nothing relevant in those two commits), so the RAG search below
(`tools/rag/rag_search.py ... --corpus all`) and direct reads of the branch writeups were used instead.

## 1. The evidence (six seeds unless said; numbers read from the branch writeups)

| # | finding | source (branch:path) |
|---|---|---|
| a | Sleep consolidates told facts into the live brain: hippocampus off after sleep 5.5/12 strict vs 0.17 with consolidated synapses off, 0 shuffled. Cost: false recall on 4 never-told questions 2.0 (no sleep) -> 3.0 (sleep, intact) -> 3.67 (hippocampus off), no better than shuffled sleep (3.17). The no-sleep brain already confabulates 2/4 (hippocampus reinstates a similar episode). | `research/live-brain-12facts-1103:brain/experiments/results/2026-11-03-live-brain-12facts.md` |
| b | Answer confidence is not a feeling-of-knowing: name-token log-prob AUC 0.47 (correct vs confabulated), answer mean log-prob 0.63-0.65, entropy 0.58-0.62. A confabulated name is said as fluently as a recalled one. | `research/false-recall-1103:brain/experiments/results/2026-11-03-false-recall-feeling-of-knowing/README.md` |
| c | Cue familiarity separates told from untold in-sample (hippocampal AUC 0.88, cortical cue response 0.79) but does not transfer: hippocampal familiarity = layer-0 word overlap ("What colour is my cat?" passes); cortical cue response reacts to the trained "my X"/"Do you remember" frame (passes 18-23/24 fresh never-told). Meaning key S_all: DEV AUROC 0.82 -> fresh 0.70; at the 85%-keep operating point no better than S0 (J 0.31 vs 0.42). S_all fixes WHICH memory, not WHETHER to recall (reworded told vs untold AUROC 0.62-0.65). | `research/abstain-in-brain-1103:.../2026-11-03-abstain-in-brain.md`; `research/abstain-v2-1103:brain/experiments/results/2026-11-03-abstain-v2.md`; `research/meaning-selection-fix-1103:.../2026-11-03-meaning-selection-realpath.md` |
| d | The gate itself works (learned-name false recall 0 of 91 gated v1, 0 of 50 gated v2), but a gated reply is not "I don't know": the pretrained 0.1B cortex mostly invents a non-learned answer ("My grandmother's name is Joanna."). One taught lesson, reinstated below the bar, gives first-person don't-know in 27/50 gated replies (13/50 without), with recitation of the lesson (11/50) and the teacher's vocative ("I don't know, Nova") because the lesson was stored as a User turn. "Recollection" as answer-token gain under reinstatement is at chance (AUROC 0.44 DEV, 0.48 fresh): a wrong episode makes the cortex just as specific. Replaying "I don't know" pairs in sleep makes a memorised default string (told correct 21 -> 3, refuses 38/48 general). | same two abstain writeups |
| e | Use does not make facts reliable at 0.1B: 12 reachable facts, k=10 tellings (~43 replays) reach 4/4 on 1/6 seeds; use vs equal replay on k=10 +0.0..+2 (4 up, 2 ties), paid for by k=1 facts; false recall ~3/4 in every sleep arm. | `research/high-use-ceiling-1103:brain/experiments/results/2026-11-03-high-use-ceiling/README.md` |
| f | Bigger cortex, same gap: 0.4B (lr 1e-3, 480, kl 3) 4.0/12 after sleep (six seeds), 1.5B (lr 5e-4) 7.0/12 (three seeds: 3, 7, 11); in-context ceiling 11/12 for both; false recall 2.0 and 1.7 of 4. | `research/cortex-size-1103b:brain/experiments/results/2026-11-03-cortex-size/2026-11-03-cortex-size.md` |

**One diagnosis for all six.** Retrieval in this brain is pattern completion with no check of the result against the
question: the hippocampus returns the nearest episode for any cue, and sleep teaches the cortex a schema ("my X ->
one of the learned names") faster than it teaches the individual bindings (a, e, f). Nothing compares what came back
with what was asked (c, d: answer-token gain cannot, because a wrong episode is just as fluent; b: fluency is not
knowledge). When retrieval is blocked, nothing in the cortex links "no memory" to a first-person answer (d).
This is the textbook profile of confabulation: memory content is intact enough, monitoring of it is missing.

**Ruled out as methods (do not retry in this form):** answer confidence (b); lexical or frame-driven cue familiarity
with bars fit on one question template (c); answer-token gain as recollection (d); replaying an "I don't know" string
in sleep (d); more replay of the same sentences (e); fixed lr/steps tuning alone (f: a ceiling gap at every size).

## 2. The biology that fits

- **Confabulation = failed post-retrieval monitoring, not failed storage.** Spontaneous confabulation follows
  ventromedial/orbitofrontal damage, usually with amnesia, and is a failure to suppress memories that do not belong to
  the present (Schnider 2003, Nat Rev Neurosci 4:662; Schnider & Ptak 1999). The strategic-retrieval account: the
  frontal lobes set the search and then check the output; confabulators skip the check, and the vmPFC "feeling of
  rightness" accepts whatever fits the schema (Moscovitch & Melo 1997, Neuropsychologia 35:1017; Gilboa et al. 2006,
  Brain 129:1399; Hebscher & Gilboa 2016, Neuropsychologia 90:46). Our consolidated cortex has exactly a schema with no
  check (a, e).
- **Source / reality monitoring.** Memories are attributed to a source by judging their features (who said it, how
  it was experienced) at retrieval (Johnson & Raye 1981; Johnson, Hashtroudi & Lindsay 1993, Psychol Bull 114:3). Our
  lesson is replayed in the teacher's voice (d): a source error. Old track: corollary-discharge authorship tag 6-seed GO
  (`research/findings/2026-08-01-agency-authorship-tag-corollary-discharge-source-monitor-6seed-GO.md`); conjunctive tag
  NO-GO because recall drives the whole shared pattern
  (`research/findings/2026-08-07-source-monitor-conjunctive-tag-NO-GO-...md`).
- **Familiarity vs recollection.** Familiarity (perirhinal) is fast, cue-based and fooled by overlap; recollection
  (hippocampal) retrieves specific content (Yonelinas 2002, J Mem Lang 46:441; Brown & Aggleton 2001, Nat Rev Neurosci
  2:51; Eichenbaum, Yonelinas & Ranganath 2007). The fast feeling-of-knowing is driven by cue familiarity, not by the
  answer (Reder & Ritter 1992; Koriat 1993, Psychol Rev 100:609) and is fooled by cue-word overlap: finding (c) is the
  human result, not a bug. Metamemory = a monitor over the object level that is learned from outcomes (Nelson & Narens
  1990). Old track: abstention is a separate familiarity signal
  (`research/findings/2026-05-22-...abstention-is-a-separate-familiarity-signal-not-a-basin-property.md`); plastic
  three-factor metacognition monitor 6-seed GO (`research/findings/2026-08-18-self-organized-metacog-monitor-GO.md`).
- **CA1 match/mismatch.** CA1 compares CA3's completed pattern with the direct entorhinal input; mismatch signals
  novelty and blocks the completed pattern from driving output (Hasselmo & Schnell 1994; Lisman & Grace 2005, Neuron
  46:703; Kumaran & Maguire 2007, J Neurosci 27:8517; Duncan et al. 2012, Hippocampus 22:389). Catalog: O'Keefe & Nadel
  misplace system and EC-III -> CA1 comparator (`~/Projects/sim-catalog/references/feature-catalog.md`, lines ~1061, 1129-1135).
- **Consolidation, schemas, interference.** A slow cortex learning many items from replay overwrites and blends them
  unless replay is interleaved with what it already knows (McClelland, McNaughton & O'Reilly 1995; McClelland 2020,
  Proc R Soc B; `research/biology/cls-interleaved-consolidation.md`). Schema-consistent items consolidate fast (Tse et
  al. 2007, Science 316:76; van Kesteren et al. 2012 SLIMM), which is also how schema-driven intrusions arise. Retrieval
  practice consolidates better than restudy (Roediger & Karpicke 2006; Antony et al. 2017, TICS 21:573); generative
  replay of the learner's own prior knowledge protects it (van de Ven, Siegelmann & Tolias 2020, Nat Commun 11:4069).

## 3. Next methods, ranked

Ranking = (how much of a-f it explains) x (cheapness of the decisive test) x (whether 1.5B leaves the problem in
place). All tests: six seeds 42/43/44/100/101/102, pre-registered before any eval run, bars/variants fit on DEV only,
the abstain-v2 fresh v2 sets (24 fresh_never, 12 fresh_told, 8 general) unless said, plus a fresh v3 set written at
pre-registration (the v2 set has been seen once). `abstain` stays default OFF; any wiring is a declared scaffold.

### M1. CA1-style comparator: verify the retrieved episode against the question (retrieve, then check)
- **Biology.** CA1 match/mismatch plus strategic-retrieval monitoring: recollection is accepted only if the retrieved
  content answers THIS cue; mismatch suppresses output (Kumaran & Maguire 2007; Gilboa et al. 2006).
- **Replaces.** F (cue familiarity: lexical/frame-driven, fresh AUROC 0.70) and R (answer-token gain: chance) as the
  gate signal. The gate itself (no reinstatement, consolidated synapses off) is kept: it already removes 100% of
  learned-name intrusions in what it gates.
- **Mechanism (brain path, `Brain.knows`).** Reinstate the top S_all episode, then score two things inside the
  cortex, no host text: (i) cue completion: per-token log-prob gain of the QUESTION's content tokens from the
  reinstated state (does this memory predict what is asked, attribute included: "colour" after "my cat is called
  Pixel" should not gain); (ii) answer agreement: the answer the cortex gives from the reinstated state vs the answer
  it gives from consolidated synapses alone (hippocampus off); accept only if both name the same learned token. (ii) is
  the CA1 comparison of two routes (trisynaptic vs direct) and needs no labels.
- **Cheapest decisive test.** Signal only, no new sleep: reuse the abstain-v2 harness and one slept brain per seed;
  compute (i), (ii) and their AND on DEV (72) and fresh. Pre-registered: fresh told-vs-never AUROC >= 0.85 for the
  chosen read-out on 6/6 seeds, and the untold-attribute subset (8 items) gated >= 75%. Only if it passes, rerun
  abstain-v2 FR+L criteria C1-C3 with M1 as the signal.
- **Falsified if** fresh AUROC < 0.75 or not above S_all (0.70) by >= 0.08 on >= 5/6 seeds; or (ii) agrees on
  never-told questions as often as on told ones (both routes share the same "my X -> name" schema, which (a) suggests
  is a real risk: then the two routes are not independent and (ii) is dead).
- **Scale.** Likely to help: the check is a reading-comprehension judgement (does "my cat is called Pixel" answer
  "what colour is my cat"), and the in-context ceiling rises from 6/12 (0.1B) to 11/12 (0.4B/1.5B). Run the DEV signal
  at 0.1B and 1.5B both (signal-only is cheap on GPU); a 0.1B failure with a 1.5B pass is a scale verdict, not a
  method one (`brain/PREREQUISITES.md`).

### M2. A lived, first-person "I don't know" exchange with a source tag (source monitoring)
- **Biology.** Source/reality monitoring: a memory carries who produced it; the brain's own past answers are
  retrieved as its own (Johnson et al. 1993; old-track authorship tag GO).
- **Replaces.** The lesson stored as a teacher User turn (27/50 don't-know, recitation 11/50, "Nova" vocative) and the
  sleep-replayed string (memorised default).
- **Mechanism.** The teacher asks one unknown question; the brain's reply is guided once to "I don't know, you haven't
  told me that." and the exchange is stored with the reply as an Assistant (self) turn; the gate reinstates that
  episode. Episodes get a speaker tag (self/other) stored with the key, so reinstatement replays the brain's own words
  in its own voice. Declared scaffold: the one guided reply (teacher), not a host reply string.
- **Cheapest decisive test.** No sleep needed (a gated reply depends only on the stored episode and the pretrained
  cortex, abstain-v2 Amendment 1): force the gate on the 36 DEV never-told questions per seed, arms W3-as-User (old) vs
  lived-self exchange vs no lesson. Pre-registered: first-person don't-know (v2 `dont_know` instrument, unchanged) >= 80%
  of gated replies on 6/6 seeds; recitation <= 5%; vocative "Nova" addressed to the user <= 5%; general questions
  answered when forced through the ungated path unchanged. Then on fresh v3.
- **Falsified if** don't-know <= 54% (the User-turn lesson) or recitation does not fall: then the cortex replays any
  stored episode verbatim and the link must be learned (M5), not stored.
- **Scale.** Probably helps a lot and costs minutes: at 1.5B the pretrained cortex may already say "I don't know" when
  gated without any lesson. Run the no-lesson forced-gate arm at 1.5B FIRST (GPU, no sleep, one evening); if 1.5B
  gated replies are >= 80% don't-know, M2 is unnecessary at the target size and drops off the list.

### M3. Retrieval practice in sleep, with the hippocampal episode as the teacher (testing effect + CLS)
- **Biology.** Retrieval is a fast route to consolidation (Antony et al. 2017); CLS: the hippocampus teaches the
  cortex by replay, and learning is driven by the cortex's own error on the replayed item, interleaved with old
  knowledge (McClelland et al. 1995).
- **Replaces.** Sleep as restudy of the told sentence, weighted by use (e: +1 of 4 at k=10; f: 4.0 and 7.0 of 11).
- **Mechanism.** In each replay, the cortex (hippocampus off) first answers the episode's own question-form cue
  (sampled by the cortex itself from the episode, not a host template); the loss is applied on the answer span against
  the episode's content, weighted by the cortex's error (large when it said a wrong learned name: the intrusion is
  corrected where it happens). KL guard unchanged.
- **Cheapest decisive test.** `brain/experiments/sleep_probe.py` at 0.4B on GPU (six seeds exist there, ~15-25 min per
  seed): arms sleep_true (current) vs sleep_retrieval, same steps and episodes. Pre-registered: recall +2/12 or more on
  >= 5/6 seeds and mean false recall not higher; shuffled control stays ~0.
- **Falsified if** recall gain < +1 mean or false recall rises: then the gap is capacity (rank 8 shared by 12 facts),
  and the next test is adapter rank/one-adapter-per-night, not the replay content.
- **Scale.** Matters more at scale: the 0.4B/1.5B gap to the ceiling is 7 and 4 facts; at 0.1B half the facts are
  unreachable, so do not run M3 at 0.1B.

### M4. Interleave the cortex's own near-miss answers in sleep (CLS interference guard against the schema)
- **Biology.** CLS interleaving and generative replay: the slow cortex keeps old mappings if they are replayed with
  the new (van de Ven et al. 2020; `research/biology/cls-interleaved-consolidation.md`); schema-driven intrusion is
  what un-interleaved schema learning produces (Tse et al. 2007; Hebscher & Gilboa 2016).
- **Replaces.** The KL guard on ordinary text only, which keeps language but lets "my X" questions about UNtold
  topics drift to learned names (a: hippocampus off false recall 3.67/4, equal to shuffled sleep).
- **Mechanism.** Before sleep, the cortex itself generates near-miss cues by recombining stored episodes (sampled, no
  host grammar) and its pre-sleep reply distribution on them is recorded; during sleep the KL guard is extended to those
  cues. No "I don't know" target is ever trained (that was the failed string replay), only "do not change what you
  would have said about things you were not told".
- **Cheapest decisive test.** sleep_probe 0.1B CPU or 0.4B GPU, arms sleep_true vs sleep_true+near-miss KL, hippocampus
  off. Pre-registered: false recall on the 4 never-told + 24 fresh_never <= 1/2 of sleep_true on 6/6 seeds, told recall
  >= 85% of sleep_true.
- **Falsified if** false recall does not halve, or recall drops > 15% (the guard and the binding fight in the same
  rank-8 adapter: an expected capacity failure at 0.1B, retest at 0.4B before calling the method).
- **Scale.** Scale alone does not remove the schema intrusion (1.5B false recall 1.7/4, 0.4B 2.0/4), so M4 stays
  relevant; risk: generated near-miss cues at 0.1B may be poor (check their quality on 20 samples first).

### M5. A learned metamemory monitor, trained on the brain's own sleep self-tests
- **Biology.** Metamemory is a monitor learned from outcomes (Nelson & Narens 1990); the old track's plastic three-
  factor ACC monitor reached 6/6 GO (`research/findings/2026-08-18-self-organized-metacog-monitor-GO.md`).
- **Replaces.** Hand-fit thresholds on 36 one-template questions (c: bars do not transfer, dev->test fire 3-5x).
- **Mechanism.** During sleep the brain quizzes itself: for stored episodes and for M4's near-miss cues, it records
  the read-outs (F, M1's i and ii, cue response) and whether the retrieved answer matched the episode (internal
  comparison, no teacher label); a small monitor (three-factor rule) learns the mapping. Its output drives the same gate.
- **Cheapest decisive test.** Offline on the abstain-v2/M1 signal dumps: train on sleep self-tests only, evaluate on
  fresh v3. Pre-registered: fresh AUROC >= 0.85 on 6/6 seeds and >= the best single read-out + 0.05.
- **Falsified if** it is not better than the best single read-out (then the features, not the read-out, are missing).
- **Scale.** Depends on M1's features; build only after M1 says which read-outs carry signal. Ranked last for that reason.

## 4. Order of work (cheapest decisive first)

1. M2 at 1.5B, no-lesson forced-gate arm (GPU, no sleep, ~1 h): decides whether M2 is needed at the target size.
2. M1 signal-only on DEV at 0.1B and 1.5B (one slept brain per seed; 1.5B six seeds needs the cortex-size runs on
   seeds 100-102 first). Gate wiring only after the AUROC criterion passes.
3. M2 at 0.1B if step 1 says it is needed (CPU, no sleep).
4. M3 and M4 as sleep_probe arms at 0.4B on GPU (they share a harness; one run set with four arms: true, +M3, +M4,
   +M3+M4), six seeds.
5. M5 only after M1.

Every result goes through `land-result`; a pass is a lead, not a GO, until it holds in the live `Brain` on a fresh
question set with a lesion that changes the reply. Never claim the brain "knows that it does not know" as an
experience; the gate output is a functional read-out.
