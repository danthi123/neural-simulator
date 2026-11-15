# Pre-registration: near-miss guard in sleep (M4) against false recall after sleep (1.5B, six seeds, AWS)

Written 2026-11-14 before any run of the test seeds. Capability: Sleep consolidation + Metacognition (false recall).

**Record brief.** After one night at 1.5B, about 7 of 12 never-told questions are answered with a told fact (mixed day,
`results/2026-11-14-sall-contrast-mixed-1.5b.md`: s 6.83, r 6.67; "Where does my aunt live?" -> Oslo on 12/12
runs). The confidence gate found the slept cortex itself confabulates with the hippocampus off (false 9.17 of 12,
`2026-11-13-confidence-gate-1.5b.md`), so the intrusion is learned in the night, not only selected by the
hippocampus. Already tried against false recall (do not retry in this form; `brain/research/2026-11-03-confabulation-
next-methods.md`): answer confidence (AUC 0.47), cue familiarity gates v1/v2 (FAIL, do not transfer), the
comparator M1 (FALSIFIED at 0.1B), the lived don't-know lesson M2 (LEAD for never-told, overrides told facts when the
gate misfires, gate misfire NOT usable at 1.5B), retrieval practice M3 (FALSIFIED for recall at 0.1B/0.4B; false
recall halved as a secondary), confidence gate (NARROWED). Lever count on this defect: 6; the 11-03 research plan
ranked M4 as an untested method of its own (CLS interleaving / generative replay: van de Ven et al. 2020; McClelland
et al. 1995; schema intrusion, Tse et al. 2007), so this is that plan's next step, not a new guess.

**Prerequisites.** Interleaved replay with a KL guard: have. Cue generation by the cortex: have at 1.5B (prototype
below). Risk named up front: the guard and the told bindings share one rank-8 adapter (capacity), and a generated
near-miss about a TOLD topic ("What is my dog's name?" from the cat fact) would hold a told answer at its pre-sleep
value: that cost shows up in told recall (N2), and it is the method's own cost, so it is not filtered.

**Mechanism (`Brain(near_miss=K)`, switch default OFF; `brain/near_miss.py`, `brain/sleep.py` guard term).** Before
each night, for every stored episode the pretrained cortex (no adapters, nothing reinstated) writes its own question
about the episode (OWN frame), then K near-miss rewrites (one greedy, K-1 sampled) in one pre-listed rewrite frame,
and its own greedy reply to each rewrite from the birth state. During sleep every step adds a second KL term (same
weight as the ordinary-text guard, own RNG, so the replay sequence is identical to the guard-off night) holding the
consolidated cortex to the pretrained one on one of those exchanges. No "I don't know" target, no host answer text.
Declared scaffold: the two frames (SCAFFOLDS.md row). Host decision: none.

**Frame choice (before any test seed; seed 7, `near_miss_proto.py`, pool40 CPU).** Frames N1-N3 in
`brain/near_miss.py`. Rule: the frame with the most usable cues over the 12 told facts at K=4 (a question, not the
own question, naming no told key word); ties -> N1. The chosen frame and its counts are added below as Amendment 1
before the AWS batch starts; nothing else may change after that.

**Test hygiene (probe only, not the brain).** A generated cue naming any evaluated never-told topic (horse, uncle,
daughter, mother, fish, aunt, nephew, father, parrot, holiday, teacher, cousin) is dropped
(`mixed_day_probe.names_eval_topic`), so false recall measures transfer to near-misses the night never saw. Every
cue, kept or dropped, is logged in the JSON (`near_miss_log`).

**Method.** `mixed_day_probe.py --arms c,s,r --self-q` (1.5B, `--steps 480 --lr 5e-4 --kl-weight 3`, slept half
counterbalanced as in the 11-14 batches, seeds 42/43/44/100/101/102), twice per seed in one AWS batch (g6.xlarge L4):
`nmoff_s*` (no flag) and `nmon_s*` (`--near-miss 4 --near-miss-frame <chosen>`). Rows re-scored with
`brain.experiments.rescore` before the table (`near_miss_table.py`).

**Criteria (arm s = the shipped 1.5B package: S0 key, gain 0.5; six-seed means; of 12 unless said).**
- N0 instrument: nmoff reproduces the 11-14 key-default/tellmix s and r told and false per seed exactly (else the
  comparison is reported but the verdict is VOID until explained); nmon logs >= 1 kept cue per seed.
- N1 false recall: nmon <= 0.5 x nmoff (the plan's bar) AND nmon < nmoff on >= 5/6 seeds.
- N2 told kept: nmon >= 0.85 x nmoff.
- N3 paraphrase: nmon >= nmoff - 1.
- N4 self (3): nmon >= nmoff - 0.5.
GO = N0-N4: land-result (skeptic, answer-rate lens), then a default flip for 1.5B with a weekly transcript.
LEAD = N0, N2-N4 and false recall lower by >= 2 on >= 5/6 seeds but above half: a second test (K, weight) is earned.
NOT A LEAD otherwise. Reported, not criteria: arm c (hippocampus off) and r, the aunt -> Oslo lure, slept vs new
told, ordinary-text drift is not measured here (named gap).

**Expected.** False recall falls on the slept facts' near-miss topics; the new (day-2) facts are untouched by the
night; told recall dips a little if cues hit told topics. If false recall does not fall at all, the schema is
not local to near-miss cues (it generalises over the "my X" frame), and the next method is guarding the frame itself
(generated "my X" questions with no statement), then M5.

## Amendment 1 (2026-11-14, after the seed-7 prototype, before any test seed)
First prototype (frames as first committed, da411bfc; pool40/41/42 CPU): usable N1 29/48, N2 14/48, N3 28/48, so the
rule picked N1. Not used: every frame wrote the cues in the SECOND person ("Where was your father born?", "What's your
cat's name?"), because the cortex imagines being asked; the evaluated questions and the lure are first-person ("Where
does my aunt live?"), so the guard would hold the wrong exchanges and the test could not inform. Changed, before any
test seed: OWN and N2/N3 put the statement in the speaker's own voice ("I told a friend: ... Later I asked them: "),
N1 asks for "a different person or thing of mine"; "usable" also requires first person (my / I / me). The prototype is
rerun on seed 7 and the same rule (most usable, ties -> N1) picks the frame; Amendment 2 records it. Nothing else changes.

## Amendment 2 (2026-11-14, after the seed-7 reruns, before any test seed): the frame guard
First-person rerun of N1-N3: usable 0, 11, 14 of 48, and the usable cues were mostly told-topic questions ("Where was
I born?", "What is the name of my dog?"); N1's own questions turned to general knowledge ("What is the capital of
France?"). Quality check failed: at 1.5B the pretrained cortex does not rewrite a told statement into a true
near-miss from a prompt. Per "Expected" above, the cue source becomes the frame itself (frame "F",
`near_miss.frame_cues`): the pretrained cortex completes five neutral first-person question onsets (declared
scaffold: "What is my", "Where does my", "Who is my", "What does my", "When is my") as a user turn from the birth
state, 4 samples each (K = 4, so about 20 cues per night, independent of the episodes), and its own replies to them
are held by the second KL term. Seed 7 prototype (pool40): 19 cues ("Who is my father?", "When is my dad's
birthday?", "Where does my money go when I send it?"), 0 about a told topic (the one flag was "account" matching
"son"). Test hygiene and criteria N0-N4 unchanged. Jobs: `--near-miss 4 --near-miss-frame F`.
