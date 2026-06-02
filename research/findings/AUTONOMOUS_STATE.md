# AUTONOMOUS CONTINUATION STATE

> Durable cross-session pointer. Any re-trigger (scheduled watchdog, new
> session, post-compaction) reads THIS first and resumes the exact next
> action without re-deriving context. Update every cycle; commit+push
> both remotes. The conversation is NOT the memory — this file + git are.

**Updated:** 2026-06-02
**Mode:** continuous autonomous (24/7; no self-imposed stopping; only an
explicit user stop/pause or a true safety boundary halts work)

## >>> DIRECTION A PREP -- ACTIVE (read THIS first; 2026-05-31) <<<

OWNER CHOSE (A) richer representation learning at scale, "but spend time in preparation ensuring we make the
most of all the time spent on compute." So: PREPARE thoroughly, run cheap-first GATES before the ~100hr.
Design-prep doc: docs/plans/2026-05-31-representation-learning-prep-direction-A.md.

INPUTS DONE (2 background agents): (1) external survey REFRAMES -- the bounded mechanisms (DG/Foldiak/random)
all attack post-hoc readout-transform toward VSA near-ortho; the 54%-wins limit is UPSTREAM representation
learning; 2 untried non-100hr levers (expansion+Hebbian Lindsay-2017; e-prop Bellec-Maass); predictive coding
ruled out (100x costlier). (2) internal map: BPTT ALREADY decisively bounded (char-level Phase 2.3a/2.3b
NEGATIVE, scale makes it WORSE), contrastive runner NEGATIVE, near-ortho floor ~0.48 set by intrinsic per-pair
overlap (FLAT across N, NOT moved by coding on the SAME activity); IF reps needed, BPTT is the wrong tool ->
better bets G.20-scaling or VSA role-binding. THE compute-protecting fact: 16-concept activity is 100%
NN-identifiable though pool-argmax recognition is 81% -> the front-end wall may be a LOSSY-READOUT artifact.

GATE 1 verdict RETRACTED by GATE 2 (finding 2026-06-01-GATE2-overturns-GATE1-...). Gate 1 concluded "28-word
representation limit" but that was CONFOUNDED: the _v17 28-word bridge was ~50 events while the 16-word
control was 200 -- unfair cross-vocab comparison. Gate 2 (controlled training pair, topographic-prior lever)
trained MATCHED 150-event 28-word bridges: baseline topo3.0 clean 16-avg pool-argmax 0.893 (single-shot k=1
0.569); strong topo10.0 0.857 (stronger prior NEUTRAL). vs _v17 50ev 0.643 and 16-word 200ev ~1.000. TWO
corrections: (1) 28-word is NOT a fundamental representation limit -- 150ev clean = 0.893, close to 16w 1.000;
(2) the single-shot ~50% wall is largely NOISE/readout (k=1 0.569 vs clean 0.893) -- temporal integration
recovers it (mirrors the real-substrate boundary). MAJOR compute implication: the premise motivating the
~100hr richer-representation-learning (28-word = hard rep wall) is substantially WRONG; cheap levers (more
training events + temporal-integration readout) carry the front-end far past the single-shot wall WITHOUT the
100hr / BPTT / new rep learning. SWEEP DONE = REFUTES THE 100hr PREMISE (seed 42 trajectory 50/150/300/500 events): clean 28-word recognition
0.643 -> 0.893 -> 0.929 -> 0.929 (RISES to ~0.93, plateaus); concept OVERLAP between-cos 0.606 -> 0.564 ->
0.495 -> 0.389 (DECREASES monotonically -- more training makes the LEARNED codes genuinely less-overlapping,
the cheap acquisition lever); single-shot 0.395->0.714 + NN 0.402->0.893 (readout helps too). So the 28-word
"wall" was UNDERTRAINING + single-shot noise; the cheap fix (more training of the EXISTING v16 arch + temporal-
integration/NN readout) reaches ~0.93 AND reduces overlap -- NO 100hr, NO BPTT, NO new mechanism. The premise
for Direction A's big run is REFUTED at 28 words. (NOTE: I STALLED here -- launched the sweep with nohup &
WITHOUT a harness-tracked waiter, so I missed completion; owner rightly annoyed. FIX: every long job gets a
run_in_background waiter, no exceptions.) MULTI-SEED CONFIRMED (seeds 42/43/44 @ 300ev): clean 28-word recognition
0.929/0.964/0.964 (mean ~0.95), overlap ~0.50 -- refutation is ROBUST, not seed-luck. SCALE TEST RUNNING
(WITH waiter b8d4xb0u6, _scale64.log): 64-word LEARNED vocab (v3: 4 motor + 20 noun + 20 verb + 20 adj),
2048 lang, sparsity 0.01, 300ev -- does "training reduces overlap" hold at 64 words (overlap stays ~0.5,
recognition ~0.9 -> cheap lever holds, no 100hr) or does the overlap floor reappear (-> a real rep-learning
target at some N)? Compare to 28-word (overlap 0.50, recognition 0.95). v3 runner + generalized capture
(--vocab-mod/--n-lang) committed. RECOVERY CONFIRMED (2026-06-02): the honest flat-distinct fix WORKS. Distinct-seed retrain (bridgeB verbs@43,
bridgeC adj@44; bridgeA nouns@42 existing) -> 192 DISTINCT FLAT codes (between-cos max 0.604) -> STRUCTURED
SVO composition (agent=noun/action=verb/patient=adj) full-3-slot QA = 1.000/1.000/1.000 (seeds 42/43/44, incl.
seed 42 where the hierarchical shortcut hit 0.000). Removing the 2nd binding level removes the nesting wall.
PASS on the REALISTIC structured distribution (the one that exposed the overclaim) + multi-seed + distinct
codes. Finding 2026-06-02-flat-distinct-RESOLVES-robust-cross-bridge-biological-composition.md. So robust
cross-bridge biological composition over structured SVO (noun/verb/adj) at 192 concepts is VALIDATED the
honest way.

INCREMENTAL TRAINING IMPLEMENTED + VERIFIED (2026-06-02, owner asked "can extended runs be incremental,
accumulating across breaks?"): YES. The GPU fragmentation is WITHIN-process (a fresh shorter process is fast;
breaks AVOID it -- it cannot "ruin" a run, only slow one marathon process). But incremental training was NOT
wired up. Added --resume-from to concept_pool_sparse_distributed (load_checkpoint the trained weights instead
of the from-scratch prior, then CONTINUE the train loop -> events ACCUMULATE). VERIFIED: A(100ev)=69%,
B(resume A +100ev=200 incremental)=75%, REF(200 one-go)=62.5% -- B>=A PROVES accumulation across the
save/break/resume boundary; B~REF within single-seed quantisation noise (16 concepts=+-6.25%/concept).
Finding 2026-06-02-incremental-resumable-training-IMPLEMENTED.md; committed both remotes (d6e0632 + this).
So extended runs (incl. full-320) can be CHUNKED across breaks, accumulating into a checkpoint -- the
fragmentation deferral reason is GONE.

TIMING MISCONCEPTION CORRECTED (2026-06-02): the recurring "fragmentation / ~17 min per bridge" narrative was
a MISDIAGNOSIS. The real per-bridge cost at the flat-distinct config (64 concepts x 400 events x 8192
lang_input, sparsity 0.007) is ~73 MIN -- bridgeD@45 took 73 min on a verified-CLEAN GPU (no python, matmul
0.164s healthy). 25,600 events x ~0.17s/event = ~73 min is just the config cost; the "17 min" expectation was
wrong (likely a smaller config). The chain b7s1jtt1g TRUNCATED because its 90-min timeout cannot fit TWO
73-min bridges (it killed bridgeE mid-train; exit 0 was a tee/no-pipefail artifact, NOT success). LESSON: size
bounded timeouts to the REAL per-job cost, and run one expensive bridge per process (the incremental "fresh
process per chunk" lesson) rather than chaining two under one timeout.

TIMING CONFIRMED: bridgeE@46 took 75 min (14:22->15:37), consistent with bridgeD@45's 73 min -> 73-75 min IS
the genuine per-bridge cost at this config (64 concepts x 400 events x 8192 lang); the doc's "~17 min" was
simply wrong for the 64-concept tier (NOT fragmentation). All 5 distinct-seed bridges now SAVED: bridgeA
noun@42 (existing), B verb@43, C adj@44, D spatial@45, E functional@46.

320 STRUCTURED COMPOSITION RESOLVES (job bh4o2reg3, 2026-06-02): structured SVO full-3-slot QA =
1.000/1.000/1.000 (seeds 42/43/44), cleanup over ALL 320 (D+E = 128 distractors), 320 codes DISTINCT
(between-cos mean 0.045, max 0.604 < 0.9, VOID-duplicate guard not triggered). SCRUTINY PASSED (5 checks):
distinct codes; STRUCTURED not random fillers (the distribution that exposed the hierarchical 0.000 overclaim);
cleanup over all 320 incl. 128 distractors (harder than 192, per-fact chance ~(1/320)^3 -> 60/60 not luck);
the harness CAN fail (hierarchical scored 0.000 seed 42 on the SAME harness); multi-seed not lucky-seed. So the
honest flat-distinct path extends robust cross-bridge biological composition from 192 to the FULL 320 "age-5"
target. Finding 2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md WRITTEN (any-bank + demo
sections pending bfuhhbthk).

>>> FULL-320 BIOLOGICAL COMPOSITION MILESTONE = COMPLETE + FULLY PROPAGATED (2026-06-02). All three results
RESOLVE + scrutinised: structured SVO 1.000/1.000/1.000 (3-seed, job bh4o2reg3); any-bank (any concept any
role, strictly harder) 0.992 mean 6-SEED 42-47 (job bc2q2z6qa, min 0.950, 119/120 facts, single miss localised
to spatial bridge); conversational demo 6/6 role+relational + absent-cue ABSTAINS (anti-artifact). Honest flat-
distinct path (5 distinct-seed bridges, single binding level) resolves the hierarchical-320 nesting wall (which
scored 0.000 at seed 42 on the SAME structured test). PROPAGATED: finding 2026-06-02-full-320-flat-distinct-
composition-RESOLVES-multiseed.md; capability_status pillar n=112 + as_of 2026-06-02 (schema tests 6/6);
CLAUDE.md milestone note; committed both remotes. Scope honesty: codes GIVEN by sparse encoding (cheating-
audit), composition GENUINE + robust at 320; per-bridge retrain ~73-75 min (the "17 min" doc was wrong).

CONVERSATION-ON-320 LAYER = essentially COMPLETE (2026-06-02, finding 2026-06-02-conversation-on-the-full-320-
substrate.md). On the validated 320 biological substrate: (1) KB CAPACITY holds to >= 15 facts PERFECT multi-
seed (relational/role/abstention all 1.000, 3x the prior ~5-fact overlapping-code cap; mechanism = distinct
codes 0.045 vs 0.70); (2) NEGATION + yes/no + who-QA (K=4) RESOLVES multi-seed: yes/no [0.9,0.9,0.8] mean 0.867
(boundary metric = the extra K=4 polarity unbind), who-QA 1.000, abstention 1.000. So the 320 substrate behaves
like a small queryable, honestly-abstaining knowledge base in spiking, multi-seed. All committed both remotes.

>>> EXACT NEXT CONCRETE ACTION (in flight): KB CEILING probe -- job b8azh6357 (_flatdist320_kb_ceiling.log,
40-min bound) tests N=30 (does the KB hold past 15?). WHEN b8azh6357 COMPLETES -> update the KB section of the
conversation finding with the ceiling result (>=30 if perfect, else the break point); add capability_status
pillar n=113 (conversation-on-320: KB capacity + negation/QA) + bump as_of; commit both remotes; then SURFACE
the consolidated conversational-on-320 picture (composition + KB + negation/QA all validated multi-seed) +
the two owner-strategic forks to the owner for steer (this is a natural checkpoint; the forks genuinely need
owner buy-in). Probe loads the cached _flatdist320_codes.npz (no retrain). Owner-strategic forks to SURFACE
(do NOT launch unilaterally): (a) 640-concept tier (10 bridges; near-ortho boundary findings say more scale at
same overlap won't help recognition, but COMPOSITION scales fine -- cheap to test if owner wants); (b) LEARNED
codes at scale (the cheating-audit frontier: 320 codes are given by sparse encoding, not learned end-to-end --
the deep brain-analogue frontier, BPTT-bounded, expensive). wiki-sync the milestone. moat/0.80 bar frozen;
honest propagation both remotes; bounded waiters; GPU/CuPy real.

HONEST CORRECTION (2026-06-02): the "full-320 biological composition RESOLVES 1.000/0.98 multi-seed" is
RETRACTED -- it was a RANDOM-FILLER artifact. On STRUCTURED facts (noun/verb/adjective, the realistic case)
the hierarchical-320 composition full-3-slot QA = 0.000/0.950/1.000 at seeds 42/43/44 -- CATASTROPHIC at seed
42 (where random fillers scored 1.000). The integration demo (seed 42, structured) caught it (0/6). MECHANISM:
the hierarchical bridge-role bind stacks a 2nd binding level (composition-role x bridge-role x code) -> the
documented NESTING/multi-hop SNR wall; at some seeds the role vectors interfere catastrophically for
structured (bridge-systematic) fillers. Recognition over the 320 distinct codes is fine; the COMPOSITION is
not robust. WHAT STANDS: within-bridge 64-concept composition (FLAT codes, no nesting) ROBUST multi-seed
(1.000/0.900/0.950). HONEST PATH to robust full-320: DISTINCT FLAT codes (retrain the bridges with DISTINCT
seeds 42-46 -> single-level composition like the within-bridge 64), NOT the hierarchical shortcut. Findings:
2026-06-02-hierarchical-320-NESTING-WALL-honest-retraction.md + banner on the milestone doc. LESSON: validate
the REALISTIC input distribution (structured facts), not random samples -- a clean abstention control on the
wrong distribution still misled. EXACT NEXT: cheap-first the flat-distinct path -- retrain a few bridges with
DISTINCT seeds (bridgeB@43, bridgeC@44; bridgeA@42 exists) -> test cross-bridge SVO composition over the
distinct flat codes; if robust (not seed-variable) the flat-distinct path works -> scale to 5 bridges. Do NOT
re-claim full-320 composition until structured-fact multi-seed is robust. moat/0.80 bar; bounded waiters;
honest propagation both remotes.

ACTIVE DIRECTION (2026-06-02, owner: "continue autonomously to lift the limits + get closer to goals"):
make the BIOLOGICAL spiking composition robust AT SCALE so conversation is built on the brain-analogue
mechanism, NOT the static engram-tag retrieval (which the owner's goal explicitly says not to build on). The
320-concept demo (g20_multibridge --sparse) WORKS but is retrieval/ranking. The biological spiking bind is
the genuine brain-analogue win but is a real-substrate BOUNDARY at scale (160-tier 0.80, temporal-integration
lifted to 0.917). LIFT IT: running the spiking relational memory + wh-QA on a 320-TIER bridge (64 concepts,
sparsity 0.007) with temporal integration (stim=300) -- _bio_compose_320tier.log. RESULT = RESOLVES,
MULTI-SEED CLEAN PASS (finding 2026-06-02-biological-composition-ROBUST-at-64-concept-scale-multiseed.md):
REAL wh-QA 1.000/0.900/0.950 (seeds 42/43/44, mean 0.95), abstention 1.000 EVERY seed -> the brain-analogue
composition is ROBUST at 64 concepts/bridge on the real 320-tier substrate (NOT a boundary like the 160-tier
0.80). Sparser 320-tier codes (between 0.350) compose CLEANER. The boundary is LIFTED via temporal
integration. Integration demo compose_bio_conversation_320_demo.py running (bytmt3a2e). EXACT NEXT: (a) the
demo transcript = biological relational conversation at 64 concepts (tangible); (b) push toward FULL 320
biological composition = the 5 bridges share seed-42 patterns (duplicate global codes), so cross-bridge
biological bind needs DISTINCT per-bridge codes (documented per-bridge-distinct-seed recovery path -- a
retrain, slow); meanwhile cross-bridge uses tags. (c) layer more conversational abilities (negation, learned
parser, generation) onto the 64-concept biological substrate. Honest scope: codes still given by sparse
encoding (cheating-audit); the COMPOSITION on top is genuine + now robust at 64. moat/0.80-bar frozen; honest propagation both remotes; bounded waiters; cheap-first.

SCALE-UP STATUS (2026-06-02): 64-word CONFIRMED 2-seed (42,43 both 0.844, overlap ~0.10 -- robust; seed 44
HUNG/slow, killed). 128-word LEARNED (v4, 4096 lang, 300ev) = IMPRACTICALLY SLOW (>4hr training, still not
done, killed) -> the practical ceiling of training fresh LEARNED orthogonal-code bridges is ~64-96 words;
beyond that, training cost explodes. PIVOT (committed to owner): use the validated FAST G.20 sparse-
distributed architecture for larger vocab -- both 160-concept AND 320-concept tiers already TRAINED + SHIPPED
(g20_sparse_bridges/ + g20_sparse_bridges_320/, 98.4% per-bridge). Running a 320-concept conversational demo
(g20_multibridge --sparse, sparsity 0.007, 5 bridges x 64; _demo320.log) to show scale concretely. LESSON:
long jobs now use BOUNDED waiters (max-iteration cap) so a hang/slow-run notifies me to reassess instead of
looping forever (the seed-44 + 128-word both stalled the old unbounded waiters). HONEST SCALING ANSWER: cheap
LEARNED recognition to ~64 words (128 too slow to train); G.20 sparse to 320 (validated); spiking composition
on top (V=320 synthetic 1.000; real-substrate 160 = boundary, temporal-integration-fixed). EARLIER scale64
detail below:
SCALE64 DONE (seed 42): cheap lever HOLDS -- overlap TINY 0.091 (did NOT
climb; codes well-separated), clean recognition 0.844, single-shot 0.378 -> the 0.844 is READOUT/SNR-limited
(inherent sparser codes: orthogonal coding needs sparsity<1/N), NOT representation-limited. PREPARATION
CONCLUSION (finding 2026-06-01-DirectionA-prep-CONCLUSION-100hr-not-warranted.md): the ~100hr is NOT
warranted -- the front-end is a cheap TRAINING+READOUT problem for v16 to ~64 words (28w ~0.95 multi-seed, 64w
~0.84), and the validated G.20 sparse-distributed architecture already covers 160-320; BPTT is independently
bounded. RECOMMEND HOLD THE 100hr; surface to owner. RUNNING (rigor completion, WITH waiter): 64-word
multi-seed (seeds 43,44 @300ev) to confirm 0.844 isn't seed-luck. EXACT NEXT: when it lands, finalize the
scale rigor; the direction call (push cheap follow-ups: matched-richness 64w / G.20-640 / consider solved) is
OWNER-STRATEGIC given A's premise is refuted -- surfaced. Cheap levers + G.20 cover the conversational vocab
range; no 100hr representation learning is justified. DISCIPLINE: a
positive control validates the INSTRUMENT not the COMPARISON (Gate 1's training-amount confound). Do NOT
launch the 100hr -- its premise is refuted at 28 words; surface to owner.
moat 7/7; 0.80 bar frozen; reuse-by-import; no autograd/protected-module edits; honest propagation both remotes.

## >>> LATEST ACTION (cheating-audit arc -- COMPLETE; 2026-05-31) <<<

CHEATING AUDIT (owner asked "are we still using templates/cheating, or is composition working?") =
COMPLETE, committed both remotes (finding 2026-05-31-cheating-audit-learned-vs-given-and-genuine-
composition.md). ANSWER: NOT cheating, honestly scoped. 16-word TRAINED pool-label 0.812 vs floor 0.125
= +68.7pp LEARNED; 28-word 0.571 vs 0.036 = +53.5pp; learned fraction erodes with vocab = the quantified
front-end wall. (c) REAL-SUBSTRATE VALIDITY (finding 2026-05-31-real-substrate-spiking-composition-
validity.md): does the validated spiking relational memory + abstention hold on REAL captured G.20 codes,
not just synthetic? bridgeA_nouns (32 concepts): REAL QA 0.800 + abstention 1.000 = RESOLVES, but honestly
DEGRADED ~20pp vs synthetic 1.000 (real codes cos 0.079 off the idealized pattern -> noisier; bind absorbs
it + perfect abstention). SCRUTINY CAUGHT A 2nd ARTIFACT: the naive 160-pool QA read 0.000 but with MAX
between-cos 1.000 -> DUPLICATE codes (all 5 bridges share byte-identical seed-42 sparse patterns) -> a global
160-way cleanup has 5-way ties -> artifactual 0, NOT a substrate boundary. Did NOT propagate the false
"boundary at scale"; corrected the finding + added a max-cos>0.95 VOID-DUPLICATE guard to the 160 probe.
qa64 already showed the algebra handles 160 DISTINCT synthetic codes (1.000); the deployed 160 substrate uses
within-bridge recall + cross-bridge engram TAGS, not global VSA. REAL-SUBSTRATE ARC COMPLETE = multi-seed
BOUNDARY: the "5-bridge" 0.800x5 was DETERMINISTIC COPIES (shared seed-42 patterns + fixed RNG), not 5
confirmations -> the real rigor = composition multi-seed (bridgeA seeds 42/43/44, n=20, varying roles/trials/
capture-noise; load_checkpoint restores the trained CSR so no wiring mismatch): REAL QA 0.900/0.650/0.850
(mean 0.800), synthetic 0.900/1.000/1.000 (mean 0.967), abstention 1.000 EVERY seed. VERDICT: composition is
GENUINE on real codes (perfect abstention -- drive-echo can't), but a multi-seed BOUNDARY (seed 43 dips 0.650
< bar while its synthetic = 1.000 -> real codes genuinely ~17pp harder, not noise). RESOLUTION: a 2.5x longer
capture window (stim 120->300, temporal integration) LIFTS real QA to 1.000/0.950/0.800 (mean 0.917, all 3
>= bar; gap to synthetic 17pp->5pp) -> the boundary is VARIANCE-limited not fundamental, fixed by the
project's validated temporal-integration denoiser (sustained encoding = cleaner code). The ALGEBRA isn't the
limit (qa64 V=160 synth 1.000); the real substrate's code NOISE is, and biology (sustained encoding) fixes
it. THREE self-caught artifacts this arc
(drive-echo + duplicate-code + deterministic-copies), none propagated, each honestly corrected, genuine
results intact = the discipline working. Finding 2026-05-31-real-substrate-spiking-composition-validity.md,
all committed both remotes. capability_status + pointer updated; wiki-sync pushed to Gitea. FRONT-END
DE-RISK DONE (cheap-first, no-retrain, _frontend_motor_dominance_probe.py; finding 2026-05-31-frontend-wall-
not-cheap-motor-rebalance-needs-redesign.md): the 28-word wall is NOT a cheap motor-rebalance -- concept
pools only separate concept words 13/24=0.54 among THEMSELVES (motors excluded), and down-weighting motors
makes it WORSE (0.571->0.464, breaks the 4 motor words). So the wall is genuinely architectural (concept-
representation separability at scale), refining v17's "motor dominance" (a symptom) -> NO cheap autonomous
lever remains; pushing recognition past ~28 words is a real retrain/redesign = OWNER-STRATEGIC (richer reps
/ more lang_input / concept-only arch / richer training; do NOT launch the ~100hr unilaterally).
>>> ARC COMPLETE + FULLY PROPAGATED. The owner's "are we cheating?" question is rigorously answered
(composition genuine + abstention-controlled; parser learned; concepts learned-at-small-scale measured;
real-substrate works, variance-limited, temporal-integration-fixable; THREE self-caught artifacts none
propagated). The tractable composition+recognition space is thoroughly validated + honestly bounded; the
next lever (learned concepts AT SCALE) is owner-strategic and de-risked as NOT-cheap. NEXT GENUINE STEP =
surface the comprehensive picture to the owner with the evidenced recommendation; await steer on the
owner-strategic direction. Do NOT fabricate make-work; do NOT launch the ~100hr or an architectural redesign
without owner buy-in (brainstorming/design + check-existing-first gate). <<<  Full rigorous answer below:
(1) Composition a template? NO -- genuine VSA algebra: generalizes (8/8 nonsense, 60/60, 3/3 multi-seed)
    AND correctly ABSTAINS on unstored facts (qa64 unknown-control 1.000 at V=160/320; drive-echo CANNOT
    abstain). (2) Parser a positional template? NO (closed) -- live REPL uses the LEARNED Hebbian parser,
    voice-invariant 3/3. (3) Concepts learned or given? PARTLY EACH, MEASURED: pool-label(trained) -
    pool-label(untrained-random) = the genuinely LEARNED fraction. V=28: 0.571 - 0.036 = +53.5pp (decisive).
    V=16: untrained floor 0.125 measured; TRAINED PENDING (_learned16 training PID 17088, ~20min, saving
    research/findings/raw/_learned16_seed42.simstate.h5). The GIVEN component is the orthogonal input
    encoding; large-V (160/320) concepts are GIVEN sparse codes (learned recognition validated only at
    small vocab). Triggered by catching one of my OWN metrics as a drive-echo ARTIFACT (bind-on-codes read
    1.000 even untrained) -- isolated to that one flawed front-end metric; abstention-controlled composition
    unaffected. Findings: 2026-05-31-cheating-audit-learned-vs-given-and-genuine-composition.md (16-word
    trained row pending) + 2026-05-31-front-end-distributed-vs-label-ARTIFACT-honest-negative.md.
EXACT NEXT: (a) when _learned16 training completes -> run
  `python -m research.findings.raw._learned_vs_given_probe --ckpt research/findings/raw/_learned16_seed42.simstate.h5`
  -> fill the 16-word TRAINED pool-label row -> complete the audit doc -> commit+push both remotes.
(b) SURFACE the honest cheating answer + the owner-strategic fork to the owner (per pointer below: tractable
  composition+P4 space is concluded; remaining big direction = V=640 richer-training, owner-strategic, do NOT
  launch unilaterally). (c) Proceed (tractable, non-100hr, real value): run the validated spiking relational
  memory on the REAL G.20 160-concept sparse substrate (bridges exist: g20_sparse_bridges/) with the
  abstention control -- the largest genuine-composition conversational artifact on a real substrate.
moat 7/7; 0.80 bar frozen; cheap-first before spiking; honest negatives are the deliverable; GPU/CuPy real.

## >>> CURRENT POINTER (read THIS first; 2026-05-31) <<<

ACTIVE ARC = BIOLOGICAL COMPOSITION (owner chose Option 2: "we absolutely want compositional
capabilities, work autonomously even with new ideas, biologically sound, catalog as needed").
POSITIVE REVISION (hardened, 2026-05-31; finding 2026-05-31-composition-REVISION-...-near-ortho-
ROLES-not-FILLERS): generalizable compositional bind/unbind (role x filler) works at 1.000 up to
K=8 with the SUBSTRATE's OVERLAPPING concept fillers (between 0.70) -- because cleanup uses
ID-separability (within>between, which the substrate HAS), NOT near-orthogonality. So the near-ortho
boundary blocks the WRONG thing (making MANY concepts near-ortho); composition only needs a FEW
near-ortho ROLE codes (agent/patient/action -- trivially feasible) x MANY ID-separable fillers.
Hardened: anti-cheat (broken binding ~chance by K=8; cleanup-bias 0.41 at K=1, an honest caveat);
role-mode controls (DISJOINT sub-pop roles FAIL ~chance -> roles must be DISTRIBUTED; overlapping
roles degrade); NOISE-robust (1.000 at 2x readout-noise std). Biologically-grounded: mean-centered
rate codes (= baseline-subtracted firing = the project's common-mode-removal) realize the +-1;
reconciles the denoiser NEGATIVE (that measured raw-symbol near-ortho separability; end-to-end
composition works via cleanup on ID-separable codes -- a different, achievable bar). Check-existing-
first found generative-replay (v2 smoke 0.02) + sequence-storage (DIRECTION-A/E) bounded; this
revision REOPENS composition. SHIPPED + VALIDATED since: (a) biological bind RESOLVED -- ON/OFF rate
coding + coincidence detection EXACTLY realizes the +-1 Hadamard with >=0 ops (verified max-diff 0.0);
(b) WORKING DEMO research/runners/compose_vsa_demo.py -- binds subject/verb/object on real substrate
concepts, answers role queries, GENERALIZES 60/60 novel sentences multi-seed (42/43/44), no training;
(c) spiking-readout de-risk RESOLVES -- composition survives Poisson spike counts at realistic firing
(1.000 at 0.5-5 spikes/neuron; the earlier BOUNDARY was MY mis-scaled spike budget, caught+corrected).
So biological compositional GENERALIZATION is VALIDATED at the mechanism level (5 axes + demo). Owner
said "go ahead, don't ask next time" -> fully autonomous, no more surfacing forks.
IN-SUBSTRATE SPIKING BIND BUILD = MILESTONE COMPLETE (2026-05-31): validated multi-seed (42,43,44) RESOLVES
to K=4 + adversarial reviewer CLEAR (7 exploit classes) + capability_status pillar n=111 promoted + owner-
facing demo shipped (compose_spiking_bind_demo.py, 12/12 novel sentences). The owner's "biologically sound"
composition is now realized IN spiking dynamics. SHIPPED+VALIDATED:
(a) PRIMITIVE 1 -- binary AND coincidence (research/findings/raw/_insubstrate_coincidence_probe.py):
a spiking neuron computes AND(role,filler) via threshold + tonic hyperpolarizing bias. seed42, RTX3090:
w=320 bias=-1000 -> BOTH=0.048 single=0.000 AND-selectivity=1.000 (perfect single rejection). Control
is geometric (role-only coinc gets role input but silent filler partner -> dark). The all-zeros at first
was sub-threshold 600pA drive (these Izh need ~2000pA); near-linear no-bias regime sharpens to clean AND
with the bias. (b) PRIMITIVE 2 -- graded gating (_insubstrate_graded_gating_probe.py): role gates, coinc
rate ~ filler magnitude (Spearman 1.000), role-OFF rate 0.000 at every filler level (perfect gating) ->
the bind preserves graded filler magnitude. (c) FULL ON/OFF BIND/UNBIND (_insubstrate_bind_unbind_probe.py):
one bridge, 8D neurons -- role_ON/OFF + fill_ON/OFF driven sources synapse into 4 coincidence banks A/B/C/D
realizing the +-1 Hadamard (bound_ON=A+B, bound_OFF=C+D); SAME layer reused for unbind; cosine cleanup on
real substrate concept codes (denoise64, projected D=800, V=16). seed42 RAW (no opponency): numpy-ceiling
1.000 all K; SPIKING recovery K1=0.933 K2=0.900 (>=0.80 RESOLVES) K3=0.756 K4=0.600 (SNR-degrade);
control at chance (0.05-0.13) throughout -> binding does REAL work, not cleanup artifact. The K>=3 degrade
is common-mode saturation (predicted: summing ON/OFF channels separately is non-canonical; the signed
DIFFERENCE is exact but re-driving the saturated channels compresses signal). FIX explored = ON/OFF opponency
(re-canonicalize superposed bound to signed form before unbind = retinal/thalamic lateral inhibition =
project's mean-centering; linear, in-substrate-realizable). OPPONENCY RESULT (D=800 seed42): lifts K1=1.000
K2=0.967 to the numpy ceiling but does NOT fix K>=3 (0.711, 0.683) -> common-mode saturation was NOT the
dominant high-load bottleneck. DIAGNOSIS (CPU Poisson two-stage capacity model, run + recorded): the K>=3
limit is finite firing-rate SNR = a READOUT-WINDOW/SPIKE-COUNT issue (Miller-like capacity), NOT a
mechanism failure. Model: window60 ~3spikes/dim K4=0.89 K6=0.78; window150 ~7spikes/dim K4=1.00 K6=0.99.
So capacity scales with the integration window (speed-accuracy tradeoff; biologically a longer readout =
more confident decision). GPU has extra noise (source-neuron stochasticity) so needs a longer window than
the ideal model, but the trend holds. DECISIVE seed-42 result (D=3200 + window 150): RESOLVES TO K=4.
spiking recovery K1=1.000 K2=1.000 K3=0.978 K4=0.833 (all >= 0.80); numpy ceiling 1.000 all K. Two SNR
levers got there (D=3200 averages cleanup over more dims; window 150 ~7 spikes/dim cuts rate noise).
SCRUTINY OF THE PASS (control elevated 0.27/0.23 at K1,2): investigated -> the numpy ALGEBRA has the SAME
elevation (0.20/0.11; chance 0.062) because codes are OVERLAPPING (between-cos mean 0.699). So spiking
control is FAITHFUL to the algebra's documented cleanup-bias floor, NOT a spiking artifact. "control==1/V"
is unachievable with overlapping fillers (mis-specified sub-clause); correct criterion = FAITHFULNESS
(spiking ctrl ~ numpy ctrl) + decisive recovery-vs-control gap (+0.73..+0.91) -- both hold. Probe verdict
corrected to faithfulness (NOT tuned to pass; algebra reference = ground truth). MULTI-SEED DONE: 3/3 seeds
RESOLVE to K=4 (K1=1.000 K2=1.000 K3=0.956 K4=0.861 mean; per-seed K4 0.833/0.833/0.917). Adversarial
reviewer CLEAR (insubstrate_spiking_bind_reviewer_verdict.md; reviewer simulated the single-Izh operating
point 0/1/2 sources -> 0.000/0.013/0.060, reproduced the overlapping-code control floor, ruled out leakage/
non-spiking/triviality/seed-sharing). capability_status pillar n=111 promoted (JSON valid, schema 6/6).
Demo compose_spiking_bind_demo.py smoke 12/12 novel sentences. ALL committed+pushed both remotes.
CAPACITY-SCALING DONE + WIKI-SYNCED. Window-300 run: HONEST NEGATIVE on the "window extends capacity to
Miller 7" hypothesis -- window 150->300 barely moved K4 (0.833->0.850); K5,6,7,8 = 0.760/0.600/0.500/0.438
(below bar). The CPU Poisson model overestimated (only spike-count noise); GPU K>=5 bottleneck is
WINDOW-INDEPENDENT (coincidence rate-resolution [rates in 0..0.05, coarse] / cross-term interference).
Honest capacity at validated operating point ~K=4. Corrected finding + capability_status (removed
window-extension over-claim, added capacity ladder); committed+pushed both. wiki-sync milestone pushed to
Gitea. CAPACITY FULLY CHARACTERIZED MULTI-SEED -> SPIKING-COMPOSITION ARC MILESTONE COMPLETE. Firing-rate lever
CONFIRMED multi-seed (42,43,44) at bias-500: K=4 0.975, K=5 0.933, K=6 0.856 (mean), every seed >= 0.80 at
K=4,5,6 -> capacity extends to K=6 (Miller 7+-2) multi-seed; control near zero. Complete honest story:
capacity is set by the coincidence FIRING RATE (K=4 clean-AND bias-1000 -> K=6 higher-rate bias-500); the
readout WINDOW does NOT extend it (window-300 negative, falsified+corrected). Finding + capability_status
(pillar n=111 tier/result/metric/summary -> K=4-6) updated, JSON valid schema 6/6; committed+pushed both.

>>> THE SPIKING-COMPOSITION ARC IS A COMPLETE, FULLY-PROPAGATED MILESTONE (mechanism + in-spiking
realization + 3 primitives + multi-seed bind/unbind RESOLVES + capacity K=6 multi-seed + adversarial CLEAR
+ demo + pillar n=111 + wiki-sync). Owner's "biologically sound" composition realized in spiking dynamics. <<<

NEXT ARC OPENED = RELATIONAL FACT-MEMORY (use the bind toward conversation). Cheap-first
(_vsa_relational_query_probe.py) RESOLVED multi-seed: SEPARATE-fact storage + cue-based retrieval =
single 1.000, relational-A(find-agent,read-patient) 1.000, two-role 1.000, control(no-false-match) 1.000
(seeds 42/43/44); superposed-B 0.475 DEGRADES (the multi-hop wall -> separate-fact storage is correct).
SPIKING version built (_insubstrate_relational_memory_probe.py, reuses bind/unbind machinery by import):
a fact = agent(x)X + action(x)Y + patient(x)Z (K=3 separate spiking bind); query = spiking-unbind agent +
cleanup-match cue, then spiking-unbind patient. RESULT D=800 bias-1000: 2/3 seeds RESOLVE (seed42 0.917/
0.917, seed43 1.000/0.917) but seed44 dips (single 0.833, relational 0.750 -- below 0.80; D=800 cleanup
margin thin). RELATIONAL FACT-MEMORY MULTI-SEED VALIDATED + DEMO SHIPPED. bias-500 3/3 PERFECT (all seeds
single=1.000 relational=1.000 control=1.000); seed-44's bias-1000 dip was bind-precision (the higher rate
= more dynamic range fixed it, same lever as K=6 capacity). Demo compose_relational_memory_demo.py smoke:
stores "dog go north"+"cat come south", answers relational queries correctly, control "(no fact found)".
Folded into finding (Downstream-capability section) + capability_status pillar n=111 summary; committed+
pushed both. So the session arc is COMPLETE + COHERENT: spiking composition (K=6 multi-seed, adversarial
CLEAR) -> queryable spiking relational fact-memory (3/3 multi-seed) -> 2 owner-facing demos. On the owner's
conversation goal.

KB-SCALING DONE: relational query holds 1.000 to N=12 facts (numpy ceiling, vocab-limited at 16 distinct
agents, all 3 seeds) and N=5 facts (spiking, seed42 bias-500, 1.000) -- separate-fact storage = each fact
an independent K=3 bind = no superposition interference. NESTED-COMPOSITION cheap-first NEGATIVE (honest):
flat phrase-as-filler ("big dog goes north", agent = bound "big dog") fails at depth-2 (descend to recover
the phrase's noun/modifier = chance 0.025-0.10; outer single-level 1.000) -- the superposition/multi-hop
wall. ARCHITECTURAL PRINCIPLE established + recorded: SEPARATE STORAGE is the universal structure mechanism
(multi-fact AND hierarchy); flat superposition/nesting hits the SNR wall. Hierarchy must use the relational-
memory pattern (store "big dog" as a {head:dog, modifier:big} fact, reference dog, recover modifier by cue),
not flat nesting. Committed+pushed (_vsa_nested_composition_probe.py + finding architectural-finding note).

>>> SESSION ARC FULLY COMPLETE + CHARACTERIZED: spiking composition (3 primitives, bind/unbind multi-seed
K=4->K=6, adversarial CLEAR) -> queryable relational fact-memory (multi-seed 3/3, scales to ~12 facts) ->
architectural principle (separate storage universal; flat nesting/superposition NEGATIVE). 2 demos, pillar
n=111, wiki-sync. ALL on the owner's conversation goal, ALL propagated both remotes. <<<

LIVE-TEXT-INPUT integration DONE: end-to-end relational fact-memory from LIVE text (drive each word through
the trained concept-pool bridge via activity_level_integration.build_substrate + capture_activity ->
live concept-pool activity -> spiking bind -> relational query) RESOLVES multi-seed (42,43,44) all 1.000;
front-end recognition 15-16/16; the bind is ROBUST to the recognition mislabel (uses distributed code, not
pool label). Probe _insubstrate_live_text_relational_probe.py; demo compose_live_text_kb_demo.py.
PARSER cheap-first DONE: voice-invariant role assignment ("dog chases cat" = "cat is chased by dog", same
agent) requires CONJUNCTIVE position*voice coding (position-only 0.000, additive 0.000, conjunctive 1.000;
seeds 42/43/44). Voice = function-word PRESENCE ("by") + relative position -- TRACTABLE features, NOT the
substrate's bounded ordered-sequence processing (the concern that nearly killed the arc; resolved). Probe
_vsa_parser_voice_probe.py. SYNTHESIS finding written (the deliverable per goal): 6 biology-translatable
insights (bind=coincidence; opponency=mean-centering; capacity=firing-rate not window; separate-storage
universal; bind robust to recognition errors; parsing needs conjunctive coding) +
2026-05-31-composition-in-spiking-substrate-SYNTHESIS.md. All committed+pushed.

SPIKING PARSER STDP-ACQUISITION = FIRST ATTEMPT DONE, honest status (NOT a boundary).
_insubstrate_parser_stdp_probe.py: a BARE STDP config (enable_stdp + plastic conj->role pathway +
simultaneous teacher) did NOT grow conj->role to firing strength -- role ensembles silent in test
(rates 0.000 at w_max=8 AND w_max=400). NOT a fundamental limit: v16 (lang_input->pool) learns exactly
this kind of input->output map via STDP, but uses embodied-Hebbian co-firing + v16 STDP params +
eligibility + a teacher protocol with correct pre->post timing -- machinery the quick probe lacks.
So the parser REPRESENTATION is validated (conjunctive coding, _vsa_parser_voice_probe.py) + its pieces
are validated (coincidence, bind); the in-substrate STDP-LEARNING is a FOCUSED SUB-ARC. Committed honestly.
LEARNED PARSER CORE = RESOLVES MULTI-SEED. The "focused sub-arc" was just "use the right learning rule":
the v16 HEBBIAN CO-FIRING rule (bridge.py:5265, pre&post-gated -> selective; hebbian_max_weight=400) was
the fix (bare spike-timing STDP failed -- a simultaneous teacher gives no pre->post order). Multi-seed
(42,43,44): 6/6 conjunctions including the active<->passive flip every seed. LEARNED (not supplied)
syntactic role assignment in-substrate. 7th insight banked (role assignment = Hebbian co-activation
learning, not fine-timing). All parser pieces validated (coincidence + Hebbian conj->role + bind).
Synthesis + capability_status (pillar n=111 summary) + probe all updated/committed both remotes.
END-TO-END LEARNED SYNTACTIC UNDERSTANDING = RESOLVES MULTI-SEED (3/3). _insubstrate_parser_bind_e2e_probe.py:
the Hebbian-learned parser assigns roles, the spiking bind stores the sentence, a relational query extracts
the agent VOICE-INVARIANTLY -- seeds 42/43/44 ALL parse 6/6, voice-invariant agent 1.000, scrambled-parse
control 0.000. "dog chases cat" (active) and "cat is chased by dog" (passive) both -> dog is the agent,
LEARNED not supplied. The FULL conversational pipeline is now validated end-to-end in spiking, multi-seed:
text -> live concept recognition (15-16/16) -> learned syntactic parsing (Hebbian conjunctive position*voice
-> role, 6/6 + flip) -> compositional bind (coincidence, K<=6) -> relational fact-memory (scales to ~12
facts) -> voice-invariant answer. 7 biology-translatable insights synthesized
(2026-05-31-composition-in-spiking-substrate-SYNTHESIS.md). All committed+pushed; capability_status pillar
n=111 + summary finalized.

>>> THE COMPOSITION ARC IS A COMPLETE END-TO-END BIOLOGY-GROUNDED CONVERSATIONAL PIPELINE, validated
multi-seed in spiking, with the scientific deliverable (7 insights) banked. On the owner's actual goal
(artificial life / brain analogue / biology-translatable insights / conversation instrumental). <<<

CONVERSATION BATCH BUILT (2026-05-31, all spiking, all multi-seed, finding 2026-05-31-conversational-
capabilities-on-the-spiking-bind.md): wh-QA (who/what 1.0/1.0/0.9), NEGATION+yes/no via a bound polarity
tag (3/3 = 1.0; insight: negation = explicit polarity ensemble, not absence), PERSISTENT KB across sessions
(3/3, no forgetting = the continual-learning premise), interactive REPL (compose_conversation_repl.py) the
owner can TALK to (teach/negate/ask, persists), conversation demo (compose_conversation_demo.py).
SCALING QUESTION RESOLVED HONESTLY (owner-prompted; _vocab_scaling_locus_note.md): the bind/COMPOSITION is
vocabulary-ROBUST (spiking cleanup 1.000 to V=320); the real ~320 limit is the RECOGNITION FRONT-END (98.4%
sparse multi-bridge; v17 28-word structural imbalance), NOT composition; plus a separate ~6-binding load cap.
Corrected two overstatements ('scaling tractable' + 'cleanup degrades with vocab', both WRONG).
SCALING FULLY ANSWERED + DEMONSTRATED: composition handles wh-QA at V=64 (3 seeds), 160, 320 ALL 1.000
(sparse codes); cleanup robust to V=640. The vocabulary limit is ENTIRELY the recognition front-end, NOT
composition. Conversation batch COMPLETE (bidirectional agent: understand/answer/negate/generate/persist +
REPL, all multi-seed). 8 insights synthesized. The TRACTABLE conversation + scaling space is comprehensively
done; everything committed both remotes.

FRONT-END distributed-vs-label = DONE, honest NEGATIVE (ARTIFACT; scrutiny caught it). Trained 28-word
bridge: pool-label 0.571, distributed-bind-QA 1.000 -- looked like a breakthrough. UNTRAINED CONTROL
(random weights, pool-label 0.036 = chance) gives bind-QA STILL 1.000 -> the metric measures the
ORTHOGONAL-DRIVE ECHO (distinct lang_input -> distinct codes even untrained), NOT learned separability.
So "distributed >> label" is an artifact; the 28-word recognition limit (57% pool-label) is REAL. Finding
2026-05-31-front-end-distributed-vs-label-ARTIFACT-honest-negative.md; committed both remotes. BROADER
honest implication recorded: captured concept codes carry a large drive-echo component (concept
separability substantially from the orthogonal INPUT encoding, not purely learned semantics) -- does NOT
undermine the COMPOSITION (bind/unbind generalizes to novel sentences, genuine), but refines scope. The
real front-end limit (learned word->concept routing) is unchanged. Possible NEXT (future): a drive-
INDEPENDENT capture/test (non-orthogonal or held-out drive) to measure LEARNED separability cleanly; or
accept the documented front-end as the hard frontier. NOTE on the dead first attempt: first training
(task b1jvm9b2g, 200 events)
TIMED OUT incomplete (~1.3s/event under demo contention = ~2hr not 28min; checkpoint never saved). RE-
LAUNCHED smoke-scale (task bnl7ff9zh, 50 events/word -- enough since v17 showed 50-200 events all give
~50% label; NO contending demos this time so it completes), saving research/findings/raw/
_v17_28word_seed42.simstate.h5. WHEN IT COMPLETES -> run
`python -m research.findings.raw._v17_distributed_vs_label_probe` (ALREADY WRITTEN + syntax-OK; loads the
bridge with matched architecture, load_checkpoint validates so the monkey-patch mismatch is caught) -> get
pool-label recognition (expect ~50% per v17) vs distributed-code bind/QA. If distributed >> label -> the
limit is a readout artifact (breakthrough); if ~equal -> the 28-word codes are genuinely inseparable (real
limit). EITHER is a real finding -> record + commit both remotes; if interesting, train seeds 43/44 for
multi-seed. The full front-end arc details:
  ANGLE (new, worth it): insight #5 (the bind uses the DISTRIBUTED code, not the pool LABEL) is established
  at 16 words (live-text 15/16 label but 1.000 bind). OPEN QUESTION: does the distributed-code bind-recovery
  EXCEED the pool-LABEL recognition at LARGER vocab, where the label drops (v17 28-word = 50% label)? The v17
  finding measured ONLY the pool-label (50%), never the distributed-code bind-recovery on the same bridge --
  so this is genuinely NEW, not re-deriving. If distributed >> label at 28 words -> the front-end limit is
  partly a READOUT artifact + the effective conversational vocab is larger -> a real path past the wall.
  BUILD REQUIRED: train a 28-word concept-pool bridge (~28-44 min/seed) + capture distributed codes + measure
  (a) pool-label recognition (b) bind/QA recovery on the captured distributed codes. CAUTION: concept_pool_
  demo_v2 uses the MODULE-LEVEL MONKEY-PATCH pattern that caused the 2026-05-14 architecture-mismatch
  retraction -- VERIFY bridge architecture matches between train + capture (the exact bug that invalidated
  the concept-concept results). Prefer extending concept_pool_demo's vocab cleanly over the v2 monkey-patch,
  OR assert architecture equality. Then adapt activity_level_integration.pool_layout for 28 pools to capture.
  This is a dedicated focused arc, not a quick probe. Propagate honestly (a re-derived 50% OR a distributed>>
  label breakthrough are BOTH real findings). prior arc (multi-hop) below.

TWO live threads:

(1) DONE = DEGRADES-WITH-FANIN (2026-05-31; finding 2026-05-31-P4-multihop-hub-reuse-DECISIVE-
DEGRADES-WITH-FANIN-...md). Multi-seed full-2hop 0.833 at fan-in 2 (>>chance 0.094) -> 0.000 at
fan-in 8. Clean 8/8 was the fan-in-1 easiest case. Bottleneck LOCATED (controller-scrutinized,
verdict survives): hop-1 flat/fine (0.83 all fan-in); entire loss at hop-2 -- querying a crowded
hub returns its many INCOMING nouns and buries the one OUTGOING edge (multitag is undirected/
aggregate-ranked). Anti-cheat held (13-14/14). Fundamental representational limit, not a tuning bug.

(2) NEXT ARC GROUNDED + BANKED (owner-aligned, ready regardless of (1)'s verdict): finding
2026-05-31-theta-multiplexing-conversational-holding-NEXT-ARC-grounding-temporal-not-
spatial-separation.md. The owner's preferred mechanism (theta-phase multiplexing, 2026-05-19
reframe) may SIDESTEP this session's separation-vs-reliability BOUNDARY because it separates
held items in TIME (theta phase slots) not in SPATIAL pattern -- routing around the exact
k-WTA knob that produced the boundary. Existing sim to adopt-from (check-existing-sims-first
directive): Ursino-Cesaretti-Pirazzini 2022 spiking Lisman-Idiart theta-gamma multi-item WM
(PMC10050512). Honest caveat recorded: 2025 Nat Neuro contests strict phase==order -> scope
to HOLDING/non-interference, not order-coding. Substrate already has the ingredients (concept
pools = gamma assemblies; parked integrated-loop Task-2 theta timing controller, reuse-by-import).

thread-(2) CHEAP GATE: DONE = RESOLVES (PASS, scrutinized) 2026-05-31. finding 2026-05-31-theta-
multiplexing-CHEAP-GATE-PASS-...-recovers-Miller-7.md. Pre-reg bar met (N=4 phaseRead 1.000>=0.90,
ctrlRead 0.217<0.50). Survived 3 scrutiny checks: decode margin +0.22 at N<=7 (confident); BOUNDARY-
ESCAPE demonstrated (overlapping cos-0.60 codes -- the regime that FAILED spatial DG separation -- held
0.989 at N=7 via phase, control collapses 0.118); capacity-realism (no-jitter cap 16 is a permissive
artifact; phase jitter 2 bins -> cap exactly 7 = Miller, recovered from theta/gamma ratio). NAMED OPEN
RISK for spiking build: cheap model assumes reader already knows each item's phase slot -> spiking build
MUST test phase-addressing LEARNABILITY + stability across encode/recall. HARD GATE PASSED.

>>> THREAD (2) CORRECTION (2026-05-31): the theta-multiplexing "next arc" is RETRACTED/DOWNGRADED, NOT
a viable new direction. On checking prior in-project work: theta-gamma multiplexing ALGEBRA was already
validated with decisive controls (2026-05-24 Direction E; 2026-05-23 FHRR N16) and its SPIKING-SUBSTRATE
composition already hit a DECISIVE 5-architecture convergent ceiling (2026-05-20-THETA-GAMMA-decisive-
honest-negative). The algebra was never the bottleneck; the substrate composition is the wall. My cheap
gate RE-DERIVED the known algebra (banners on both theta-multiplex docs). The night synthesis (2026-05-31-
NIGHT-ARC-...) had ALREADY pivoted correctly to P4. Residual value kept: the cross-arc insight (temporal
sidesteps the spatial DG boundary) + Miller-7-under-jitter. Do NOT build a theta-multiplexing spiking
arc -- it is a re-tread of ceiling'd work. <<<

DIRECTIONAL FIX DONE = RESCUED-but-BIMODAL (2026-05-31; finding 2026-05-31-P4-multihop-directional-fix-
RESCUES-per-bar-but-BIMODAL-...md). Multi-seed OUT full-2hop at fan-in 8 = 0.583 >= 0.50 bar (vs undirected
ANY 0.000) => RESCUED per the unmoved frozen bar. BUT bimodal: seed 42 = 8/8, seed 44 = 6/8, seed 43 = 0/8.
The directional filter (hop-2 hub query -> hub-first tags only) isolates big_red correctly on ALL seeds;
seed 43's 0/8 is weak UNDERLYING big_red binding on that bridge, not a filter bug. So directional removes the
hub-crowding bottleneck (strict win 0.583 vs 0.000) and EXPOSES residual per-seed binding-quality variance
as multi-hop's next limit. Directional multi-hop = REAL but NOT-UNIFORMLY-ROBUST.

INTEGRATION DONE (2026-05-31, commit 24ea2d4): directional filter SHIPPED into research/runners/
g20_multibridge.py -- _tag_matches_direction pure helper (8 unit tests, no GPU) + query_concept gains
direction='any'(default,backward-compat)/'out'/'in' + return_ranked + new "trace X" 2-hop command. GPU
smoke (160-concept seed 42): existing what-is unaffected; "trace apple" -> "apple relates to big, which
relates to red, new, angry". MULTI-HOP ARC COMPLETE end-to-end (characterized clean->DEGRADES->RESCUE-
bimodal + the validated fix shipped). g20_multibridge no longer byte-unmodified (this is the deliberate
shipped capability); no protected/frozen/moat module touched; backward-compat by construction.

GRID/CONJUNCTIVE BIOLOGICAL ARC = CONCLUDED 2026-05-31 (check-existing-first + cheap-first + scrutiny).
Survey: TEM/tensor-product conjunctive BINDING already covered (2026-05-06 Pick 4) AND the binding algebra
is already validated (FHRR) -> binding was never the blocker. The sharp grid idea (modular REDUNDANT coding)
cheap probe = CANNOT-CONCLUDE (instrument-invalid: M=1 control passes at all densities + id metric saturated
because the RAW activity is ALREADY 16/16 ID-separable, within 0.896 > between 0.768). CLARIFICATION (refines
my own DG "fundamental" overclaim): the substrate activity is already ID-separable (why retrieval works); the
unmet bar is NEAR-ORTHOGONALITY (between->~0) for clean VSA binding -- spiking DG reaches 0.66, clean k-WTA
0.45, neither near-0; and the spiking within-collapse is an implementation artifact (deterministic top-k is
stable). Findings: 2026-05-31-modular-coding-probe-INSTRUMENT-INVALID-...md + survey + DG-boundary banner.
NET: cheap biological VSA-near-orthogonal symbol-grounding is unmet; CONVERGES on night-synthesis P3(c) =
accept the oracle near-orthogonal code as an engineering component + advance the validated P4 retrieval stack.

TRACE BIMODALITY DIAGNOSED 2026-05-31 (finding 2026-05-31-P4-multihop-trace-bimodality-DIAGNOSED-...md):
it is per-pair x per-seed RECALL-STRENGTH (an engram-binding lottery), NOT a filter flaw, NOT seed-global.
Stim the tag, read target rank/32: big->red rank 2(s42)/8(s43,buried)/1(s44) -- mirrors multi-hop 8/0/6.
Other pairs weak on OTHER seeds (hot->dry rank4 s44; cold->wet rank8 s44; s43's hot->dry is rank1 strongest).
Where target falls below trace's top-3, multi-hop misses. Actionable in principle (strengthen weak bindings).
REINFORCEMENT-FIX = NEGATIVE (bf2wbr7n7 landed): re-encoding is an UNSTABLE random walk, not a reliable fix
(s43 big->red 8->1->2->2->1 fixed; s44 cold->wet 1->18->2->27->2 wild oscillation; s42 control stable). The
single-pass sparse engram capture is HIGH-VARIANCE; reinforcement adds variance not monotonic strengthening.
Reliable fix = deeper BALANCED-TEACHER encode (drive both concepts strongly in ONE controlled pass in
encode_pair_engram_sparse) -- a real sparse-encode change + re-validation, DEFERRED (marginal polish on an
already-shipped+sound capability). MULTI-HOP ARC COMPLETE: clean->hub-crowding DEGRADES->directional RESCUE->
bimodality DIAGNOSED (recall-strength lottery)->simple fix NEGATIVE->deeper fix specified+deferred.

NEAR-ORTHO BOUNDARY now DEFINITIVE (3 methods, 2026-05-31). Foldiak learned anti-Hebbian decorrelation (a
genuinely-new, check-existing-first'd, biology-grounded mechanism; finding 2026-05-31-foldiak-learned-
decorrelation-BOUNDARY-...md) = BOUNDARY: it DOES actively decorrelate to near-ortho (between 0.299, beating
the fixed-random floor 0.488) BUT over-sparsifies -> within collapses 0.484 + 7/16 dead codes (the near-ortho
is partly a dead-code artifact). So 3 independent coding methods -- spiking DG (0.66, within-collapse), fixed
random projection (0.45 floor, reliable), learned decorrelation (0.30, over-sparsified) -- all sit on the SAME
separation-vs-reliability frontier; NONE reaches near-ortho + reliable + all-alive. The near-orthogonality bar
for clean VSA binding is a GENERAL property of the substrate activity's structure, NOT method-specific. The
oracle near-ortho code (G.20 Kanerva-SDM) is genuinely IRREDUCIBLE from the substrate activity = an engineering
component, not a shortcut a cleverer code removes. The biologize-the-VSA-symbol line is DEFINITIVELY boundary-
characterized + banked as the biology-translatable deliverable.

MULTI-HOP TRACE BIMODALITY FIXED + SHIPPED (2026-05-31): root cause = SharedPoolMember.encode_pair's SPARSE
path silently OMITTED teacher_pA (used function default 100 vs configured self.teacher_pA=500; non-sparse path
passed it -- a real inconsistency bug). Teacher-strength probe validated 100->500 lifts weak big->red rank
8->2 stably (saturates, no over-drive, no harm to strong). One-line fix shipped (pass teacher_pA=self.teacher_pA
in the sparse call). POST-FIX diagnostic through the SHIPPED path: ALL 12 (pair x seed) now rank <=2 (was
big->red s43 r8, hot->dry s44 r4, cold->wet s44 r8) -> per-pair-per-seed lottery ELIMINATED -> trace bimodality
fixed at the determinant level (hop-2 needs top-3; all now <=2). 66 g20 tests pass. CONFIRMING (deferred,
advisable): full directional-multi-seed re-test (expect uniform pass) + multitag benchmark re-validation at
teacher=500. POST-FIX END-TO-END CONFIRMED (multi-seed directional re-test; finding 2026-05-31-P4-multihop-
POST-teacher-fix-...md) WITH AN HONEST REVISION: the teacher fix was the REAL win -- undirected multi-hop at
fan-in 8 jumped 0.000 -> 0.750; this makes the DIRECTIONAL filter (shipped earlier as "the fix") roughly
NEUTRAL/slightly-negative now (0.708 vs 0.750), because strong bindings surface the outgoing edge even
undirected. Directional treated the SYMPTOM (hub-crowding); teacher fix treated the CAUSE (weak bindings) ->
symptom gone -> directional no longer load-bearing (retained as harmless semantic choice; earlier directional-
RESCUES finding bannered superseded). Multi-hop now ~0.71-0.75 multi-seed at fan-in 8 (up from 0.00 undirected)
but NOT uniformly 8/8; residual variance is now HOP-1 CROSS-bridge encoding (noun->hub, encode_partial path,
NOT targeted by the intra-bridge teacher fix) -- a specified deferred lever (strengthen teacher in
encode_partial_pair_engram_sparse). MULTI-HOP ARC COMPLETE+FIXED: clean->DEGRADES->directional RESCUE(symptom)
->DIAGNOSED->reinforcement NEGATIVE->teacher_pA bug FOUND+FIXED(cause,real win)->end-to-end confirmed. HONEST
FRAMING CORRECTION: the encode_pair change (sparse teacher 100->500) was OVERSTATED as a "bug fix" -- the
encode_partial docstring says sparse teacher=100 is "the VALIDATED capture recipe", so 100 was likely a
DELIBERATE recipe (what the multitag 90% was validated at); the change is better framed as an EMPIRICALLY-
SUPPORTED RECIPE CHANGE (teacher probe no-harm + diagnostic + multi-hop improvement). MULTITAG FUNCTIONALLY
RE-VALIDATED at 500 (g20 scripted: "what is apple"->big 896, red 627 both correct; "is apple big?"->Yes); full
multi-seed 90% benchmark re-validation remains the rigorous DEFERRED confirmation. encode_partial (CROSS-bridge
sparse path) DELIBERATELY left at teacher=100 (its docstring-validated recipe) -- NOT changed, to avoid
compounding un-re-validated recipe changes; the hop-1 cross-bridge lever stays deferred pending proper
re-validation. Net: the intra-bridge teacher=500 change is empirically net-positive + functionally re-validated.

STRATEGIC FORK RESOLVED WITH EVIDENCE (2026-05-31; finding 2026-05-31-near-ortho-boundary-is-FUNDAMENTAL-
not-capacity-...md): the near-ortho floor is FLAT at ~0.48 from N=4 to N=16 concepts (delta +0.002) ->
near-ortho is unreachable even at 4 concepts -> the boundary is per-pair-overlap-FUNDAMENTAL, NOT capacity-
limited. So a "richer substrate" (more dims/concepts at the same overlap) would NOT help; the only biological
escape is months-scale richer TRAINING (intrinsically less-overlapping concept reps -- Phase-2 BPTT, previously
toy-scale falsified), which is HIGH-COST + UNCERTAIN. EVIDENCED RECOMMENDATION: ACCEPT the oracle near-ortho
code as an engineering component + advance the validated P4 stack (the deliverable). The months-scale escape
is an explicit OWNER decision, not autonomous.

EXACT NEXT CONCRETE ACTION (updated end-of-turn): the biological-symbol-grounding investigation is COMPLETE
with an evidenced recommendation (accept oracle + advance P4). The multi-hop arc is COMPLETE + the teacher_pA=
500 recipe change SHIPPED + CONFIRMED net-positive: the recipe change touches ONLY intra-bridge encode_pair;
the pre->post per-seed diagnostics ARE the multitag-accuracy re-validation for it (intra-bridge top-3 retrieval
9/12 -> 12/12, strict improvement) and cross-bridge encode_partial is UNCHANGED (left at teacher=100) -> NO
regression possible (only intra-bridge changed, and it improved). Plus functional multitag re-validation
passed ("what is apple"->big+red correct) + teacher probe shows no over-drive. So the change is validated.
A larger-K multitag-accuracy probe is a nice-to-have rigor upgrade, NOT needed. CROSS-BRIDGE TEACHER LEVER =
VALIDATE-FIRST NEUTRAL (NOT shipped): probe (seed 42, 6 cross-bridge noun->adj pairs) shows teacher=100 -> 6/6
AND teacher=500 -> 6/6 (EQUAL, both perfect) -> cross-bridge encoding is ALREADY strong at 100; the
encode_partial lever gives no gain. Validate-first avoided an unnecessary recipe change. IMPLICATION: the
residual multi-hop variance (seed 44 = 4/8) is NOT encoding-strength (cross-bridge already 6/6); it is
seed-specific STRUCTURAL variance in the fan-in-8 chaining -- irreducible by encoding strength. So the
MULTI-HOP ROBUSTNESS INVESTIGATION IS DEFINITIVELY CLOSED: intra-bridge teacher=500 was the real AND
sufficient lever; NO remaining cheap P4 encoding lever exists. NET: the working P4 stack IS the deliverable;
remaining big directions are OWNER-STRATEGIC (months-scale richer-training escape; ~100hr V=640 -- do NOT
launch unilaterally). Any NEW biological direction must be check-existing-first'd (theta-gamma + grid +
decorrelation arcs all explored/bounded this session). The honest boundary characterization IS the biology-
translatable deliverable per the owner's frame. Surface the evidenced recommendation to the owner.
P4 is well-advanced (160/320 concepts, multitag 90%, directional trace, hierarchy, yes/no, tokenize);
the biological-composition line is boundary-banked. Genuinely-open next directions all need a real design
effort (new biological subsystem mechanism) -- do that via brainstorming->design->cheap-first, check-existing-
first FIRST (theta-gamma + grid arcs were both already-explored). Do NOT unilaterally launch the ~100hr V=640.
moat 7/7;
0.80 bar frozen; cheap-first before spiking; honest negatives/clarifications are the deliverable.
moat 7/7; 0.80 bar frozen; cheap-first before spiking; honest negatives are the deliverable; GPU/CuPy real.


## DG-BIOLOGIZATION CONCLUDED 2026-05-31 = FUNDAMENTAL BOUNDARY; ACTIVE ARC = P4 conversational capability

>>> DG-biologization line CLOSED at a clean fundamental separation-vs-reliability BOUNDARY (finding
2026-05-31-DG-biologization-FUNDAMENTAL-BOUNDARY-...md). The DG separation MECHANISM is confirmed (0.82->0.18)
but no DG SIZE threads separation AND within-concept reliability: 800-sparse separated(0.27)/unstable(0.24);
4000-sparse stable(0.6-0.8)/unseparated(0.66-0.76) -- same competitive-k-WTA tradeoff curve, sweet-spot never
reached; CA3 collapses separation further. The oracle lookup's orthogonality is IRREDUCIBLE on this substrate.
Coherent night deliverable: integrated-loop VOID -> ceiling audit (representational) -> denoiser NEGATIVE ->
3-arc DG convergence -> DG gate PASS -> DG-composition NULL -> this fundamental boundary. Honest biology-
translatable scientific deliverable BANKED. (Controller mis-tuning in one 4000 re-run was caught + corrected;
boundary is the clean tradeoff curve, not an artifact.) <<<

ACTIVE = P4: advance the VALIDATED conversational capability (instant-runnable: g20_sparse_bridges 160-concept
+ g20_sparse_bridges_320 320-concept, multitag retrieval 90% / engram 87.5% / cross-bridge encode / hierarchy /
tokenization). FIRST STEP: confirm the working stack runs (g20_multibridge --sparse / g20_160word_demo), then
advance the highest-value extension toward conversational capability: candidates (a) multi-hop reasoning over
stored associations [known open gap, corrected-NEGATIVE], (b) scale toward 640 concepts [D8 infra scaffolded],
(c) cleaner interactive chat. reuse-by-import; moat 7/7; 0.80 bar frozen; honest negatives are the deliverable.

P4 STEP 1 DONE (clean-condition PASS) + DECISIVE SCRUTINY IN FLIGHT. Multi-hop 2-hop transitive on the
160-concept multitag stack = 8/8 PASS under CLEAN conditions (all-distinct words, anti-cheat 8/8, vs 0.25 prior /
0.094 chance) -- but the EASIEST case (no hub competition at hop-2); mechanism is chained ~100% single-hops via a
shared tag-name middle term, NOT learned inference. DECISIVE hub-reuse+multi-seed scrutiny RUNNING (subagent
a3d6187f2cb233796, _multihop_hubreuse_test.py: hub fan-in 2/4/8 x seeds 42/43/44). ROBUST(>=0.50 at fan-in 8)
-> real multi-hop reasoning capability -> build multi-hop chat demo. DEGRADES-WITH-FANIN -> bounded by hub
crowding (characterize curve). NEGATIVE -> clean 8/8 didn't generalize. [superseded marker for prior in-flight:], _multihop_reasoning_test.py): MULTI-HOP reasoning on the
validated 160-concept g20_multibridge --sparse stack. Encode 2-hop chains (A->B, B->C; A->C NOT directly
encoded), test whether CHAINING the 90% single-hop multitag (query A->B, query B->C) gives reliable transitive
inference. PRE-REG: WORKS if 2-hop transitive >=0.50 (> prior corrected-NEGATIVE 0.25); PARTIAL if >0.25 but
<0.50; NEGATIVE if <=0.25/chance. Controller scrutinizes (A->C genuinely not direct? hop-1 finds B? 2-hop
degradation = hop1*hop2 or worse from drift/loops?). FOLLOW-UP: WORKS -> real multi-hop reasoning capability;
build a multi-hop chat demo + characterize multi-seed. PARTIAL/NEGATIVE -> characterize the chaining limit
(noise compounding / loops) honestly, then next P4 extension (scale-to-640 [D8 infra] or interactive chat).

## (CONCLUDED) DECISION POINT 2026-05-31: three arcs converge on DG PATTERN-SEPARATION

Denoiser arc CONCLUDED = NEGATIVE (finding 2026-05-31-denoiser-arc-NEGATIVE-...-three-arcs-converge-on-
DG-pattern-separation.md). Biologizing shortcut-2 (oracle lookup) via activity grounding FAILS the
{2,3,5} bar: temporal integration denoises VARIANCE (CV ~1.63/sqrt(k) confirmed) but the activity-
grounded symbol is SEPARABILITY-limited (not variance) -- L=3 0.69 / L=5 0.57 plateau below bar at k=32.
The attractor cleanup is CATASTROPHICALLY WORSE (near-chance 0.23-0.26; it needs separable patterns,
collapses on the overlapping activity symbols). Sanity-checked NOT a usage bug (attractor recovers clean
vocab 100% at noise<=0.20). The oracle lookup's irreducible value = the ORTHOGONALITY the substrate
activity lacks. Honest NEGATIVE = the deliverable.

>>> CA3 DIAGNOSTIC DONE = INCONCLUSIVE/CONFOUNDED (CA3 saturated 0.946 active -> within/between 0.90 are artifacts; DG dense 0.37 -> not the separated regime; did NOT cleanly test CA3 on separated codes). Clean DG-side test P5 (larger DG: separation AND stability?) RUNNING (PID 10091, _dg_size_lever_probe, n_dg 800 vs 4000). If larger DG threads sparse-fraction-but-many-active -> stable+separated = resolution; else fundamental boundary (P3). Earlier note kept: CONVERGENCE (the strategic finding): THREE independent arcs now prescribe the SAME missing substrate
mechanism -- DG-style PATTERN SEPARATION:
  - integrated-loop (2026-05-30): wm binding needs stable+lesionable selectivity -> DG pattern-sep.
  - D-arc capacity (2026-05): dedicated-pool geometry erodes -> DG pattern-sep.
  - denoiser (2026-05-31): activity-grounded symbol not separable -> DG pattern-sep (orthogonalize
    before composition).
The project HAS a validated DG (trisynaptic loop, P1 D.12: DG cosine 0.218 from input 0.800, 58pp
orthogonalization). The convergent next arc: insert DG pattern-separation between substrate raw activity
and the composition-symbol derivation, then re-test whether DG-separated activity grounds a composable
symbol. DEEPER arc -> DECISION POINT for owner (this is the third arc to land on DG; it is the strongest-
evidenced direction the project has). DG GATE DONE = PASS (finding 2026-05-31-DG-separation-gate-PASS-...md). The hippocampal DG
ORTHOGONALIZES the overlapping concept activity: pool between-concept 0.806 -> DG 0.296 (sparsity 0.044)
/ 0.169 (sparsity 0.018), bracketing P1's validated 0.218. Multi-seed 42/43/44; genuine trained-substrate
activity (denoise64 caches, baseline 0.82 reproduced); isolation verified (ec at noise floor, no
lang_input); positive control reproduces P1 (0.800->0.218); dg_max 0.59 (no degenerate pairs). Controller
scrutiny: caught an ABANDONED first attempt (untrained pools 0.24 + degenerate silent DG) -- the subagent
independently fixed the SAME two flaws; final verified against JSONs.
LOAD-BEARING CAVEAT (carried to build): separation is SPARSITY-DEPENDENT (k-WTA) -- holds at sparse <=~0.05
(biological; P1 0.007-0.014), degrades if DG driven dense (0.16->DG 0.54, 0.81->0.81). The gate reached
the sparse band by tuning drive/FFi; the BUILD must drive DG into the sparse regime via WIRING, not
hand-tuning -- the build's first risk.

>>> DG ARC GATED-IN. BUILD (next): route concept activity -> DG (sparse k-WTA) -> derive the composition
symbol from the DG-SEPARATED activity -> re-test composition clears the 0.80 bar at {2,3,5} (the bar the
raw-activity symbols FAILED). If YES: oracle lookup biologized via DG pattern-separation = all 3 shortcuts
removable (artificial-life milestone). If NO: DG separation necessary-but-not-sufficient (narrower honest
boundary). Build must (a) wire DG into sparse regime, (b) preserve sparse DG code as the symbol, (c) keep
FHRR composition + moat byte-unchanged. reuse-by-import; no autograd. Cheap-first first: derive symbols
from the gate's DG-separated activity + run mean-of-k + argmax composition (reuse _denoiser_cheap_probe
machinery) before any heavier build. <<<

DG-COMPOSITION DECISIVE TEST DONE = NULL (finding 2026-05-31-DG-composition-NULL-...-needs-CA3-
completion.md). DG-symbol composition WORSE than pool baseline at every load (L2 0.41/L3 0.37/L5 0.33 vs
pool 0.83/0.69/0.58), barely above chance. Mechanism: separation is EXCELLENT (between-concept DG-symbol
cosine 0.18/0.10) but within-concept RELIABILITY collapses -- sparse DG silent on one obs-half for ~1/3-1/2
of words; storage vs query DG of the SAME concept near-disjoint (k-WTA picks different winners) -> unbind
recovers noise. Classic SEPARATION-vs-RELIABILITY tension (gate dose-response: sparse separates/unstable,
dense stable/no-separation; no single DG operating point gives both). 'no-silent' column is a vocab-collapse
artifact (disregarded). DG pattern-separation = NECESSARY-BUT-NOT-SUFFICIENT.

>>> RESOLUTION (biology prescribes it): CA3 PATTERN COMPLETION. The trisynaptic loop is DG->CA3 precisely
because DG separation alone is unstable. CA3 is a recurrent attractor that COMPLETES a sparse/partial DG
pattern to a STABLE stored ensemble -- the within-concept reliability the DG code lacks. P1 validated CA3
completion (D.13, cosine 0.748). Convergent prescription REFINES: not DG-alone but the FULL trisynaptic loop
(DG separates [confirmed 0.82->0.18], CA3 completes/stabilizes). NEXT TEST: drive concept->DG->CA3, TRAIN CA3
ensembles per concept (D.13 direct-CA3: co-fire full pattern + ca3_swr_burst gate to store; recall by
partial/noisy DG drive), derive symbol from the CA3 (completed, stable) code, re-test composition {2,3,5}.
HONEST RISK: D.13 was seed-variable (direct-CA3 passed 0.748; EC-driven FAILED) -> CA3 reliability on the
DG-separated concept activity is uncertain. FIXES -> trisynaptic loop biologizes the oracle lookup
(artificial-life milestone). Cannot-both-separate-and-complete -> deeper honest boundary. reuse-by-import
(builder/validate_trisynaptic_loop D.13 methodology byte-unchanged); no autograd; moat/FHRR byte-unchanged. <<< Standing reframes hold (0.80
bar frozen; moat 7/7; reuse-by-import; no new autograd; honest negatives are the deliverable). <<<

---

## CONCLUDED ARC 2026-05-30 (night): biologize shortcut-2 (activity-grounded symbol DENOISER)

Owner delegated ("whatever you think most productive, keeping goals in mind"). Per the top-level
goal (artificial life / biology-translatable; capabilities instrumental; honest negatives under
strict biology ARE the deliverable) -> chose the BIOLOGY-FAITHFUL path over a scaffold ceiling-break.

CONTEXT (from "check existing sims first" survey + the May-22 findings): phase-coded FHRR
composition is BUILT + VALIDATED -- spiking_phasor_fhrr.py (Orchard, PASS {2,3,5}), resonate_fire_
fhrr.py (Frady-Sommer RF + separated-TPAM cleanup, PASS {2,3,5}), identity-level integration
0.96-0.99 multi-seed. It rests on 3 engineered shortcuts: (1) function-first bind/unbind = BIOLOGIZED
(RF); (3) argmax-over-vocab cleanup = BIOLOGIZED (attractor TPAM); (2) ORACLE LOOKUP (fixed clean
symbol per concept) = STILL ENGINEERED. The May-22 activity-level integration tried to remove
shortcut 2 (derive symbol from real activity) -> NEGATIVE: substrate per-neuron activity CV~1.63
(160% noise); even composition-only collapses to 0.36 (<<0.80). Re-specified: a faithful activity-
grounded symbol needs an ATTRACTOR / TEMPORAL-INTEGRATION DENOISER (CV 1.63 -> ~0.20, the regime
where it composes >0.80). Shortcuts 2+3 COUPLED: a biological attractor grounds AND denoises.

THE ARC: build the denoiser between substrate activity and the FHRR composition layer; reuse the
validated FHRR composition + attractor (TPAM) machinery byte-unchanged; frozen 0.80 bar at loads
{2,3,5}, multi-seed, leakage-guarded.
CHEAP-FIRST GATE (next concrete step, CPU): reuse research/findings/raw/activity_level_integration.py
(captures 3200-dim per-neuron concept-pool activity; measured CV 1.63; composes via byte-unchanged
spiking_phasor_fhrr) -- INSERT a denoiser (temporal integration over k observations: CV~1.63/sqrt(k),
k~66 -> 0.20; AND/OR an attractor settle like the validated TPAM) BEFORE symbol derivation; measure
(a) post-denoiser CV, (b) composition accuracy vs 0.80. If any denoiser gets composition >0.80 (or CV
near 0.20) -> the denoiser arc is VIABLE -> design + build properly. If NONE -> honest NEGATIVE (the
substrate is irreducibly noisy for single-pass activity grounding; the oracle lookup is irreducible
on this substrate) = a biology-translatable deliverable. Cheap-first BEFORE designing big (the
falsify-cheaply discipline). resonate_fire_fhrr.ResonateFireTPAM is the reusable attractor denoiser.
Standing: reuse-by-import; no new autograd; no protected/frozen/moat edit; moat 7/7; 0.80 bar frozen.

CHEAP-FIRST GATE DONE = VIABLE (finding doc 2026-05-30-denoiser-cheap-first-VIABLE-temporal-
integration-denoises-activity-grounded-symbol-CV-falls-as-1-over-sqrt-k.md). k-curve (3-seed,
comp-only): k=1 0.34/0.36/0.41 (reproduces NEGATIVE baseline); k=8 L=2 0.849 PASS; k=16 L=2 0.936
PASS, L=3 0.802 PASS, L=5 0.659 (rising, extrapolates ~0.80 at k~32-48). CV falls ALMOST EXACTLY as
1.63/sqrt(k) (1.518/1.079/0.787/0.552/0.395 vs 1.63/1.15/0.82/0.58/0.41) => the substrate noise is
INDEPENDENT across observations (averageable), NOT correlated. So TEMPORAL INTEGRATION (sustained
encoding) genuinely denoises the activity-grounded symbol; the oracle-lookup shortcut IS biologizable;
required k grows with load. HONEST CAVEAT: 16 cached obs -> bootstrap-overlap may make exact k modestly
optimistic (CV law is overlap-independent so viability is robust; exact k needs more obs).

>>> CORRECTED 2026-05-31: the cheap-first 16-obs "VIABLE" was OPTIMISTIC. The rigorous 64-obs DISTINCT
confirmation (NO substrate confound -- RECOG_CACHE=phase1_800ev constant, both captures used it) shows
temporal integration ALONE is INSUFFICIENT for L>=3. <<<

64-OBS RESULT (distinct, k up to 32; finding doc CORRECTED + banner): CV still falls EXACTLY as
1.63/sqrt(k) (variance-reduction mechanism real) BUT composition PLATEAUS below 0.80 for higher loads:
  k=32 (CV 0.294): L=2 0.834 PASS (only at large k); L=3 0.694; L=5 0.575 (both BELOW bar, plateauing).
The 16-obs cheap-first inflated via vocab/storage observation overlap (16 obs -> cleanup-target vocab
shares obs with storage symbols). HONEST: temporal integration is a real VARIANCE denoiser but the
activity-derived symbol has a residual QUALITY/SEPARABILITY limit (not variance) that averaging cannot
fix -> at higher load, inter-concept crosstalk dominates. BOUNDARY for temporal-integration-ALONE
(L=2 only). NO confound (verified RECOG_CACHE constant).

KEY: the probe used a SIMPLE argmax cleanup, NOT the biological attractor. The May-22 'shortcuts 2+3
coupled' insight = an attractor GROUNDS + DENOISES + its fixed points are clean/separable. So the
residual is exactly what the attractor cleanup should fix.

>>> NEXT = CAPSTONE (well-motivated): temporal-integration denoiser + ResonateFireTPAM ATTRACTOR cleanup
(cleanup_separated), end-to-end on the 64-obs activity-grounded symbols, validate 0.80 bar {2,3,5}.
Does the attractor's recurrent settling lift L=3/L=5 above 0.80 where simple argmax couldn't? If YES ->
activity-grounded symbol biologizable WITH the coupled attractor (all 3 shortcuts removed). If NO ->
activity grounding is fundamentally separability-limited on this substrate (honest biology-translatable
boundary). Build: reuse 64-obs cache denoise64_seed{N}.npz + mean-of-k + resonate_fire_fhrr.
ResonateFireFHRR composition + ResonateFireTPAM.cleanup_separated (read its self-test for the validated
theta_low/high/n_anneal/abstain_threshold). RF + TPAM are time-stepped (slow) -> modest trials, can use
cleanup_separated fast= path if needed. reuse-by-import; spiking_phasor_fhrr / resonate_fire_fhrr / moat
byte-unchanged; no autograd. <<<


P4 PIVOT IS READY (instant, no training): the validated G.20 multitag conversational stack's bridges EXIST on disk -- g20_sparse_bridges/bridge{A-E}_*_sparse.simstate.h5 (160 concepts) + g20_sparse_bridges_320/*_sparse64.simstate.h5 (320 concepts). Runnable now via g20_160word_demo / g20_multibridge --sparse (cross-bridge encode + multitag retrieve + hierarchy + tokenization). DECISION LOGIC: the corrected 4000-low-drive DG test (PID 10146, watcher blw3vj8hv) is the DG-line DECIDER. RESOLVE (4000 reaches sparsity ~0.05 with WITHIN>>0.235 & BETWEEN<=0.5) -> continue trisynaptic line (CA3 next, carefully tuned). CANNOT-REACH-SPARSE or UNSTABLE -> the DG-biologization separated+stable compositional symbol is BLOCKED by a tuning-sensitive separation-reliability tension across systematic attempts (mechanism confirmed, assembly unreached) = honest biology-translatable BOUNDARY, BANK it, PIVOT to P4 (advance the working stack: candidate extensions = multi-hop reasoning [known gap], scale-to-640 [D8 infra scaffolded], or interactive chat). worth-GPU-time frame favors P4 if DG boundaries.
Re-run 64-obs (kill-safe): python -u -m research.findings.raw._denoiser_cheap_probe --capture-obs 64
--distinct --k-list 4 8 16 24 32 (GPU/CuPy for capture).

---

## CONCLUDED ARC 2026-05-30 (PM): conversational-ceiling AUDIT (owner chose "audit the ceiling")

Integrated-loop wm-emergence arc CONCLUDED (two-horns VOID, below). Owner picked A=pivot,
then for the next arc chose "audit the ceiling" over building phase-coded VSA. Survey found
the conversational line already ran 8+ decisive arcs (theta-gamma cue-supp / gentle-replay
6th-arc local-optimum 0.458 / SPEAR phase-multiplex 0.00 / Pirazzini / generative-replay /
staged-recurrence) all NEGATIVE/VOID, framed as a REPRESENTATION ceiling prescribing
phase-coded VSA (Orchard spiking-phasor FHRR; resonate_fire_fhrr.py exists). Audit verifies
that premise before the big build.

PHASE 1 DONE (no GPU, code-read; finding doc 2026-05-30-ceiling-audit-phase1-headline-numbers-
conflate-pipelines-composition-IS-decodable-at-0.46.md):
  - SPEAR full_acc=0.00 gated on RAW FIRING RATE @650 moat (spear...runner.py:515-528). The
    SPEAR units-bug hypothesis (cosine ranked @650 -> trivial abstain) is FALSIFIED (readout
    is genuinely firing-rate scale).
  - BUT headline numbers CONFLATE pipelines: 6th/8th-arc full_acc (0.458/0.315) is lang_output
    COSINE gated @ COMPOSITIONAL_UNIFIED_THRESHOLD=0.1977 (cosine scale); SPEAR is firing-rate
    sum @650. Non-comparable -> "8 arcs converge on ~0.46/0.00 ceiling" is a loose framing.
  - Composition IS decodably represented at ~0.46 TRUSTWORTHY gated emission (6th-arc cosine,
    calibrated gate, margins 0.064-0.118) -> the "composition not a structured decodable
    object / phase-coded VSA needed to make it representable AT ALL" framing is OVERSTATED.
  - Honest: the LITERAL pre-registered artifact (a)="raw vs gated" was NOT met (both gated);
    reported the related conflation + ceiling-reframe instead. Does NOT dissolve the ceiling
    (~0.46 is a real cap below 0.80).

PHASE 2 IN FLIGHT (owner said "ok" -> proceed): decisive latent-composition decode probe.
Subagent a3a208e2fea58fb08 (background) builds a throwaway probe (research/findings/raw/
_ceiling_audit_phase2_decode.py) reusing the EXACT 6th-arc machinery: generative_replay_pfc_
frame_runner.py FULL arm = unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe
+ _encode_facts + _unified_compositional_pairs + _compositional_query_ranked + consolidation_
trainer.run_concept_replay_phase + PFC-frame priming (the ~0.46 regime). Captures the composed
lang_output state + cosine-readout decision + true answer per query (>=200 instances, several
seeds); trains a HELD-OUT linear (sklearn LogisticRegression) + NN decoder with EPISODE-LEVEL
group k-fold (train/test never share an episode -> no leakage); compares decoder held-out acc
(B) vs cosine readout acc (A) on identical test sets.
PRE-REGISTERED (frozen): READOUT-LIMIT if B >= 2x A (and >> chance); REPRESENTATIONAL-CEILING
if no decoder beats A by >= +0.10; INCONCLUSIVE else. Decoder is ANALYSIS-ONLY (sklearn/numpy
CPU linear probe; NOT a sim learning rule, NOT autograd). No protected/frozen/moat/sim edit
(throwaway script only). CONTROLLER forms the official verdict + scrutinizes (a READOUT-LIMIT
result is the surprising/strong claim -> scrutinize it harder than a FAIL: episode-level split
real? regime check ~0.46? chance baseline? class balance?).
PHASE 2 ATTEMPT 1 INVALID (subagent a3a208e2 ended early w/o a valid result; controller
diagnosed): the probe script is STRUCTURALLY SOUND (episode-level GroupKFold no-leakage, diff
pairs/episode, primary lang_output + secondary pool states, pre-reg verdict) BUT FAILS THE
REGIME CHECK -- on the captured data the cosine readout top1 = 0.0 (0/8), only 2/8 predictions
are even adjectives (predicts "dog"/"go" for "small" answers). primary_state dim = 2048 => it
ran on the FULL validated substrate (loaded unified_per_regime/phase1/seed42.simstate.h5, 27MB,
EXISTS), so this is a PIPELINE-REPRODUCTION bug, NOT a substrate-scale issue: the probe does not
reproduce generative_replay_pfc_frame_runner's FULL arm (the ~0.46 regime). Per pre-registration
(STOP if regime wildly off) the decode comparison is INVALID -> NOT trusted.
FIX SUBAGENT (a2649185) STEP A + B DONE; STEP C decisive run IN FLIGHT:
  STEP A DONE: the REAL generative_replay_pfc_frame_runner REPRODUCES -- full_acc 0.40 (seed42 N2),
    0.4583 (3-seed N3). The 0.46 regime is REAL, NOT a regression.
  STEP B DONE (probe fixed to RUNNER-BLEND regime 0.4545, in band) + KEY REFINEMENT FINDING: the
    6th-arc "full_acc 0.46" is a BLEND of DIRECT-retrieval queries (easy, high acc) + COMPOSITIONAL
    queries (hard). The COMPOSITIONAL-cosine readout ALONE is ~0.0-0.30, NOT 0.46. This SHARPENS
    Phase 1: composition-only is decodable at ~0-0.30 (lower than the blended 0.46 implied); the
    Phase-1 doc's "composition decodable at ~0.46" should be read as the BLENDED number, not the
    compositional-only number. (Pending exact A from STEP C.)
  STEP C DECISIVE DONE = REPRESENTATIONAL-CEILING-CONFIRMED (conclusion doc
    2026-05-30-ceiling-audit-CONCLUSION-representational-confirmed-0.46-was-a-blend-VSA-warranted.md).
    3 seeds x 24 episodes, 120 instances, chance 0.25: compositional cosine A=0.24 (answer-subspace)
    / 0.04 (full-vocab); held-out linear B=0.21, NN 0.204, B_best 0.21; secondary pool-firing decoder
    0.218/0.208. ALL ~chance. Verdict REPRESENTATIONAL (B_best 0.21 < A 0.24 + 0.10). Scrutiny passed:
    episode-level no-leakage (decoders at chance not above); secondary state 16-dim WELL-sampled (not
    underdetermined) yet still chance -> composition genuinely not decodable from lang_output OR pool
    firing. Only throwaway probe changed; protected/runner/sim/compose untouched (verified).
  HONEST SELF-CORRECTION: Phase-1's "composition decodable at ~0.46" over-read the BLEND; compositional-
    only is ~chance & not decodable by any held-out decoder. Phase-1 doc banner-corrected.

>>> AUDIT CONCLUDED. The conversational-composition ceiling is REPRESENTATIONAL, not readout-limited.
The phase-coded vector-symbolic (Orchard spiking-phasor FHRR) arc is WARRANTED by a VERIFIED premise.
The audit gate PASSES. <<<

NEXT (owner decision point, surfaced): proceed to the phase-coded VSA arc DESIGN (brainstorm ->
design doc -> cheap-first probe gate -> spiking build under frozen-verdict discipline). The rhythm
must CARRY composition as spike phase so the composed state is a STRUCTURED DECODABLE object (the
exact thing this audit proved is missing). resonate_fire_fhrr.py exists as the spiking-phasor
primitive (reuse-by-import seed); check Orchard 2023/24 + Frady-Sommer resonator networks. Big new
arc -> recommend a proper design pass, NOT a reflexive build; cheap-first probe MUST gate the spiking
build. DEFAULT (no steer): begin the VSA arc brainstorm/design (check existing sims first). Standing
reframes hold (biology-grounded conflict-resolution; 0.80 bar frozen; moat 7/7; reuse-by-import; no
new autograd; honest negatives are the deliverable).

---

## CONCLUDED ARC 2026-05-30: phase-factored integrated closed-loop (goal pivot after closing the D-arc)

D-arc CLOSED at pillar n=110 (cross-bridge FHRR-scaffold capacity track done;
synthesis bbbf98f). Pivoted to the goal-aligned work: the phase-factored
integrated loop — composition as emergent from online theta-ordered episodic
encode + offline shuffled-replay consolidation (resolves the encode-order
conflict the parked Q5 loop stalled on, 2026-05-19). Design 45fe0a7; plan
c1e79b7 (subagent-driven, Task 6 controller-only).

Progress:
- Task 0 grounding pin: DONE (6bee885; 4 pass / 5 skip).
- Task 1 cheap-first falsification probe: DONE (19ef6f1 + strengthen 23ae76e).
  Controller caught the first version was CIRCULAR; strengthened the
  residual-coupling to a GENUINE measurement. Verdict RESOLVES (gate met;
  4e1d10f) but coupling_demonstrated FALSE at toy scale.
  SUBSTRATE CAVEAT (carry forward): toy uses near-orthogonal reps (tiny
  common-mode); the D-arc MEASURED real reps have LARGE common-mode
  (0.18-0.68), so consolidation moves real reps substantially and the
  residual-coupling could be real. The spiking build MUST test index
  survival on real reps + keep the consolidation-updates-index path as
  insurance.

UPDATE 2026-05-30 (Tasks 0-5 DONE; decisive run BLOCKED on instrument soundness):
- Tasks 0-5 complete: grounding pin; cheap-first probe RESOLVES (caught circular,
  strengthened; commit 4e1d10f); two-phase controller built (29 tests; reviewer
  CLEAR Task 4); verdict-reuse pin; no-harm (protected byte-empty, moat 7/7).
- Task 6 decisive orchestrator built (KILL-SAFE per-load; commit 3b04148).
- FULL-SCALE GROUNDING PROBE (32 min, N=2 seed42) = INSTRUMENT UNSOUND (commit
  e8cf685 + diagnosis 45e02d9). v1 wm=0.5 < 0.90 bar (decisive run would VOID).
  Problem 1 (BLOCKER): wm readout is a role->filler BINDING query, near-chance,
  no_bg_gate doesn't collapse it. Problem 2: ep collapses for the genuine task
  (full ep=0.0 vs v1 ep=1.0) = substrate caveat materialized.
- The grounding probe did its job: 32-min catch vs ~8-hr VOID. Decisive run NOT
  launched.
DIAGNOSIS COMPLETE (raw-counts sink, commit 42c851a): the wm failure is
NON-SELECTIVE RETRIEVAL, not gate calibration. Driving a role lights up ALL 8
filler pools ~equally; correct filler top only 1/8 -> chance. PARTIAL SUCCESS:
v1 ep=1.0 -- the two-phase ORDER readout WORKS.

UPDATE 2026-05-30 (decisive ITERATION 1: phase-restructure fix tried; EP DECOUPLING
VALIDATED, WM blocked by a SUBSTRATE-LEVEL cause; findings a54f2a9, fix 06b13c1):
The fix the prior NEXT-ACTION prescribed (move selectivity training to its correct
phase) was implemented faithfully: Phase 1 now FREEZES the selectivity gates so the
in-order pass writes only the ORDER INDEX (engram + theta-gamma slot order); Phase 2
runs the validated v16 SHUFFLED teacher co-fire + STDP to build selectivity, plus the
SWR consolidation. Full-scale v1 re-probe (seed 42, N=2): V1_EP=1.000, V1_WM=0.000.
  - VALIDATED: ep stayed 1.0 through the restructure -> the two-phase ENCODE-ORDER
    DECOUPLING is real (order written online, untouched by offline selectivity). Half
    the two-phase thesis confirmed.
  - NEGATIVE (characterized): wm still non-selective. Checkpoint probes show the
    topographic PRIOR alone gives clean 2/2 role->filler selectivity, but repeated
    selectivity-STDP over epochs ERODES the prior's margin (unbound fillers creep up,
    overtake by ~ep6) in EVERY variant (phase placement, role-only co-fire, off-target
    suppression, SWR on/off). Root cause = STDP-selectivity INSTABILITY on this
    substrate. The deep tension: the STABLE selectivity source (prior) is
    lesion-INVARIANT (can't satisfy lesion-collapse); the lesion-ABLATABLE source
    (STDP) is UNSTABLE. No mechanism here is both. Same substrate theme as the D-arc
    geometry erosion. This is the binding-retrieval problem localized to representation
    stability -- NOT a phase-placement bug (which is now fixed).

EXACT NEXT ACTION (DECISION POINT -- surfaced to user 2026-05-30): the wm instrument
cannot be made sound by controller-side wiring; it needs a selectivity CARRIER that is
both stable and lesion-ablatable. This is a deeper redesign, not a tweak. Two
biology-grounded candidates, both genuine next steps:
  (A) [RECOMMENDED] Hippocampal DG pattern-separation as the selectivity carrier (DG
      orthogonalizes reps AND is a lesionable subsystem -> satisfies lesion-collapse) --
      replaces STDP-on-cortex as the selectivity source. Where the D-arc independently
      pointed. Largely BUILT + P1-validated (trisynaptic loop: D.12 separation 3/3,
      D.13 completion). Function-matched to the failure (DG's job IS clean separation).
      Still a multi-day redesign: wire DG as the wm selectivity carrier, re-validate
      the instrument (v1 wm>=0.90), re-review, then decisive run.
  (B) [DE-RISKED OUT as cheap reuse] homeostatic stabilization. Cheapest-first probe
      (this turn) found the bridge's existing homeostasis (enable_synaptic_scaling,
      Turrigiano 2008, bridge.py:5797-5827) is postsynaptic-RATE-HOMOGENIZING (pulls
      every filler pool toward one target rate) -- that works AGAINST selectivity
      (which needs bound-filler high / unbound low). A genuine stabilizer here is
      divisive normalization across the filler population, which is a NEW learning rule
      (conflicts with reuse-by-import / no-new-rule discipline). So B is NOT a clean
      cheap reuse; A is the better-grounded path.
Both reuse-by-import only (no new autograd; no protected/frozen/moat edits). Decisive
multi-seed run STAYS UNLAUNCHED (v1 wm < 0.90, correctly). The frozen verdict + the
two-phase controller + the grounding discipline all held; the ep result is a real
partial win banked. FORK SURFACED TO OWNER 2026-05-30: (A) commit to the multi-day DG
selectivity-carrier redesign [recommended, goal-aligned: lifts the substrate ceiling
that blocks BOTH this arc and the D-arc]; OR park the integrated-loop instrument here
as an honest characterized NEGATIVE + banked ep win, and pick the next goal-aligned
arc.
OWNER STEER 2026-05-30: "Proceed as suggested" -> path A (DG selectivity carrier).

DE-RISK DONE 2026-05-30 = GO (commit b809ac7+ this update). Read confirmed path A is
wireable by reuse-by-import: (1) the built bridge ALREADY has DG/trisynaptic regions
(_build_bridge -> build_biological_brain_regions(enable_hippocampus_consolidation=True),
integrated_loop_gate.py:762 / phase_factored imports it). (2) The engram API
(start/commit/stimulate/clear/delete_engram_tag) is a dict keyed by name
(bridge.py:2483) -> MULTIPLE concurrent per-binding recordings supported, region-filtered
commits, tag stimulation -- all existing blessed-reuse methods. (3) The ep readout ALREADY
proves the path works: _episodic_order_readout (phase_factored_loop_gate.py:401-429)
stimulate_tag -> DG-separated CA3 completion -> role-pool peaks, ep=1.0. The wm readout
(606-649) currently does NOT use it -- it drives a bare role code + relies on the eroding
cortical dlpfc_verb->filler STDP selectivity. The fix: route wm role->filler retrieval
through the SAME DG/engram path.

EXACT NEXT ACTION = BUILD path A (controller file phase_factored_loop_gate.py ONLY; no
new rule/autograd; no protected/frozen/moat edit; engram API reuse byte-unchanged):
  1. At encode (Phase 1 loop ~269-356), commit a PER-BINDING engram tag per (role,filler)
     -- start_engram_recording("pf_ep%d_bind%d") around each binding's BG-gated co-fire,
     commit_engram_tag(region_filter capturing the role pool + filler pool + ca3). Keep
     the whole-episode tag (ep needs it).
  2. Rework the wm readout (606-649) to retrieve via the engram path: at a role query,
     stim the queried role's per-binding tag(s) (multitag stim-recall variant FIRST --
     87.5%/90% validated, the higher-reliability path; CA3 partial-cue completion as
     fallback) and rank filler pools by reactivation. Keep the DEFAULT_THRESHOLD gate +
     the _wm_raw passive sink.
  3. PRESERVE the 7-lesion partition: no_hippo_store/no_binding (SHARED) remove tag/
     assembly -> wm collapses WITH ep; no_bg_gate (HELPER_WM) degrades the per-binding
     gated co-fire -> wm collapses, ep survives. Map each lesion in comments; re-run the
     tiny-synth lesion probe to confirm.
  4. GATE: v1 full-scale probe (~3.5 min): require v1 wm>=0.90 AND v1 ep>=0.90. Iterate
     (multitag -> CA3-completion). If v1 sound -> re-review (readout changed post-Task-4)
     -> decisive run (controller-only, smell-test). If neither engram variant makes v1
     wm sound -> honest NEGATIVE (DG/engram per-binding retrieval also can't make the wm
     instrument sound at this scale) -> park.
Decisive multi-seed run stays UNLAUNCHED until v1 wm>=0.90.

SUBAGENT DONE + COMMITTED 2026-05-30 (cb6834b, pushed both remotes; HONEST partial-
verification status). The engram-based wm retrieval is built (per-binding tags at encode
+ stimulate_tag/CA3-completion retrieval, multitag). CONTROLLER VERIFICATION STATUS:
  R2 (engram path real, not relabeled STDP): CLEAR (from diff).
  R4 (rng faithful: deterministic sorted tag order, _make_pairs sole shared-rng consumer;
     integrated_loop_core.py byte-empty): CLEAR.
  Tests: 71/71 phase-factored+engram+moat PASS (controller re-ran, CPU). New test is a
     real behavioral spy on engram-API calls, not an impl echo.
  Tiny-synth partition: ep-side CORRECT (no_hippo_store ep=0, rest ep=1); wm-side FLOORED
     at 0 for ALL modes (scale artifact -> tiny-synth CANNOT validate the wm partition;
     R3 REQUIRES full scale).
  R1 (v1 raw-count selectivity smell-test) + R3 (wm-side 7-lesion partition at FULL
     scale): IN FLIGHT. Subagent's full-ladder probe (PID 6256, research.findings.raw.
     _pf_full_ladder_probe) is STILL RUNNING on GPU; watcher bbw6or6pg waits for it +
     dumps the table. After it lands: read the table for R3, then run _pf_v1_probe.py
     (~3.5 min, has the _WM_RAW_SINK) for R1 raw counts. Subagent CLAIMS v1 wm=1.0 ep=1.0,
     bound filler out-firing distractors ~13-50x -- MUST be controller-confirmed (scrutinize
     a PASS harder than a FAIL).
R3 DONE 2026-05-30 = VOID (full-scale partition table, N=2 seed42; finding doc
2026-05-30-phase-factored-decisive-iteration2-engram-wm-SOUND-but-VOID-two-horns-characterized.md):
  v1=(1.0,1.0) SOUND; full=(0.5,1.0); no_binding=(0.5,1.0); no_shared_clock=(0.5,1.0);
  no_hippo_store=(0.0,0.0); no_bg_gate=(0.5,1.0); no_sequencing=(0.5,1.0);
  no_cls_replay=(0.5,1.0); no_neuromod_timing=(0.5,0.0).
  The frozen verdict VOIDs at the discrimination check: no_binding (SHARED) must drop BOTH
  <=0.40 but wm=0.5 -> "not emergent-from-integration / wiring artifact" (and no_bg_gate /
  no_sequencing / no_cls_replay independently fail their checks). wm is FLAT 0.5 for full +
  6/7 lesions, 0.0 only under no_hippo_store -> the per-binding engram tag is a LOCALIZED
  hippocampal-store LOOKUP, lesion-invariant except removing the store. Drilled query passes,
  novel-recombination query fails (=0.5); v1 scores only the drilled query (=1.0).

>>> DO NOT LAUNCH THE DECISIVE RUN. <<< It would VOID identically at every seed (the
discrimination failure is STRUCTURAL -- the engram store is lesion-invariant by construction,
not stochastic). Running ~8 hr to reconfirm a structural VOID violates the grounding discipline.

TWO HORNS NOW CHARACTERIZED (both VOID-certified by the pre-registered verdict, OPPOSITE reasons):
  iter 1 STDP selectivity: EMERGENT but UNSTABLE -> VOID (unsound, v1 wm<0.90).
  iter 2 DG/engram store:   STABLE but NOT EMERGENT -> VOID (non-discriminating).
No mechanism here is both stable-enough-for-soundness AND emergent-enough-for-the-partition.
The integrated-loop wm-emergence thesis (role-filler binding retrieval) is NOT supported on
this substrate. BANKED + unaffected: ep-decoupling validated (ep=1.0 both iters); engram v1
soundness is genuinely SELECTIVE (R1 probe buzja4s2j confirming true filler >> distractors).

R1 DONE 2026-05-30: v1 retrieval DECISIVELY SELECTIVE at full scale (true filler 15x-400x over
distractors; cleanest queries true ~6000-6900 vs distractors <=15). The engram store is a
reliable, sharply selective role-filler memory -- the v1 soundness PASS is real (scrutinized
harder than a FAIL, holds up massively). Finding doc finalized + committed (7f662e0 + R1
sharpen). ARC iteration-2 fully recorded.

>>> INTEGRATED-LOOP wm-EMERGENCE ARC CONCLUDED (honest VOID, two horns characterized). <<<
Next per DEFAULT A: pivot to the next GOAL-ALIGNED arc (conversational capability / artificial
life), banking (i) ep-decoupling validated, (ii) engram = reliable selective role-filler memory,
(iii) the two-horns substrate finding. Do NOT auto-start path B (loop-gated readout = deeper
redesign, reopens instability) without explicit owner steer. When picking the next arc, honor
the standing reframes: check existing biology-grounded sims FIRST; build conversation on the
biological conflict-resolution mechanisms (SPEAR theta-multiplexing / theta-gamma / generative
replay), NOT static retrieval/RAG; bug-discovery-first on chance results; 0.80 multi-seed bar
frozen; moat 7/7 never weakened.

DECISION POINT (surfaced to owner; recommend A): (A) PARK the integrated-loop wm-emergence
thesis as a characterized VOID + bank the ep win, pivot to the next goal-aligned arc
(conversational / artificial-life). (B) attempt a THIRD mechanism -- loop-GATED engram readout
(retrieval depends on BG gate + shared clock + binding so lesions collapse wm) -- deeper
redesign, re-opens the instability risk (walks back toward horn 1), real risk of re-VOID.
DEFAULT (no owner steer): A -- record the two-horns VOID as the arc's honest conclusion + pick
the next goal-aligned arc; do NOT auto-start B (a deeper redesign warrants an explicit steer).

PRE-STAGED RE-REVIEW (R1/R3 detail; the wm readout changed post-Task-4):
  R1 [SELECTIVITY SMELL-TEST -- the load-bearing one]: scrutinize the _wm_raw raw filler
     counts on the scored v1 queries. A real PASS = the TRUE filler fires HIGH and the
     other 7 fire LOW (selective retrieval). A FALSE pass = all 8 fillers fire ~equally
     and the gated top is correct only by luck / because v1's drilled query trivially
     matches. Scrutinize a PASS HARDER than a FAIL. If v1 wm>=0.90 but raw counts show
     non-selective firing, it is NOT sound -> treat as NO-GO.
  R2: the engram path is REAL (per-binding tags + stimulate_tag), not a relabeled
     re-introduction of the eroding cortical dlpfc_verb->filler STDP.
  R3: 7-lesion partition holds at FULL scale (not just tiny-synth): SHARED
     (no_binding/no_shared_clock/no_hippo_store) collapse BOTH; HELPER_WM (no_bg_gate)
     collapses wm not ep; HELPER_EP (no_sequencing/no_cls_replay) collapses ep not wm --
     and each collapses for the RIGHT mechanistic reason via the new engram path.
  R4: RNG faithfulness (_make_pairs SOLE shared-rng consumer; any new tag-stim-order rng
     is a dedicated cross-mode-identical local rng); integrated_loop_core.py byte-empty;
     4 validated subsystems byte-unchanged; no new rule/autograd; moat 7/7; ep still 1.0.
Only if the re-review is CLEAR on all four -> re-confirm decisive cache empty
(research/findings/raw/phase_factored_decisive_cache/ -- verified empty 2026-05-30) ->
run the controller-only decisive multi-seed (phase_factored_decisive.py, seeds 42/43/44,
ladder N=2/4/8) -> mandatory smell-test on the recorded JSON -> honest propagation both
remotes. If NO-GO at v1 -> honest NEGATIVE finding (DG/engram per-binding retrieval also
cannot make the wm instrument sound at this scale) -> park the integrated-loop instrument
+ bank the ep-decoupling win + surface the next goal-aligned arc.
(D8 smoke killed 2026-05-30; marginal post-closure.)

OLD NEXT ACTION (superseded): Task 2 — build the two-phase controller +
order-preserving index readout in the spiking bridge
(research/runners/phase_factored_loop_gate.py), reusing 4 validated
subsystems byte-unchanged (engram-tag API, consolidation_trainer /
Phase-1.3 SWR replay, concept_pool_demo v16 binding, abstention_gate) +
the parked theta-gamma controller (integrated_loop_gate.py). Expose
run_rung(N, seed) emitting the rung shape integrated_loop_core.
integrated_loop_verdict consumes. Tiny-synth CPU-testable; heavy GPU run
is Task 6 (controller-only). Then Task 3 verdict-reuse, Task 4 adversarial
review BEFORE the decisive run, Task 5 no-harm, Task 6 controller-only
decisive multi-seed run.

GPU note: D8 speedup smoke (methodology validation) still running in
background; Task 2 is CPU-buildable so it does not wait.


## [HISTORY ARCHIVED 2026-05-31] older arcs moved to AUTONOMOUS_STATE_ARCHIVE.md

To keep THIS file under the 256KB Read limit (the local watchdog Reads it every
cycle; a >256KB Read errors), the 2026-05-21..05-27 history was moved verbatim to
`research/findings/AUTONOMOUS_STATE_ARCHIVE.md` (zero loss). Archived content:
D-arc Direction Q/P/3/4/R; capability pillars n=105..n=110; D6 V=160 + D7 V=320
sparse-distributed validation; the integrated-loop necessity-instrument 5-route
terminal line; the 2026-05-21 cumulative-deliverable + multiple "preserved state"
blocks; the 2026-05-27 WDDM perf finding (multi-process parallelism is a no-op on
Windows). Full per-arc detail also lives in the dated `research/findings/*.md` docs
+ `INDEX.md`. The LIVE pointer is at the TOP of this file; live-reference (frozen
bars + watchdog guarantee + crash-recovery lesson) is immediately below.

## Pre-registered acceptance / frozen bars (NEVER tuned)

`integrated_loop_core.py` `_IL_*`: V1_MIN 0.90, SCI_MIN 0.80,
LESION_MAX 0.40, SCALE_TOL 0.10, ladder (2,4,8), MIN_SEEDS 3. No-confab
moat `research/runners/abstention_gate.py` + test 7/7 byte-identical.
GPU (CuPy) for every real/decisive run; numpy only for `--tiny-synth`.

## Continuation guarantee (TWO watchdogs — installed)

1. **LOCAL, GPU-capable** — Windows Scheduled Task `SimAutonomousWatchdog`
   runs `scripts/autonomous_watchdog.ps1` every 20 min. Conservative
   stall-gate: fires ONLY if no git commit for >40 min AND no active
   claude/python-sim process AND no fresh `.watchdog.lock`. On stall it
   re-invokes local `claude.exe -p` (bypassPermissions, `--add-dir` repo)
   with a prompt to read THIS file and continue the exact next action
   INCLUDING PENDING-LOCAL-GPU steps. Audit log:
   `research/findings/raw/autonomous_watchdog.log`. This is the primary
   guarantee for GPU-bound work. (Re-verify: `schtasks /query /tn
   SimAutonomousWatchdog`; re-register via `scripts/autonomous_watchdog.ps1`
   contract if missing.)
2. **REMOTE claude.ai routine -- DISABLED 2026-05-20 (owner
   correction: budget consumed; do NOT re-enable or replace).**
   The prior `sim-autonomous-continuation-watchdog` routine
   (`trig_01W7vwnpv4JYWUMjzwHaEKK6`) was disabled by `RemoteTrigger`
   update `enabled: false` after consuming the routine budget.
   Continuity going forward is the LOCAL Windows Scheduled Task
   ONLY. Do NOT create/enable/replace this routine. See
   `memory/feedback_no_claude_routines_for_continuity.md`.

If the local watchdog is missing, RE-CREATE it before other work.
The local watchdog is a fallback for genuine session death --
it is NOT a justification to stop early in a working session.
The in-session discipline NEVER stops on a promise: the next
concrete tool call is always in the same turn after every commit;
ending a turn with "AUTONOMOUS_STATE points the next session at X"
is itself the promise-stall pattern the discipline forbids
(owner-corrected 2026-05-20).

## CRASH-RECOVERY (2026-05-28 ~17:32 EDT): D7 production died with Claude crash; relaunched with proper detachment

Claude desktop crashed during autoupdate (took D7 production PID 30216 with it -- I had wrongly claimed earlier it was "detached," but `-NoNewWindow` kept it attached to the harness's console process group, so the console tree got reaped). KILL-SAFE caches saved 12/15 cells (A_nouns, B_verbs, C_adj, D_spatial all 3 seeds; E_functional/seed42 was at 10%, lost the ~21 min partial).

Relaunched at 17:32:57 as PID 26928 with `Start-Process -WindowStyle Hidden` (creates a separate process group with its own console, no shared console with the harness -> survives client crashes). Cache-skip verified working (each of 12 cached cells "completes" in ~1.1 min by just loading the bridge + activity npz). Will then train E_functional seeds 42/43/44 (~225 min each), then run cross-bridge probe inline. ETA ~05-30 05:00 EDT.

LESSON for future launches: Use `-WindowStyle Hidden` (or omit `-NoNewWindow`) so the python process gets its own console + survives client death. `-NoNewWindow` is convenient for live log piping but makes the process die with the harness. The KILL-SAFE per-cell caches did their job here -- they're the load-bearing recovery mechanism, not the detachment.

