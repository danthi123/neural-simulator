# Direction A preparation CONCLUSION: the ~100hr representation-learning run is NOT warranted -- 2026-06-01

Owner chose (A) richer representation learning at scale, "but spend time in preparation ensuring we make the
most of all the time spent on compute." The preparation is complete. **Recommendation: do NOT launch the
~100hr run. The front-end is covered by cheap levers + an already-validated architecture.**

## The evidence chain (all committed, both remotes)
1. **External survey + internal asset map:** BPTT is already decisively bounded for this goal (char-level
   Phase 2.3a/2.3b NEGATIVE, "scale makes it worse"); the existing contrastive runner is NEGATIVE; post-hoc
   transforms on fixed activity are bounded (DG/Foldiak/random). So the obvious big bets were all dead ends.
2. **Gate 1 (later corrected):** measured the 28-word codes and concluded "representation limit" -- but this
   was confounded (the bridge was undertrained, 50 events, vs a 200-event 16-word control). Retracted.
3. **Gate 2 + training-events sweep (the decisive result, multi-seed):** with adequate training the 28-word
   "wall" disappears. Clean recognition 0.643(50ev) -> 0.893(150) -> 0.929(300) -> 0.929(500); concept
   OVERLAP DECREASES with training (0.606 -> 0.389). Multi-seed (42/43/44 @ 300ev): 0.929/0.964/0.964
   (~0.95). The "wall" was undertraining + single-shot noise; the cheap lever (more training of the EXISTING
   architecture + temporal-integration readout) does what the 100hr was supposed to buy.
4. **Scale test (64-word learned vocab, seed 42, 300ev):** the cheap lever HOLDS. Overlap is TINY (0.091 --
   codes very well separated, did NOT climb at 64 words); clean recognition 0.844. With overlap 0.091 but
   single-shot 0.378, the 0.844 is READOUT/SNR-limited (sparse codes), NOT representation-limited -- the
   cheap-lever regime. Honest caveat: 64 words forces sparser input codes (orthogonal coding requires
   sparsity < 1/N), which is most of the 0.95 -> 0.844 drop and is INHERENT to the orthogonal scheme.

## The vocab-scale picture (what covers each range -- no 100hr anywhere)
- **<= ~64 words, LEARNED v16 (orthogonal codes):** cheap lever (adequate training + temporal-integration /
  NN readout). 28w ~0.95 multi-seed; 64w ~0.84 (readout-limited, sparse-code headwind). ~1-2 GPU-hr/bridge.
- **160-320 words:** the project's G.20 sparse-DISTRIBUTED architecture (Kanerva K-of-N, a DIFFERENT code
  scheme that sidesteps the orthogonal sparsity headwind) is ALREADY validated at 98-100% per bridge. That
  is the right substrate for larger vocab -- no new representation learning needed.
- **Composition on top:** the in-substrate spiking VSA bind/unbind (validated multi-seed) composes whatever
  separable concept codes the front-end provides.

## Why the 100hr is not warranted
The premise that motivated it -- a hard representation wall at the front-end -- is refuted. The front-end is
a TRAINING + READOUT problem in the cheap regime up to ~64 words (the v16 architecture), and the larger-vocab
regime is already handled by the validated G.20 sparse architecture. BPTT (the obvious 100hr target) is
independently bounded. So a ~100hr representation-learning run would either repeat bounded work (BPTT) or
buy something the cheap levers + G.20 already deliver.

## What WOULD justify real compute (if the owner wants to push further) -- all far cheaper than 100hr
- A multi-seed 64-word confirmation + a matched-richness 64-word run (8192 lang) to nail the sparse-code
  headwind (~few GPU-hr).
- Train the G.20 sparse architecture to the documented 640-concept tier (linear in bridge count, ~hrs/bridge)
  -- the validated route to larger vocab.
- Strengthen the in-substrate composition's real-substrate boundary (temporal-integration readout already
  lifts it; a dedicated readout tier is cheap).

## Bottom line
The preparation did exactly what it was for: it found that the expensive run's premise was wrong (at a cost
of ~10 GPU-hr of gates instead of ~100hr on a false premise). Recommend holding the 100hr; apply the cheap
levers (adequate training + temporal-integration readout) for the v16 front-end and the validated G.20 sparse
architecture for larger vocab. Surface to owner for the call on whether to push the (cheap) 64-word multi-seed
/ G.20-640 characterizations or consider the front-end sufficiently solved.
