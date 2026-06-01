# Genuine spiking composition on the REAL deployed substrate (validity test) -- 2026-05-31

## Question
qa64 showed the spiking relational memory + wh-QA handles 160/320 concepts on SYNTHETIC gen_sparse codes.
Does it survive on the REAL concept codes captured from the deployed G.20 sparse bridges -- the actual
substrate representations, with their real noise/overlap -- not just idealized sparse patterns? This is the
"does it work on the real thing" validity step (the same rigor that historically caught architecture-
mismatch bugs).

## Method
For each word, capture the REAL concept code = shared_concept_pool activity when driving lang_input(word)
through the TRAINED sparse bridge (load_checkpoint validates architecture -> mismatch caught not silent).
Run the validated spiking SVO fact-memory + wh-QA + abstention control (bias-500 operating point) on those
real codes; compare head-to-head against the SYNTHETIC gen_sparse codes for the same words.
Probe: research/findings/raw/_insubstrate_real_substrate_qa_probe.py (32) + _..._qa160.py (160).

## Result -- single bridge (bridgeA_nouns, 32 concepts), seed 42

| Codes | wh-QA (who+what-obj+what-act all correct) | abstention control | mean cos to idealized pattern | between-concept cos |
|-------|------------------------------------------:|-------------------:|------------------------------:|--------------------:|
| REAL (captured)  | 0.800 | 1.000 | 0.079 | 0.170 |
| SYNTHETIC (gen_sparse) | 1.000 | 1.000 | -- | -- |

RESOLVES (real QA 0.80 >= bar, abstention perfect), but with an HONEST caveat under scrutiny:

- The REAL captured codes are NOT the idealized sparse patterns: mean cos(real, synthetic) = 0.079. The
  substrate's actual concept representation (after FS inhibition + Izhikevich dynamics) differs substantially
  from the topographic-prior TARGET pattern. So this is a genuine validity test, not a re-run of qa64.
- The real codes ARE separable: between-concept cos 0.170 (low) -> cleanup has signal to work with.
- But composition is DEGRADED on real codes: REAL QA 0.800 vs SYNTHETIC 1.000 = a ~20pp gap. The real-
  substrate structure (noisier, cos-0.079 off the ideal) costs ~20pp of wh-QA accuracy. The bind absorbs it
  and still clears the bar at 32 concepts with PERFECT abstention, but it is not free.
- 0.800 at n_trials=10 is exactly at the bar (8/10; wide binomial CI). Treated as a directional PASS, not a
  comfortable one. The decisive scale test is 160 (n=20, the harder cross-bridge case).

## Result -- naive 160-pool (5 bridges) = INVALID TEST (probe artifact, caught by the smell test), NOT a boundary
First 160 run: QA 0.000, abstention 1.000, and the giveaway -- **160-wide between-concept cos mean 0.191 but
MAX 1.000**. Max cos 1.000 means two pooled concepts have IDENTICAL codes. Diagnosis: all 5 G.20 bridges
regenerate their sparse patterns with generate_sparse_patterns(32, 2000, 100, seed=42) -- the SAME seed ->
BYTE-IDENTICAL patterns across all 5 bridges (the documented 320-tier "all 5 train with seed 42 -> identical
pattern set" issue). So bridge A concept i and bridge B concept i capture near-identical real codes. Naively
pooling 160 = 32 distinct patterns x 5 duplicates; a GLOBAL cleanup then has 5-way ties for every concept ->
exact-word QA collapses to ~0 (the cleanup returns one of the 5 identical-code copies, wrong word label 4/5
of the time). This is MY test's flaw, NOT a substrate boundary:

- qa64 ALREADY showed the composition handles 160 DISTINCT (synthetic) codes at 1.000 -- the algebra scales.
- The DEPLOYED 160-concept substrate does NOT provide 160 distinct codes in ONE cleanup space; it provides
  32 distinct x 5 bridges with SHARED per-index patterns. The deployed system never does a global 160-way
  cleanup: it does WITHIN-bridge recall (32-concept cleanup) + CROSS-bridge association via engram TAGS (the
  validated multitag mechanism), which is unaffected by the shared patterns.
- So "global 160-way VSA cleanup over the real bridges" is the wrong test for THIS substrate; the probe now
  asserts and reports max-cos > 0.95 as a duplicate-code instrument-invalidity (honest-by-construction).

## Result -- VALID scale test: within-bridge real-code QA across all 5 bridges (each 32 distinct concepts), seed 42

| Bridge | REAL QA | abstention | (synthetic QA) |
|--------|--------:|-----------:|---------------:|
| A nouns | 0.800 | 1.000 | 1.000 |
| B verbs | PENDING | | |
| C adj | PENDING | | |
| D spatial | PENDING | | |
| E functional | PENDING | | |

This is the valid scale test (5 x 32 distinct within-bridge codes, no cross-bridge duplicate artifact). If
all 5 RESOLVE -> real-substrate composition is robust across the deployed bridges (5x the single-bridge
evidence). The cross-bridge conversational case is handled by the validated engram-tag mechanism, not a
global VSA cleanup.

## Honest scope (carried from the cheating audit)
The real codes are pool activity in response to orthogonal lang_input drives, so they retain a drive-echo
component (separability partly from the input encoding). The point of THIS test is narrower and valid: given
the substrate's REAL concept representations (whatever their learned/given mix), does the genuine spiking
composition + abstention still hold? At 32 concepts: yes, degraded ~20pp. The composition is genuine (the
abstention control -- which drive-echo cannot pass -- is perfect); the real substrate just makes the cleanup
harder than idealized sparse.
