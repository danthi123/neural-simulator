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

## Result -- full ensemble (160 concepts, 5 bridges, cross-bridge facts), seed 42
PENDING (_real_substrate_qa160 in flight; n_trials=20). RESOLVES (>=0.80 + abstention) -> the largest
genuine-composition conversational artifact in the project runs on the REAL deployed 160-concept substrate.
PARTIAL/below-bar -> honest boundary: real-substrate cross-bridge structure degrades composition at scale,
characterize the gap (vs the synthetic-code qa64 1.000 at V=160).

## Honest scope (carried from the cheating audit)
The real codes are pool activity in response to orthogonal lang_input drives, so they retain a drive-echo
component (separability partly from the input encoding). The point of THIS test is narrower and valid: given
the substrate's REAL concept representations (whatever their learned/given mix), does the genuine spiking
composition + abstention still hold? At 32 concepts: yes, degraded ~20pp. The composition is genuine (the
abstention control -- which drive-echo cannot pass -- is perfect); the real substrate just makes the cleanup
harder than idealized sparse.
