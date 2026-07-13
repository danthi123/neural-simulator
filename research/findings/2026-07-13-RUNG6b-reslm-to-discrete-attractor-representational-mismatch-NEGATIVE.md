# RUNG 6b — reslm-front-end → D3 discrete-attractor for UNBOUNDED tracking: a characterized representational-mismatch NEGATIVE (the reslm's codes don't expose the clean per-clause slots the relational attractor needs)

**Date:** 2026-07-13
**Runner:** `research/runners/_rung6b_reslm_plus_d3_attractor_derisk.py` (reuse-by-import: reslm `ReservoirStates` + `make_reference_tracking_task` + `discrete_attractor_rnn`; numpy-CPU; NO `sim/` edit).
**Verdict:** NEGATIVE with the fixed random read → **SURPASSED same-cycle by candidate #2 (a LEARNED decorrelating read): 6-seed GO, attractor 1.000 at held-out-deeper lengths.** The workflow in action — the negative was an undiscovered mechanism, and the ranked next-gate's #2 candidate closed it.

## ⭐ RESOLVED — candidate #2 (learned decorrelating read) is 6-seed GO
The root cause (below) was that the reslm's per-clause codes are correlated (cosine ~0.4), so the attractor's *random* input map cannot separate the referent slots. Fix (`--learned`): fit a ridge read that maps each reslm subject code → a clean one-hot referent slot (the 6 codes are distinct, so a trained read separates them), then feed `[decode(reslm(a)) ; decode(reslm(b))]` as the attractor's per-clause input. **Result (6-seed 42/43/44/100/101/102): learned-read attractor@deeper = 1.000 every seed** (even exceeding the D3 clean-code control 0.75, since the learned read yields near-perfect one-hot slots), while the **reservoir-alone read fades to 0.203 (chance)** — the attractor's persistence + the learned decorrelating read are BOTH load-bearing. Anti-cheats hold: markov/retention floors at chance (the deeper-length tracking needs the composed history), reservoir-alone fades. ⇒ **unbounded discourse-referent tracking WORKS on the reslm substrate** (the reslm reads each clause → a learned decorrelating read → clean slots → the discrete attractor holds the referent to held-out-DEEPER lengths where the reservoir alone fades). **Honest scope:** the learned read is SUPERVISED on the KNOWN referent set (a bounded-referent scaffold; a read that GENERALIZES to novel/open referents is the frontier — same bounded-vs-open pattern as the rest of the arc). This is the same decorrelation lever (PPMI/learned-read) that recurs across the project. The finding title reflects the INITIAL fixed-read negative; the mechanism is RESOLVED for bounded referents.

---


## The idea (Rung 6 → 6b)
Rung 6 (`2026-07-13-RUNG6-...`) showed the reslm's own reservoir tracks the discourse referent at SHORT range and FADES with distance. The plan for unbounded tracking: use the reslm as the emergent per-clause ENCODER and the validated D3 discrete-attractor (`2026-07-09-D3-language-reference-tracking-GO.md`) as the GLOBAL tracker — feed the reslm's per-clause reservoir state as the attractor's per-clause input `X_t` on a token-encoded possession-TRANSFER task ("subj_a gives subj_b" per clause; holder=b iff holder==a).

## The result (seed 42; chance holder 1/6=0.167)
| variant | attractor@deeper | reservoir-alone | lesion | markov | retention |
|---|---|---|---|---|---|
| reslm NATURAL per-clause code | 0.200 | 0.200 | 0.200 | 0.174 | 0.200 |
| reslm STRUCTURED 2-slot read (reslm(a)⊕reslm(b)) | 0.298 | 0.225 | 0.200 | — | — |
| **CLEAN entity codes (positive control)** | **0.746** (n_per_len 800) / **0.871** (2500) | — | — | — | — |

- **The attractor MECHANISM is fine** — on the D3 clean entity codes it tracks the holder to held-out-deeper lengths (0.746–0.871, step-transition 0.999, matching the D3 finding). The failure is NOT the attractor.
- **The reslm codes feed it poorly** — the natural per-clause reservoir code lands at chance (0.200); even a structured 2-slot read (reslm(a) and reslm(b) in separate halves, mirroring the D3 clean code's subj-half/obj-half) reaches only 0.298 (≪ 0.746). The reslm evidence is barely load-bearing (lesion 0.200 ≈ natural 0.200).

## The diagnosis (systematic, positive-control-isolated)
The relational transition δ(holder, clause) = "b if holder==a else holder" requires the attractor to READ **a** and **b** as clean, comparable SLOTS from `X_t` (compare a to the tracked holder; conditionally output b). The D3 clean codes put a and b in **separable ±1 sub-codes** → the attractor learns δ (0.75+). The reslm's reservoir state is a **next-token predictor's distributed, blurred spike-rate code** — it does NOT expose a and b as clean linearly-separable slots (the running-cumulative even blends them; the per-clause structured read of the reservoir's subject codes is still too noisy/entangled). This is the SAME representational tension as the composer idealization (the exact-inverse VSA algebra demands clean decorrelated slots; a learned messy code does not provide them) and the D3 finding's own relational-δ residual (relational composition needs clean per-step slot structure, unlike a lookup DFA).

## Root cause (a0 measurement — the reslm codes are distinct but CORRELATED)
Measured the reslm per-subject codes directly: **6/6 distinct, but pairwise cosine mean 0.399 (max 0.589, min 0.230)** — moderately correlated, NOT the near-orthogonal sparse ±1 slots the D3 clean codes provide. The relational δ (compare a-slot to the tracked holder, conditionally emit b-slot) needs cleanly SEPARABLE slots; at cosine ~0.4 the six referent codes overlap enough that a random-projection read (the attractor's untrained `Wi`) cannot separate them → chance. This is the SAME decorrelation tension that runs through the whole conversational arc (the exact-inverse VSA algebra demands decorrelated codes; the point-neuron whitening limit; PPMI local normalization for generalization). So the gap is **code correlation**, and the fix is a read that DECORRELATES the reslm codes into clean referent slots — a LEARNED read (candidate #2), not merely more data.

## ⇒ The next-mechanism gate (NOT a wall)
Rung 6 (short-range emergent tracking) STANDS (6-seed GO). Unbounded tracking via the reslm→attractor composition needs a mechanism to give the attractor clean per-clause slot structure. Ranked candidates for the next research gate:
1. **DATA lever (cheapest first):** the clean codes improved 0.746→0.871 with more data (800→2500); train the attractor's input map on MORE reslm-coded narratives — does it learn to read the noisier reslm slots? (a bounded run; partial-close likely, full-close uncertain given the 0.30-vs-0.75 gap).
2. **A LEARNED reslm→attractor interface:** a trained projection (not the attractor's own random Wi) that maps the reslm state to the K-referent slot space — i.e. learn to read the referent identity from the reslm code first, then track (the D3 "dense per-step observation" done as a learned read, not assumed clean).
3. **Reframe the composition:** maybe the discrete-attractor is the wrong partner — a referent-tracking read-out trained END-TO-END on the reslm's own state trajectory (the reslm tracks AND generates), rather than a separate attractor with its own input encoding. (This is closer to how an LM learns coreference — one network.)

NEXT CONCRETE ACTION: research-gate the three candidates (a-1 our own record on learned-read vs fixed-projection referent decoding + the D3 relational-δ residual; then cheap-first de-risk #1 the data lever, then #2 the learned interface). The reslm's SHORT-range emergent tracking (Rung 6) is the standing positive; unbounded tracking is the open frontier.

Reuse-by-import; NO `sim/` edit. Runner: `_rung6b_reslm_plus_d3_attractor_derisk.py` (`--structured` for the 2-slot diagnostic).
