---
status: qualified
lane: gap#66 / board #108
type: finding
date: 2026-09-02
---

# Tightening the decode-escalation near-tie gate (0.02 -> 0.008) is CORRECTNESS-SAFE but is NOT the #108 R1 latency lever: the gate already fires narrowly (~4%, all at the seed-44 near-tie margin); the ~1303 ms regression is cupy-specific

**STATUS: qualified.** The #108 R1 verdict was: the `enable_decode_escalation` seed-44 fix RESTORES correctness on all 6 seeds (recall parity 0 mismatches + moat hold) but warm routed-recall MEDIAN latency runs ~1303 ms on 4/6 cupy seeds vs the verify's <1000 ms bar. The hypothesis under test: the near-tie gate fires too BROADLY, paying finer-re-read latency on candidates that were never going to flip; tighten it and the median drops under 1000 ms while the genuine seed-44 near-tie (its thin coarse margin) still resolves.

**The numpy diagnosis REFUTES that hypothesis.** The gate does not fire broadly, so tightening its trigger cannot recover the latency. It IS a correctness-safe hardening and can't-hurt, so it ships; but whether the median clears 1000 ms is decided ONLY by the queued faithful cupy re-verify, because on numpy escalation-ON adds NO measurable latency at all.

## What was measured (numpy, real wikidata_100k via the production load path, seed 44 = the hole seed)

Artifact: `research/findings/raw/_escalation_gating_tighten_smoke.json` (runner `research/runners/_escalation_gating_tighten_smoke.py`; cache-OFF -- the escalation gate's decisions are codebook-cache-independent, finding 2026-09-01 STEP 1; RSS ~1.4 GB, ~2 min).

1. **The seed-44 near-tie margin is razor-thin, and the loose 0.02 gate is ~9x looser than it needs to be.** Direct read of the stored composite (no query): unbinding the ACTION role and cleaning up over the full 23,914-word codebook, the coarse winner is `pelagonians` and the true `located_in_the_administrative_territoria` is the RUNNER-UP (rank 1) by `s_win - s_true = 0.002224`<!--derived--> of mean-cos (artifact `seed44_action_role_margin_direct.margin_win_minus_true`) -- reproducing the finding exactly. So the trigger only needs to reach ~0.0022<!--derived--> to catch it, vs the 0.02 default.

2. **The 0.02 gate fires on only ~4% of recall queries, and every observed finer-decode FLIP is at the seed-44 margin.** On a small probe (24 random recall cues + the seed-44 cue), exactly ONE query escalates -- the seed-44 cue -- at BOTH margin 0.02 and margin 0.008; the other 24 random recalls fire on NEITHER (artifact `loose.fire_rate` = `tight.fire_rate` = 0.04). The only flip is at the seed-44 margin (artifact `loose.max_flip_margin` == `tight.max_flip_margin`). 24 random cues with zero fires makes a broad (>~10%) fire-rate statistically implausible, so the gate is genuinely narrow.

3. **Tightening 0.02 -> 0.008 does NOT drop the fire-rate (0.04 -> 0.04) and does NOT change any recall answer** (0 answer-diffs on the probe; seed-44 still resolves to `culture_of_west_virginia` AND still fires). Because there are no candidates in the (0.008, 0.02] band on the sample, tightening removes no re-reads there.

4. **On numpy, escalation-ON adds NO measurable latency.** Median what_does over the probe: escalation OFF `1039 ms`, ON-margin=-1 (selection scan runs, nothing ever fires) `1014 ms`, ON-loose 0.02 `1027 ms`, ON-tight 0.008 `1025 ms` -- all within run-to-run noise (~+-25 ms), and an earlier low-load run read ~900-935 ms for all four. So the numpy median is INSENSITIVE to escalation and to the trigger margin, and this smoke does NOT reproduce the #108 ~1303 ms cupy median AT ALL. The regression is therefore cupy-backend-specific (the extra branch's cupy execution -- finer-re-resonate bridge builds and/or per-query selection kernel launches), not the trigger margin.

## The fix (additive / guarded, correctness-safe)

- `RFPhasorComposer.__init__`: `decode_escalate_margin` default **0.02 -> 0.008** (`research/runners/rf_phasor_composer.py`). **0.02 stays reachable as an explicit ESCAPE** (`decode_escalate_margin=0.02`) for A/B or rollback; `enable_decode_escalation` is still default-OFF (byte-identical when OFF, unchanged). The re-read LOGIC is untouched -- only WHEN it fires.
- Threaded the override end-to-end: `load_developed_brain(..., decode_escalate_margin=None)` (None keeps the composer default) and `_knowledge_scale_100k_production_verify.py --decode-escalate-margin` -- so the loose gate is a one-arg A/B from the production path.
- Unit-pinned: `tests/test_decode_escalation_seed44_hole.py::test_escalation_tightened_trigger_margin_default` (default 0.008, period 2000, 0.02 escape reachable); the existing byte-identical-when-OFF + moat + aggressive-margin tests still pass (8/8).

<!--derived-->
**Why 0.008 and not tighter.** (The margin-swing values below are QUOTED from the prior root-cause finding `research/findings/2026-09-01-seed44-recall-hole-ROOT-CAUSED-phase-quantization-decode-escalation-fix.md` + its artifact `research/findings/raw/_seed44_decode_margin_diag.json`; the span is derived from them.) The trigger must catch every near-tie a FINER readout could flip. That finding measured the seed-44 fact's mean-cos margin swing under readout refinement: coarse `+0.0022` -> period-2000 `-0.0047` -> closed-form `-0.0055` (the true word overtakes), a span of ~`0.0077`. A candidate whose coarse gate-margin exceeds ~0.0077 cannot be rescued by a finer period, so it never needed the re-read. `0.008` sits just above that measured span (loses NO recovery the 0.02 gate made -- confirmed here: `max_flip_margin` 0.0022 < 0.008, and 0 answer-diffs) while being 3.6x the seed-44 coarse margin (ample headroom for seed-44 + its unprobed thin-margin siblings) and 2.5x tighter than 0.02.

## Verdict + honest residual

- **Correctness: preserved.** seed-44 resolves at 0.008 (and still fires); 0 recall-answer changes vs the loose gate on the probe; the tightened band provably loses no readout-flippable recovery (0.008 > the ~0.0077<!--derived--> refinement span). The 6-seed correctness (recall parity + moat) that #108 R1 already established with escalation ON is unaffected -- 0.008 is a strict subset of the 0.02 firing set with headroom.
- **Latency: UNRESOLVED by this numpy smoke, by construction.** numpy shows no escalation regression at all, so it cannot show a fix either. The tighten is correctness-hardening; it is NOT expected to move the cupy median much (the fire-rate it trims is ~0 on the sample). Whether it -- or anything short of accepting ~1.3 s -- clears 1000 ms is decided by the QUEUED faithful 6-seed cupy re-verify.
- **If the cupy median stays >1000 ms** (the likely outcome given the numpy diagnosis), the honest options are: **(a)** owner-accept ~1.3 s -- it is within the owner's stated 1.1-1.3 s tolerance -- and re-run the verify with the latency bar set to that tolerance rather than 1000 ms; or **(b)** attack the CUPY-SPECIFIC per-query cost of the escalation branch directly (a byte-identical cheapening of the per-query near-tie selection scan -- e.g. vectorizing the winner-code gather so it is one GPU gather instead of a ~200-element Python-loop stack per role -- and/or capping the finer-re-resonate work), which is the mechanism a tighter trigger margin cannot touch. This is a follow-up, not a wall: the capability (correct + <1000 ms) is not abandoned, only re-pointed at the real driver.

## Scope / TERMS

This does NOT flip a production default (`enable_decode_escalation` stays default-OFF; the #108 100k default flip remains NO-GO pending the latency resolution + the #94 confidence cupy re-verify). Per docs/TERMS.md this is an additive lever + a diagnosis, not a capability "closed". "0 answer-diffs" is an exact compare over the 25-cue probe, NOT a global byte-identical hash -- the global no-recovery-lost guarantee rests on `max_flip_margin` (0.0022<!--derived-->) < tightened margin (0.008) plus the measured ~0.0077<!--derived--> refinement span, not on an exhaustive compare.

**Queued (do NOT run here -- GPU busy, one brain-loading proc at a time):** the SAME 6-seed cupy soak `SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._knowledge_scale_100k_cacheon_6seed --enable-decode-escalation` on `tools/gpu_queue.sh`, guarded to run only once the 0.008 default is on the primary checkout. It re-measures recall parity + moat + the latency median (all 6 seeds) at the tightened gate -- the real gate on the latency question above.
