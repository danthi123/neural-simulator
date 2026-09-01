---
status: qualified
lane: gap#66
type: finding
date: 2026-09-01
---

# The seed-44 oracle-parity recall hole was a period-200 PHASE-READOUT QUANTIZATION flip — closed by a confidence-gated finer-period decode-escalation (NOT the codebook-cache; NOT routing)

**STATUS: seed-44 gate now PASSES (root-caused + FIXED); full 6-seed re-confirmation in flight.** The seed-44 recall hole is root-caused and FIXED with an additive, default-OFF, byte-identical-when-OFF lever (`enable_decode_escalation`).
Decisive integration result: with the lever ON, the seed-44 production-load verify passes the oracle-parity gate with **0 mismatches / 558 checks** (was 2), recall 1.0, moat 0, status GO. The byte-identity + moat guarantees are unit-asserted across seeds 42/43/44.
The remaining 5 seeds already passed oracle-parity BEFORE this change (escalation OFF), so the running full 6-seed numpy soak (+ the queued faithful cupy soak) only RE-confirms no-regression; this finding does not headline a 6-seed GO until that lands. The prior 2026-08-31 soak was PARTIAL only on seed 44's oracle-parity (2 mismatches from ONE fact).

## STEP 1 (decisive) — the miss is PRE-EXISTING, NOT cache-induced. The codebook-cache byte-identical claim HOLDS.

The failing query `query_patient(berkeley_county_virginia, located_in_the_administrative_territoria)` returns `None` on the tiered store **both with the codebook-cache ON and with it OFF** (reproduced on numpy: `off=None on=None`).
The query hot path is `query_patient -> _scan_first_match -> _cleanup_all` (the batched matched-filter), and `_cleanup_all` **does not read the codebook cache at all** — only the single `_cleanup` does, and that is reached only when RENDERING a patient AFTER a match is already found.
The reported miss is a `None` (no match found), which originates entirely in `_scan_first_match`, off the cached path. So the codebook-cache lever (commit 6bbe779a9, board #192) is **not** the cause and its byte-identical claim is intact on this path. The hole is a separate, pre-existing tiered-store decode defect.

## STEP 2 — root cause: the RF phase readout quantizes to 1/period (0.005 at period=200), coarser than a real inter-word cleanup margin

The RF first-spike phase readout is `phase = ((period - spike_step) % period) / period` (`sim/bridge.py:rf_read_phases`) — **quantized to 1/period = 1/200 = 0.005**. For the target fact's composite, unbinding the ACTION role and cleaning up over the full 23,914-word codebook gives (numpy; measurements in `research/findings/raw/_seed44_decode_margin_diag.json`):

| composite / unbind | ACTION decode | true-word mean-cos | `pelagonians` mean-cos | verdict |
|---|---|---|---|---|
| stored (npz, resonate-built) + period-200 resonate unbind | `pelagonians` | 0.32606 | 0.32829 | **WRONG (margin -0.0022)** |
| stored + period-2000 resonate unbind (finer readout) | `located_in...` | 0.33342 | 0.32872 | correct (+0.0047) |
| stored + closed-form unbind (infinite resolution) | `located_in...` | 0.33428 | 0.32879 | correct (+0.0055) |
| ideal (encode_fast) + period-200 resonate unbind (the ORACLE's path) | `located_in...` | 0.3385 | 0.33 | correct (+0.0085) |

The true action word `located_in_the_administrative_territoria` is the RUNNER-UP by **0.0022** of mean-cos — well inside the 0.005 readout quantization.
Two correlated quantization noise sources stack against it: (a) the stored composite was built by the neural resonate (it drifts from the ideal `encode_fast` by L-inf 0.0336 / mean 0.0152), and (b) the query-time resonate unbind quantizes again.
Only the STORED+resonate combination flips it; the oracle (ideal composite) and the closed-form (infinite resolution) both decode correctly — which is exactly why the oracle finds the fact and the tiered store misses it. As `period` grows the decode converges to the closed-form and the flip disappears (period >= 400 already fixes THIS fact).
It is **not** a routing miss: all 3 `berkeley_county_virginia` facts are correctly co-located in shard 271; the fact is present, only mis-decoded. Both seed-44 mismatches (the `what_does` None AND the `ask_yes_no` unknown) come from this ONE action-role flip.

## Other seeds harbor the SAME hole, unprobed (seed-INDEPENDENT substrate)

The tiered LTM is loaded from the bundle manifest at **seed=42 for every test seed** (`developed_brain_io.load_developed_brain` fast path -> `ShardedPhasorStore.load` uses `manifest.seed`, ignoring the test seed; the test seed only reseeds the 2-fact conversation BUFFER and the verify's probe SAMPLING).
So the store — composites + codebook — is byte-identical across all 6 seeds, and this hole exists identically in all of them. Only seed 44's oracle-agent sample happened to include `berkeley_county_virginia`; the other seeds' samples did not probe it.
So there are almost certainly OTHER thin-margin holes in the same store that no seed's held sample happened to probe; the fix is therefore a GLOBAL mechanism fix, not a fact-specific patch. Escalation recovers exactly the class whose true word is a thin-margin runner-up (the finer readout converges to the closed-form and reveals it); the only irreducible residual is a fact whose IDEAL (closed-form / infinite-resolution) decode is itself wrong — a genuine D=128 capacity collision that no period fixes — which the oracle would equally miss, so it is a recall<1.0 event, not an oracle-parity failure.

## The fix — confidence-gated finer-period decode-escalation (`enable_decode_escalation`, additive, default OFF)

`RFPhasorComposer.enable_decode_escalation` (threaded through `ShardedPhasorStore` composer_kwargs and `developed_brain_io.load_developed_brain(enable_decode_escalation=)`): in `_scan_first_match`, a fact still viable on the earlier cue roles whose stored `role` decoded (coarse argmax) to a word OTHER than the cued value, but for which the cued value is a **near-tie runner-up** (winner mean-cos - value mean-cos <= `decode_escalate_margin`, default 0.02), is re-unbound at a **finer resonate period** (`decode_escalate_period`, default 2000) — a longer-integrated, more faithful neural readout — and its match bit is set iff the finer decode now argmaxes to the cued value.
Biology: a difficulty-dependent decision time (an uncertain, near-tie readout triggers longer evidence integration before committing — the speed-accuracy trade-off / drift-diffusion decision-time), which is squarely in scope for "speed is secondary, slow-but-faithful biology."

**Guarantees (each asserted, not merely reasoned):**
- **Byte-identical when OFF** (data-asserted, `tests/test_decode_escalation_seed44_hole.py`): the default is OFF, the `_scan_first_match` restructure runs the identical computation, and even with the margin turned up so escalation fires on EVERY candidate, ON == OFF on a full stored-fact battery (no clean answer ever changes). The end-to-end store showed 0 OFF-vs-ON diffs on a 40-real-cue sample.
- **Moat-safe by construction**: an out-of-vocabulary cue value (an unknown agent / unknown relation) is never in `self.concepts`, so escalation is skipped and the abstain path is unchanged; and the finer readout converges to the ideal representation, so a fact that does not genuinely encode the cued value is never promoted (escalation only RECOVERS a truly-stored fact the coarse readout dropped). Measured 0 confabulations on the moat battery with escalation ON, plus a wrong-patient `ask_yes_no` still returns `unknown`.
- **Latency common-case unchanged**: the finer re-resonate touches only the rare near-tie candidates. Common-cue query 805->813 ms (numpy, ~1% overhead); an escalating query pays ~+320 ms once. The scale-battery MEDIAN (the gate's latency bar) is unaffected (near-tie holes are not in the scale sample).

## Verdict (re-verified)

- **Seed 44** (`--enable-codebook-cache --enable-decode-escalation`, numpy): oracle-parity **0** mismatches / 558 checks (was 2), recall 1.0, moat 0, lat_med 969 ms — **status GO** (was UNDEFINED).
- **6-seed soak** (42/43/44/100/101/102), decode-escalation ON, numpy: IN PROGRESS at commit time (seed 44 landed GO first; the other 5 already passed oracle-parity with escalation OFF, so this only re-confirms no-regression). Aggregate artifact `_knowledge_scale_100k_cacheon_escal_6seed_numpy.json`; faithful cupy re-run queued on `tools/gpu_queue.sh` (`_knowledge_scale_100k_cacheon_escal_6seed_cupy.json`).

## Scope / honesty boundary

This CLOSES the seed-44 oracle-parity GATE FAILURE (the specific #66 recall hole). It does NOT by itself flip any production default: `enable_decode_escalation` ships default-OFF (a reviewed flip is the next step), and the tiered LTM's agent-hash router + the codebook cleanup remain declared host scaffolds for the capacity de-risk (the faithful versions are a learned/spiking cue->sub-population router and a spiking cleanup). Per docs/TERMS.md, this is a gate pass + an additive lever, not a capability "closed" (which additionally requires on-by-default + scaffold-retired).

**Artifacts (committed):** `research/findings/raw/_seed44_decode_margin_diag.json` — the decode-margin measurements + the seed-44 gate result (escalation ON: 0 oracle mismatches / 558, recall 1.0, moat 0, GO; was 2). **Raw run outputs (on disk, uncommitted, per repo convention):** per-seed `research/findings/raw/_knowledge_scale_100k_cacheon_s{seed}.json`, the running numpy 6-seed aggregate `_knowledge_scale_100k_cacheon_escal_6seed_numpy.json`, and the queued faithful cupy re-run `_knowledge_scale_100k_cacheon_escal_6seed_cupy.json`. Superseded PARTIAL: `research/findings/2026-08-31-codebook-cache-6seed-soak-PARTIAL-seed44-oracle-parity-recall-hole.md`.
