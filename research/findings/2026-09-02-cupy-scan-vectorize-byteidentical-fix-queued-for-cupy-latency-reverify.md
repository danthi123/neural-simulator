---
status: qualified
lane: gap#66 / board #108
type: finding
date: 2026-09-02
mechanism: escalate-role-match-vectorized-gather
artifact: research/findings/raw/_cupy_scan_vectorize/byteident.json
---

# Vectorized the `_escalate_role_match` winner-code gather (board #108 cupy-latency fix): 0 answer/index mismatches vs the pre-fix Python-loop across 3 seeds on the real 100k bundle; the cupy median re-verify is QUEUED, not yet measured

**STATUS: qualified.** This closes the exact mechanism the 2026-09-02 diagnosis
([`2026-09-02-escalation-gating-tighten-latency-correctness-safe-not-the-lever.md`](2026-09-02-escalation-gating-tighten-latency-correctness-safe-not-the-lever.md))
pointed at as the real #108 R1 latency driver: "vectorizing the winner-code gather so it is one GPU gather
instead of a ~200-element Python-loop stack per role." The vectorization is DONE and PROVEN correctness-safe
(numpy, real bundle, 3 seeds, 0 divergences). Whether it actually drops the cupy median under 1000ms is
UNMEASURED here (numpy shows no escalation-branch latency signal at all, same as the prior diagnosis) — that
verdict is queued on `tools/gpu_queue.sh`, guarded so it only runs once this fix is on the primary checkout.

## The hotspot + the fix

`RFPhasorComposer._escalate_role_match` (`research/runners/rf_phasor_composer.py`) re-examines every fact
candidate whose coarse role-decode did not match the cued value, to check whether the cued value is a
near-tie runner-up. The line that gathered each candidate's coarse-winner concept code was a per-candidate
Python loop:

```python
win_codes = np.stack([self.concepts[words[i]] for i in cand])   # (m, D) -- ORIGINAL
```

`cand` can be LARGE — for the first cue role of a query, `prior_mask` is all-True (nothing has narrowed the
fact set yet), so `cand` is essentially every fact whose decoded role-word isn't the cued value, i.e. close
to the FULL K-fact store minus the few matches. On cupy this per-candidate loop pays a host<->device sync on
every one of those `self.concepts[w]` device-array touches — the cupy-specific driver the prior finding
diagnosed (numpy escalation-ON/OFF showed no latency difference at all; only cupy showed the ~1303ms
regression).

**The fix**: gather the same rows with ONE vectorized fancy-index into a cached (V,D) codebook matrix:

1. `_ensure_codebook_cache()` (already existed, board #192, `enable_codebook_cache` opt-in) now ALSO builds
   `self._concept_row = {w: i for i, w in enumerate(self.words)}` in the SAME pass as `self._cb_frac`
   (the (V,D) codebook), so the word->row map can never drift out of alignment with the codebook — both
   invalidate together on the same `len(self.words)` vocab-growth check the existing cache already used.
2. `_escalate_role_match` now calls `self._ensure_codebook_cache()` UNCONDITIONALLY (independent of the
   `enable_codebook_cache` flag — that flag still only gates the separate `_cleanup` fast path) and replaces
   the loop with:
   ```python
   row_idx = np.fromiter((self._concept_row[words[i]] for i in cand), dtype=np.int64, count=len(cand))
   win_codes = self._cb_frac[row_idx]   # VECTORIZED_WINCODE_GATHER -- one fancy-index gather
   ```
   `row_idx` construction is a plain host dict-lookup loop (no device traffic — dict lookups never touch
   cupy arrays); the ONLY backend operation is the single indexed gather `self._cb_frac[row_idx]`, which is
   the standard `xp` fancy-indexing every other batched path in this file already relies on.

The marker `VECTORIZED_WINCODE_GATHER` sits on the gather line itself (not on an infra comment elsewhere),
so a grep-guarded queue command can verify the actual hotspot line changed, not just that some nearby
scaffolding landed.

## Byte-identity proof (the hard correctness gate)

Artifacts: `research/findings/raw/_cupy_scan_vectorize/byteident.json` (structured per-seed counts +
backend + preconditions) + `research/findings/raw/_cupy_scan_vectorize/byteident.txt` (human-readable log)
(runner `research/runners/_cupy_scan_vectorize_byteident.py`; numpy backend — forced via
`os.environ["SIM_BACKEND"]="numpy"` inside the script itself, defense-in-depth against the GPU being busy
with another brain-loading proc, plus a runtime `sim.backend.get_backend()` assertion that refuses to run if
cupy resolves anyway).

Methodology: load the REAL shipped 100k wikidata bundle
(`/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k`, 78,857 facts, vocab 23,914, 395 shards)
through the exact production path (`developed_brain_io.load_developed_brain(..., enable_decode_escalation=
True, enable_codebook_cache=True)`), then run the SAME probe set TWICE on the SAME already-loaded agent: once
with `RFPhasorComposer._escalate_role_match` monkeypatched to a byte-for-byte reproduction of the PRE-FIX
method (verified against `git diff` at fix time — the only line that differs from the shipped file is the
`win_codes = np.stack(...)` assignment), once with the REAL shipped (post-fix, vectorized) method — no
reproduction risk on that side. A recall query is read-only (no RNG advance, no `self.kb`/cache mutation), so
re-running identical cues under the two implementations is a valid apples-to-apples comparison.

Per seed (15 recall probes + 10 moat/abstention cues, sampled the same way as the established
`_knowledge_scale_100k_production_verify.py` methodology; seed 44 additionally gets the documented near-tie
cue `query_patient("berkeley_county_virginia", "located_in_the_administrative_territoria")` from
[`2026-09-01-seed44-recall-hole-ROOT-CAUSED-phase-quantization-decode-escalation-fix.md`](2026-09-01-seed44-recall-hole-ROOT-CAUSED-phase-quantization-decode-escalation-fix.md),
so the escalation branch is provably EXERCISED, not merely present-but-inert):

| seed | decoded answers compared | answer mismatches | `_scan_first_match` index calls compared | index mismatches | verdict |
|---|---|---|---|---|---|
| 42 | 25 | **0** | 50 | **0** | IDENTICAL |
| 43 | 25 | **0** | 50 | **0** | IDENTICAL |
| 44 | 26 (incl. the explicit near-tie cue) | **0** | 52 | **0** | IDENTICAL |

Seed 44's explicit near-tie cue resolves to `culture_of_west_virginia` under BOTH implementations, matching
the documented fix (confirms the escalation branch actually ran a re-examination on this cue, not just that
the two implementations agree on cues where escalation never fires). Total: 76 decoded-answer comparisons +
152 selected-fact-index comparisons, 0 divergences. Per `docs/TERMS.md`'s "byte-identical" entry (asserted in
the data via exact compare, not inferred from reading the code) — this is an exact compare over the probed
cue set, not a global hash; the guarantee that `win_codes` is elementwise identical rests on construction
(`self._cb_frac[i] == self.concepts[self.words[i]]` for every `i`, by the SAME invariant `_ensure_codebook_
cache` already relies on for the `enable_codebook_cache` fast path — established board #192) plus this exact
compare showing no observable divergence on the cues probed.

## Verdict + honest residual

- **Correctness: PROVEN on the probed set.** 0/76 answer mismatches, 0/152 selected-index mismatches, across
  3 seeds on the real 100k bundle, with the seed-44 near-tie explicitly exercised. Ships as a drop-in
  replacement for the hotspot line — no other behavior changed.
- **Latency: UNMEASURED here, by design.** This is a numpy correctness proof; the prior diagnosis already
  established numpy shows NO escalation-branch latency signal at all (with or without this fix), so a numpy
  timing comparison would be uninformative. The queued cupy re-verify is the only instrument that can answer
  whether the median clears the <1000ms bar.
- **Scope**: does not flip any production default (`enable_decode_escalation` stays default-OFF pending the
  latency resolution, unchanged from the prior finding). Additive fix only.

**Queued (NOT run here — GPU held by another brain-loading proc; guarded on the `VECTORIZED_WINCODE_GATHER`
marker so it only fires once this fix is merged to `main`)**, on `tools/gpu_queue.sh`: the SAME 6-seed cupy
cacheon soak `SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._knowledge_scale_100k_cacheon_6seed
--enable-decode-escalation` used by the prior R1 verify, wrapped in a `grep -q VECTORIZED_WINCODE_GATHER
research/runners/rf_phasor_composer.py` guard, writing its own aggregate output artifact (not yet produced —
this run is queued, not yet executed).
Controller gate: recall 0-mismatch + moat 0-confab + latency median <1000ms on all 6 seeds -> #108 latency
cleared. If the median still exceeds 1000ms after this fix, the owner-stated fallback (accept ~1.1-1.3s) from
the prior finding still applies — this fix does not change that fallback, only gives it the best remaining
shot at clearing the harder bar first.

Branch: `research/cupy-scan-vectorize`.
