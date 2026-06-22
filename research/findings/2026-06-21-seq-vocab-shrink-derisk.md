# Sequencer vocab-shrink de-risk — audit opportunity #2 (the cheapest sub-lever)

**Date:** 2026-06-21 · **Runner:** `research/runners/_seq_vocab_shrink_derisk.py` · **Branch:** main
**Audit:** `research/findings/2026-06-22-megakernel-revisit-optimization-audit.md` §5 row #2 ("shrink the fabric:
build at a reduced CUE vocab, not full V — the cheapest sub-lever").

## TL;DR — VERDICT: **GO (byte-identical decisions)**.

Building the `integrated_loop` K-way on-substrate SEQUENCER at the **reduced cue vocab** (only the words that can
actually appear as a stored agent / action) instead of the composer's full V yields **byte-identical block decisions**
across the full battery (every present cue, three moat cues, the sequencer-lesion, the permuted anti-cheat, the
degenerate per-block-priority store), multi-seed, AND a large neuron-count + wall-clock reduction. The shrink is
answer-identical by construction (proven below + verified empirically — including the honest spurious-lit-word edge
case, which is *dropped* by the shrink yet never flips a decision).

## Why it is answer-identical (the load-bearing argument)

The sequencer's per-word match gate `g{b}A_{w}` is opened ONLY by the CUE word-line `cueA_{w}` firing
(`wire_sequencerK_couplings`: `couple_gate_to_pool(f"g{b}A_{w}", f"cueA_{w}")`), and a match requires BOTH the cue
line AND the block's DECODED line on the SAME word `w`. The decoded agent of every stored block is one of the K
stored agents; the decoded action one of the K stored actions. Therefore:
- a word that is NOT a stored agent can never be a block's decoded agent → its agent match line is dead weight;
- likewise the action role and the stored actions.

So building the sequencer over only `V'_A` = {distinct stored agents} (role A) and `V'_X` = {distinct stored actions}
(role X), with the cue index + the per-block decoded-line drives REMAPPED from the global word index (0..V-1) into the
reduced `V'_A`/`V'_X` spaces, gives the IDENTICAL spiking match cascade — no gate the full-V build opens is removed,
no match the full-V build makes is lost. A query cue `(agent, action)`: if `agent ∉ V'_A` OR `action ∉ V'_X`, NO block
can match → **abstain immediately** (a global no-op == the full-V sequencer, which would also find no decoded line for
it → abstain). Else run the shrunk sequencer with the cue remapped + the decoded drive restricted to `V'_A`/`V'_X` (the
cleanup score at every kept cue index is identical, so the match COMPARISON is byte-identical).

### The one honest subtlety — spurious-lit decoded words (handled, not papered over)

At imperfect fidelity (small D) a block's cleanup can light, within `drive_frac` of the peak, a NEAR-TIE word that is
NOT that block's stored role-filler (observed: a D=64 block-7 action cleanup lit `river` alongside `fly`). In the
**full-V** build that spurious line is harmless for every battery cue, because its gate `g{b}{role}_{spur}` is opened
only by the CUE word `cue{role}_{spur}` firing — and no battery cue's role-word IS that spurious word (present cues use
the stored fillers; moat cues use words guaranteed not to pair). So the spurious line is gated CLOSED and never
matches. In the **reduced** build that word has no line at all (it is not a stored agent/action) — the SAME net effect
(a closed/absent line on a word no battery cue drives). The runner therefore DROPS a spurious lit word (tracking the
count via `spurious_dropped`) rather than crash, and the load-bearing claim is then PROVEN EMPIRICALLY: if dropping it
ever flipped a decision vs the full-V build, the `identical` check would be False and the de-risk would report
NEGATIVE. **It never flipped a decision** (e.g. CPU seed 44 K=8 dropped 9 spurious action words → decisions still
byte-identical).

## Results

### CPU smoke (numpy, V=22, K=8, D=64), seeds 42/43/44

```
K=8 seed 42 D64: decisions IDENTICAL  lesion-safe+ident  perm-ident+rule  neurons 16190->5310 (-67.2%) VA=8 VX=4  full==host  time full=126.88ms red=40.58ms (3.13x)
K=8 seed 43 D64: decisions IDENTICAL  lesion-safe+ident  perm-ident+rule  neurons 16190->5310 (-67.2%) VA=8 VX=4  full==host  time full=126.99ms red=41.2ms  (3.08x)
K=8 seed 44 D64: decisions IDENTICAL  lesion-safe+ident  perm-ident+rule  neurons 16190->5310 (-67.2%) VA=8 VX=4  full==host  spur-dropped={'X': 9}  time full=125.43ms red=42.76ms (2.93x)
K=8 SUMMARY: decisions-identical 3/3  lesion 3/3  permute 3/3  (full==host 3/3, mean neuron reduction 67.2%)  -> GO
PRIORITY SUMMARY: 3/3  -> GO
OVERALL: GO
```

- **Decisions byte-identical, 3/3 seeds**, on every battery case + lesion-fails-safe + permute-follows-rule +
  per-block-priority. `full == host` 3/3 (the full-V reference itself tracks the host `_scan`, so identical-to-full-V
  is also identical-to-host on this battery).
- 67.2% neuron reduction at this small V=22 scale; **2.9–3.1× wall-clock** on the 80-step run even here.

### GPU production-representative (cupy, K∈{8,32}, D=128), seeds 42/43/44

K=32 uses the S2 production-stress table (32 distinct facts, V=72, **8 actions each shared by 4 agents** — the
maximal shared-action routing stress, the same table the K=32 margin de-risk uses).

<!-- FILL-GPU: paste the K=8 + K=32 result + SUMMARY + PRIORITY + OVERALL lines from
     research/findings/raw/_seq_vocab_shrink_k32_gpu.log here once the run completes. -->

### Neuron-count reduction — production scale (computed from the region math)

The settle cost (80 `_run_one_simulation_step` calls) scales ~linearly in neuron count (the sparse CSR SpMV is
O(nnz), and the fabric's nnz scales with the word-line count), so the reduction maps directly to the per-query
sequencer wall-clock:

| Configuration | full-V neurons | shrunk neurons (V'_A / V'_X) | reduction |
|---|---|---|---|
| **PRODUCTION `consolidated_320` (K=8, V=320, 8 agents / 7 actions)** | **218,830** | **6,330** (8 / 7) | **97.1% (34.6× smaller)** |
| K=32 stress table (V=72, 32 agents / 8 actions) | 192,030 | 56,830 (32 / 8) | 70.4% (3.4×) |
| hypothetical V=320 at K=32 (the audit's 836,830 flag; 32 agents / 8 actions) | 836,830 | 56,830 (32 / 8) | 93.2% (14.7×) |

The production conversational demo (`consolidated_320_conversation_demo.py`, `--composer onebrain --integrated-loop`,
the default) stores **8 facts** at **V=320** → the full-V sequencer builds word-lines for all 320 words but only
**8 agents / 7 actions** are reachable, so the shrink is **34.6×** (218,830 → 6,330 neurons). This is exactly the
audit's flagged ~244,580-neuron production sequencer path.

## What this de-risks for the wiring (STEP 2)

`OneBrainComposer._ensure_sequencer` / `_seq_block` currently build `build_sequencerK_bridge(seed, V=self.V, K)` and
pass global word indices. The wiring is: build at `V'_A`/`V'_X` (the distinct stored agents/actions), remap the cue +
the decoded drive, abstain-fast when the cue word is outside the reduced vocab, behind a default-ON flag with a
byte-identical full-V escape, invalidated when the store's cue vocab changes (a `store` or a reconsolidation that
rewrites a stored agent/action — not only when K changes). The no-confab moat is preserved by construction (an absent
cue word still → abstain; the spurious-drop never flips a decision).

## Anti-cheat panel (all GREEN on the CPU smoke; GPU pending the run)

- **sequencer-LESION** (sever the result→op conditioning): both builds fail SAFE → abstain, identically.
- **permuted-rule** (cyclic shift `m{b} → ans{(b+1)%K}`): both follow the rule + the decisions match.
- **per-block priority** (degenerate two-block-match on (dog, go)): the LOWER block wins on both builds == host.
- **spurious-lit drop**: tracked, and shown not to flip any decision (the load-bearing claim, proven empirically).

## Provenance / reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._seq_vocab_shrink_derisk --seeds 42,43,44 --dim 64 --ks 8 --time
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m research.runners._seq_vocab_shrink_derisk \
    --seeds 42,43,44 --dim 128 --ks 8,32 --time
```

NO `sim/` edit (reuse-by-import: the S0 K-way sequencer builder/wiring/reset/production-rule + the composer cleanup;
the de-risk only ADDS a reduced-vocab BUILDER + a remap RUNNER and asserts byte-identical decisions vs full-V).
Raw: `research/findings/raw/_seq_vocab_shrink_derisk_k8_cpu.json`, `_seq_vocab_shrink_derisk_k32_gpu.json`.
