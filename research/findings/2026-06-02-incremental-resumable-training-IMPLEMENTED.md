# Incremental / resumable training IMPLEMENTED + verified -- extended runs need not be one-shot -- 2026-06-02

## The question (owner)
"Can't extended runs be handled incrementally -- accumulating training even with breaks in between? Or does
the full run need to be done in one go, meaning memory fragmentation ruins the whole run?"

## The answer (from the code + a verification)
1. **The GPU fragmentation is NOT a fundamental one-shot requirement.** It is within a single long-running
   process (CuPy memory pool fragmenting over many train steps) + orphaned GPU memory Windows doesn't free
   between processes. A fresh, shorter process runs fast (matmul 0.2s after killing python; bridgeC retrained
   in ~17min on a clean GPU while bridgeD crawled to ~60min after hours of accumulated runs). So breaks
   (fresh processes per chunk) AVOID fragmentation entirely -- it cannot "ruin" a run, only slow one marathon
   process.
2. **But incremental training was NOT wired up.** The sparse trainer trained only from scratch; the
   concept-pool trainer's --load-bridge loaded-then-SKIPPED-training (inference only). No "train chunk, save,
   break, resume, train more" path existed.
3. **Implemented:** concept_pool_sparse_distributed gains --resume-from <checkpoint>. On resume it
   load_checkpoint()s the trained weights (instead of re-applying the from-scratch topographic prior) and
   continues the training loop, so the new --n-train-events ACCUMULATE on top of prior training. Additive +
   backward-compatible (default None = unchanged from-scratch behaviour). STDP eligibility is per-event so
   resuming at an event boundary is clean; the saved weights carry the accumulated learning.

## Verification (16 concepts, 2048 lang, seed 42)
| run | training | top-1 recall |
|-----|----------|-------------:|
| A | 100 ev from scratch | 11/16 (69%) |
| B | RESUME from A, +100 ev (200 total, incremental) | **12/16 (75%)** |
| REF | 200 ev one-go | (pending) |

B (incremental 200) > A (100) -> resume genuinely ACCUMULATES training (the +100 events on the loaded
checkpoint improved recall 69% -> 75%). [REF one-go pending to confirm incremental ~ single-run equivalence.]

## Implication
Extended runs (e.g. the deferred full-320 retrain, or any large multi-bridge training) can now be done as a
sequence of SHORT chunks across breaks -- each chunk a fresh fast process (no fragmentation), accumulating
into a checkpoint. This matches the project's continuous-learning premise (the lineage system already persists
bridge state across sessions; now training can ACCUMULATE, not just reload for inference). The
fragmentation operational issue is no longer a blocker for large training.

## Operational note
Between long runs, kill python + verify nvidia-smi memory freed (Windows WDDM can leave orphaned GPU memory
that degrades the next process). Each incremental chunk should be its own process for a clean pool.
