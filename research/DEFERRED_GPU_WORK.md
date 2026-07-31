# DEFERRED WORK — needs a free GPU (self-contained; execute without re-deriving anything)

**Purpose.** Items blocked ONLY on GPU headroom, written so a future session can run them directly. Each entry
carries its trigger, the exact command, a PRE-REGISTERED prediction, and what to do with either outcome. If an
entry's prediction is refuted, that is a result — record it, do not quietly restate the claim.

**How to use this file:** check the TRIGGER, run the COMMAND, compare against the PREDICTION, then act on the
OUTCOME table and delete the entry (or mark it resolved with its finding path).

---

## D-1 · Is the `n > 15000` chunking threshold stale for the argsort implementation?

**Status:** OPEN · raised 2026-07-31 by the adversarial bug hunt · severity MEDIUM · owner: autonomous

### Trigger — how to know the GPU is free enough
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader     # need >= 18000 MiB
ps -eo args | grep -c '[g]ap4_onbridge'                       # want 0 (the crux is done)
```
Both conditions must hold. **~15 GB free is NOT enough** — see the budget below.

### Why it is deferred rather than done
`sim/connectivity.py:132` routes `n > 15000` to the chunked generator. The question is whether the NON-chunked
branch, taken at `n <= 15000`, actually fits in VRAM at n=15000. Measured cost of that branch:

| term | size at n=15000 |
|---|---|
| ~7 coexisting `(n,n)` float32 arrays (`distances`, `prob_dist`, `prob_trait`, `conn_prob`, `log_prob`, `gumbel_noise`, `perturbed`) | **6.3 GB** |
| int64 argsort output (`cp.argsort(perturbed, axis=1)`) | **1.8 GB** |
| Thrust sort temporaries | **unmeasured** |

That is ~8.1 GB known with an unbounded tail, against a card also carrying other work. Running it while the crux
occupies the GPU risks OOM-ing the roadmap's load-bearing dependency mid-run, which is why it was held.

### The command
```bash
SIM_BACKEND=cupy .venv/bin/python - <<'PY'
import numpy as np, logging, cupy as cp
logging.disable(logging.INFO)
from sim.connectivity import generate_spatial_connections_gpu
from sim.config import CoreSimConfig
from sim.backend import get_backend
xp, _ = get_backend()
for n in (12000, 15000):                      # 15000 is the last n that takes the NON-chunked branch
    cp.get_default_memory_pool().free_all_blocks()
    base = cp.get_default_memory_pool().used_bytes()
    pos = xp.asarray((np.random.default_rng(0).random((n, 3)) * 100).astype(np.float32))
    tr  = xp.asarray(np.zeros(n, dtype=np.int32))
    try:
        m = generate_spatial_connections_gpu(n, 16, pos, tr, CoreSimConfig(), log_fn=lambda *a, **k: None)
        peak = cp.get_default_memory_pool().total_bytes()
        print("n=%-6d OK   nnz=%-9d peak_pool=%.2f GB" % (n, m.nnz, (peak - base) / 1e9))
    except Exception as e:
        print("n=%-6d FAIL %s: %s" % (n, type(e).__name__, str(e)[:90]))
    del pos, tr
    cp.get_default_memory_pool().free_all_blocks()
PY
```

### Pre-registered prediction
**n=15000 on the non-chunked branch will exceed ~8 GB of pool and may OOM on a card with other work.** If so the
threshold is stale for the argsort implementation and should come DOWN (the chunked path is now known-correct and
tested, so lowering it is cheap and safe).

### Outcomes
| result | action |
|---|---|
| n=15000 **OOMs**, or peak > ~8 GB | Lower the threshold in `sim/connectivity.py:132` to where peak stays under ~4 GB (start at 8000, measure). Add the measured peaks to the comment there, replacing the argpartition-era reasoning. |
| n=15000 **fits comfortably** (< 4 GB peak) | The threshold is fine. Record the measurement in the comment so nobody re-raises this, and close D-1 as REFUTED. |
| n=12000 already OOMs | The problem is worse than flagged — the non-chunked branch is unusable well below the threshold. Escalate to HIGH and lower the threshold aggressively. |

### Related items on the SAME never-previously-executed path (do them in the same session)
- **The VRAM budget is derived from a cost model the code no longer uses.** `connectivity.py:496,515,521,527`
  justify `60 bytes per chunk row` and the 35% factor by "argpartition internals (thrust sort)". The shipped
  implementation is `cp.argsort(perturbed, axis=1)[:, -k:]`, whose int64 `(chunk_n, n)` output is 8n bytes/row and
  which the `[:, -k:]` slice keeps alive as its `.base` until `del top_k_indices`. Re-derive the constant against
  argsort and update the comments. **No large allocation needed — this is arithmetic plus a measurement at modest
  n**, so it can be done any time; it is listed here only because it belongs with D-1.
- **Latent OOM at very large n.** `chunk_size = max(64, ...)` floors at 64 rows, but the fallback guard tests only
  a SINGLE row against 25% of free VRAM. At n=500000 with 1 GB free the guard does not fire and peak is ~1.92 GB.
  **This one needs only ~2 GB and was already runnable**; if it has not been done by the time D-1 runs, do it
  alongside.

### Provenance
- The path had **never executed successfully in the repo's history** — a `NameError` on the first chunk predates
  the module extraction. Fixed 2026-07-31 (`4ac14cd7`), regression test in
  `tests/test_spatial_connectivity_chunked.py` (verified to fail on reintroduction).
- Backend portability fixed separately (`7c96ea30`): all four spatial generators were dead under
  `SIM_BACKEND=numpy`.
- These residuals come from the adversarial review of that fix, which correctly judged my original verification
  inadequate: it ran at n=300/1200 where `chunk_size` evaluates to n, so `num_chunks == 1` and chunking was never
  exercised at all.
