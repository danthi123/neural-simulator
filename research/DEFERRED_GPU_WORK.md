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

---

## N-1 · Lane D: give the rate-Hebbian rule an INPUT-DEPENDENT fixed point (Miller-MacKay subtractive normalization)

**Status:** READY TO BUILD · no GPU headroom needed (the regression suite runs in ~45 s beside the crux) · raised
2026-07-31 · owner: autonomous

### The evidence that forces it
With the drive fixed (init weight 120, `hebb_max` 1200, drive 1200) lane D's V1 fires and the gates SPLIT:

| gate | measured | required | reads |
|---|---|---|---|
| `rsa_vs_host` | **0.827** | 0.60 | **PASSES** |
| `orient_decode` | 0.281 | host ref 0.984 | fails |
| OSI post | 0.195 | 0.50 | fails |

RSA is dominated by WHICH inputs a unit listens to; OSI and decode need a GRADED response. So the rule selects the
right support and cannot grade it — exactly what `w_j* = hebbian_max_weight` predicts, since that fixed point is
INPUT-INDEPENDENT and can only express a binary partition {gated → bound} vs {ungated → decayed}.
Drive is excluded as the cause: firing rate rose 9.6× (0.00047 → 0.00450) while OSI did not move at all
(0.199 → 0.195 across every condition).

### The build — mirror a mechanism this engine ALREADY has and has PROVEN
Do NOT write Oja. The engine already ships **Miller-MacKay 1994 subtractive normalization** as
`btsp_mean_subtract` (`sim/config.py:393`, implemented `sim/bridge.py:8307-8330`), validated in the gap#5
consolidation work. It is also the better-motivated mechanism for THIS problem: Miller-MacKay is literally the
orientation/ocular-dominance development model, whereas Oja is a PCA rule.

Add `hebbian_mean_subtract: float = 0.0` mirroring it exactly:
1. compute the raw Hebbian increment `_h` for gated synapses
2. `scatter_add` `_h` and a count into per-POSTsynaptic-cell buffers (`cupyx.scatter_add`, `cp.add.at` fallback)
3. subtract each cell's mean increment, so `sum_j dw_ij = 0` BY CONSTRUCTION
4. `0.0 ⇒ OFF ⇒ byte-identical` (the existing flag's contract)

Why this grades what the current rule cannot: forcing the per-cell increment sum to zero makes afferents COMPETE,
so a synapse can only grow at another's expense. The fixed point then depends on relative input correlation
instead of collapsing to the bound.

### Test at the KNOWN operating point (do not re-derive it)
```bash
SIM_BACKEND=numpy .venv/bin/python -m research.runners._b1_v1_selforg_onbridge_derisk   --seeds 42 --init-weight-mean 120 --hebb-max 1200 --drive-pA 1200   --coact-thresh 0.0002 --homeo-target 0.002 --n-inh 0 --dev-steps 6000 --out <path>
```

### Pre-registered prediction and kill criterion
**Prediction:** `orient_decode` rises from 0.281 toward the host reference and OSI rises above 0.50, while
`rsa_vs_host` stays ≥ 0.60. **KILL CRITERION:** if OSI and decode do NOT move with mean-subtract on, the
input-independent-fixed-point diagnosis is REFUTED and the residual is elsewhere — record that, do not retune.

### Traps already paid for, do not re-pay
- `--hebb-max` defaults to **70**: any init weight above it is clipped on step 1, which silently ERASES the
  independent variable (three weights read identically). Raise the bound with the weight, always.
- `--drive-pA` exists and defaults to 1200; a lowercase-only flag scan misses it.
- The drive is IMAGE-MODULATED (`image * drive_pA`), so a uniform-current probe does not match the runner.
- Weight, not drive, is binding: at weight 20 a 10× drive increase leaves `v1_rate` at exactly 0.00000.
