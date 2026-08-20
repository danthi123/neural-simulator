---
type: finding
status: live
date: 2026-08-20
mechanism: cupy-sim-step-backend-coercion
lane: infra-cupy
seeds: [42]
instrument: byte-identity determinism (numpy) + a cupy organ-step reproduction + a full-faculty cupy chat turn, all adversarially re-verified
artifacts:
  - research/findings/raw/_cupy_step_fix/evidence.json
gate: tests/test_production_chat_gpu_smoke.py
---
# The full-faculty GPU chat server hung on a cupy sim-step crash (scipy connections on the cupy backend) — now fixed byte-identically

Artifact: research/findings/raw/_cupy_step_fix/evidence.json

**One line.** The full-faculty chat server HUNG on the cupy (GPU) backend because a co-resident organ (the SURPRISE
organ) writes its connection matrix with `import scipy.sparse; bridge.cp_connections = sp.csr_matrix(...)` — a SciPy
matrix on the cupy backend — so the simulation step's `effective_connections_matrix.T @ fired_2col` became a
SciPy-@-cupy matmul that raises "Implicit conversion to a NumPy array is not allowed" (caught + logged CRITICAL, so
the turn silently ground on garbage). Fixed at the step boundary with a backend-coercion that is a **byte-identical
no-op on numpy**; the full-faculty fluent GPU server now runs. (Found + fixed via a parallel-attack workflow lane,
adversarially verified, orchestrator-re-verified.)

## Root cause + fix
`_run_one_simulation_step` now calls `_ensure_connections_on_backend()` FIRST: if `cp_connections` is already the
active backend's `csp` sparse type it returns immediately (on numpy `csp` IS `scipy.sparse`, so a scipy matrix
already matches → never rebuilt → byte-identical); otherwise it rebuilds a backend `csr_matrix` from the CSR arrays
via `cp.asarray` (which accepts host OR device data, covering both the host-data scipy case and a scipy-container-
holding-cupy-`.data` hybrid). It stores the backend-native matrix back, so later steps are a no-op. This is generic —
it fixes ANY co-resident organ whose `cp_connections` is foreign, not just surprise. Also: the 3 mock-stat
`// self._stats_sync_counter` sites now guard with `max(1, ...)` (a ZeroDivision edge). This is the STEP-side sibling
of the earlier `tocoo` BUILD-crash fix (same scipy-hybrid class).

## Verification (see evidence.json)
- **numpy byte-identity:** `tests/test_determinism.py` 9/9 (run 3×) AND a sha256 of a 200-step numpy firing hash
  matches pre-fix — the transpose-matvec computes the same conductances on numpy.
- **cupy step fixed:** the surprise organ builds + steps on cupy (0.0 Hz confirm vs 4.98 Hz contradict — correct
  discrimination); load-bearing proof: monkeypatching the coercion to a no-op reproduces the exact crash.
- **full-faculty cupy turn:** `webapp.server.brain_chat` on `SIM_BACKEND=cupy` returns a grounded answer over two
  turns; `grep 'Implicit conversion'` = 0.

## Impact
Unblocks the **full-faculty fluent GPU server** (all organs, not the reduced-faculty demo), the **continuous tick on
cupy**, and **idle BTSP consolidation** (which is cupy-gated). The GPU-smoke test (`tests/test_production_chat_gpu_smoke.py`)
guards the default chat on cupy; extending it to a full-faculty turn is a small follow-up.
