---
type: finding
status: live
date: 2026-08-19
mechanism: production-chat-cupy-backend-correctness
artifacts:
  - research/findings/raw/_prod_chat_cupy_fix/evidence.json
  - research/findings/raw/_prod_chat_cupy_fix/cupy_chat_before_crash.txt
  - research/findings/raw/_prod_chat_cupy_fix/cupy_chat_after_fix.txt
  - research/findings/raw/_prod_chat_cupy_fix/gate_selftest.txt
gate: tests/test_set_pathway_weights_backend_safe.py
---
# The production GPU chat path was 400-crashing: `set_pathway_weights` called SciPy `.tocoo()` on a cupy-hybrid CSR

**One line.** The default `/api/brain-chat` turn — `tiny-demo` + the spiking Qwen mouth — returned HTTP **400
on the cupy (GPU) backend** for every request, because building the brain called `set_pathway_weights(...,
add_missing=True)` which ran SciPy's `.tocoo()` on a CSR whose `.data` was a cupy array, raising
`TypeError: Implicit conversion to a NumPy array is not allowed`. Fixed in [`sim/bridge.py`](sim/bridge.py) by
rebuilding the COO from host CSR arrays; the GPU chat now returns **200** with a real spiking-forward answer.
Guarded by a CPU-runnable regression gate that fails on the old code.

## How it was found (and why it was invisible)
While re-verifying one affect faculty (ledger #13, mood→manner) on the GPU — the one row the observe-vs-drive audit
could not exercise on numpy — the cupy chat crashed at brain load. The audit's whole main pass, and the
onebrain-parser bind that introduced this call (board #43), were verified on **numpy only**. So the cupy production
path had never been executed end-to-end, and a GPU-only crash in a *wired, on-by-default, "verified"* faculty
shipped undetected. This is exactly the residual the 2026-08-11 production-integration entry named: the static PI
gate proves a path is *reachable + default-on*, not that it *runs without crashing on the production backend*.

## Root cause (`sim/bridge.py:4823`, the `add_missing=True` branch)
`self.cp_connections` arrives at `set_pathway_weights("onebrain_parser", …)` as a **SciPy CSR whose `.data` is a
cupy array** (a hybrid container built by the onebrain-parser pool bind). The branch did:
```python
existing_coo = self.cp_connections.tocoo(copy=False)   # SciPy's pure-Python tocoo -> np.array(cupy .data) -> TypeError
```
SciPy's `.tocoo()` calls `np.array()` on the cupy `.data`, which CuPy forbids. Reached on **every** production GPU
chat build: `tiny-demo` → `BrainConversationalAgent` → `make_pool1_onebrain_composer` → `_bind_parser_onto_pool`.
Full stack in [`cupy_chat_before_crash.txt`](research/findings/raw/_prod_chat_cupy_fix/cupy_chat_before_crash.txt).

## The fix
The function already decodes the host CSR arrays (`indptr`/`indices`/`data`) higher up (bridge.py:4769-4771). The
fix rebuilds the existing COO from those instead of calling `.tocoo()`:
```python
existing_row_host = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr).astype(np.int64))
existing_col_host = np.asarray(indices, dtype=np.int64)
existing_data_host = np.asarray(data)   # current (updated) host weights
# ... then cp.asarray(...) to the backend for the concatenate + COO->CSR rebuild.
```
This is **equivalent on numpy** (CSR order is preserved, so the rebuilt COO is identical to the old `.tocoo()`) and
**correct on cupy** (never routes cupy data through SciPy). The `_remap_gate_indices_after_rebuild` call now takes
the same host-derived backend arrays.

## Verification (see [`evidence.json`](research/findings/raw/_prod_chat_cupy_fix/evidence.json))
- **Cupy production chat: 400 → 200.** A pure-default `brain_chat(BrainChatRequest(session, message))` on
  `SIM_BACKEND=cupy` now returns `status_code=200`, `answer="The dog chased the cat. The cat eats fish."`,
  `renderer="off-bridge Qwen-0.5B (spiking forward)"` —
  [`cupy_chat_after_fix.txt`](research/findings/raw/_prod_chat_cupy_fix/cupy_chat_after_fix.txt).
- **Regression gate is legitimate.** [`tests/test_set_pathway_weights_backend_safe.py`](tests/test_set_pathway_weights_backend_safe.py)
  wraps `cp_connections` in a proxy whose `.tocoo()` raises the exact cupy error while every other attribute
  forwards to a real numpy CSR: it **passes on the fix** and **fails on the old code** with that error, and runs
  on a GPU-less CI. Both directions recorded in
  [`gate_selftest.txt`](research/findings/raw/_prod_chat_cupy_fix/gate_selftest.txt).
- **No regression.** `tests/test_determinism.py` → 9 passed. The 3 failures in the older
  `tests/test_set_pathway_weights.py` are **pre-existing** cupy-only tests (they call `.get()` on numpy int) that
  fail identically on the pre-edit code — not caused by this change.

## The class (why the gates missed it) and the follow-ups
The bug is a member of a class the record already flags: **a default-on production faculty exercised only on numpy,
so a cupy-backend crash ships.** The observe-vs-drive audit and the onebrain-parser verification both ran on numpy.
Follow-ups (filed, not done here):
1. A **GPU-present smoke of the default chat** (`brain_chat` returns 200 on cupy) that runs when a GPU is available
   — the class-level guard the specific gate cannot provide on CPU CI.
2. Understand **why `cp_connections` is a SciPy-hybrid on the cupy backend** in the onebrain-parser pool (the API
   fix is defensive and the end-to-end turn works, but a scipy container on cupy is worth understanding).
3. Make the 3 cupy-only tests in `test_set_pathway_weights.py` backend-agnostic so they run in CPU CI.
