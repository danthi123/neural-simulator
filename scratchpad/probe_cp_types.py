"""Pre-prereg empirical probe for `research/runners/_gnw_continuous_branchpoint_verify.py`.

Committed per adversarial-review fix round 2026-09-24 (journal wf_16244c9d-ca5, agentId a6dd960fd33710903,
issue 3): the "32/32 live cp_* arrays captured" and "scipy CSR .copy() is independent" preconditions cited by
that runner's docstring/prereg were checked with this exact script before the file was written, but the script
was never committed -- it lived only in the session scratchpad and its runs.jsonl provenance rows were an
unstaged diff that would have been lost with the worktree. This is that script, unchanged in method, committed
alongside its own provenance so the preconditions are traceable rather than asserted from "see commit history".

Two independent checks, each printed and asserted:
  1. cp_completeness -- every `cp_*` attribute on a LIVE `_gnw_neural_swap_intention_derisk.build()` bridge that
     is not None is enumerated, and split into "would `_full_snapshot` capture it" (has both `.copy()` and
     `.shape`) vs "missing". This is the runtime claim G1 in the runner re-measures on every seed; this script
     is the one-off precondition check that motivated adding G1 as an executable gate rather than an assumption.
  2. csr_copy_independent -- a standalone scipy.sparse CSR mutate-after-copy check: `.copy()` must produce a
     structurally and numerically independent object, which is what G4 (`no_aliasing`) and the fork-restore path
     both rely on for `cp_connections`. This does not touch the substrate; it is a property of scipy itself,
     checked here rather than assumed from the library docs.

Usage: `SIM_BACKEND=numpy bash tools/memcap.sh 2 -- .venv/bin/python scratchpad/probe_cp_types.py`
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp

from research.runners._gnw_neural_swap_intention_derisk import build

# ── check 1: cp_completeness precondition ────────────────────────────────────────────────────────────────────
S = build(seed=42)
bridge = S["bridge"]
missing = []
included = []
for k, v in vars(bridge).items():
    if not k.startswith("cp_"):
        continue
    has_copy = hasattr(v, "copy")
    has_shape = hasattr(v, "shape")
    if has_copy and has_shape:
        included.append((k, type(v).__name__))
    else:
        missing.append((k, type(v).__name__, v if not hasattr(v, "__len__") else "..."))
print(f"[cp_completeness] included={len(included)} missing_from_full_snapshot_filter={len(missing)}")
non_none_missing = [m for m in missing if m[1] != "NoneType"]
for m in non_none_missing:
    print("  LIVE-BUT-MISSING (would be a G1 failure):", m)
print(f"[cp_completeness] of {len(missing)} missing names, {len(non_none_missing)} are LIVE "
      f"(non-None) -- the rest are disabled subsystems and are NOT a completeness failure")

# ── check 2: scipy CSR .copy() independence precondition (mutate-after-copy) ────────────────────────────────────
src = sp.random(50, 50, density=0.2, format="csr", random_state=0)
src.data[:] = np.arange(src.data.size, dtype=np.float64)
dup = src.copy()
same_object = dup.data is src.data
src.data[:] = -1.0  # mutate the ORIGINAL in place after copying
dup_unaffected = bool(np.all(dup.data != -1.0)) and bool(np.any(dup.data >= 0.0))
print(f"[csr_copy_independent] same_data_object={same_object} dup_unaffected_by_mutation={dup_unaffected}")

ok = (len(non_none_missing) == 0) and (not same_object) and dup_unaffected
print(f"[probe_cp_types] ALL PRECONDITIONS HOLD: {ok}")
if not ok:
    raise SystemExit(1)
