"""Capture the flag-OFF identity golden for step 0a of `docs/drafts/bg-tonic-loop/PACKET.md`.

Run ONCE on the UNEDITED merged tree (research/bg-tonic-loop @ c29c91071) BEFORE any
`BRAIN_BG_PRESET_CELLS` / `BRAIN_BG_BOUNDED_COND` switch code exists. `tests/test_bg_tonic_loop_identity.py`
rebuilds the same cases on the edited tree with both switches UNSET and asserts every hash matches:
every instance array of the bridge (sha256 over dtype+shape+bytes) plus the global RNG state
(numpy legacy MT19937 + python `random`) after build, through BOTH build paths — a plain
`build_selector_bridge` call and a full `NeuralSeamSelector.ensure_built()` (neural-seam
core_config_updates + gates + baseline + warmup).

Usage:  SIM_BACKEND=numpy SIM_NO_PROVENANCE=1 python -u tests/_capture_bg_loop_identity_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# flag-OFF capture: the switches must be unset regardless of the ambient environment
os.environ.pop("BRAIN_BG_PRESET_CELLS", None)
os.environ.pop("BRAIN_BG_BOUNDED_COND", None)

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

GOLDEN_PATH = os.path.join(_HERE, "_bg_loop_identity_golden.json")


def _sha_array(v) -> str:
    from sim.backend import to_host
    a = np.ascontiguousarray(np.asarray(to_host(v)))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rng_state_hash() -> str:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(np.asarray(keys, dtype=np.uint32).tobytes())
    h.update(repr((int(pos), int(has_gauss), float(cached))).encode())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def snapshot(bridge) -> dict:
    out = {}
    for name in sorted(vars(bridge)):
        v = vars(bridge)[name]
        if hasattr(v, "tobytes") and hasattr(v, "dtype"):
            out[name] = _sha_array(v)
    out["rng_state"] = rng_state_hash()
    return out


def build_plain(seed: int):
    from research.runners._vocal_action_selector_gate import build_selector_bridge
    return build_selector_bridge(int(seed))


def build_seam(seed: int):
    from research.runners.bg_action_selection_production_organ import NeuralSeamSelector
    org = NeuralSeamSelector(seed=int(seed))
    org.ensure_built()
    return org._bridge


CASES = {
    "bridge_plain_s9": (lambda: build_plain(9)),
    "bridge_plain_s16": (lambda: build_plain(16)),
    "seam_s9": (lambda: build_seam(9)),
    "seam_s16": (lambda: build_seam(16)),
}


def capture() -> dict:
    from sim.backend import get_backend
    _xp, backend = get_backend()
    cases = {}
    for name, builder in CASES.items():
        cases[name] = snapshot(builder())
        print(f"captured {name}: {len(cases[name])} hashes")
    return {"backend": backend,
            "device": "cpu" if backend == "numpy" else "gpu",
            "switches": "BRAIN_BG_PRESET_CELLS unset, BRAIN_BG_BOUNDED_COND unset",
            "cases": cases}


if __name__ == "__main__":
    golden = capture()
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN_PATH} ({len(golden['cases'])} cases, backend {golden['backend']})")
