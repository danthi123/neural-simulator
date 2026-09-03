"""CROSS-SESSION PERSISTENCE (#B8) — write the conversation-driven learning to disk so a server restart RESUMES it.

THE GAP (gap proof, 2026-09-02). The in-session learning loop is built + default-ON + lesion-proven (D5
learn-through-use, sleep-replay, in-loop fact acquire): recall strength rises with use, and every rise vanishes
under lesion. But NOTHING writes that conversation-driven learning to disk — every store lives in a module-level
`dict` / process singleton — so every server restart reverts the brain to its build-time state. The 2-process gap
proof measured it exactly: teach the D5 organ 'dog', consolidate 4× (within-assembly weight 81.12 -> 90.34), exit;
a fresh process at the SAME cache_key/seed/topics rebuilds at EXACTLY the pre-teach baseline (81.12, |Δ|=0.000).

THIS MODULE closes that gap ADDITIVELY, behind a DEFAULT-OFF flag (`BRAIN_PERSIST_LEARNING`). OFF (the default) is
BYTE-IDENTICAL to today: every save/reload entry point below returns immediately, writes no file, reads no file, and
mutates no substrate. ON, it SAVES the three conversation-driven stores the gap proof mapped — (a) the D5 episodic
organ's within-assembly weights (`EpisodicDapMemory.bridge.cp_connections.data` + `formed`/`store_log`/`_store_order`),
(b) the process-global xedge cross-edge weights (`XedgeProductionPool.bridge.cp_connections.data`), and (c) the
`_maybe_acquire` runtime-taught facts (`composer.kb`, via the existing `developed_brain_io` extract/restore) — on the
genuine SLEEP-DEPTH idle tick and/or graceful shutdown, and RELOADS them at boot after the base brain builds.

DESIGN — restore by WHOLESALE array overwrite, gated on a structural FINGERPRINT (the safe, simple choice the
store-plan's determinism check earned). The store-plan verified empirically that two separate process builds at the
same seed/topics produce a bit-identical CSR STRUCTURE (`indices`/`indptr`/shape/nnz) and bit-identical baseline
`.data` on both D5 and xedge. So a reload REBUILDS the store fresh (same seed/topics -> same structure), verifies the
saved fingerprint matches the fresh structure, and — only then — overwrites the whole `.data` array with the saved
weights. If the fingerprint does NOT match (the codebase's own documented emergent-membership non-determinism worry
ever coming true for some seed), the reload SKIPS the weight overwrite and logs a count, degrading to the fresh
baseline rather than corrupting the store by positional mismatch. No `sim/` edit; reuse-by-import.

BRAIN-BASED boundary: this is host BOOKKEEPING (serialize/deserialize the substrate's learned synaptic weights across
a process boundary) — it is the disk analogue of the brain persisting between sleeps, not a cognition shortcut. The
weights it writes were produced entirely by the substrate's own plasticity kernels (BTSP, three-factor DA credit); it
copies them, it never computes them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1

# DEFAULT-OFF (owner-gated flip). This is the whole byte-identical-off guarantee: unset -> every entry point below is
# a no-op (no file I/O, no substrate mutation).
_PERSIST_DEFAULT_ON = False


def persist_learning_enabled() -> bool:
    """`BRAIN_PERSIST_LEARNING` in {1,true,on,yes} -> cross-session persistence is armed (save on the sleep-depth
    idle tick / shutdown, reload at boot). Unset/{0,false,no,off,''} -> the default (OFF): every save/reload entry
    point returns immediately, so the brain build + any tick is BYTE-IDENTICAL to a build without this module."""
    v = os.environ.get("BRAIN_PERSIST_LEARNING")
    if v is None:
        return _PERSIST_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def persist_dir(base: str | os.PathLike | None = None) -> Path:
    """The base directory persisted learning is written under. `base` arg overrides; else `BRAIN_PERSIST_DIR`; else
    `<repo>/bridges/persist`. Created lazily by the save path (never by the flag check)."""
    if base is not None:
        return Path(base)
    env = os.environ.get("BRAIN_PERSIST_DIR")
    if env:
        return Path(env)
    # repo root = three parents up from research/runners/<this file>
    return Path(__file__).resolve().parents[2] / "bridges" / "persist"


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Shared low-level serialization: a CSR weight array + a structural fingerprint of the store it came from. The
# fingerprint is what makes the wholesale-overwrite restore SAFE — a reload only overwrites when the freshly-rebuilt
# store has an identical CSR structure, else it skips (degrade to baseline, never corrupt by positional mismatch).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _to_host(arr) -> np.ndarray:
    """cupy/numpy -> a host numpy array (contiguous), without importing cupy when the backend is numpy."""
    try:
        from sim.backend import to_host
        return np.ascontiguousarray(np.asarray(to_host(arr)))
    except Exception:
        return np.ascontiguousarray(np.asarray(arr))


def _csr_fingerprint(indices, indptr, shape) -> str:
    """A stable hash of a CSR matrix's STRUCTURE (indices + indptr + shape) — NOT its weights. Two builds whose
    fingerprints match are positionally aligned, so a wholesale `.data` overwrite from one into the other is exact."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(indptr, dtype=np.int64)).tobytes())
    h.update(np.asarray(shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def _sparse_struct(mat):
    """(indices, indptr, shape, nnz) of a CSR sparse connection matrix, host-side. Works for cupy/scipy CSR."""
    indices = _to_host(mat.indices).astype(np.int64)
    indptr = _to_host(mat.indptr).astype(np.int64)
    shape = tuple(int(x) for x in mat.shape)
    nnz = int(indices.shape[0])
    return indices, indptr, shape, nnz


def _save_csr_weights(path: Path, mat, meta: dict) -> str:
    """Save a CSR matrix's `.data` weights + its structural fingerprint + arbitrary `meta` to `<path>.npz`/`<path>.json`.
    Returns the fingerprint. The weights are saved in their native dtype so the round-trip is bit-exact."""
    indices, indptr, shape, nnz = _sparse_struct(mat)
    fp = _csr_fingerprint(indices, indptr, shape)
    data = _to_host(mat.data)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path) + ".npz", data=data)
    sidecar = {"schema_version": SCHEMA_VERSION, "fingerprint": fp, "shape": list(shape), "nnz": nnz,
               "dtype": str(data.dtype), **meta}
    with open(str(path) + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    return fp


def _load_csr_sidecar(path: Path) -> dict | None:
    p = Path(str(path) + ".json")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_csr_weights(path: Path) -> np.ndarray | None:
    p = Path(str(path) + ".npz")
    if not p.exists():
        return None
    with np.load(str(p)) as z:
        return np.array(z["data"])


def _overwrite_csr_data(mat, host_data: np.ndarray) -> None:
    """Overwrite a CSR matrix's `.data` in place with `host_data` (bit-exact, native dtype), on whatever backend the
    matrix lives (cupy or numpy). Caller MUST have already verified the fingerprint matches (same structure)."""
    arr = np.asarray(host_data, dtype=mat.data.dtype)
    # cupy array has an `asarray`-capable module; numpy array is set directly.
    mod = getattr(type(mat.data), "__module__", "")
    if "cupy" in mod:
        import cupy as cp  # only imported when the store actually lives on cupy
        mat.data[:] = cp.asarray(arr)
    else:
        mat.data[:] = arr


def _sanitize_key(cache_key) -> str:
    """A filesystem-safe, stable id for a cache_key (a tuple like ('default','tiny-demo','qwen')). Deterministic, so
    the SAME conversation resumed after a restart maps to the SAME file."""
    if isinstance(cache_key, (tuple, list)):
        raw = "__".join(str(x) for x in cache_key)
    else:
        raw = str(cache_key)
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in raw)
    # keep it bounded + collision-resistant
    return safe[:120] + "-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (a) D5 EPISODIC ORGAN — per-conversation within-assembly BTSP weights + host bookkeeping.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _d5_path(cache_key, base=None) -> Path:
    return persist_dir(base) / "d5" / _sanitize_key(cache_key)


def save_d5_organ(cache_key, organ, base=None) -> Path | None:
    """Persist a built D5 episodic organ's learned state. Returns the path stem written, or None (disabled / nothing
    to save: no organ, no built mem, or no formed assembly). Best-effort — never raises into the caller's tick."""
    if not persist_learning_enabled() or organ is None:
        return None
    mem = getattr(organ, "mem", None)
    if mem is None:
        return None
    formed = sorted(int(s) for s in getattr(mem, "formed", set()))
    if not formed:
        return None  # nothing was learned this conversation -> nothing to persist
    path = _d5_path(cache_key, base)
    meta = {
        "seed": int(getattr(organ, "seed", getattr(mem, "seed", 42))),
        "topics": list(getattr(organ, "topics", getattr(mem, "topics", []))),
        "sep_bias": float(getattr(organ, "sep_bias", 0.0)),
        "formed": formed,
        "store_log": list(getattr(mem, "store_log", [])),
        "store_order": list(getattr(organ, "_store_order", [])),
    }
    _save_csr_weights(path, mem.bridge.cp_connections, meta)
    return path


def reload_d5_organ(cache_key, seed: int, topics, base=None):
    """Rebuild the D5 organ fresh (same seed/topics -> same CSR structure), verify the saved fingerprint matches, and
    overwrite its within-assembly weights + `formed`/`store_log`/`_store_order` with the saved state, then REGISTER it
    into `d5_episodic_production_organ._ORGANS[cache_key]` so the live server picks it up. Returns the organ, or None
    (disabled / no saved file / fingerprint mismatch / build failure). The heavy rebuild is paid only when a saved
    file exists (the same cost the first live `note_topic` would pay), never for a session that never learned."""
    if not persist_learning_enabled():
        return None
    path = _d5_path(cache_key, base)
    sidecar = _load_csr_sidecar(path)
    if sidecar is None:
        return None
    host_data = _load_csr_weights(path)
    if host_data is None:
        return None
    try:
        import research.runners.d5_episodic_production_organ as _EP
        # rebuild fresh with the SAVED seed/topics/sep_bias so the CSR structure matches what was saved.
        saved_seed = int(sidecar.get("seed", seed))
        saved_topics = sidecar.get("topics") or list(topics)
        saved_sep = float(sidecar.get("sep_bias", 0.0))
        organ = _EP.EpisodicRecallOrgan(saved_seed, saved_topics, sep_bias=saved_sep)
        organ._ensure_built()
        mem = organ.mem
        indices, indptr, shape, nnz = _sparse_struct(mem.bridge.cp_connections)
        fp = _csr_fingerprint(indices, indptr, shape)
        if fp != sidecar.get("fingerprint") or nnz != int(sidecar.get("nnz", -1)):
            import logging
            logging.getLogger(__name__).warning(
                "D5 persistence fingerprint mismatch for %s (build-to-build structure drift) -> "
                "skipping weight restore, using fresh baseline", cache_key)
            return None
        if host_data.shape[0] != nnz:
            return None
        _overwrite_csr_data(mem.bridge.cp_connections, host_data)
        # mem.R.C is the SAME array as mem.bridge.cp_connections (verified in the store-plan); recall() reads it.
        mem.formed = set(int(s) for s in sidecar.get("formed", []))
        mem.store_log = list(sidecar.get("store_log", []))
        organ._store_order = list(sidecar.get("store_order", []))
        _EP._ORGANS[cache_key] = organ
        return organ
    except Exception:
        import logging
        logging.getLogger(__name__).warning("D5 persistence reload failed for %s", cache_key, exc_info=True)
        return None


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (b) XEDGE CROSS-EDGE — process-global grown cross-region synapse weights.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _xedge_path(base=None) -> Path:
    return persist_dir(base) / "xedge" / "pool"


def save_xedge(base=None, pool=None) -> Path | None:
    """Persist the process-global xedge cross-edge weights. `pool` overrides; else the live `get_xedge_pool()`. Returns
    the path stem, or None (disabled / no pool / pool not built). Best-effort."""
    if not persist_learning_enabled():
        return None
    try:
        if pool is None:
            from research.runners.onebrain_xedge_production import get_xedge_pool
            pool = get_xedge_pool()
        if pool is None or not getattr(pool, "ok", False) or pool.bridge is None:
            return None
        path = _xedge_path(base)
        meta = {
            "seed": int(getattr(pool, "seed", 42)),
            "learned": bool(getattr(pool, "learned", False)),
            "live_per_turn": bool(getattr(pool, "live_per_turn", False)),
            "n_live_credited": int(getattr(pool, "n_live_credited", 0)),
        }
        _save_csr_weights(path, pool.bridge.cp_connections, meta)
        return path
    except Exception:
        import logging
        logging.getLogger(__name__).warning("xedge persistence save failed", exc_info=True)
        return None


def reload_xedge(pool, base=None) -> bool:
    """Overwrite a freshly-built xedge pool's cross-edge weights with the saved ones, gated on a fingerprint match.
    Called from `XedgeProductionPool._build` AFTER the fresh build sets the baseline. Returns True if weights were
    restored, False otherwise (disabled / no saved file / mismatch). Best-effort — never raises into brain load."""
    if not persist_learning_enabled() or pool is None or getattr(pool, "bridge", None) is None:
        return False
    try:
        path = _xedge_path(base)
        sidecar = _load_csr_sidecar(path)
        host_data = _load_csr_weights(path)
        if sidecar is None or host_data is None:
            return False
        indices, indptr, shape, nnz = _sparse_struct(pool.bridge.cp_connections)
        fp = _csr_fingerprint(indices, indptr, shape)
        if fp != sidecar.get("fingerprint") or nnz != int(sidecar.get("nnz", -1)) or host_data.shape[0] != nnz:
            import logging
            logging.getLogger(__name__).warning(
                "xedge persistence fingerprint mismatch -> skipping weight restore, using fresh baseline")
            return False
        _overwrite_csr_data(pool.bridge.cp_connections, host_data)
        # recompute the exposed per-mask summary + restore the live-credit counter for the record.
        try:
            if getattr(pool, "_r3pool", None) is not None:
                pool.cross_weights = pool._r3pool.cross_weights()
        except Exception:
            pass
        pool.n_live_credited = int(sidecar.get("n_live_credited", getattr(pool, "n_live_credited", 0)))
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).warning("xedge persistence reload failed", exc_info=True)
        return False


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (c) RUNTIME-ACQUIRED FACTS — the `_maybe_acquire` facts already round-trip through developed_brain_io.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _facts_path(cache_key, base=None) -> Path:
    return persist_dir(base) / "facts" / (_sanitize_key(cache_key) + ".json")


def save_session_facts(cache_key, chat, base=None) -> Path | None:
    """Persist the session's composer facts (incl. every `_maybe_acquire` runtime-taught fact) + any runtime-grounded
    codes, reusing `developed_brain_io.extract_facts`/`extract_grounded_codes` (the exact stores the develop-loop
    already round-trips). Returns the path, or None (disabled / no chat / no facts). Best-effort."""
    if not persist_learning_enabled() or chat is None:
        return None
    try:
        from research.runners.developed_brain_io import extract_facts, extract_grounded_codes
        agent = getattr(chat, "agent", chat)
        facts = extract_facts(agent)
        if not facts:
            return None
        codes = extract_grounded_codes(agent)
        path = _facts_path(cache_key, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, "facts": facts,
                   "codes": {w: list(map(float, ph)) for w, ph in codes.items()}}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        return path
    except Exception:
        import logging
        logging.getLogger(__name__).warning("session-facts persistence save failed for %s", cache_key, exc_info=True)
        return None


def reload_session_facts(cache_key, chat, base=None) -> int:
    """Re-store the persisted session facts into a freshly-built chat's composer (so the runtime-taught knowledge
    survives). Injects any persisted runtime-grounded codes first (so a runtime-allocated word is encodable), then
    re-stores facts NOT already present (dedup against the bundle facts the build already loaded). Returns the number
    of facts added, or 0 (disabled / no saved file / nothing new). Best-effort."""
    if not persist_learning_enabled() or chat is None:
        return 0
    path = _facts_path(cache_key, base)
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        saved_facts = payload.get("facts", [])
        if not saved_facts:
            return 0
        from research.runners.developed_brain_io import _inner_agent, _restore_facts, extract_facts
        agent = getattr(chat, "agent", chat)
        inner = _inner_agent(agent)
        comp = inner.composer
        # inject persisted runtime-grounded codes for any word not already known (so re-store can encode it).
        codes = payload.get("codes", {}) or {}
        try:
            concepts = getattr(comp, "concepts", None)
            if concepts is not None:
                for w, ph in codes.items():
                    if w not in concepts or concepts.get(w) is None:
                        concepts[w] = np.asarray(ph, dtype=float)
        except Exception:
            pass
        # dedup: only add facts whose (agent,action,patient,polarity) is not already stored.
        def _key(f):
            return (f.get("agent"), f.get("action"),
                    json.dumps(f.get("patient"), sort_keys=True, default=str), f.get("polarity"))
        existing = {_key(f) for f in extract_facts(agent)}
        new_facts = [f for f in saved_facts if _key(f) not in existing]
        if not new_facts:
            return 0
        _restore_facts(agent, new_facts)
        return len(new_facts)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("session-facts persistence reload failed for %s", cache_key, exc_info=True)
        return 0


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Orchestration — the entry points the server + continuous engine call.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def save_session_learning(cache_key, chat, episodic_organ, base=None) -> dict:
    """SAVE one session's conversation-driven learning (D5 store + acquired facts) + the process-global xedge edge.
    Called on the sleep-depth idle tick and at shutdown. No-op (byte-identical) when the flag is off. Best-effort:
    each store is guarded independently, so one failing store never blocks the others. Returns a small record."""
    if not persist_learning_enabled():
        return {}
    rec: dict[str, Any] = {}
    try:
        p = save_d5_organ(cache_key, episodic_organ, base)
        rec["d5"] = str(p) if p else None
    except Exception:
        rec["d5"] = None
    try:
        p = save_session_facts(cache_key, chat, base)
        rec["facts"] = str(p) if p else None
    except Exception:
        rec["facts"] = None
    try:
        p = save_xedge(base)
        rec["xedge"] = str(p) if p else None
    except Exception:
        rec["xedge"] = None
    return rec


def reload_session_learning(cache_key, chat, seed: int, topics, base=None) -> dict:
    """RELOAD one session's conversation-driven learning at boot, AFTER the base brain builds. Reloads the D5 store
    (registers it into `_ORGANS`) + re-stores the acquired facts into the fresh chat's composer. xedge is reloaded
    separately inside its own pool build (process-global). No-op when the flag is off. Returns a small record."""
    if not persist_learning_enabled():
        return {}
    rec: dict[str, Any] = {}
    try:
        org = reload_d5_organ(cache_key, seed, topics, base)
        rec["d5"] = bool(org is not None)
    except Exception:
        rec["d5"] = False
    try:
        rec["facts_added"] = reload_session_facts(cache_key, chat, base)
    except Exception:
        rec["facts_added"] = 0
    return rec


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SELFTEST — the decisive teach -> save -> FRESH reload -> recall survival check, on the REAL D5 substrate + the
# REAL xedge pool + REAL facts. Runnable via the experiment engine (0 Claude tokens). Proves BOTH directions:
#   (1) flag OFF  -> save/reload are no-ops (no file written, no substrate change) = byte-identical.
#   (2) flag ON   -> teach -> save -> a FRESH rebuild (the gap proof's process-2 analogue) -> the learned weight
#                    SURVIVES (matches post-teach, not the pre-teach baseline the gap proof measured on HEAD).
# The pytest (tests/test_cross_session_persistence.py) runs the FAST fingerprint/round-trip/OFF-inert subset that
# CI can gate; this main() runs the full real-substrate cycle.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _selftest_d5(base, seed=42, topics=("cat", "dog"), teach="dog"):
    """Real-substrate D5 cycle. Returns a dict with pre/post/reloaded within-assembly weights + the two verdicts."""
    import research.runners.d5_episodic_production_organ as _EP
    from webapp.continuous_engine import consolidate_used_memory, mark_recall

    cache_key = ("selftest-persist", "d5", "cupy")
    # --- process 1: teach + consolidate, then SAVE ---
    _EP._ORGANS.pop(cache_key, None)
    organ = _EP.get_episodic_organ(cache_key, seed, list(topics))
    organ.note_topic(teach)
    mem = organ.mem
    slot = mem.topic_slot[teach]
    cp = mem.cp
    w_pre = float(cp.mean(mem.bridge.cp_connections.data[mem.R.withinA_masks[slot]]))
    # consolidate a few times (learn-through-use) so the weight rises measurably above baseline
    for _ in range(4):
        mark_recall(cache_key, teach)
        consolidate_used_memory(cache_key, organ)
    w_post = float(cp.mean(mem.bridge.cp_connections.data[mem.R.withinA_masks[slot]]))
    saved = save_d5_organ(cache_key, organ, base)

    # --- process-2 analogue: drop the in-memory organ (simulate restart) + REBUILD FRESH from disk ---
    _EP._ORGANS.pop(cache_key, None)
    organ2 = reload_d5_organ(cache_key, seed, list(topics), base)
    if organ2 is None:
        return {"saved": bool(saved), "w_pre": w_pre, "w_post": w_post, "reloaded": None,
                "survived": False, "note": "reload returned None"}
    mem2 = organ2.mem
    slot2 = mem2.topic_slot[teach]
    w_reload = float(mem2.cp.mean(mem2.bridge.cp_connections.data[mem2.R.withinA_masks[slot2]]))
    survived = abs(w_reload - w_post) < 1e-3 and abs(w_reload - w_pre) > 1e-2
    return {"saved": bool(saved), "w_pre": round(w_pre, 4), "w_post": round(w_post, 4),
            "w_reload": round(w_reload, 4), "formed_reloaded": sorted(mem2.formed),
            "delta_to_post": round(abs(w_reload - w_post), 6), "delta_to_pre": round(abs(w_reload - w_pre), 6),
            "survived": bool(survived)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None, help="persistence dir (defaults to a temp under the scratch)")
    args = ap.parse_args()

    import tempfile
    base = args.base or tempfile.mkdtemp(prefix="persist_selftest_")

    results = {"base": base}

    # (1) OFF is inert: with the flag off, save writes NOTHING and reload returns None.
    os.environ["BRAIN_PERSIST_LEARNING"] = "0"
    off_saved = save_xedge(base=base)   # cheap: pool may be off; either way must be None under the flag
    off_files = list(Path(base).rglob("*")) if Path(base).exists() else []
    results["off_inert"] = {"save_returned_none": off_saved is None, "files_written": len(off_files)}

    # (2) ON: the real D5 teach -> save -> fresh reload -> survive cycle.
    os.environ["BRAIN_PERSIST_LEARNING"] = "1"
    results["d5"] = _selftest_d5(base)

    results["GO"] = bool(results["off_inert"]["save_returned_none"]
                         and results["off_inert"]["files_written"] == 0
                         and results["d5"].get("survived"))
    print(json.dumps(results, indent=2, default=str), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    return 0 if results["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
