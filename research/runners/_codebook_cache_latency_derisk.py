"""Codebook-caching latency de-risk (board #192 -- decouple codebook caching from the DG sparse index).

CONTEXT. The 100k production verify (`research/findings/raw/_knowledge_scale_100k_production_verify.json`)
is GO on recall (1.0) and moat (0 confab) but NO-GO on LATENCY: warm routed-recall median ~1.1 s (p95 ~2.2 s)
vs a <1 s bar. A cProfile of the per-shard hot loop (one routed shard = one RFPhasorComposer holding ~200 facts
over the full V~24k shared codebook) shows the single largest cost bucket is NOT the spiking resonate (~27%) but
the O(V) CLEANUP codebook operations (~40%), dominated by REBUILDING the codebook matrix from the `concepts`
dict on every query:
  * rf_phasor_composer.py:844  `_cleanup_all`  cb = np.stack([np.exp(2j*pi*self.concepts[w]) for w in words])
      -- the (V, D) phasor matrix, rebuilt 2x/query (once per cue role), ~0.19 s/query (1.545s tottime / 8q).
  * rf_phasor_composer.py:763  `_cleanup` (single, used by _render->unbind) -- a Python per-word mean(cos())
      loop over ALL V words, ~0.10 s/query (0.777s tottime / 8q).

This is EXACTLY the caching the DG sparse-index path already gets for free via `_ensure_dg_index` (which caches
`_dg_codebook`), which is why turning the index ON previously showed a ~25% latency win that a prior finding
attributed to codebook-CACHING as a side-effect, NOT the sharding/index. Board #192 = expose that caching on the
DEFAULT path, independent of the DG index (whose separate NO-GO came from a resonate noise-calibration mismatch,
unrelated to caching).

THE LEVER (low-risk, byte-identical, default-on-able). Cache the (V, D) codebook ONCE per vocab state
(rebuild only when len(words) changes -- the same invalidation rule `_dg_built_V` already uses) and:
  * `_cleanup_all` (words is None): matmul against the cached PHASOR codebook instead of re-stacking it.
  * `_cleanup`   (words is None): one vectorized cos-sum argmax against the cached FRACTIONAL codebook instead
    of the Python per-word loop.
Answers are byte-identical: the cached matrix IS the matrix the current code rebuilds; argmax is over the same
values. This de-risk realizes the lever as an RFPhasorComposer SUBCLASS (NO sim/ or runner edit) and:
  (1) VERIFIES byte-identity: for many cues, cached answers == baseline answers (query_patient / ask_yes_no /
      query_agent), 0 mismatches required;
  (2) MEASURES latency baseline vs cached at ~one-routed-shard scale (V~24k, K~200, D=128).

Run (CPU, bounded, RSS < ~1 GB):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._codebook_cache_latency_derisk \
      --json research/findings/raw/_codebook_cache_latency_derisk.json
For the REAL 100k bundle end-to-end confirmation (heavier), see the gpu_queue/pool command in the task return.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.tiered_fact_store import encode_fast


class CachedCodebookRFPhasorComposer(RFPhasorComposer):
    """RFPhasorComposer + a cached (V, D) cleanup codebook (phasor + fractional), rebuilt only on vocab growth.

    Overrides ONLY the two full-vocabulary cleanup paths (`words is None`); every `words`-scoped call (polarity,
    trace) and the spiking-cleanup / sparse-index branches fall through to the parent unchanged. The cached
    matrices are exactly what the parent rebuilds each call, so decode is byte-identical."""

    def _ensure_codebook_cache(self):
        V = len(self.words)
        if getattr(self, "_cb_cache_V", -1) == V and getattr(self, "_cb_frac", None) is not None:
            return
        # fractional-cycle codebook (V, D) aligned to self.words (the single-cleanup convention)
        self._cb_frac = np.stack([np.asarray(self.concepts[w], dtype=float) for w in self.words])
        # phasor codebook (V, D) (the batched-cleanup convention)
        self._cb_z = np.exp(2j * np.pi * self._cb_frac)
        self._cb_cache_V = V

    def _cleanup(self, rec_phases, words=None):
        if self.enable_spiking_cleanup or (self.enable_sparse_index and words is None):
            return super()._cleanup(rec_phases, words)
        if words is None:
            self._ensure_codebook_cache()
            # identical to the parent's per-word mean(cos()) argmax, vectorized against the cached codebook
            # (sum vs mean differ only by the /D constant -> argmax identical)
            scores = np.cos(2.0 * np.pi * (np.asarray(rec_phases)[None, :] - self._cb_frac)).sum(axis=1)
            return self.words[int(np.argmax(scores))]
        return super()._cleanup(rec_phases, words)

    def _cleanup_all(self, rec, words=None):
        if (self.enable_sparse_index and words is None) or words is not None:
            return super()._cleanup_all(rec, words)
        if len(rec) == 0:
            return []
        self._ensure_codebook_cache()
        rec_z = np.exp(2j * np.pi * np.asarray(rec))                       # (K, D)
        sims = (rec_z @ self._cleanup_conj(self._cb_z).T).real             # (K, V), same op as the parent
        return [self.words[int(j)] for j in np.argmax(sims, axis=1)]


def _populate(comp, agents, actions, patients):
    for a, ac, p in zip(agents, actions, patients):
        fd = {"agent": a, "action": ac, "patient": p, "polarity": "AFFIRM"}
        comp.kb.append((fd, encode_fast(comp, fd)))


def _time_queries(comp, agents, actions, n, warm=3):
    for i in range(min(warm, len(agents))):
        comp.query_patient(agents[i], actions[i])
    lat = []
    K = len(agents)
    for i in range(n):
        a, ac = agents[i % K], actions[i % K]
        t = time.perf_counter()
        comp.query_patient(a, ac)
        lat.append(time.perf_counter() - t)
    lat = np.array(lat) * 1000.0
    return {"median_ms": round(float(np.median(lat)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "mean_ms": round(float(np.mean(lat)), 2), "n": int(n)}


def main():
    ap = argparse.ArgumentParser(description="codebook-cache latency de-risk (#192)")
    ap.add_argument("--V", type=int, default=24000, help="vocab size (~ real bundle 23,914)")
    ap.add_argument("--K", type=int, default=200, help="facts in one routed shard (~78,857/395)")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-timing", type=int, default=25)
    ap.add_argument("--n-identity", type=int, default=120)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {"params": {"V": a.V, "K": a.K, "D": a.D, "seed": a.seed}}
    rng = np.random.default_rng(a.seed)
    vocab = [f"w{i}" for i in range(a.V)]
    agents = [f"w{int(rng.integers(0, a.V))}" for _ in range(a.K)]
    actions = [f"w{int(rng.integers(0, a.V))}" for _ in range(a.K)]
    patients = [f"w{int(rng.integers(0, a.V))}" for _ in range(a.K)]

    base = RFPhasorComposer(seed=a.seed, D=a.D, vocab=vocab)
    cached = CachedCodebookRFPhasorComposer(seed=a.seed, D=a.D, vocab=vocab)
    _populate(base, agents, actions, patients)
    _populate(cached, agents, actions, patients)

    # --- (1) byte-identity: cached decode == baseline decode over many cues ---
    mism = []
    n_id = min(a.n_identity, a.K)
    for i in range(n_id):
        ag, ac, pt = agents[i], actions[i], patients[i]
        if base.query_patient(ag, ac) != cached.query_patient(ag, ac):
            mism.append({"kind": "query_patient", "cue": [ag, ac]})
        if base.ask_yes_no(ag, ac, pt) != cached.ask_yes_no(ag, ac, pt):
            mism.append({"kind": "ask_yes_no", "cue": [ag, ac, pt]})
        if base.query_agent(ac, pt) != cached.query_agent(ac, pt):
            mism.append({"kind": "query_agent", "cue": [ac, pt]})
    # a few known-abstain moat cues
    for j in range(20):
        ua = f"zzz_unknown_{j}"
        if base.query_patient(ua, actions[j % a.K]) != cached.query_patient(ua, actions[j % a.K]):
            mism.append({"kind": "moat", "cue": [ua, actions[j % a.K]]})
    out["identity"] = {"checked_cues": n_id * 3 + 20, "n_mismatches": len(mism), "mismatches": mism[:10]}

    # --- (2) latency baseline vs cached ---
    out["latency_baseline"] = _time_queries(base, agents, actions, a.n_timing)
    out["latency_cached"] = _time_queries(cached, agents, actions, a.n_timing)
    b = out["latency_baseline"]["median_ms"]
    c = out["latency_cached"]["median_ms"]
    out["speedup_median"] = round(b / c, 3) if c else None
    out["median_reduction_pct"] = round(100.0 * (b - c) / b, 1) if b else None

    byte_identical = len(mism) == 0
    faster = c < b
    out["byte_identical"] = byte_identical
    out["cached_faster"] = faster
    out["go"] = bool(byte_identical and faster)
    out["status"] = "GO" if out["go"] else "NO-GO"

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("wrote", a.json)
    print(json.dumps({k: out[k] for k in ("params", "identity", "latency_baseline", "latency_cached",
                                          "speedup_median", "median_reduction_pct", "byte_identical",
                                          "cached_faster", "status")}, indent=2, default=str))
    return 0 if out["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
