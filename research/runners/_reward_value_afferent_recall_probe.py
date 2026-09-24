"""A10 follow-up round (re-review of 4b6a9cf66, item 3): is A10's OWN recall call history-dependent? Governed by
research/findings/2026-09-24-reward-value-spiking-afferent-PREREG-AMENDMENT-3.md (committed before this runner's first
run). An instrument measurement that backs a DECLARATION, not a capability verdict.

WHY. `webapp/reward_value_afferent_chat.spiking_reward_value` asks the brain for the expected patient with
`chat.inner.what_does(agent, action)` (under the production-default composer that is `OneBrainComposer.query_patient`,
a spiking recall on the composer's own bridge; under `BRAIN_COMPOSER_KIND=rf` it is the host `RFPhasorComposer`). That
call runs OUTSIDE the organ snapshot isolation, and with the flag on it is the turn's first recall: production's own
recalls (the comprehension gate's `known_binding`, the surprise block's `p_stored`) run later in the same turn. If a
recall leaves state that a later recall or other later code reads, the flag changes what production computes.

WHAT IT RUNS (one process, the arms' env for the seed, numpy; the chat is the one the battery worker's first
`brain_chat` call builds, via `webapp.server._build_chat_brain("tiny-demo", "stub")`):
  H(x) = sha256 over everything reachable from x: arrays by dtype/shape/bytes, sparse matrices by data/indices/indptr,
  scalars by repr, lists/tuples/dicts/sets by content, objects by their attributes (and __slots__), numpy/Python
  generator objects by state; modules, functions, classes and methods by qualified name; other C objects (locks,
  kernels) by type name only (counted and reported as `opaque`).
  0. build; S0 = H(chat.inner), per-attribute H of the composer and of its bridge, the numpy and Python global RNG
     states, the composer's `last_trace`.
  1. recall #1 what_does(dog, chase)   -- the call A10 makes first in a turn            -> p1, S1
  2. recall #2 what_does(dog, chase)   -- production's first recall after A10's        -> p2, S2
  3. recall #3 what_does(dog, chase)                                                   -> p3, S3
  4. what_does(cat, eat), then recall #4 what_does(dog, chase)                         -> q, S4; p4, S5
  sensitivity control: flip one element of the composer bridge's largest dense array; H must change; put it back;
  H must return to its value.

REPORTED (the evidence the declaration cites):
  * value_history_independent: p1 == p2 == p3 == p4 (and p1 is the stored patient);
  * state_after_converges: S1 == S2 == S3 (the state a later recall leaves does not depend on whether A10's ran);
  * state_touched_by_first_recall: S0 != S1, with the attributes that differ (code that runs BETWEEN A10's recall and
    production's first recall sees S1 instead of S0);
  * rng_untouched: the numpy and Python global generators are unchanged across every recall;
  * last_trace per call.
The Verdict below is GO iff value_history_independent AND state_after_converges AND rng_untouched ("history-
independent at the module level"); its preconditions are the build, the composer class the arms use, and the
sensitivity control. A NO-GO is a finding for the declaration, not a failure of the runner.

ADDED BY AMENDMENT-4 (run 2; run 1 read UNDEFINED on two instrument defects: an exact-class precondition where
AMENDMENT-3 names a class family, and no bridge found for the pool-bound rf composer):
  * isolated_first: `webapp.reward_value_afferent_chat.isolated_recall` from the fresh state S0 (first use, where A10
    calls it), then the hash and both generators compared with S0; isolated_warm: the same after the five
    production recalls. A second Verdict (`verdict_isolated`) is GO iff both leave the hash and the generators
    unchanged with an exact restore, and the isolated value equals the first production recall's value. It needs a
    sensitivity control that the unisolated recall DOES leave a trace (else the check could not fail).
  * global_rng_callers on recalls #1 and #2: every call into numpy's / Python's global generator during the recall,
    by function and the three calling frames (randn counts are samples).
  * deques and mappingproxies are hashed by content (run 1 hashed them by type only).

Run (local, under memcap; run 2 writes recall_probe_run2.json beside run 1):
  bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m \
      research.runners._reward_value_afferent_recall_probe --seed 7 \
      --out research/findings/raw/_reward_value_afferent_derisk/v4/recall_probe_run2.json
  (add --composer rf and write under v4/rf/ for the forced-rf composer)
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import random
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_NAMED = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType,
          types.CodeType, type, functools.partial, staticmethod, classmethod, property)
_SCALARS = (type(None), bool, int, float, complex, str, bytes, np.generic)


class _Tag:
    __slots__ = ("b",)

    def __init__(self, b):
        self.b = b


def _host_bytes(a):
    return (a.get() if hasattr(a, "get") and not isinstance(a, np.ndarray) else a).tobytes()


def deep_hash(root, max_nodes=20_000_000):
    """sha256 over everything reachable from `root` (see the module docstring). Returns (hexdigest, stats)."""
    from webapp.reward_value_afferent_chat import _is_dense, _is_sparse
    h = hashlib.sha256()
    seen = {}
    stats = {"nodes": 0, "arrays": 0, "array_bytes": 0, "sparse": 0, "objects": 0, "opaque": 0, "opaque_types": {}}
    stack = [root]
    while stack:
        x = stack.pop()
        if isinstance(x, _Tag):
            h.update(x.b)
            continue
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes:
            raise RuntimeError("deep_hash: more than %d nodes reachable" % max_nodes)
        if isinstance(x, _SCALARS):
            h.update(b"s" + type(x).__name__.encode() + b":" + repr(x).encode())
            continue
        if isinstance(x, _NAMED):
            h.update(b"n" + str(getattr(x, "__qualname__", type(x).__name__)).encode())
            continue
        xid = id(x)
        if xid in seen:
            h.update(b"r%d" % seen[xid])
            continue
        seen[xid] = len(seen)
        if _is_dense(x):
            stats["arrays"] += 1
            if getattr(x.dtype, "kind", "") == "O":
                h.update(b"A" + str(x.shape).encode())
                stack.extend(reversed(list(np.asarray(x).ravel())))
            else:
                b = _host_bytes(x)
                stats["array_bytes"] += len(b)
                h.update(b"a" + str((x.dtype, x.shape)).encode())
                h.update(b)
        elif _is_sparse(x):
            stats["sparse"] += 1
            h.update(b"p" + str((type(x).__name__, x.shape)).encode())
            for part in (x.data, x.indices, x.indptr):
                h.update(_host_bytes(part))
        elif isinstance(x, (list, tuple, collections.deque)):
            h.update(b"l" + type(x).__name__.encode() + b"%d" % len(x))
            stack.extend(reversed(list(x)))
        elif isinstance(x, (dict, types.MappingProxyType)):
            h.update(b"d%d" % len(x))
            items = sorted(x.items(), key=lambda kv: repr(kv[0]))
            for k, v in reversed(items):
                stack.append(v)
                stack.append(_Tag(b"k" + repr(k).encode()))
        elif isinstance(x, (set, frozenset)):
            h.update(b"S" + repr(sorted(repr(e) for e in x)).encode())
        elif isinstance(x, np.random.RandomState):
            st = x.get_state()
            h.update(b"R" + st[1].tobytes() + repr(st[2:]).encode())
        elif isinstance(x, np.random.Generator):
            h.update(b"G" + repr(x.bit_generator.state).encode())
        elif isinstance(x, random.Random):
            h.update(b"P" + repr(x.getstate()).encode())
        else:
            attrs = {}
            if hasattr(x, "__dict__"):
                try:
                    attrs.update(vars(x))
                except TypeError:
                    pass
            for cls in type(x).__mro__:
                for s in getattr(cls, "__slots__", ()) or ():
                    if isinstance(s, str) and s not in attrs and hasattr(x, s):
                        try:
                            attrs[s] = getattr(x, s)
                        except Exception:
                            pass
            if attrs or hasattr(x, "__dict__"):
                stats["objects"] += 1
                h.update(b"O" + type(x).__qualname__.encode())
                for k in sorted(attrs, reverse=True):
                    stack.append(attrs[k])
                    stack.append(_Tag(b"." + k.encode()))
            else:
                stats["opaque"] += 1
                tn = type(x).__qualname__
                stats["opaque_types"][tn] = stats["opaque_types"].get(tn, 0) + 1
                h.update(b"?" + tn.encode())
    return h.hexdigest(), stats


def _attr_hashes(obj):
    out = {}
    if obj is None or not hasattr(obj, "__dict__"):
        return out
    for k in sorted(vars(obj)):
        try:
            out[k] = deep_hash(getattr(obj, k))[0]
        except Exception as e:
            out[k] = "error:%s" % type(e).__name__
    return out


def _rng_hashes():
    st = np.random.get_state()
    return {"numpy": hashlib.sha256(st[1].tobytes() + repr(st[2:]).encode()).hexdigest(),
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}


def _unwrap_composer(chat):
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    return comp


def _bridge_of(comp):
    """The bridge the composer's recall runs on: `b` (OneBrainComposer and its pool-bound subclass), `bridge`, or the
    pool #1 substrate's bridge (`_pool1.bridge`, Pool1BoundComposer: an RFPhasorComposer whose RF ops run on pool #1)."""
    for b in (getattr(comp, "b", None), getattr(comp, "bridge", None),
              getattr(getattr(comp, "_pool1", None), "bridge", None)):
        if b is not None:
            return b
    return None


def _composer_family(comp):
    """Class names in the composer's MRO (Pool1BoundComposer IS an RFPhasorComposer; the pool-bound onebrain class IS
    a OneBrainComposer), so the precondition matches the family the arms' env builds, not one exact class."""
    return [c.__name__ for c in type(comp).__mro__] if comp is not None else []


def _trace(comp):
    try:
        return repr(getattr(comp, "last_trace", None))[:2000]
    except Exception as e:
        return "error:%s" % type(e).__name__


def snapshot(chat):
    comp = _unwrap_composer(chat)
    t0 = time.time()
    h, stats = deep_hash(chat.inner)
    return {"H_inner": h, "stats": stats, "composer_attrs": _attr_hashes(comp),
            "bridge_attrs": _attr_hashes(_bridge_of(comp)), "rng": _rng_hashes(), "last_trace": _trace(comp),
            "hash_seconds": round(time.time() - t0, 2)}


def _diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _largest_dense(bridge):
    from webapp.reward_value_afferent_chat import _is_dense
    best = None
    for k, v in vars(bridge).items():
        if _is_dense(v) and v.size and getattr(v.dtype, "kind", "") in "fc" and (best is None or v.size > best[1].size):
            best = (k, v)
    return best


class _RngCallers:
    """Count every call into numpy's and Python's GLOBAL generators (module-level functions, which is how sim/bridge.py
    reaches them: `cp.random.randn` with cp = numpy, `np.random.seed`, `random.seed`) during a block, by function and
    the three calling frames. A measurement aid only; the functions are restored on exit."""
    _NP = ("seed", "set_state", "random", "random_sample", "rand", "randn", "normal", "uniform", "randint", "choice",
           "permutation", "shuffle", "standard_normal", "binomial", "poisson", "exponential")
    _PY = ("seed", "setstate", "random", "uniform", "randint", "choice", "shuffle", "gauss", "sample", "randrange")

    def __init__(self):
        self.counts = {}
        self._saved = []

    def _wrap(self, mod, name, label):
        fn = getattr(mod, name, None)
        if fn is None:
            return
        self._saved.append((mod, name, fn))
        counts = self.counts

        def w(*a, **k):
            f = sys._getframe(1)
            chain = []
            for _ in range(3):
                if f is None:
                    break
                chain.append("%s:%d:%s" % (os.path.relpath(f.f_code.co_filename, _REPO), f.f_lineno, f.f_code.co_name))
                f = f.f_back
            key = "%s.%s <- %s" % (label, name, " <- ".join(chain))
            n = (int(np.prod([int(x) for x in a])) if (label == "numpy" and name in ("randn", "rand") and a
                                                     and all(isinstance(x, (int, np.integer)) for x in a)) else 1)
            counts[key] = counts.get(key, 0) + n
            return fn(*a, **k)
        setattr(mod, name, w)

    def __enter__(self):
        for n in self._NP:
            self._wrap(np.random, n, "numpy")
        for n in self._PY:
            self._wrap(random, n, "python")
        return self

    def __exit__(self, *exc):
        for mod, name, fn in reversed(self._saved):
            setattr(mod, name, fn)
        return False


def run(seed, composer, out_path):
    from research.runners._reward_value_afferent_derisk import arm_env
    os.environ.update(arm_env("off_a", seed, composer))
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    from tools.verdict import Verdict
    import webapp.server as S

    t0 = time.time()
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    build_s = round(time.time() - t0, 1)
    comp = _unwrap_composer(chat)
    bridge = _bridge_of(comp)
    family = _composer_family(comp)
    rec = {"runner": "_reward_value_afferent_recall_probe", "seed": int(seed), "composer_mro": family,
           "seed_kind": "dev-calibration (NOT a 6-seed gate seed)", "composer_forced": composer,
           "source": source, "build_s": build_s, "composer_class": type(comp).__name__ if comp is not None else None,
           "composer_flags": {k: getattr(comp, k, None) for k in ("integrated_loop", "enable_batched", "trace",
                                                                  "enable_fact_shard", "enable_sparse_index",
                                                                  "persistent_store", "_fused")},
           "bridge_class": type(bridge).__name__ if bridge is not None else None}

    # sensitivity control: the hash sees a one-element change in the composer bridge and returns when it is undone
    sens = {"ran": False}
    S0 = snapshot(chat)
    big = _largest_dense(bridge) if bridge is not None else None
    if big is not None:
        name, arr = big
        old = arr.ravel()[0].copy() if hasattr(arr.ravel()[0], "copy") else arr.ravel()[0]
        arr.ravel()[0] = old + 1.0
        h_mut = deep_hash(chat.inner)[0]
        arr.ravel()[0] = old
        h_back = deep_hash(chat.inner)[0]
        sens = {"ran": True, "array": name, "changed_when_mutated": h_mut != S0["H_inner"],
                "returned_when_undone": h_back == S0["H_inner"]}
    rec["sensitivity"] = sens

    # AMENDMENT-4: the ISOLATED A10 recall (webapp/reward_value_afferent_chat.isolated_recall) from the fresh state S0,
    # i.e. at first use, exactly where A10 calls it in a turn. Everything reachable from chat.inner and both global
    # generators must be as they were; the production recalls below then start from S0.
    import webapp.reward_value_afferent_chat as RVA

    def iso(label, ref):
        t = time.time()
        p, r = RVA.isolated_recall(chat, "dog", "chase")
        after = snapshot(chat)
        return {"label": label, "value": None if p is None else str(p), "record": r,
                "seconds": round(time.time() - t, 2), "hash_unchanged": after["H_inner"] == ref["H_inner"],
                "rng_unchanged": after["rng"] == ref["rng"],
                "composer_attrs_changed": _diff(ref["composer_attrs"], after["composer_attrs"]),
                "bridge_attrs_changed": _diff(ref["bridge_attrs"], after["bridge_attrs"])}

    iso_first = iso("first use (fresh state S0)", S0)
    rec["isolated_first"] = iso_first

    calls = []

    def recall(a, v, trace=False):
        before = _rng_hashes()
        t = time.time()
        tr = _RngCallers() if trace else None
        try:
            if tr is not None:
                with tr:
                    p = chat.inner.what_does(a, v)
            else:
                p = chat.inner.what_does(a, v)
            err = None
        except Exception as e:
            p, err = None, "%s: %s" % (type(e).__name__, e)
        after_snap = snapshot(chat)
        calls.append({"cue": [a, v], "value": None if p is None else str(p), "error": err,
                      "seconds": round(time.time() - t, 2), "rng_unchanged": before == after_snap["rng"],
                      "global_rng_callers": None if tr is None else tr.counts,
                      "H_inner_after": after_snap["H_inner"], "last_trace_after": after_snap["last_trace"]})
        return p, after_snap

    p1, S1 = recall("dog", "chase", trace=True)
    p2, S2 = recall("dog", "chase", trace=True)
    p3, S3 = recall("dog", "chase")
    q, S4 = recall("cat", "eat")
    p4, S5 = recall("dog", "chase")
    rec["calls"] = calls
    rec["S0"] = {k: S0[k] for k in ("H_inner", "stats", "rng", "last_trace", "hash_seconds")}

    vals = [p1, p2, p3, p4]
    value_hi = bool(all(str(p) == str(p1) for p in vals) and p1 is not None)
    state_conv = bool(S1["H_inner"] == S2["H_inner"] == S3["H_inner"])
    rng_ok = bool(all(c["rng_unchanged"] for c in calls))
    rec.update({
        "values": {"p1": p1, "p2": p2, "p3": p3, "q_cat_eat": q, "p4_after_other_recall": p4},
        "value_history_independent": value_hi,
        "state_after_converges": state_conv,
        "state_after_other_recall_returns": bool(S5["H_inner"] == S1["H_inner"]),
        "state_touched_by_first_recall": bool(S0["H_inner"] != S1["H_inner"]),
        "composer_attrs_changed_by_first_recall": _diff(S0["composer_attrs"], S1["composer_attrs"]),
        "bridge_attrs_changed_by_first_recall": _diff(S0["bridge_attrs"], S1["bridge_attrs"]),
        "composer_attrs_changed_by_second_recall": _diff(S1["composer_attrs"], S2["composer_attrs"]),
        "bridge_attrs_changed_by_second_recall": _diff(S1["bridge_attrs"], S2["bridge_attrs"]),
        "composer_attrs_changed_by_other_recall": _diff(S3["composer_attrs"], S4["composer_attrs"]),
        "bridge_attrs_changed_by_other_recall": _diff(S3["bridge_attrs"], S4["bridge_attrs"]),
        "last_trace_first_vs_second_equal": S1["last_trace"] == S2["last_trace"],
        "rng_untouched": rng_ok,
    })

    v = Verdict("A10 follow-up: A10's recall call (chat.inner.what_does) is history-independent at the module level")
    v.require("the tiny-demo chat built", comp is not None, expect=True)
    want = "RFPhasorComposer" if composer == "rf" else "OneBrainComposer"
    v.require("composer is of the family the arms use (%s or a subclass; LTM tier off, not TieredFactStore)" % want,
              family, expect=lambda f: want in f and "TieredFactStore" not in f)
    v.require("the recall's bridge was found (for the sensitivity control)", bridge is not None, expect=True)
    v.require("recall #1 returns the stored patient (cat)", None if p1 is None else str(p1), expect="cat")
    v.require("sensitivity: the hash changes on a one-element bridge change and returns when undone",
              bool(sens.get("changed_when_mutated") and sens.get("returned_when_undone")), expect=True)
    history_independent = value_hi and state_conv and rng_ok
    decided = v.decide(go=history_independent, verbose=True)
    rec.update({"go": bool(decided["go"]), "status": decided["status"], "preconditions": decided["preconditions"],
                "verdict": decided})

    # AMENDMENT-4: the isolated recall again, now at a warm state (after five production recalls), and its verdict
    S_warm = snapshot(chat)
    iso_warm = iso("warm (after the production recalls)", S_warm)
    rec["isolated_warm"] = iso_warm
    vi = Verdict("A10 follow-up (AMENDMENT-4): the ISOLATED A10 recall leaves no state and no global-generator change "
                 "(module level)")
    vi.require("the tiny-demo chat built", comp is not None, expect=True)
    vi.require("composer is of the family the arms use (%s or a subclass; not TieredFactStore)" % want,
               family, expect=lambda f: want in f and "TieredFactStore" not in f)
    vi.require("sensitivity: the hash changes on a one-element bridge change and returns when undone",
               bool(sens.get("changed_when_mutated") and sens.get("returned_when_undone")), expect=True)
    raw_leaves_trace = bool(rec["state_touched_by_first_recall"] or not rng_ok)
    vi.require("sensitivity: an UNISOLATED recall changes the state hash or a global generator (else no test)",
               raw_leaves_trace, expect=True)
    vi.require("the isolated recall took its snapshot (first use and warm)",
               bool(iso_first["record"].get("isolated") and iso_warm["record"].get("isolated")), expect=True)
    iso_clean = all(x["hash_unchanged"] and x["rng_unchanged"] and x["record"].get("restored_exact") is True
                    and (x["record"].get("rng") or {}).get("host_rngs_unchanged") is True
                    for x in (iso_first, iso_warm))
    iso_value = iso_first["value"] == (None if p1 is None else str(p1)) and iso_warm["value"] == iso_first["value"]
    rec["isolated_value_equals_first_production_recall"] = bool(iso_value)
    decided_i = vi.decide(go=bool(iso_clean and iso_value), verbose=True)
    rec.update({"go_isolated": bool(decided_i["go"]), "status_isolated": decided_i["status"],
                "preconditions_isolated": decided_i["preconditions"], "verdict_isolated": decided_i})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(json.dumps({"status": decided["status"], "status_isolated": decided_i["status"],
                      "value_history_independent": value_hi,
                      "state_after_converges": state_conv, "state_touched_by_first_recall":
                      rec["state_touched_by_first_recall"], "rng_untouched": rng_ok,
                      "composer": rec["composer_class"], "out": out_path}), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--composer", choices=("onebrain", "rf"), default=None,
                    help="force BRAIN_COMPOSER_KIND (default: unset -> the production default, onebrain)")
    ap.add_argument("--out", default="research/findings/raw/_reward_value_afferent_derisk/v4/recall_probe.json")
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    sys.exit(run(a.seed, a.composer, a.out))
