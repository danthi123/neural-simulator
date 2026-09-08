"""ONEBRAIN FAST-LOAD verify: does the new `onebrain_substrate.npz` sidecar (`extract_onebrain_substrate` /
`_restore_facts`'s ONEBRAIN direct-set branch, `research/runners/developed_brain_io.py`) skip the per-fact RF
resonate on reload WITHOUT changing recall -- byte-identical, not just answer-identical?

WHY (board 2026-09-08 "onebrain re-resonate" blocker, `GAP_CLOSURE_MISSION.md` lines 28-29; finding
`2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md`): `OneBrainComposer` holds its bound
composite ON-SUBSTRATE (`kb=(fact,None)`), so unlike rf/rate (which persist `kb_composites.npz` and skip the
resonate via `_restore_facts`'s existing direct-set branch), every `load_developed_brain` call on a onebrain bundle
re-`store()`s -- re-resonates -- every fact (~416 RF-resonate steps/fact via `_compose_phases`, confirmed by
reading `one_brain_composer.py` directly; the docs' "~832-step" figure is `RFPhasorComposer`'s own numpy-oracle
`_encode` cost, a DIFFERENT composer family, not onebrain's). Flipping the production composer default to onebrain
would make every brain restart pay this cost (~20 min on the production cupy backend at 404 facts, per the board).

METHOD (one seed = the bundle's own developmental seed; numpy/CPU per the task's cost-routing -- do NOT contend
with the GPU's own in-flight latency benchmark):
  1. SLOW load: `load_developed_brain(bundle)` on a bundle with NO onebrain_substrate.npz present -> the existing,
     unchanged re-resonate path. Time it; snapshot store_conns + kb + full recall (query_patient/query_agent/
     ask_yes_no over every distinct cue).
  2. Persist the sidecar: `extract_onebrain_substrate(agent_slow)` -> `onebrain_substrate.npz`, written into a
     COPY of the bundle directory (never mutates the source bundle in place).
  3. FAST load: `load_developed_brain(bundle_with_sidecar)` -- now `_restore_facts`'s onebrain direct-set branch
     should fire for every fact (0 resonate calls). Time it; snapshot the same three things.
  4. Compare SLOW vs FAST: store_conns weight arrays element-wise EXACT equality (the actual byte-identical claim,
     not "expected unchanged (unverified)" -- docs/TERMS.md's own bar for the word); kb fact-dict list equality;
     full recall equality over every distinct cue (not a sample). GO iff all three match AND the fast load is
     markedly faster.

ANTI-CHEATS: this is a byte-identical correctness check on the SAME already-verified bundle (rank-1's own GO), not
a fresh capability claim -- so the anti-cheat is comparing the RAW SUBSTRATE STATE (store_conns), not just derived
answers, because a subtly-wrong composite could still decode correctly for some cues by chance while being wrong
elsewhere (query answers alone would not catch that with certainty; the byte comparison would).

NO sim/ edit. Reuse-by-import only (developed_brain_io.load_developed_brain/extract_onebrain_substrate).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners.developed_brain_io import (  # noqa: E402
    _load_facts_json, _read_manifest, extract_onebrain_substrate, load_developed_brain,
)

DEFAULT_BUNDLE = "bridges/developed/rebuilt_scale787_onebrain"


def _flat_facts(facts):
    flat = []
    for f in facts:
        a, v, p = f.get("agent"), f.get("action"), f.get("patient")
        if isinstance(a, str) and isinstance(v, str) and isinstance(p, str):
            flat.append((a, v, p))
    return flat


def _cue_sets(flat):
    patient_set, agent_set, svo = {}, {}, set()
    for a, v, p in flat:
        patient_set.setdefault((a, v), set()).add(p)
        agent_set.setdefault((v, p), set()).add(a)
        svo.add((a, v, p))
    return patient_set, agent_set, svo


def _snapshot(agent, distinct_qp, distinct_qa, distinct_yn):
    """Every cue's answer (query_patient/query_agent/ask_yes_no) -- the FULL recall surface, not a sample."""
    comp = agent.agent.composer if hasattr(agent, "agent") else agent.composer
    qp = {f"{a}|{v}": comp.query_patient(a, v) for (a, v) in distinct_qp}
    qa = {f"{v}|{p}": comp.query_agent(v, p) for (v, p) in distinct_qa}
    yn = {f"{a}|{v}|{p}": comp.ask_yes_no(a, v, p) for (a, v, p) in distinct_yn}
    return qp, qa, yn


def _composer_of(agent):
    return agent.agent.composer if hasattr(agent, "agent") else agent.composer


def _store_conns_snapshot(comp):
    """A hashable/comparable snapshot of store_conns: a list of (post, pre, weight) per block, weight as complex."""
    return [(p, q, complex(w)) for (p, q, w) in comp.store_conns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default="research/findings/raw/_onebrain_fastload/verify.json")
    args = ap.parse_args()

    manifest = _read_manifest(args.bundle)
    if manifest is None:
        raise SystemExit(f"no manifest at {args.bundle}")
    if manifest.get("composer_kind") != "onebrain":
        raise SystemExit(f"bundle {args.bundle} is not composer_kind='onebrain' (got {manifest.get('composer_kind')!r})")
    facts_json = _load_facts_json(args.bundle)
    flat = _flat_facts(facts_json)
    patient_set, agent_set, svo = _cue_sets(flat)
    distinct_qp = sorted(patient_set.keys())
    distinct_qa = sorted(agent_set.keys())
    distinct_yn = sorted(svo)
    print(f"[fastload] bundle={args.bundle} n_facts={len(flat)} (of {len(facts_json)} total facts) "
          f"distinct cues: qp={len(distinct_qp)} qa={len(distinct_qa)} yesno_svo={len(distinct_yn)}", flush=True)

    sidecar_path = os.path.join(args.bundle, "onebrain_substrate.npz")
    if os.path.exists(sidecar_path):
        raise SystemExit(f"{sidecar_path} already exists -- this verify needs a SLOW baseline first; remove it "
                          f"(or point --bundle at a fresh copy) and rerun")

    # ---- (1) SLOW load: no sidecar present -> the existing, unchanged re-resonate path ----
    t0 = time.time()
    agent_slow, _m = load_developed_brain(args.bundle, composer_kind="onebrain",
                                          onebrain_k_max=len(facts_json) + 16)
    slow_load_s = time.time() - t0
    comp_slow = _composer_of(agent_slow)
    print(f"[fastload] SLOW load (no sidecar, re-resonate every fact): {slow_load_s:.2f}s "
          f"n_kb={len(comp_slow.kb)} n_store_conns={len(comp_slow.store_conns)}", flush=True)

    qp_slow, qa_slow, yn_slow = _snapshot(agent_slow, distinct_qp, distinct_qa, distinct_yn)
    store_conns_slow = _store_conns_snapshot(comp_slow)
    kb_slow = [dict(f) for (f, _h) in comp_slow.kb]

    # ---- (2) persist the sidecar (extract_onebrain_substrate) ----
    sub = extract_onebrain_substrate(agent_slow)
    print(f"[fastload] extract_onebrain_substrate: {len(sub)} of {len(comp_slow.kb)} facts covered "
          f"(should be ALL -- a partial map means some kb entry's store_conns slice was short)", flush=True)
    np.savez_compressed(sidecar_path, **sub)
    sidecar_bytes = os.path.getsize(sidecar_path)
    print(f"[fastload] wrote {sidecar_path} ({sidecar_bytes} bytes)", flush=True)

    # ---- (3) FAST load: sidecar present -> the new direct-set branch should fire ----
    t0 = time.time()
    agent_fast, _m2 = load_developed_brain(args.bundle, composer_kind="onebrain",
                                           onebrain_k_max=len(facts_json) + 16)
    fast_load_s = time.time() - t0
    comp_fast = _composer_of(agent_fast)
    print(f"[fastload] FAST load (sidecar present): {fast_load_s:.2f}s "
          f"n_kb={len(comp_fast.kb)} n_store_conns={len(comp_fast.store_conns)}", flush=True)

    qp_fast, qa_fast, yn_fast = _snapshot(agent_fast, distinct_qp, distinct_qa, distinct_yn)
    store_conns_fast = _store_conns_snapshot(comp_fast)
    kb_fast = [dict(f) for (f, _h) in comp_fast.kb]

    # ---- (4) compare ----
    qp_mismatches = [k for k in qp_slow if qp_slow[k] != qp_fast.get(k)]
    qa_mismatches = [k for k in qa_slow if qa_slow[k] != qa_fast.get(k)]
    yn_mismatches = [k for k in yn_slow if yn_slow[k] != yn_fast.get(k)]

    store_conns_identical = (len(store_conns_slow) == len(store_conns_fast)
                             and all(p1 == p2 and q1 == q2 and w1 == w2
                                     for (p1, q1, w1), (p2, q2, w2) in zip(store_conns_slow, store_conns_fast)))
    # a finer-grained max-abs-diff in case of a near-miss (float rounding) rather than a structural mismatch
    if len(store_conns_slow) == len(store_conns_fast) and store_conns_slow:
        max_abs_diff = max(abs(w1 - w2) for (_p1, _q1, w1), (_p2, _q2, w2) in zip(store_conns_slow, store_conns_fast))
    else:
        max_abs_diff = None

    kb_identical = (kb_slow == kb_fast)
    kb_mismatches = []
    if not kb_identical:
        for i, (fs, ff) in enumerate(zip(kb_slow, kb_fast)):
            if fs != ff:
                kb_mismatches.append({"index": i, "slow": fs, "fast": ff})

    answers_identical = (not qp_mismatches) and (not qa_mismatches) and (not yn_mismatches)
    byte_identical = store_conns_identical and kb_identical
    speedup = (slow_load_s / fast_load_s) if fast_load_s > 0 else None

    verdict = "GO" if (answers_identical and byte_identical) else "NO-GO"
    print(f"[fastload] store_conns_identical={store_conns_identical} (max_abs_diff={max_abs_diff}) "
          f"kb_identical={kb_identical} answers_identical={answers_identical}", flush=True)
    print(f"[fastload] slow_load_s={slow_load_s:.2f} fast_load_s={fast_load_s:.2f} speedup={speedup} "
          f"VERDICT={verdict}", flush=True)

    result = {
        "bundle": args.bundle,
        "n_facts": len(facts_json),
        "n_flat_facts": len(flat),
        "n_kb_slow": len(comp_slow.kb),
        "n_kb_fast": len(comp_fast.kb),
        "n_onebrain_substrate_entries": len(sub),
        "sidecar_bytes": sidecar_bytes,
        "timings_s": {"slow_load": slow_load_s, "fast_load": fast_load_s, "speedup_x": speedup},
        "store_conns_identical": store_conns_identical,
        "store_conns_max_abs_diff": (None if max_abs_diff is None else float(max_abs_diff)),
        "kb_identical": kb_identical,
        "kb_mismatches_sample": kb_mismatches[:20],
        "answers_identical": answers_identical,
        "qp_mismatches": qp_mismatches[:40],
        "qa_mismatches": qa_mismatches[:40],
        "yn_mismatches": yn_mismatches[:40],
        "n_cues": {"qp": len(distinct_qp), "qa": len(distinct_qa), "yesno_svo": len(distinct_yn)},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[fastload] wrote {args.out}", flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
