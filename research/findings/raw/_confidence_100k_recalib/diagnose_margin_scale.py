"""DIAGNOSTIC (board #94 / #108 R3, 2026-09-02): is `margin_norm` genuinely scale-invariant?

Measures, for the SAME clean correct recall (`asimov_isaac employer university_of_boston`, present verbatim in
BOTH bundles) on the shipped 15k core and the board-#108 100k bundle:
  * the production trace's per-role `margin` (raw cosine diff) and `margin_norm` ((top-runner)/top),
  * the mean_role_conf the metacog organ actually reads,
  * AND (via a read-only monkeypatch of `RFPhasorComposer._cleanup_all_score_stats`) the FULL candidate-score
    DISTRIBUTION for each decoded role: winner value, runner-up value, mean/std of the non-winner candidates,
    and the number of candidate words -- so we can see WHICH term drifts with vocabulary size.

If `margin_norm` is scale-invariant it should read ~the same at both scales. If the runner-up (an order
statistic = max over V-1 candidates) inflates with V while the winner and the candidate mean/std hold, then the
drift is extreme-value inflation of the single runner-up, and a distribution-relative decisiveness read (z-score
of the winner above the candidate bulk) would be scale-robust.

Run (numpy, single seed, light):
  SIM_BACKEND=numpy .venv/bin/python \
    research/findings/raw/_confidence_100k_recalib/diagnose_margin_scale.py --bundle 15k --seed 42
  SIM_BACKEND=numpy .venv/bin/python \
    research/findings/raw/_confidence_100k_recalib/diagnose_margin_scale.py --bundle 100k --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)
os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)
os.environ["BRAIN_CLAIM_MOAT"] = "0"

import numpy as np                                                                   # noqa: E402

import webapp.server as S                                                            # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo  # noqa: E402
from research.runners.developed_brain_io import _inner_agent                          # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore                        # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore                  # noqa: E402
from research.runners import rf_phasor_composer as RFC                                # noqa: E402

BUNDLES = {
    "15k": "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k",
    "100k": "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k",
}
Q = "who does asimov isaac work for?"
EXPECTED_SVO = ["asimov_isaac", "employer", "university_of_boston"]

# ── read-only instrumentation: capture the full candidate-score distribution for every _cleanup_all_score_stats
# call, keyed by winner word, WITHOUT changing what the method returns (byte-identical production path). ──────────
_DIST_CAP: list = []
_orig_stats = RFC.RFPhasorComposer._cleanup_all_score_stats


def _instrumented_stats(self, rec, words=None):
    out = _orig_stats(self, rec, words=words)   # the real, unchanged production computation
    try:
        w = words if words is not None else self.words
        if len(rec) and len(w) > 1:
            rec_z = np.exp(2j * np.pi * np.asarray(rec))
            cb = np.stack([np.exp(2j * np.pi * self.concepts[x]) for x in w])
            sims = (rec_z @ self._cleanup_conj(cb).T).real / float(self.D)   # (K, V)
            for i in range(len(rec)):
                row = sims[i]
                order = np.argsort(row)
                top_i = int(order[-1]); run_i = int(order[-2])
                nonwin = np.delete(row, top_i)
                _DIST_CAP.append({
                    "winner_word": str(w[top_i]),
                    "n_candidates": int(len(w)),
                    "top_raw": float(row[top_i]),
                    "runner_raw": float(row[run_i]),
                    "nonwin_mean": float(nonwin.mean()),
                    "nonwin_std": float(nonwin.std()),
                    "nonwin_p50": float(np.percentile(nonwin, 50)),
                    "nonwin_p90": float(np.percentile(nonwin, 90)),
                    "nonwin_p99": float(np.percentile(nonwin, 99)),
                    "margin_raw": float(row[top_i] - row[run_i]),
                    "margin_norm": (float((max(row[top_i], 0.0) - max(row[run_i], 0.0)) / (max(row[top_i], 0.0) + 1e-9))
                                    if row[top_i] > 0 else 0.0),
                    # candidate distribution z-score of the winner (scale-robust decisiveness):
                    "winner_z": (float((row[top_i] - nonwin.mean()) / (nonwin.std() + 1e-9))),
                })
    except Exception as e:  # noqa: BLE001
        _DIST_CAP.append({"error": repr(e)})
    return out


RFC.RFPhasorComposer._cleanup_all_score_stats = _instrumented_stats


def build_chat(bundle_path, seed):
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(bundle_path, extra_kwargs={"enable_codebook_cache": True,
                                                             "enable_decode_escalation": True})
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


_sid = [0]


def ask(chat):
    ck = (f"s{_sid[0]:04d}", "tiny-demo", "stub")
    _sid[0] += 1
    S._BRAIN_CHATS[ck] = chat
    r = S.brain_chat(S.BrainChatRequest(session=ck[0], message=Q, brain="tiny-demo",
                                        reset=False, rich=True, renderer="stub"))
    return json.loads(bytes(r.body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=list(BUNDLES), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    t0 = time.time()
    chat = build_chat(BUNDLES[args.bundle], args.seed)
    _DIST_CAP.clear()
    d = ask(chat)
    dt = time.time() - t0

    cf = d.get("confidence_forthcoming") or {}
    mc = d.get("metacog") or {}
    # the roles the trace exposes (what mean_role_confidence averages)
    roles = (d.get("roles") or [])
    role_slim = [{"word": r.get("word"), "margin": r.get("margin"), "margin_norm": r.get("margin_norm"),
                  "confidence": r.get("confidence")} for r in roles]

    # distribution capture rows whose winner is one of the target SVO tokens (the roles that feed confidence)
    tgt = set(EXPECTED_SVO)
    tgt_dist = [x for x in _DIST_CAP if x.get("winner_word") in tgt]

    out = {
        "bundle": args.bundle, "bundle_path": BUNDLES[args.bundle], "seed": args.seed,
        "question": Q, "expected_svo": EXPECTED_SVO, "elapsed_s": dt,
        "recalled_svo": d.get("recalled_svo"), "abstained": d.get("abstained"),
        "recall_correct": (d.get("recalled_svo") == EXPECTED_SVO),
        "n_sentences": d.get("n_sentences"),
        "mean_role_conf": mc.get("mean_role_conf"),
        "confident": cf.get("confident"), "cf_reason": cf.get("reason"),
        "trace_roles": role_slim,
        "target_role_distributions": tgt_dist,
        "n_dist_captures_total": len(_DIST_CAP),
    }
    outp = args.out or os.path.join(_HERE, f"diagnose_{args.bundle}_seed{args.seed}.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"[{dt:.0f}s] bundle={args.bundle} recall_correct={out['recall_correct']} "
          f"mean_role_conf={out['mean_role_conf']} confident={out['confident']} n={out['n_sentences']}")
    for x in tgt_dist:
        print(f"   role winner={x['winner_word']:>22}  top={x['top_raw']:.4f} runner={x['runner_raw']:.4f} "
              f"m_raw={x['margin_raw']:.4f} m_norm={x['margin_norm']:.4f} "
              f"nonwin_mean={x['nonwin_mean']:.5f} nonwin_std={x['nonwin_std']:.5f} "
              f"p99={x['nonwin_p99']:.4f} z={x['winner_z']:.2f} V={x['n_candidates']}")
    print(f"   wrote {os.path.relpath(outp, _REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
