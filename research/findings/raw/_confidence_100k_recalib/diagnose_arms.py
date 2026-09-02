"""DIAGNOSTIC arm-separation (board #94/#108 R3, 2026-09-02): mean margin_norm vs mean winner-z (SNR) for the
CLEAN, NOISE-DEGRADED, and LESION arms at a given bundle scale -- so a scale-invariant read (winner_z) can be
placed with a real clean-vs-degraded gap at BOTH scales.

Mirrors the R3 verify's exact clean/degraded arms (Gaussian phase jitter NOISE_SIGMA=0.12 on the target fact's
composite). Captures, per turn, the mean over the target SVO roles of: margin_norm ((top-runner)/top), and
winner_z ((top-nonwin_mean)/nonwin_std). Read-only monkeypatch of _cleanup_all_score_stats (production path
unchanged).

  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_confidence_100k_recalib/diagnose_arms.py --bundle 15k
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_confidence_100k_recalib/diagnose_arms.py --bundle 100k
"""
from __future__ import annotations
import argparse, json, os, sys, time

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

BUNDLES = {"15k": "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k",
           "100k": "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"}
Q = "who does asimov isaac work for?"
EXPECTED_SVO = ["asimov_isaac", "employer", "university_of_boston"]
NOISE_SIGMA = 0.12
TGT = set(EXPECTED_SVO)

_CAP: list = []
_orig = RFC.RFPhasorComposer._cleanup_all_score_stats


def _patched(self, rec, words=None):
    out = _orig(self, rec, words=words)
    try:
        w = words if words is not None else self.words
        if len(rec) and len(w) > 1:
            rec_z = np.exp(2j * np.pi * np.asarray(rec))
            cb = np.stack([np.exp(2j * np.pi * self.concepts[x]) for x in w])
            sims = (rec_z @ self._cleanup_conj(cb).T).real / float(self.D)
            for i in range(len(rec)):
                row = sims[i]; order = np.argsort(row)
                ti = int(order[-1]); ri = int(order[-2])
                nonwin = np.delete(row, ti)
                top = float(row[ti]); runner = float(row[ri])
                _CAP.append({"winner": str(w[ti]), "top": top, "runner": runner,
                             "margin_norm": (float((max(top,0.)-max(runner,0.))/(max(top,0.)+1e-9)) if top>0 else 0.),
                             "winner_z": float((top - float(nonwin.mean()))/(float(nonwin.std())+1e-9))})
    except Exception as e:  # noqa: BLE001
        _CAP.append({"error": repr(e)})
    return out


RFC.RFPhasorComposer._cleanup_all_score_stats = _patched


def build_chat(bundle_path, seed):
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(bundle_path, extra_kwargs={"enable_codebook_cache": True, "enable_decode_escalation": True})
    inner = _inner_agent(agent); inner.composer = TieredFactStore(inner.composer, ltm)
    return ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer()), inner.composer


_sid = [0]


def ask(chat):
    ck = (f"s{_sid[0]:04d}", "tiny-demo", "stub"); _sid[0] += 1
    S._BRAIN_CHATS[ck] = chat
    r = S.brain_chat(S.BrainChatRequest(session=ck[0], message=Q, brain="tiny-demo", reset=False, rich=True, renderer="stub"))
    return json.loads(bytes(r.body))


def arm_stats(d):
    _CAPn = [x for x in _CAP if x.get("winner") in TGT]
    # take the last occurrence of each target role (the trace-producing decode)
    by = {}
    for x in _CAPn:
        by[x["winner"]] = x
    mns = [by[k]["margin_norm"] for k in by]
    zs = [by[k]["winner_z"] for k in by]
    mc = d.get("metacog") or {}; cf = d.get("confidence_forthcoming") or {}
    return {"mean_margin_norm": float(np.mean(mns)) if mns else None,
            "mean_winner_z": float(np.mean(zs)) if zs else None,
            "mean_role_conf_reported": mc.get("mean_role_conf"),
            "confident": cf.get("confident"), "n_sentences": d.get("n_sentences"),
            "recall_correct": (d.get("recalled_svo") == EXPECTED_SVO), "abstained": d.get("abstained"),
            "per_role": by}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", choices=list(BUNDLES), required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import logging; logging.disable(logging.INFO)
    t0 = time.time()
    chat, composer = build_chat(BUNDLES[a.bundle], a.seed)

    sh = composer.ltm.shard_for(EXPECTED_SVO[0])
    idx = next(i for i, (f, _c) in enumerate(sh.kb) if f.get("agent") == EXPECTED_SVO[0] and f.get("action") == EXPECTED_SVO[1])
    base_fact, base_comp = sh.kb[idx]; base_comp = np.array(base_comp, copy=True)

    res = {}
    os.environ.pop("BRAIN_METACOG_LESION", None)
    _CAP.clear(); res["clean"] = arm_stats(ask(chat))

    rng = np.random.default_rng(1000 + a.seed)
    sh.kb[idx] = (base_fact, base_comp + rng.normal(0.0, NOISE_SIGMA, size=base_comp.shape))
    _CAP.clear(); res["degraded"] = arm_stats(ask(chat))
    sh.kb[idx] = (base_fact, base_comp)

    os.environ["BRAIN_METACOG_LESION"] = "1"
    _CAP.clear(); res["lesion"] = arm_stats(ask(chat))
    os.environ.pop("BRAIN_METACOG_LESION", None)

    dt = time.time() - t0
    out = {"bundle": a.bundle, "seed": a.seed, "noise_sigma": NOISE_SIGMA, "elapsed_s": dt, "arms": res}
    outp = a.out or os.path.join(_HERE, f"arms_{a.bundle}_seed{a.seed}.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    for arm in ("clean", "degraded", "lesion"):
        r = res[arm]
        print(f"[{a.bundle}] {arm:>9}: margin_norm={r['mean_margin_norm']} winner_z={r['mean_winner_z']} "
              f"role_conf={r['mean_role_conf_reported']} confident={r['confident']} n={r['n_sentences']} "
              f"correct={r['recall_correct']} abstain={r['abstained']}")
    print(f"   wrote {os.path.relpath(outp, _REPO)} ({dt:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
