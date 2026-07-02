"""EMERGE-24 / toward-language — ONLINE GROWTH: the emergent producer LEARNS A NEW GROUNDED FACT LIVE (mid-life, on the
same spiking bridge, no reset), then PRODUCES the new fact, RETAINS the old facts (no catastrophic forgetting), and
still ABSTAINS on a truly-novel cue (the moat holds through growth). This is the growth axis of the toward-language
chain (EMERGE-23 producer + EMERGE-14 on-bridge incremental learning) -- a brain that grows its knowledge from
experience, the master directive's core, biology-native (spiking, emergent, no `sim/` edit).

MECHANISM: the three-block content-bearing codes (POS-class = grammar / content = the specific word / family =
generalization) live on a pre-allocated DENSE cross-column coincidence pool; a fact is LEARNED by the committed `sim/`
three-term kernel raising the right permanences. GROWTH = potentiating a NEW fact's adjacent-pair pathways on the SAME
bridge that already holds the base facts (the new word's columns pre-exist at the sub-connected p_init, so learning a
new fact is a permanence-rise, exactly as a fixed-topology cortex grows an engram). Because each fact uses a DISTINCT
verb, the new fact's pathway is disjoint from the old ones -> learning it does NOT overwrite them (no catastrophic
forgetting). The new subject (fox) is a fully-disjoint code with no family, so BEFORE teaching it ABSTAINS (genuinely
unknown), and AFTER teaching it produces the taught fact -> the "learns live" signature.

TASK: base facts "dog chased ball" / "cat ate fish" (trained first); then teach "fox saw seed" LIVE. Check (i) fox
ABSTAINS before teaching (genuinely new); (ii) fox -> "fox saw seed" after teaching (learned-new); (iii) dog/cat still
produce their facts after teaching (RETENTION, no forgetting); (iv) zzz still ABSTAINS (moat). ANTI-CHEATS: learned-new
+ retention + moat accuracies; pre-teach-abstain = 1.0 (the fact was genuinely absent); dAP-LESION collapses; no
teacher; multi-seed. Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_emerge24_online_growth.json")

CLASS_COLS = {"NOUN": [0, 1, 2], "VERB": [3, 4, 5]}
CONTENT = {"dog": [6, 7], "cat": [8, 9], "ball": [10, 11], "fish": [12, 13], "chased": [14, 15], "ate": [16, 17],
           "fox": [18, 19], "seed": [20, 21], "saw": [22, 23], "zzz": [30, 31]}
FAMILY = {"dog": [24, 25], "cat": [26, 27]}                                      # fox: none (a clean, disjoint new word)
WORDCLASS = {"dog": "NOUN", "cat": "NOUN", "ball": "NOUN", "fish": "NOUN", "fox": "NOUN", "seed": "NOUN", "zzz": "NOUN",
             "chased": "VERB", "ate": "VERB", "saw": "VERB"}
FRAME = ["NOUN", "VERB", "NOUN"]
BASE = [["dog", "chased", "ball"], ["cat", "ate", "fish"]]                       # the pre-existing knowledge
NEW = ["fox", "saw", "seed"]                                                     # taught LIVE, mid-life
ACT_TH = 2


def word2cols(w):
    return list(CLASS_COLS[WORDCLASS[w]]) + list(CONTENT[w]) + list(FAMILY.get(w, []))


def distinguishing_cols(w):
    return list(CONTENT[w]) + list(FAMILY.get(w, []))


class GrowthProducer:
    def __init__(self, seed=42, lesion=False):
        self.nE = 8
        self.M = 1 + max(c for w in WORDCLASS for c in word2cols(w))
        self.b, self.ci, self.row, self.col = build_pool_bridge(self.M, self.nE, seed, act_th=ACT_TH,
                                                                coincidence=(not lesion))
        self.z = np.zeros(self.M * self.nE); self.lesion = lesion

    def _sdr(self, w):
        return set(c * self.nE + 0 for c in word2cols(w))

    def _dist_sdr(self, w):
        return set(c * self.nE + 0 for c in distinguishing_cols(w))

    def learn(self, facts, epochs):
        """Incrementally potentiate the given facts' adjacent-pair pathways on the CURRENT bridge (no reset -> growth)."""
        for _ in range(epochs):
            for sent in facts:
                for a, bnext in zip(sent, sent[1:]):
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(a), self._sdr(bnext),
                                        self.z, 0.14, 0.02, 1.0)

    def _predict_primed(self, active, thresh=-40.0):
        if getattr(self.b, "cp_v_apical", None) is None and not self.b.core_config.enable_coincidence_detection:
            return set()
        ab = np.zeros(len(self.ci), bool)
        for i in active:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return set()
        vap = _host(vap)[self.ci]
        return set(int(i) for i in np.where(vap > thresh)[0])

    def generate(self, subject):
        out = [subject]; pos = [WORDCLASS[subject]]
        active = self._dist_sdr(subject)
        for step in range(1, len(FRAME)):
            primed = self._predict_primed(active)
            if not primed:
                break
            pc = Counter(int(i) // self.nE for i in primed)
            want = FRAME[step]
            cand = [w for w in WORDCLASS if WORDCLASS[w] == want and w != "zzz"]
            scores = {w: sum(pc.get(c, 0) for c in CONTENT[w]) for w in cand}
            if not scores or max(scores.values()) == 0:
                break
            nxt = max(scores, key=scores.get)
            out.append(nxt); pos.append(WORDCLASS[nxt]); active = self._dist_sdr(nxt)
        if len(out) < len(FRAME):
            return ["<abstain>"], []
        return out, pos


def _run_arm(seed, arm, epochs_base, epochs_new):
    p = GrowthProducer(seed=seed, lesion=(arm == "lesion"))
    p.learn(BASE, epochs_base)                                                   # phase 1: the pre-existing knowledge
    pre_abstain = float(p.generate("fox")[0] == ["<abstain>"])                   # fox unknown BEFORE teaching
    p.learn([NEW], epochs_new)                                                   # phase 2: teach the new fact LIVE
    learned = float(p.generate("fox") == (NEW, FRAME))                           # the new fact now produces
    retention = np.mean([p.generate(f[0]) == (f, FRAME) for f in BASE])          # old facts survive (no forgetting)
    moat = float(p.generate("zzz")[0] == ["<abstain>"])                          # moat holds through growth
    return arm, {"pre_teach_abstain": pre_abstain, "learned_new": learned,
                 "retention": float(retention), "moat": moat}


ARMS = ["htm", "lesion"]


def _demo(seed=42, epochs_base=80, epochs_new=80):
    p = GrowthProducer(seed=seed)
    p.learn(BASE, epochs_base)
    print("\n=== EMERGE-24 online growth (learn a new fact LIVE; no transformer) ===")
    print(f"  base knowledge (trained): {[' '.join(f) for f in BASE]}")
    print(f"  BEFORE teaching: cue 'fox' -> {' '.join(p.generate('fox')[0])}   (should ABSTAIN -- fox is unknown)")
    print(f"  ...teaching '{' '.join(NEW)}' LIVE on the same bridge...")
    p.learn([NEW], epochs_new)
    for subj, note in [("fox", "LEARNED-NEW (taught live)"), ("dog", "RETAINED (no forgetting)"),
                       ("cat", "RETAINED"), ("zzz", "MOAT (still abstains)")]:
        out, pos = p.generate(subj)
        print(f"  cue '{subj}' -> {' '.join(out)}   [{' '.join(pos) if pos else 'ABSTAIN'}]  ({note})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs-base", type=int, default=80)
    ap.add_argument("--epochs-new", type=int, default=80)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0], a.epochs_base, a.epochs_new); return 0
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    print(f"base {[' '.join(f) for f in BASE]} | NEW (taught live) {' '.join(NEW)} | frame {FRAME}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs_base, a.epochs_new); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] pre-teach-ABSTAIN {h['pre_teach_abstain']:.2f} | LEARNED-NEW {h['learned_new']:.2f} "
                  f"| RETENTION {h['retention']:.2f} | MOAT {h['moat']:.2f} || lesion-learned {d['lesion']['learned_new']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        pre, ln, ret, moat = m("htm", "pre_teach_abstain"), m("htm", "learned_new"), m("htm", "retention"), m("htm", "moat")
        les = m("lesion", "learned_new")
        go = bool(ln >= 0.90 and ret >= 0.90 and moat >= 0.90 and pre >= 0.90 and ln >= les + 0.30)
        if go:
            verdict = (f"GO -- the emergent producer LEARNS A NEW GROUNDED FACT LIVE (mid-life, on the same spiking bridge), "
                       f"PRODUCES it ({ln:.2f}), RETAINS the old facts ({ret:.2f}, no catastrophic forgetting), and still "
                       f"ABSTAINS on a novel cue ({moat:.2f}, the moat holds through growth). BEFORE teaching, the new subject "
                       f"ABSTAINS ({pre:.2f} -- genuinely unknown, not pre-existing); dAP-LESION collapses learning ({les:.2f}); "
                       f"no teacher; multi-seed. Growth = potentiating the new fact's disjoint pathway on the pre-allocated "
                       f"coincidence pool via the committed sim/ three-term kernel. => a brain that GROWS its knowledge from "
                       f"experience, biology-native, NO sim/ edit -- the master directive's core, on the toward-language chain.")
        else:
            miss = []
            if ln < 0.90: miss.append(f"learned-new {ln:.2f} < 0.90")
            if ret < 0.90: miss.append(f"retention {ret:.2f} < 0.90 (catastrophic forgetting)")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if pre < 0.90: miss.append(f"pre-teach-abstain {pre:.2f} < 0.90 (fact not genuinely new)")
            if ln < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({ln:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune epochs_new / block sizes vs ACT_TH; "
                       "online growth is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge24_online_growth", "verdict": verdict,
               "mechanism": "online growth on the emergent producer: potentiate a NEW fact's disjoint adjacent-pair pathways "
                            "on the SAME bridge (pre-allocated dense coincidence pool + committed sim/ three-term kernel); a "
                            "distinct verb keeps the new pathway disjoint -> the old facts are retained (no catastrophic "
                            "forgetting); the new subject is a disjoint code -> abstains before, produces after; sim/ unchanged",
               "task": "train base facts, teach a new fact LIVE, verify learned-new + retention + moat + pre-teach-abstain; "
                       "dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs_base": a.epochs_base, "epochs_new": a.epochs_new, "act_th": ACT_TH},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the new fact uses a DISTINCT verb (no shared-verb high-order ambiguity in the object slot). "
                              "Online growth of facts that SHARE a verb/object with the base (requiring high-order context to "
                              "disambiguate) + growth that OVERRIDES a generalization prior are the named next integration "
                              "steps. The deep residual is open-world SEMANTICS (knowledge acquisition beyond told facts)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge24] VERDICT: {verdict}", flush=True)
    print(f"[emerge24] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
