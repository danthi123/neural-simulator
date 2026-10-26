"""Arms (f) injection stress and (h) quench necessity, organ level, from the emo-bounded-cond worktree. The flag comes from the
ENVIRONMENT (as in production). Driver-side only (no repo edit): a wrapper on SimulationBridge.inject_explicit_wiring puts an
outlier appraisal neuron (s101's a/b/d, capacitance C) into one off-channel-capable ensemble BEFORE the wiring is built.
Usage: stress.py <flag 0|1> inject <tag> seeds...   |   stress.py <flag 0|1> quench <tag> seeds..."""
import os, sys, json, time, re
WT = "/home/dant123/Projects/sim-rev/wt-emobc"
flag, mode, tag = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(x) for x in sys.argv[4:]]
os.environ["BRAIN_EMOTION_CATEGORY_BOUNDED"] = flag
os.environ["SIM_BACKEND"] = "numpy"; os.environ["PYTHONHASHSEED"] = "0"; os.environ["SIM_NO_PROVENANCE"] = "1"
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
os.chdir(WT); sys.path.insert(0, WT)
import numpy as np
from sim import SimulationBridge
from sim.backend import to_host
from research.runners import emotion_category_organ as E
from research.runners import emotion_category_dev as D
assert E.__file__.startswith(WT)
CATS = E.CATS
SWAP = D.SITUATION_SWAP

INJECT = {"idx": None}
_orig_wire = SimulationBridge.inject_explicit_wiring
def _wired(self, plan, output_inhibitory_indices=None):
    if INJECT["idx"] is not None:
        idx, (cval, a, b_, d) = INJECT["idx"]
        self.cp_izh_C[idx] = cval; self.cp_izh_a[idx] = a; self.cp_izh_b[idx] = b_; self.cp_izh_d_increment[idx] = d
    return _orig_wire(self, plan, output_inhibitory_indices)
SimulationBridge.inject_explicit_wiring = _wired

def layout():
    o = E.EmotionCategoryOrgan(seed=1); o._ensure()
    return {c: int(o.app_idx[c][3]) for c in CATS}

def protocol(org, seed, tag):
    res = {}
    for name, swap in (("intact", False), ("permuted", True)):
        key = (tag, name, seed); reads = {}
        for c in CATS:
            for _ in range(2):
                org.observe(key, D.situation_text(c, swap))
            r = org.read(key)
            reads[c] = {"cat": r["category"], "slot": max(r["slot_counts"].values()), "inv": r["read_invariant"], "co": r["co_commit"],
                        "latched": r["latched"], "oor_count_session": r["oor_count_session"], "readback": r["bounded_readback"]}
        want = {c: (SWAP[c] if swap else c) for c in CATS}
        res[name] = {"reads": reads, "ok": all(reads[c]["cat"] == want[c] and reads[c]["inv"] for c in CATS)}
    return res

def quench_gain(org):
    b = org._bridge
    idx = b._transmission_gate_indices_gpu[E.GATE_QUENCH]
    g = np.asarray(to_host(b.cp_transmission_gain))[np.asarray(to_host(idx))]
    return {"n_syn": int(g.size), "gain_min": float(g.min()), "gain_max": float(g.max()), "gate_value": float(b._transmission_gate_values[E.GATE_QUENCH])}

out = {"flag": flag, "mode": mode, "seeds": seeds, "runs": {}}
path = "/home/dant123/Projects/sim-rev/emobc_scratch/battery/stress_%s.json" % tag
if mode == "inject":
    L = layout()
    for s in seeds:
        for ens in CATS:
            for cval in (39.6, 53.6):
                INJECT["idx"] = (L[ens], (cval, 0.023, -1.877, 62.984))
                t = time.time()
                org = E.EmotionCategoryOrgan(seed=s).train()
                r = protocol(org, s, "st")
                ok = r["intact"]["ok"] and r["permuted"]["ok"]
                eng = {"bounded_flag": org.bounded, "readback": sorted({x["readback"] for v in ("intact", "permuted") for x in r[v]["reads"].values()}),
                       "oor_session_max": max(x["oor_count_session"] for v in ("intact", "permuted") for x in r[v]["reads"].values()),
                       "injected_C": float(to_host(org._bridge.cp_izh_C)[L[ens]])}
                out["runs"]["%d|%s|%s" % (s, ens, cval)] = {"ok": bool(ok), "intact_ok": r["intact"]["ok"], "permuted_ok": r["permuted"]["ok"],
                    "intact": {c: v["cat"] for c, v in r["intact"]["reads"].items()}, "permuted": {c: v["cat"] for c, v in r["permuted"]["reads"].items()}, "eng": eng}
                print("[flag=%s] seed %d inject %-8s C=%-5s ok=%s intact=%s permuted=%s eng=%s %.0fs" % (flag, s, ens, cval, ok, r["intact"]["ok"], r["permuted"]["ok"], eng, time.time() - t), flush=True)
                json.dump(out, open(path, "w"), indent=1)
elif mode == "pinned":
    for s in seeds:
        INJECT["idx"] = None
        t = time.time()
        org = E.EmotionCategoryOrgan(seed=s).train()
        r = protocol(org, s, "pin")
        ok = r["intact"]["ok"] and r["permuted"]["ok"]
        eng = {"bounded_flag": org.bounded, "readback": sorted({x["readback"] for v in ("intact", "permuted") for x in r[v]["reads"].values()}),
               "oor_session_max": max(x["oor_count_session"] for v in ("intact", "permuted") for x in r[v]["reads"].values())}
        out["runs"][str(s)] = {"ok": bool(ok), "intact_ok": r["intact"]["ok"], "permuted_ok": r["permuted"]["ok"],
            "intact": {c: (v["cat"], v["slot"]) for c, v in r["intact"]["reads"].items()}, "permuted": {c: (v["cat"], v["slot"]) for c, v in r["permuted"]["reads"].items()}, "eng": eng}
        print("[flag=%s] PINNED seed %d ok=%s intact=%s permuted=%s eng=%s %.0fs" % (flag, s, ok, out["runs"][str(s)]["intact"], out["runs"][str(s)]["permuted"], eng, time.time() - t), flush=True)
        json.dump(out, open(path, "w"), indent=1)
elif mode == "quench":
    for s in seeds:
        res = {}
        for qarm in ("open", "closed"):
            INJECT["idx"] = None
            org = E.EmotionCategoryOrgan(seed=s).train()
            if qarm == "closed":
                org._bridge.set_transmission_gate(E.GATE_QUENCH, 0.0)
            gain = quench_gain(org)
            key = ("q", qarm, s); reads = {}
            for c in CATS:
                for _ in range(2):
                    org.observe(key, D.situation_text(c))
                r = org.read(key)
                reads[c] = {"cat": r["category"], "co": r["co_commit"], "latched": r["latched"], "slot_counts": r["slot_counts"],
                            "readback": r["bounded_readback"], "oor_session": r["oor_count_session"]}
            res[qarm] = {"reads": reads, "quench": gain}
            print("[flag=%s] seed %d quench %-6s gain=%s reads=%s" % (flag, s, qarm, gain, {c: (v["cat"], v["latched"]) for c, v in reads.items()}), flush=True)
        out["runs"][str(s)] = res
        json.dump(out, open(path, "w"), indent=1)
print("done")
