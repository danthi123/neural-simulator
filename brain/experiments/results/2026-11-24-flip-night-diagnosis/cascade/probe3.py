# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (cascade test, pool40 CPU fp32)
# Expected-fail: no (a read-out on banked homes)
# Flip-night cascade-vs-weights probe (scratch; pool40 only). Night-15 trunk battery on copies of the final trunk home,
# opened exactly like brain.eval.life.probe(), in three conditions: A as banked, B every question from a fresh state,
# C told H/rotating questions before greet and the name rows.   python probe3.py <seed> <home> <life.json> <out.json>
import json, os, re, shutil, sys, time, socket
import torch
from brain import clock
from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval import life

seed, home, lj, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
NIGHT = 15
res = json.load(open(lj))
assert res["seed"] == seed and res["days"] == 15
brain_kw = res["brain_kw"]
sched, ledger = life.build_life(seed, 15, "dev")
assert json.loads(json.dumps(ledger)) == res["ledger"], "ledger differs from the banked life"
qs = life.battery(seed, NIGHT, ledger)
banked = {p["arm"]: p["rows"] for p in res["probes"] if p["night"] == NIGHT and p.get("line", "trunk") == "trunk" and not p.get("tag")}
assert [q["text"] for q in qs] == [r["you"] for r in banked["intact"]], "battery differs from the banked rows"

REF = re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai", re.I)                       # classify.py
REF2 = re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai|i think|i can |i'd|i'm glad|i hope|i remember|i'll", re.I)  # fresh2.py
def own(t): return bool(re.search(r"\b(my|mine)\b", t, re.I) or re.match(r"I(?:'m| am| | \w+)", t.strip())) and not REF.search(t)
def own2(t):
    t = t.strip()
    return bool(re.search(r"\b(my|mine)\b", t, re.I) or re.match(r"I(?:'m| am| | \w+)", t)) and not REF2.search(t) and "my friend" not in t.lower()
def told(r): return bool((ledger.get(r.get("item") or "") or {}).get("telling"))
def count(rows):
    t = [r for r in rows if told(r)]
    return {"n_told": len(t), "own": sum(own(r["brain"]) for r in t), "own_strict2": sum(own2(r["brain"]) for r in t),
            "refusal": sum(bool(REF.search(r["brain"])) for r in t), "ok": sum(bool(r.get("ok")) for r in t),
            "refusal_loop": sum(bool(REF.search(r["brain"])) for r in t) >= 4}

def order_C(qs):
    opens = [q for q in qs if q["kind"].startswith("open-")]
    first = [q for q in qs if q["kind"] in ("H", "rotating") and ledger[q["item"]].get("telling")]
    rest = [q for q in qs if q not in opens and q not in first]
    return opens + first + rest

torch.set_num_threads(os.cpu_count())
cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
sim = life.SimClock(); clock.set(sim.now)
work = os.path.join(os.path.dirname(os.path.abspath(out)), f"work_s{seed}"); os.makedirs(work, exist_ok=True)
R = {"seed": seed, "night": NIGHT, "host": socket.gethostname(), "device": "cpu fp32", "brain_kw": brain_kw, "arms": {}}
T0 = time.time()
for arm in ("hippocampus-off", "intact"):
    h_off, c_off = life.ARMS[arm]
    R["arms"][arm] = {"banked": {"rows": banked[arm], **count(banked[arm])}}
    for cond in ("A", "B", "C"):
        cp = os.path.join(work, f"probe_L{NIGHT}_{arm}_{cond}")
        shutil.rmtree(cp, ignore_errors=True); shutil.copytree(home, cp)
        t0 = time.time()
        sim.to(life.day_start(NIGHT + 1, True))           # the life ran --weekend
        life._reseed(seed, 500 + NIGHT)
        b = Brain(cp, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
        try:
            assert not b.pending()
            b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
            if cond == "A":
                rows = life.run_battery(b, qs, ledger, NIGHT)
            elif cond == "B":
                rows = []
                for q in qs:
                    b.state = [x.clone() for x in b.base]
                    rows += life.run_battery(b, [q], ledger, NIGHT)
            else:
                rows = life.run_battery(b, order_C(qs), ledger, NIGHT)
        finally:
            b.release(); shutil.rmtree(cp, ignore_errors=True)
        rec = {"rows": rows, **count(rows), "seconds": round(time.time() - t0, 1)}
        if cond == "A":
            m = [x["brain"] == y["brain"] for x, y in zip(rows, banked[arm])]
            rec["match_banked"] = sum(m); rec["match_banked_told"] = sum(a for a, x in zip(m, rows) if told(x)); rec["n_rows"] = len(m)
        R["arms"][arm][cond] = rec
        print(seed, arm, cond, {k: v for k, v in rec.items() if k != "rows"}, flush=True)
        for r in rows:
            if not r["kind"].startswith("open-"): print("    ", r["kind"], r.get("item"), "|", r["you"], "->", r["brain"][:90], flush=True)
        json.dump(R, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    a = R["arms"][arm]
    a["verdict"] = "UNDEFINED" if a["A"]["own"] < 4 else "WEIGHTS" if a["B"]["own"] >= 4 else "CASCADE" if a["B"]["own"] <= 1 else "BETWEEN (B 2-3/8)"
    print(seed, arm, "VERDICT", a["verdict"], flush=True)
R["wall_seconds"] = round(time.time() - T0, 1)
json.dump(R, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
clock.set(None)
