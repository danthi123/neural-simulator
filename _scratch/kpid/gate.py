# Capability: Lifelong learning
# Record: brain/research/2026-11-23-kp-identity-prereg.md (section 2, Step 0 gate); _scratch/kpgate_w/gate_night_g.py, drift_read.py
# Expected-fail: yes, named in the prereg (masking may give back the language win; the cost may sit in the weights; one night may not show the K-P cost)
"""K-P-U Step-0 one-night gate (prereg 2026-11-23-kp-identity section 2). Host-only harness, scratch.

Arms: B (keep_bank None), KP ("prior"), KPU ("prior_u"), KP0 ("prior_0"); one night each from the same B night-15 snapshot
and sleep seed, through brain.eval.one_night (as _scratch/kpgate_w/gate_night_g.py). Reads on every after-home:
  (a) drift: _scratch/kpid/drift_read.py (a device-aware copy of kpgate_w/drift_read.py) on the snapshot with the
      after-adapters as the extra state -> held-out KL/NLL, role_kl U/A on HELDOUT, prose KL, replayed-episode NLL;
  (b) self-name and (c) refusal: life.probe() on the after-home, arms intact + hippocampus-off, questions
      battery(seed, 16, ledger) (ledger from build_life(seed, 15, "dev"), cross-checked against the B life JSON),
      6 replicates: r=0..2 as asked, r=3..5 with the nova row's text replaced by its `reworded` form; probe(seed=S+1000*r).
Subcommands (each writes one JSON and skips work whose JSON exists):
  night ARM SRC DST SEED          -> DST/ (home) + DST.result.json      [KPU needs --bank-from <KP after-home>]
  drift ARM SRC DST SEED OUT      -> OUT (drift JSON of DST's adapters against snapshot SRC)
  probe ARM DST SEED OUT          -> OUT (rows + G7/G8 scores)
  score DIR                       -> DIR/gate_read.json (+ printed table): the section-2 bars, pooled over the seeds found
  smoke DIR                       -> a 0.1B CPU end-to-end run on a tiny home (tiny bank), then score
Common: --model 1.5b --device cpu|cuda --cuda-precision fp16|fp32 --life-json PATH --brain-kw JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

ARMS = {"B": None, "KP": "prior", "KPU": "prior_u", "KP0": "prior_0"}
REPS = 6
NIGHT = 16
LIFE_JSON = os.path.join(REPO, "_scratch", "stage2", "read", "lifeS1F_s{}.json")

# test2 prereg section 5 regexes, copied verbatim (life_costs.py is not yet written; prereg 2026-11-23-kp-identity section 3)
G8_PAT = re.compile(r"^\W*(?:(?:i'?m|i am) sorry,?\s*)?(?:but\s+)?(?:i\s+(?:don'?t|do not)\s+(?:know|understand)\b|i'?m not sure\b|i am not sure\b)")
G8_PHRASE = "don't know what you mean"
FRIEND_PAT = r"\b(?:my name is|i'?m called|i am called|call me|i'?m|i am)\s+{friend}\b"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _cortex(a):
    from brain.cortex import Cortex, resolve_model
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA is not available")
    return Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)


def _brain_kw(a, seed):
    """The B life's own Brain kwargs (life JSON brain_kw), minus sleep_seed (passed as the night seed), overridden by
    --brain-kw. The life's sleep_seed must equal SEED."""
    kw = {}
    lj = a.life_json.format(seed) if a.life_json and a.life_json != "none" else None
    if lj:
        kw = dict(json.load(open(lj))["brain_kw"] or {})
        if kw.pop("sleep_seed", seed) != seed:
            raise SystemExit(f"{lj}: brain_kw sleep_seed != {seed}")
    kw.update(json.loads(a.brain_kw or "{}"))
    kw.pop("keep_bank", None)
    return kw, lj


def _tiny_bank(a):
    if a.bank_n or a.bank_tokens or a.bank_night:
        if not a.smoke_ok:
            raise SystemExit("--bank-n/--bank-tokens/--bank-night are smoke-only")
        import brain.brain as bb
        bb.KEEP_BANK_N = a.bank_n or bb.KEEP_BANK_N
        bb.KEEP_BANK_TOKENS = a.bank_tokens or bb.KEEP_BANK_TOKENS
        bb.KEEP_BANK_NIGHT = a.bank_night or bb.KEEP_BANK_NIGHT


# ---------------------------------------------------------------- night
def cmd_night(a, cortex=None):
    from brain.brain import Brain, KEEP_BANK_FRAME, KEEP_BANK_SEED_OFFSET
    from brain.eval.one_night import one_night
    from brain import clock
    arm, src, dst, seed = a.arm, a.src, a.dst, a.seed
    if os.path.exists(dst + ".result.json"):
        print("skip night (result exists)", dst)
        return json.load(open(dst + ".result.json"))
    shutil.rmtree(dst, ignore_errors=True)    # a cut night leaves a partial home: redo it
    _tiny_bank(a)
    kw, lj = _brain_kw(a, seed)
    bank = None
    if arm == "KPU":
        if not a.bank_from:
            raise SystemExit("KPU needs --bank-from <KP after-home>")
        bf = os.path.join(a.bank_from, "keep_bank.json")
        blob = json.load(open(bf))
        if blob.get("frame") != KEEP_BANK_FRAME or blob.get("seed") != seed + KEEP_BANK_SEED_OFFSET:
            raise SystemExit(f"{bf}: frame {blob.get('frame')!r} seed {blob.get('seed')} is not the K-P bank of seed {seed}")
        bank = {"from": bf, "sha256": _sha(bf)}
    elif a.bank_from:
        raise SystemExit("--bank-from is for KPU only")
    meta = json.load(open(os.path.join(src, "meta.json")))
    if meta.get("keep_bank") is not None:
        raise SystemExit(f"{src} is not a B snapshot (keep_bank={meta['keep_bank']!r})")
    _t = [meta["last_sleep"] + 86400.0]
    clock.set(lambda: (_t.__setitem__(0, _t[0] + 1e-3) or _t[0]))
    T = {}
    orig = Brain._keep_bank

    def timed(self, *x, **k):
        t = time.time(); r = orig(self, *x, **k); T["bank_s"] = T.get("bank_s", 0) + time.time() - t; return r
    Brain._keep_bank = timed
    cortex = cortex or _cortex(a)
    t0 = time.time()
    try:
        r = one_night(src, dst, keep_bank=ARMS[arm], cortex=cortex, sleep_seed=seed,
                      bank_from=bank["from"] if bank else None, **kw)
    finally:
        Brain._keep_bank = orig
        clock.set(None)
    T["night_total_s"] = time.time() - t0
    r.update({"arm": arm, "seed": seed, "src": src, "timing": T, "strategy": getattr(cortex, "strategy", None),
              "brain_kw": kw, "life_json": lj, "src_adapters_sha256": _sha(os.path.join(src, "cortex_adapters.pt"))})
    bp = os.path.join(dst, "keep_bank.json")
    if os.path.exists(bp):
        r["bank_sha256"] = _sha(bp)
        if bank:
            r["bank_copied_from"] = bank
            if r["bank_sha256"] != bank["sha256"]:
                raise SystemExit(f"KPU bank hash {r['bank_sha256'][:12]} != K-P bank {bank['sha256'][:12]}")
    json.dump(r, open(dst + ".result.json", "w"), indent=1, default=str)
    print("RESULT", json.dumps(r, default=str))
    return r


# ---------------------------------------------------------------- drift (a)
def cmd_drift(a):
    if os.path.exists(a.out):
        print("skip drift (exists)", a.out)
        return
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH", "")) if p), DRIFT_MODEL=a.model, DRIFT_DEVICE=a.device, DRIFT_PRECISION=a.cuda_precision)
    if a.sets:
        env["SETS"] = a.sets
    label = f"kpid_{a.arm}_s{a.seed}_{os.getpid()}"
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "drift_read.py"), a.src, label, a.out,
                    os.path.join(a.dst, "cortex_adapters.pt")], env=env, check=True)


# ---------------------------------------------------------------- probes (b, c)
def norm(s):
    from brain.eval.life_leak import _norm
    return _norm(s)


def is_refusal(reply):
    t = norm(reply)
    return bool(G8_PAT.match(re.split(r"[.!?]", t)[0])) or G8_PHRASE in t


def answerable(row):
    k, it = row["kind"], row.get("item")
    return ((k == "H" and it not in ("dan", "nova")) or k == "rotating" or k == "neardup") and row.get("told") is not False


def non_identity(row):
    return (row["kind"] == "H" and row.get("item") not in ("dan", "nova")) or row["kind"] in ("rotating", "neardup", "never")


def score_rows(rows, friend="Dan"):
    """rows: probe rows tagged with arm and rep. G7 = nova rows ok False; friend-name rate on nova rows (report only);
    G8a = refusal rate on answerable rows, pooled and per arm; G8b = per (arm, rep) battery share over non-identity rows."""
    fp = re.compile(FRIEND_PAT.format(friend=re.escape(friend)), re.I)
    nova = [r for r in rows if r["kind"] == "H" and r.get("item") == "nova"]
    ans = [r for r in rows if answerable(r)]
    out = {"G7_n": len(nova), "G7_wrong": sum(1 for r in nova if not r.get("ok")),
           "friend_name": sum(1 for r in nova if fp.search(norm(r["brain"]))),
           "G8a_n": len(ans), "G8a_refusals": sum(1 for r in ans if is_refusal(r["brain"])), "G8a_by_arm": {},
           "G7_by_rep_kind": {}, "G8b_battery_share": {}}
    out["G7_rate"] = out["G7_wrong"] / out["G7_n"] if nova else None
    out["G8a_rate"] = out["G8a_refusals"] / out["G8a_n"] if ans else None
    for arm in sorted({r["arm"] for r in rows}):
        aa = [r for r in ans if r["arm"] == arm]
        out["G8a_by_arm"][arm] = (sum(1 for r in aa if is_refusal(r["brain"])) / len(aa)) if aa else None
    for kind in ("asked", "reworded"):
        nn = [r for r in nova if r["nova_text"] == kind]
        out["G7_by_rep_kind"][kind] = (sum(1 for r in nn if not r.get("ok")) / len(nn)) if nn else None
    for key in sorted({(r["arm"], r["rep"]) for r in rows}):
        ni = [r for r in rows if (r["arm"], r["rep"]) == key and non_identity(r)]
        out["G8b_battery_share"][f"{key[0]}/r{key[1]}"] = (sum(1 for r in ni if is_refusal(r["brain"])) / len(ni)) if ni else None
    return out


class GateClock:
    """life.SimClock whose to() always lands on the morning after the gate night (after-home last_sleep + 10 h, the
    09:00 after a 23:00 night, as the life's probes): life.probe's day_start(night + 1) assumes the life's own calendar,
    which the gate night (last_sleep + 1 day) does not follow."""
    def __init__(self, morning):
        self.morning, self.base, self.n = float(morning), float(morning), 0

    def now(self):
        self.n += 1
        return self.base + self.n * 1e-3

    def to(self, t):
        self.base, self.n = self.morning, 0


def cmd_probe(a, cortex=None):
    from brain import clock
    from brain.eval import life
    if os.path.exists(a.out):
        print("skip probe (exists)", a.out)
        return json.load(open(a.out))
    kw, lj = _brain_kw(a, a.seed)
    seed = a.seed
    sched, ledger = life.build_life(seed, a.days, "dev")
    if lj:
        saved = json.load(open(lj))["ledger"]
        if json.loads(json.dumps(ledger)) != saved:
            raise SystemExit(f"ledger of build_life({seed}, {a.days}) differs from {lj}")
    meta = json.load(open(os.path.join(a.dst, "meta.json")))
    sim = GateClock(meta["last_sleep"] + 10 * 3600)
    clock.set(sim.now)
    cortex = cortex or _cortex(a)
    qs0 = life.battery(seed, NIGHT, ledger)
    nova_rw = life.LB.ALL["nova"].reworded
    work = a.out + ".work"
    os.makedirs(work, exist_ok=True)
    rows, recs, t0 = [], [], time.time()
    reps = list(range(REPS)) if not a.reps else [int(x) for x in a.reps.split(",")]
    try:
        for r in reps:
            kind = "asked" if r < 3 else "reworded"
            qs = [dict(q) for q in qs0]
            if kind == "reworded":
                for q in qs:
                    if q["kind"] == "H" and q["item"] == "nova":
                        q["text"] = nova_rw
            recs_r = life.probe(a.dst, cortex, NIGHT, life.LIGHT_ARMS, qs, ledger, sim, seed + 1000 * r, kw, work)
            for rec in recs_r:
                for row in rec["rows"]:
                    rows.append({**row, "arm": rec["arm"], "rep": r, "nova_text": kind})
                recs.append({"rep": r, "arm": rec["arm"], "summary": rec["summary"], "seconds": rec["seconds"],
                             "adapter_norm": rec["adapter_norm"]})
            print(f"probe {a.arm} s{seed} rep {r} done ({time.time() - t0:.0f}s)", flush=True)
    finally:
        clock.set(None)
        shutil.rmtree(work, ignore_errors=True)
    out = {"arm": a.arm, "seed": seed, "home": a.dst, "night": NIGHT, "reps": reps, "arms": list(life.LIGHT_ARMS),
           "nova_reworded": nova_rw, "brain_kw": kw, "life_json": lj, "strategy": getattr(cortex, "strategy", None),
           "morning": sim.morning, "seconds": round(time.time() - t0, 1),
           "scores": score_rows(rows, kw.get("friend", meta.get("friend", "Dan"))), "battery": recs, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("PROBE", a.arm, seed, json.dumps({k: v for k, v in out["scores"].items() if not isinstance(v, dict)}))
    return out


# ---------------------------------------------------------------- score
def _drift_vals(d):
    st = d["states"]
    snap = st["home"]["sets"]
    after = next(v for k, v in st.items() if k != "home")["sets"]
    rk = after["HELDOUT"].get("role_kl", {})
    return {"kl_snap": snap["HELDOUT"]["kl"], "kl_after": after["HELDOUT"]["kl"],
            "dKL": after["HELDOUT"]["kl"] - snap["HELDOUT"]["kl"], "nll_after": after["HELDOUT"]["nll_on"],
            "role_A": (rk.get("A") or [None])[0], "role_U": (rk.get("U") or [None])[0],
            "prose_kl": after.get("prose_ctrl", {}).get("kl"),
            "replayed_nll": after.get("episodes_replayed", {}).get("nll_on")}


def _engagement(home, res, seed, days=15):
    """K-P section 3.1 bank bars (G0): 512 texts, >= 90% distinct, mean >= 40 tokens, < 10% told key, record present."""
    from brain.eval import life
    bp = os.path.join(home, "keep_bank.json")
    if not os.path.exists(bp):
        return {"pass": False, "why": "no bank file"}
    b = json.load(open(bp))
    T = b["texts"]
    _, ledger = life.build_life(seed, days, "dev")
    keys = life.told_keys(ledger, days)
    told = sum(1 for t in T if life._has(t, keys))
    rec = res.get("keep_record") or {}
    e = {"n": len(T), "distinct": len(set(T)) / max(len(T), 1), "mean_tokens": b.get("mean_tokens"),
         "told_key_share": told / max(len(T), 1), "frame": b.get("frame"), "record": bool(rec),
         "mean_keep_kl": rec.get("mean_keep_kl"), "kept_pos_share": rec.get("kept_pos_share")}
    ok = (e["n"] == 512 and e["distinct"] >= 0.9 and (e["mean_tokens"] or 0) >= 40 and e["told_key_share"] < 0.10
          and e["record"])
    if res.get("keep_bank") == "prior_u":
        ok = ok and e["kept_pos_share"] is not None and 0.3 <= e["kept_pos_share"] <= 0.95 and (e["mean_keep_kl"] or 0) > 0
    e["pass"] = bool(ok)
    return e


def cmd_score(a):
    D = a.dir
    seeds = sorted({int(m.group(1)) for f in os.listdir(D) for m in [re.match(r"probe_B_s(\d+)\.json$", f)] if m})
    per = {}
    for s in seeds:
        per[s] = {}
        for arm in ARMS:
            pj, dj, rj = (os.path.join(D, f"{x}_{arm}_s{s}.json") for x in ("probe", "drift", "home"))
            rj = os.path.join(D, f"home_{arm}_s{s}.result.json")
            if not (os.path.exists(pj) and os.path.exists(dj) and os.path.exists(rj)):
                continue
            p, d, r = json.load(open(pj)), json.load(open(dj)), json.load(open(rj))
            per[s][arm] = {**_drift_vals(d), **{k: v for k, v in p["scores"].items()},
                           "engagement": _engagement(os.path.join(D, f"home_{arm}_s{s}"), r, s, a.days)
                           if arm != "B" else None, "strategy": p.get("strategy"), "mean_keep": r.get("mean_keep")}
    mean = lambda v: (sum(v) / len(v)) if v and None not in v else None

    def pooled(arm, k):
        return mean([per[s][arm][k] for s in seeds if arm in per[s]]) if all(arm in per[s] for s in seeds) else None

    def rate(arm, num, den):
        if not all(arm in per[s] for s in seeds):
            return None
        n = sum(per[s][arm][den] for s in seeds)
        return sum(per[s][arm][num] for s in seeds) / n if n else None
    out = {"dir": D, "seeds": seeds, "per_seed": per, "pooled": {}, "bars": {}}
    for arm in ARMS:
        out["pooled"][arm] = {"dKL": pooled(arm, "dKL"), "kl_after": pooled(arm, "kl_after"),
                              "replayed_nll": pooled(arm, "replayed_nll"), "role_A": pooled(arm, "role_A"),
                              "role_U": pooled(arm, "role_U"), "G7": rate(arm, "G7_wrong", "G7_n"),
                              "G8a": rate(arm, "G8a_refusals", "G8a_n"),
                              "friend_name": rate(arm, "friend_name", "G7_n")}
    P = out["pooled"]
    have = lambda *xs: all(x is not None for x in xs)
    if have(P["B"]["G7"], P["KP"]["G7"], P["B"]["G8a"], P["KP"]["G8a"]):
        informative = not (P["KP"]["G7"] - P["B"]["G7"] < 0.10 and P["KP"]["G8a"] <= P["B"]["G8a"] + 0.03)
    else:
        informative = None
    out["bars_2_4_informative"] = informative
    for var in ("KPU", "KP0"):
        v, B, K = P[var], P["B"], P["KP"]
        if not have(v["dKL"], B["dKL"], K["dKL"]):
            out["bars"][var] = "MISSING"
            continue
        b = {}
        b["1_language"] = (v["dKL"] <= K["dKL"] + 0.30) and all(per[s][var]["dKL"] < per[s]["B"]["dKL"] for s in seeds)
        b["1_dir_per_seed"] = {s: per[s][var]["dKL"] < per[s]["B"]["dKL"] for s in seeds}
        b["2_selfname"] = (v["G7"] <= B["G7"] + 0.10 and v["G7"] <= K["G7"]) if have(v["G7"], B["G7"], K["G7"]) else None
        b["3_writes"] = (v["replayed_nll"] <= B["replayed_nll"] + 0.2) if have(v["replayed_nll"], B["replayed_nll"]) else None
        b["4_refusal"] = (v["G8a"] <= B["G8a"] + 0.03 and v["G8a"] <= K["G8a"]) if have(v["G8a"], B["G8a"], K["G8a"]) else None
        eng = [per[s][var]["engagement"]["pass"] for s in seeds]
        b["5_engagement"] = all(eng)
        b["mechanism_A_ratio"] = (v["role_A"] / K["role_A"]) if have(v["role_A"], K["role_A"]) and K["role_A"] else None
        if not b["5_engagement"]:
            verdict = "UNDEFINED (engagement)"
        elif informative is None:
            verdict = "MISSING (B/KP probes)"
        elif informative is False:
            core = b["1_language"] and b["3_writes"]
            mech = b["mechanism_A_ratio"] is not None and b["mechanism_A_ratio"] >= 1.25
            verdict = ("PASS (bars 2/4 uninformative; mechanism engaged)" if core and mech else
                       "UNDEFINED (bars 2/4 uninformative; A-position KL within 1.25x K-P)" if core else "FAIL")
        else:
            verdict = "PASS" if all(b[k] for k in ("1_language", "2_selfname", "3_writes", "4_refusal")) else "FAIL"
        b["verdict"] = verdict
        out["bars"][var] = b
    json.dump(out, open(os.path.join(D, "gate_read.json"), "w"), indent=1, default=str)
    f = lambda x: "-" if x is None else f"{x:.3f}"
    print("arm   dKL    kl_after replNLL role_A role_U  G7    G8a   friend")
    for arm in ARMS:
        q = P[arm]
        print(f"{arm:5s} {f(q['dKL'])} {f(q['kl_after'])} {f(q['replayed_nll'])} {f(q['role_A'])} {f(q['role_U'])} "
              f"{f(q['G7'])} {f(q['G8a'])} {f(q['friend_name'])}")
    print("bars 2/4 informative:", informative)
    print(json.dumps(out["bars"], indent=1, default=str))
    return out


# ---------------------------------------------------------------- smoke (0.1B CPU, tiny home and bank)
def cmd_smoke(a):
    from brain.brain import Brain
    from brain.cortex import Cortex, resolve_model
    from brain.eval import life
    from brain import clock
    D = a.dir
    os.makedirs(D, exist_ok=True)
    seed = 319
    src = os.path.join(D, f"src_s{seed}")
    kwj = json.dumps({"sleep_steps": 4})
    cortex = Cortex(model_path=resolve_model("0.1b"), device="cpu")
    if not os.path.exists(src):
        sched, _ = life.build_life(seed, 2, "dev")
        sim = life.SimClock()
        clock.set(sim.now)
        for day in sched:
            sim.to(life.day_start(day["day"]))
            b = Brain(src, cortex=cortex, sleep_seed=seed, sleep_steps=4)
            for t in day["turns"][:4]:
                b.hear(t["text"])
            b.save()
            sim.to(life.day_start(day["day"]) + life.STEP_HOURS * 3600)
            b.close()
        clock.set(None)
    common = ["--model", "0.1b", "--device", "cpu", "--life-json", "none", "--brain-kw", kwj, "--days", "2",
              "--bank-n", "8", "--bank-tokens", "48", "--bank-night", "4", "--smoke-ok"]
    run = lambda *x: main(list(x) + common, cortex=cortex)
    for arm in ARMS:
        dst = os.path.join(D, f"home_{arm}_s{seed}")
        extra = ["--bank-from", os.path.join(D, f"home_KP_s{seed}")] if arm == "KPU" else []
        run("night", arm, src, dst, str(seed), *extra)
        run("drift", arm, src, dst, str(seed), os.path.join(D, f"drift_{arm}_s{seed}.json"), "--sets",
            "HELDOUT,episodes_replayed,prose_ctrl")
        run("probe", arm, dst, str(seed), os.path.join(D, f"probe_{arm}_s{seed}.json"), "--reps", "0,3")
    out = main(["score", D, "--days", "2"])
    r = out["per_seed"][seed]
    assert set(r) == set(ARMS), r.keys()
    assert all(r[x]["G7_n"] == 4 for x in ARMS)   # 2 reps x 2 arms, one nova row each
    k = json.load(open(os.path.join(D, f"home_KPU_s{seed}.result.json")))
    assert k["bank_sha256"] == json.load(open(os.path.join(D, f"home_KP_s{seed}.result.json")))["bank_sha256"]
    assert 0.0 < k["keep_record"]["kept_pos_share"] < 1.0, k["keep_record"]
    z = json.load(open(os.path.join(D, f"home_KP0_s{seed}", "keep_bank.json")))
    assert z["frame"] == ""
    print("SMOKE OK", {x: (r[x]["dKL"], r[x]["G7_rate"], r[x]["G8a_rate"]) for x in ARMS},
          "kept_pos_share", k["keep_record"]["kept_pos_share"])


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("night", "drift", "probe", "score", "smoke"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--cuda-precision", default="fp32", choices=("fp16", "fp32"))
    ap.add_argument("--life-json", default=LIFE_JSON, help="the B life JSON ({} = seed); 'none' = no kwargs, no ledger check")
    ap.add_argument("--brain-kw", default=None, help="JSON dict overriding the life's brain_kw")
    ap.add_argument("--bank-from", default=None)
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--reps", default=None, help="comma list of replicate indices (default 0..5)")
    ap.add_argument("--sets", default=None, help="drift_read SETS filter (default all)")
    ap.add_argument("--bank-n", type=int, default=None)
    ap.add_argument("--bank-tokens", type=int, default=None)
    ap.add_argument("--bank-night", type=int, default=None)
    ap.add_argument("--smoke-ok", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    x = a.args
    if a.cmd == "night":
        a.arm, a.src, a.dst, a.seed = x[0], x[1], x[2], int(x[3])
        return cmd_night(a, cortex)
    if a.cmd == "drift":
        a.arm, a.src, a.dst, a.seed, a.out = x[0], x[1], x[2], int(x[3]), x[4]
        return cmd_drift(a)
    if a.cmd == "probe":
        a.arm, a.dst, a.seed, a.out = x[0], x[1], int(x[2]), x[3]
        return cmd_probe(a, cortex)
    if a.cmd == "score":
        a.dir = x[0]
        return cmd_score(a)
    a.dir = x[0]
    return cmd_smoke(a)


if __name__ == "__main__":
    main()
