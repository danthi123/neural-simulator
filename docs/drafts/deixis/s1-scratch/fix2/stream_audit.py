"""Round-3 scratch: the step-0b audit defined on a SHARED, RECORDED exposure stream (answers the BLOCKING finding).

The toy world WRITES its exposure stream to disk (one canonical JSONL file per seed and arm, sha256 over its bytes).
The rate replica READS that file (never regenerates the world), and so does the spiking run (here: a stated spiking-noise
model standing in for the bridge, below). Both record the sha256 they read; the audit refuses (UNDEFINED) unless the two
match. Scripts copied unchanged from scratch/fix/: replica3.py (world, train, bars). Nothing here edits them.

SPIKING-NOISE MODEL (stated; the stand-in for the 0b bridge, two levels):
  * every post unit is NPN = 12 neurons (the 0b assemblies), each with its own trace; a route's weight is the mean over
    its 12 x 12 synapses (pre side: the 1500-pA token assembly fires on every occurrence, so the pre trace is the token's
    presence times a per-occurrence jitter N(1, PRE_SD));
  * the post trace of a neuron is a spike count: Poisson(LAM_SPK * (g * y + SPONT)) / LAM_SPK, where y is the world's drive
    (1 participant, 0.2 present, 1/0 pool), LAM_SPK the expected count of a fully driven neuron in one 40-ms token window
    (0.19 spikes/ms measured, 2.6: about 8), g a per-neuron gain drawn once per seed from lognormal(0, GAIN_SD) (the
    heterogeneity spread of the measured cells), SPONT a spontaneous rate on every neuron;
  * an occurrence is missed by the presynaptic floor with probability MISS;
  * the per-occurrence unit conversion of the 0b smoke is off by a factor drawn once per run: lam x U(1/CONV, CONV),
    kap x U(1/KCONV, KCONV).
  moderate: LAM_SPK 8, GAIN_SD 0.25, SPONT 0.01, PRE_SD 0.05, MISS 0.02, CONV 1.25, KCONV 1.10
  harsh   : LAM_SPK 3, GAIN_SD 0.40, SPONT 0.03, PRE_SD 0.10, MISS 0.05, CONV 1.50, KCONV 1.25

AUDIT (per arm and cell; items computed exactly as the 0b module will):
  A0 stream identity: the spiking run's and the replica's recorded stream sha256 are equal, else UNDEFINED;
  A1 bar pattern: per seed, the pass/fail of every GO bar (S0, S1, S2a, S2b, S3, S4, S5) is the same in the spiking
     weights and in the replica's weights on that seed's shared stream; >= 13/15 seeds;
  A2 route side: per seed, every scored S0 route (must-fire >= theta_fire, class W <= theta_silent) is on the same side
     of its bar in both, ignoring a disagreement only when BOTH values lie within TOL = 1 engine unit (0.05 of W_MAX)
     of the bar; >= 13/15 seeds;
  Spearman of the scored routes: REPORTED only (not a bar; on independent streams the reviewer measured 0.46).
"""
import os, sys, json, hashlib, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replica3 as R

HERE = os.path.dirname(os.path.abspath(__file__)); SDIR = os.path.join(HERE, "streams")
T = 4800; TAU = 400; NPN = 12; TOL = 0.05
CENTRE = {"covpre": 6, "cov": 6, "hebb": 3, "hebbwta": 3, "bcm": 1}
NOISE = {"moderate": dict(LAM_SPK=8.0, GAIN_SD=0.25, SPONT=0.01, PRE_SD=0.05, MISS=0.02, CONV=1.25, KCONV=1.10),
         "harsh": dict(LAM_SPK=3.0, GAIN_SD=0.40, SPONT=0.03, PRE_SD=0.10, MISS=0.05, CONV=1.50, KCONV=1.25)}


# ── the stream file (what the toy world writes; both learners read it) ─────────────────────────────────────────
def write_stream(seed, T=T, arm="base", **kw):
    os.makedirs(SDIR, exist_ok=True)
    rows = R.world(seed, T, **kw)
    path = os.path.join(SDIR, "stream_%s_s%d.jsonl" % (arm, seed))
    with open(path, "w") as f:
        f.write(json.dumps({"schema": "deixis-toy-stream-v1", "seed": seed, "T": T, "arm": arm, "world_kw": kw,
                            "TOK": R.TOK, "POST": R.POST}, sort_keys=True) + "\n")
        for t, (x, y, m) in enumerate(rows):
            f.write(json.dumps({"t": t, "x": [int(i) for i in np.flatnonzero(x)],
                                "y": [[int(i), round(float(y[i]), 6)] for i in np.flatnonzero(y)],
                                "S": m["S"], "A": m["A"]}, sort_keys=True) + "\n")
    return path, sha256(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def read_stream(path):
    """Returns (rows, sha256). rows: list of (x, y) vectors in replica3's TOK / POST order (checked)."""
    sha = sha256(path); rows = []
    with open(path) as f:
        head = json.loads(f.readline())
        assert head["TOK"] == R.TOK and head["POST"] == R.POST, "stream vocabulary differs from the replica's"
        for line in f:
            d = json.loads(line); x = np.zeros(len(R.TOK)); y = np.zeros(len(R.POST))
            x[d["x"]] = 1.0
            for i, v in d["y"]: y[i] = v
            rows.append((x, y))
    return rows, sha


# ── the replica on the stream (unchanged arithmetic: replica3.train) ───────────────────────────────────────────
def replica_on_stream(path, rule, lt):
    rows, sha = read_stream(path)
    W = R.train([(x, y, None) for x, y in rows], rule, lt / TAU, TAU)[len(rows)]
    return W, sha


# ── the spiking-noise stand-in on the SAME stream file ─────────────────────────────────────────────────────────
def spiking_on_stream(path, rule, lt, noise, run_seed, alpha=0.005, warm=200, kap=1.0):
    rows, sha = read_stream(path); P = NOISE[noise]; rng = np.random.default_rng(run_seed)
    nT, nP = len(R.TOK), len(R.POST)
    g = rng.lognormal(0.0, P["GAIN_SD"], size=(nP, NPN))
    lam = (lt / TAU) * rng.uniform(1 / P["CONV"], P["CONV"]); kp = kap * rng.uniform(1 / P["KCONV"], P["KCONV"])
    decay = 1.0 / TAU

    def post(y):
        return rng.poisson(P["LAM_SPK"] * (g * y[:, None] + P["SPONT"])) / P["LAM_SPK"]

    W = np.zeros((nT, nP, NPN))
    ybar = np.mean([post(y) for _, y in rows[:warm]], axis=0); th = ybar ** 2
    for x, y in rows:
        xs = x * (rng.random(nT) >= P["MISS"]) * rng.normal(1.0, P["PRE_SD"], nT)
        yn = post(y)
        if rule == "cov": W += lam * xs[:, None, None] * (yn - ybar)[None]; W *= (1 - decay)
        elif rule == "covpre": W += lam * xs[:, None, None] * ((yn - ybar)[None] - kp * W)
        elif rule == "hebb": W += lam * xs[:, None, None] * yn[None]; W *= (1 - decay)
        elif rule == "hebbwta":
            yy = yn.copy(); pm = yn[:R.NPOOL].mean(axis=1); act = [k for k in range(R.NPOOL) if pm[k] > 0.5]
            if len(act) > 1:
                drive = [pm[k] + float(x @ W[:, k, :].mean(axis=1)) + 1e-6 * k for k in act]; keep = act[int(np.argmax(drive))]
                for k in act:
                    if k != keep: yy[k] = 0.0
            W += lam * xs[:, None, None] * yy[None]; W *= (1 - decay)
        elif rule == "bcm": W += lam * xs[:, None, None] * (yn * (yn - th))[None]; W *= (1 - decay)
        np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * yn; th = (1 - alpha) * th + alpha * yn * yn
    return W.mean(axis=2), sha


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────────────
ZERO_NULL = {t: 0.0 for t in R.TOK}
BARS = R.KEYS_NF   # S0, S1, S2a, S2b_nf, S3_nf, S4_nf, S5_nf (the GO bars of 3.5; no null needed)


def bar_pattern(W):
    b = R.bars(W, ZERO_NULL, 0.0)
    return {k: bool(b[k]) for k in BARS}


def scored_routes(W):
    """Every S0 route with its kind, mirroring replica3.bars' S0 enumeration route by route."""
    r = lambda t: W[R.ti[t]]; out = {}
    for t, ks in R.TARGET.items():
        for k in ks: out["%s>%s" % (t, k)] = (float(r(t)[R.pi[k]]), "fire")
    out["N_SELF>SELF"] = (float(r("N_SELF")[R.NPOOL + R.ENT.index("SELF")]), "fire")
    for n in R.NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        out["%s>%s" % (n, bearer)] = (float(r(n)[R.NPOOL + R.ENT.index(bearer)]), "fire")
        for e in R.ENT:
            if e != bearer: out["%s>%s" % (n, e)] = (float(r(n)[R.NPOOL + R.ENT.index(e)]), "silent")
    for t in R.TOK:
        for j, pool in enumerate(R.POOLS):
            if t in R.TARGET and pool in R.TARGET[t]: continue
            if t == "PREV" and pool in ("GIV", "ADR"): continue
            if t in ("IDIO", "PLUR"): continue
            out["%s>%s" % (t, pool)] = (float(r(t)[j]), "silent")
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]:
        for e in R.ENT: out["%s>%s" % (t, e)] = (float(r(t)[R.NPOOL + R.ENT.index(e)]), "silent")
    return out


def side(v, kind):
    return v >= R.TH_FIRE if kind == "fire" else v <= R.TH_SILENT


def route_disagreements(Ws, Wr):
    a, b = scored_routes(Ws), scored_routes(Wr); bad = []
    for k, (vs, kind) in a.items():
        vr = b[k][0]; bar = R.TH_FIRE if kind == "fire" else R.TH_SILENT
        if side(vs, kind) != side(vr, kind) and not (abs(vs - bar) <= TOL and abs(vr - bar) <= TOL):
            bad.append((k, round(vs, 3), round(vr, 3)))
    return bad


def spearman(Ws, Wr):
    a, b = scored_routes(Ws), scored_routes(Wr); ks = sorted(a)
    x = np.array([a[k][0] for k in ks]); y = np.array([b[k][0] for k in ks])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def audit(per_seed):
    """per_seed: list of dicts with sha_spk, sha_rep, pat_spk, pat_rep, bad. Returns the audit verdict."""
    if any(p["sha_spk"] != p["sha_rep"] for p in per_seed):
        return dict(verdict="UNDEFINED", reason="stream sha256 mismatch")
    a1 = sum(p["pat_spk"] == p["pat_rep"] for p in per_seed); a2 = sum(len(p["bad"]) == 0 for p in per_seed)
    n = len(per_seed); need = 13 if n == 15 else int(np.ceil(13 * n / 15))
    return dict(verdict="PASS" if (a1 >= need and a2 >= need) else "FAIL", A1=a1, A2=a2, n=n)


if __name__ == "__main__":
    seeds = list(range(1, 16)); rules = ["covpre", "cov", "hebb", "hebbwta", "bcm"]
    out = {"T": T, "TAU": TAU, "TOL": TOL, "noise": NOISE, "centre": CENTRE, "seeds": seeds, "streams": {}}
    paths = {}
    for s in seeds:
        p, h = write_stream(s); paths[s] = p; out["streams"][s] = h
    # replica weights on the shared streams, per rule at its own centre
    rep = {r: {s: replica_on_stream(paths[s], r, CENTRE[r]) for s in seeds} for r in rules}
    for noise in ("moderate", "harsh"):
        print("== noise model:", noise, NOISE[noise], flush=True)
        spk = {g: {s: spiking_on_stream(paths[s], g, CENTRE[g], noise, run_seed=9000 + 100 * s + rules.index(g)) for s in seeds}
               for g in rules}
        res = {}
        for g in rules:                      # g: the rule that actually produced the spiking weights
            gobars = sum(all(bar_pattern(spk[g][s][0]).values()) for s in seeds)
            for ref in rules:                # ref: the arm label whose replica is the audit reference
                per = []
                for s in seeds:
                    Ws, shs = spk[g][s]; Wr, shr = rep[ref][s]
                    per.append(dict(sha_spk=shs, sha_rep=shr, pat_spk=bar_pattern(Ws), pat_rep=bar_pattern(Wr),
                                    bad=route_disagreements(Ws, Wr), rho=spearman(Ws, Wr)))
                a = audit(per); a["rho_median"] = float(np.median([p["rho"] for p in per]))
                a["rho_min"] = float(np.min([p["rho"] for p in per]))
                a["bad_examples"] = [p["bad"][:3] for p in per if p["bad"]][:3]
                res["%s|%s" % (g, ref)] = a
                print("  generated-by %-7s (GO bars pass %2d/15) vs replica-of %-7s : audit %-9s A1 %2d/15 A2 %2d/15 rho med %.3f min %.3f" % (
                    g, gobars, ref, a["verdict"], a.get("A1", -1), a.get("A2", -1), a["rho_median"], a["rho_min"]), flush=True)
            res["%s|GO_bars" % g] = gobars
        out[noise] = res
    # the reviewer's case: independent streams (world(s) vs world(s+1000)), COV-PRE replica vs COV-PRE replica
    per = []
    for s in seeds:
        p2, _ = write_stream(s + 1000, arm="indep")
        Wa, sa = rep["covpre"][s]; Wb, sb = replica_on_stream(p2, "covpre", CENTRE["covpre"])
        per.append(dict(sha_spk=sb, sha_rep=sa, pat_spk=bar_pattern(Wb), pat_rep=bar_pattern(Wa), bad=route_disagreements(Wb, Wa),
                        rho=spearman(Wb, Wa)))
    ind = dict(rho_median=float(np.median([p["rho"] for p in per])), rho_min=float(np.min([p["rho"] for p in per])),
               A1_same=sum(p["pat_spk"] == p["pat_rep"] for p in per), A2_same=sum(len(p["bad"]) == 0 for p in per),
               verdict=audit(per)["verdict"])
    out["independent_streams_covpre"] = ind
    print("== independent streams (s vs s+1000), COV-PRE replica both sides:", ind, flush=True)
    # an unengaged arm (the rule never ran: weights stay 0) and the sha guard
    z = np.zeros((len(R.TOK), len(R.POST)))
    out["unengaged"] = dict(GO_bars=bar_pattern(z), note="sum|dw| = 0 -> UNDEFINED by the engagement record, never a FAIL")
    print("== unengaged arm (all weights 0):", out["unengaged"], flush=True)
    json.dump(out, open(os.path.join(HERE, "stream_audit.json"), "w"), indent=1, default=str)
