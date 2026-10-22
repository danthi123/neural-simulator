"""_deixis_audit_check — step-0a module 5 (0a(ii)): the audit of design 3.4.

Ported, arithmetic UNCHANGED, from `s1-scratch/fix2/stream_audit.py`, run on
the RECORDS instead of regenerated worlds: the spiking side trains under the
stated spiking-noise model on the recorded streams (read through their
manifests, refusing a sha mismatch vs the manifest — packet test line 99),
the replica side reads the weights module 4 recorded in replica.json, and the
audit compares them per seed (A0, A1, A2, the reported Spearman) for every
pair (rule g that produced the spiking weights, rule r whose replica is the
reference) at each arm's centre, under `--noise moderate,harsh`.

Ports:
  * stream_audit: NOISE/CENTRE/T/TAU/NPN/TOL, spiking_on_stream (W over
    TOKM's 35 rows; the unit-conversion draws lam and kap ONCE per run and
    are returned for the wrong-unit report), scored_routes, side,
    route_disagreements (per-route bar, both sides within TOL of it),
    spearman, audit (sha mismatch -> UNDEFINED; else 13/15 bars).
  * stream_audit's controls: the independent-stream control (base stream s
    vs indep stream s+1000, COV-PRE both sides — A0 records DIFFERENT file
    shas BY DESIGN, so its verdict is UNDEFINED and the A1/A2 counts are
    reported beside it, never made circular) and the unengaged control (all
    weights 0: the GO bars evaluate, and the engagement record — not a FAIL —
    owns the reading).
Differences from the scratch (records, not new arithmetic): the bar pattern
is score()'s 8-key GO dict with the CALIBRATION's per-target thresholds on
both sides (score is ported unchanged in module 4; the scratch predates the
per-target thresholds), the rules iterate in the SCRATCH's order
(covpre, cov, hebb, hebbwta, bcm) so run_seed 9000+100*s+index reproduces
the recorded noise draws exactly, and sha256 equality is the file's real
bytes (A0) rather than a regenerated stream's header.

CLI (packet command 117): --streams $OUT/streams --thresholds
$OUT/calibration.json --replica $OUT/replica.json --noise moderate,harsh
--seeds 1-15 --out $OUT/audit_check.json.  Host class: pool, memcap 4.
Refuses: a threshold file whose sha256 differs from replica.json's recorded
one; any manifest/stream sha mismatch; a replica.json missing a rule/seed.
--selftest builds synthetic streams + a mini replica in a temp dir and runs
main() end to end (including both controls), plus the sha-corruption refusal.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import _deixis_rule_replica as R  # noqa: E402

# stream_audit constants (unchanged) — NPN = 12 replica sites (line 37)
T = 4800
TAU = 400
TOL = 0.05
NPN = 12
CENTRE = {"covpre": 6, "cov": 6, "hebb": 3, "hebbwta": 3, "bcm": 1}
NOISE = {
    "moderate": dict(LAM_SPK=8.0, GAIN_SD=0.25, SPONT=0.01, PRE_SD=0.05,
                     MISS=0.02, CONV=1.25, KCONV=1.10),
    "harsh": dict(LAM_SPK=3.0, GAIN_SD=0.40, SPONT=0.03, PRE_SD=0.10,
                  MISS=0.05, CONV=1.50, KCONV=1.25),
}
# the scratch's rule order (stream_audit line 180): run_seed = 9000 + 100*s
# + index over THIS list, reproducing the recorded noise draws exactly.
RULES = ["covpre", "cov", "hebb", "hebbwta", "bcm"]


def spiking_on_stream(rows, g, lam_tau, level, tau=TAU, warm=200, alpha=0.005,
                      kap=1.0, run_seed=0):
    """stream_audit.spiking_on_stream (lines 87-115) over recorded (x, y)
    rows: per-site lognormal gain, Poisson output spikes, the unit
    conversion CONV/KCONV drawn once per run.  Returns
    (W.mean over the NPN replica sites, lam_factor, kap_factor) — the two
    draws recorded for the wrong-unit report."""
    P = NOISE[level]
    rng = np.random.default_rng(run_seed)
    nT, nP = len(R.TOKM), len(R.POST)
    gg = rng.lognormal(0.0, P["GAIN_SD"], size=(nP, NPN))
    lamf = rng.uniform(1 / P["CONV"], P["CONV"])
    kapf = kap * rng.uniform(1 / P["KCONV"], P["KCONV"])
    lam = (lam_tau / tau) * lamf
    decay = 1.0 / tau

    def post(y):
        return rng.poisson(
            P["LAM_SPK"] * (gg * y[:, None] + P["SPONT"])) / P["LAM_SPK"]

    W = np.zeros((nT, nP, NPN))
    ybar = np.mean([post(y) for _, y in rows[:warm]], axis=0)
    th = ybar ** 2
    for x, y in rows:
        xs = x * (rng.random(nT) >= P["MISS"]) * rng.normal(
            1.0, P["PRE_SD"], nT)
        yn = post(y)
        if g == "cov":
            W += lam * xs[:, None, None] * (yn - ybar)[None]
            W *= (1 - decay)
        elif g == "covpre":
            W += lam * xs[:, None, None] * ((yn - ybar)[None] - kapf * W)
        elif g == "hebb":
            W += lam * xs[:, None, None] * yn[None]
            W *= (1 - decay)
        elif g == "hebbwta":
            yy = yn.copy()
            pm = yn[:R.NPOOL].mean(axis=1)
            act = [k for k in range(R.NPOOL) if pm[k] > 0.5]
            if len(act) > 1:
                drive = [pm[k] + float(x @ W[:, k, :].mean(axis=1))
                         + 1e-6 * k for k in act]
                keep = act[int(np.argmax(drive))]
                for k in act:
                    if k != keep:
                        yy[k] = 0.0
            W += lam * xs[:, None, None] * yy[None]
            W *= (1 - decay)
        elif g == "bcm":
            W += lam * xs[:, None, None] * (yn * (yn - th))[None]
            W *= (1 - decay)
        np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * yn
        th = (1 - alpha) * th + alpha * yn * yn
    return W.mean(axis=2), float(lamf), float(kapf)


def bar_pattern(W, th):
    """The 8-key GO pattern with the calibration's per-target thresholds:
    score()'s dict (S0-S5 + the class-W marker bar), both sides alike."""
    go, _mk = R.score(W, th["P"], th["E"])
    return {k: bool(go[k]) for k in sorted(go)}


def scored_routes(W, th):
    """stream_audit.scored_routes with per-target bars: every scored S0
    route -> ("tok>target", (value, kind, bar))."""
    r = lambda t: W[R.tim[t]]
    out = {}
    for t, ks in R.TARGET.items():
        for k in ks:
            out["%s>%s" % (t, k)] = (float(r(t)[R.pi[k]]), "fire",
                                     th["P"][0])
    out["N_SELF>SELF"] = (
        float(r("N_SELF")[R.NPOOL + R.ENT.index("SELF")]), "fire",
        th["E"][0])
    for n in R.NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        out["%s>%s" % (n, bearer)] = (
            float(r(n)[R.NPOOL + R.ENT.index(bearer)]), "fire", th["E"][0])
        for j, e in enumerate(R.ENT):
            if e != bearer:
                out["%s>%s" % (n, e)] = (
                    float(r(n)[R.NPOOL + j]), "silent", th["E"][1])
    for t in R.TOK:
        for j, pool in enumerate(R.POOLS):
            if t in R.TARGET and pool in R.TARGET[t]:
                continue
            if t == "PREV" and pool in ("GIV", "ADR"):
                continue
            if t in ("IDIO", "PLUR"):
                continue
            out["%s>%s" % (t, pool)] = (float(r(t)[j]), "silent",
                                        th["P"][1])
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]:
        for j, e in enumerate(R.ENT):
            out["%s>%s" % (t, e)] = (float(r(t)[R.NPOOL + j]), "silent",
                                     th["E"][1])
    return out


def side(v, kind, bar):
    return v >= bar if kind == "fire" else v <= bar


def route_disagreements(Ws, Wr, th):
    a, b = scored_routes(Ws, th), scored_routes(Wr, th)
    bad = []
    for k, (vs, kind, bar) in a.items():
        vr, _kind_r, bar_r = b[k]
        if side(vs, kind, bar) != side(vr, kind, bar_r) and not (
                abs(vs - bar) <= TOL and abs(vr - bar) <= TOL):
            bad.append((k, round(vs, 3), round(vr, 3)))
    return bad


def spearman(Ws, Wr, th):
    a, b = scored_routes(Ws, th), scored_routes(Wr, th)
    ks = sorted(a)
    x = np.array([a[k][0] for k in ks])
    y = np.array([b[k][0] for k in ks])
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def audit(per_seed):
    """stream_audit.audit, unchanged: A0 sha equality is a precondition
    (mismatch -> UNDEFINED), then A1/A2 vs need = 13 of 15."""
    if any(p["sha_spk"] != p["sha_rep"] for p in per_seed):
        return {"verdict": "UNDEFINED", "reason": "stream sha256 mismatch"}
    a1 = sum(p["pat_spk"] == p["pat_rep"] for p in per_seed)
    a2 = sum(len(p["bad"]) == 0 for p in per_seed)
    n = len(per_seed)
    need = 13 if n == 15 else int(np.ceil(13 * n / 15))
    return {"verdict": "PASS" if (a1 >= need and a2 >= need) else "FAIL",
            "A1": a1, "A2": a2, "n": n, "need": need}


def _rng(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="0a(ii): the design-3.4 audit on recorded streams")
    ap.add_argument("--streams")
    ap.add_argument("--thresholds")
    ap.add_argument("--replica")
    ap.add_argument("--noise")
    ap.add_argument("--seeds")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest()))
        return 0
    missing = [n for n, v in (("--streams", args.streams),
                              ("--thresholds", args.thresholds),
                              ("--replica", args.replica),
                              ("--noise", args.noise),
                              ("--seeds", args.seeds),
                              ("--out", args.out)) if not v]
    if missing:
        ap.error("the following arguments are required: "
                 + ", ".join(missing))
    seeds = _rng(args.seeds)
    levels = [s for s in args.noise.split(",") if s]
    th = R.load_thresholds(args.thresholds)
    th_sha = R.sha256(args.thresholds)
    by_arm, by_key = R.load_manifests(args.streams)  # sha-verifies everything
    with open(args.replica) as f:
        rep = json.load(f)
    if rep.get("thresholds_sha256") != th_sha:
        raise ValueError(
            "replica.json was built on different thresholds (%s vs %s)"
            % (rep.get("thresholds_sha256"), th_sha))
    for r in RULES:
        if r not in rep["rules"]:
            raise ValueError("replica.json has no rule %r" % r)
        for s in seeds:
            if str(s) not in rep["rules"][r]["weights"]:
                raise ValueError("replica.json: rule %r has no seed %d"
                                 % (r, s))

    out = {"schema": "deixis-audit-check-v1", "T": T, "TAU": TAU, "TOL": TOL,
           "NPN": NPN, "noise": {k: NOISE[k] for k in levels},
           "centre": {r: rep["rules"][r]["centre"] for r in RULES},
           "seeds": seeds,
           "thresholds": th, "thresholds_sha256": th_sha,
           "replica_sha256": R.sha256(args.replica),
           "streams": {}, "matrix": {}, "unit_conversion": {},
           "controls": {}}
    rows_cache = {}

    def base(s):
        if s not in rows_cache:
            ent = by_key[("base", s)]
            rows, sha = R.read_stream(ent["file"])
            rows_cache[s] = (rows, sha, os.path.basename(ent["file"]))
        return rows_cache[s]

    for s in seeds:
        _r, sha, fn = base(s)
        out["streams"][str(s)] = {"file": fn, "sha256": sha,
                                  "matches_replica":
                                      sha == rep["rules"]["covpre"]["sha"]
                                      [str(s)]}

    for level in levels:
        mat = {}
        conv = {}
        spk = {}
        for g in RULES:
            centre = rep["rules"][g]["centre"]
            gobars = 0
            spk[g] = {}
            lam_draws, kap_draws = [], []
            for s in seeds:
                rows, sha, _fn = base(s)
                W, lamf, kapf = spiking_on_stream(
                    rows, g, centre, level, run_seed=9000 + 100 * s
                    + RULES.index(g))
                spk[g][s] = (W, sha)
                lam_draws.append(lamf)
                kap_draws.append(kapf)
                if all(bar_pattern(W, th).values()):
                    gobars += 1
            conv[g] = {"lam_min": min(lam_draws),
                       "lam_max": max(lam_draws),
                       "kap_min": min(kap_draws),
                       "kap_max": max(kap_draws)}
            for ref in RULES:
                per = []
                for s in seeds:
                    Ws, shs = spk[g][s]
                    Wr = np.array(rep["rules"][ref]["weights"][str(s)],
                                  dtype=float)
                    shr = rep["rules"][ref]["sha"][str(s)]
                    per.append({
                        "sha_spk": shs, "sha_rep": shr,
                        "pat_spk": bar_pattern(Ws, th),
                        "pat_rep": bar_pattern(Wr, th),
                        "bad": route_disagreements(Ws, Wr, th),
                        "rho": spearman(Ws, Wr, th)})
                a = audit(per)
                a["rho_median"] = float(np.median([p["rho"] for p in per]))
                a["rho_min"] = float(np.min([p["rho"] for p in per]))
                a["bad_examples"] = [p["bad"][:3] for p in per
                                     if p["bad"]][:3]
                a["sha_all_match"] = all(p["sha_spk"] == p["sha_rep"]
                                         for p in per)
                mat["%s|%s" % (g, ref)] = a
            mat["%s|GO_bars" % g] = gobars
        out["matrix"][level] = mat
        out["unit_conversion"][level] = conv
        n_pass = sum(1 for k, v in mat.items()
                     if isinstance(v, dict) and v.get("verdict") == "PASS")
        n_cell = sum(1 for k, v in mat.items() if isinstance(v, dict))
        print("noise %-8s cells %d PASS / %d" % (level, n_pass, n_cell),
              flush=True)

    # the reviewer's control: base stream s vs indep stream s+1000, COV-PRE
    # both sides — A0 records DIFFERENT file shas BY DESIGN, so the verdict
    # is UNDEFINED and A1/A2 sit beside it, never made circular.
    if ("indep", seeds[0] + 1000) in by_key:
        per = []
        centre = rep["rules"]["covpre"]["centre"]
        for s in seeds:
            rows_i, sha_i = R.read_stream(
                by_key[("indep", s + 1000)]["file"])
            Wb = R.train_m(rows_i, "covpre", centre / 400.0, 400)
            Wa = np.array(rep["rules"]["covpre"]["weights"][str(s)],
                          dtype=float)
            sa = rep["rules"]["covpre"]["sha"][str(s)]
            per.append({"sha_spk": sha_i, "sha_rep": sa,
                        "pat_spk": bar_pattern(Wb, th),
                        "pat_rep": bar_pattern(Wa, th),
                        "bad": route_disagreements(Wb, Wa, th),
                        "rho": spearman(Wb, Wa, th)})
        ind = {"rho_median": float(np.median([p["rho"] for p in per])),
               "rho_min": float(np.min([p["rho"] for p in per])),
               "A1_same": sum(p["pat_spk"] == p["pat_rep"] for p in per),
               "A2_same": sum(len(p["bad"]) == 0 for p in per),
               "sha_differ_all": all(p["sha_spk"] != p["sha_rep"]
                                     for p in per),
               "verdict": audit(per)["verdict"]}
        out["controls"]["independent_streams_covpre"] = ind
        print("independent streams (s vs s+1000), COV-PRE both sides:",
              ind, flush=True)
    else:
        out["controls"]["independent_streams_covpre"] = {
            "verdict": "SKIPPED", "reason": "no manifest_indep for seeds "
                                            "%d-%d" % (seeds[0] + 1000,
                                                       seeds[-1] + 1000)}
    # an unengaged arm (the rule never ran: weights stay 0)
    z = np.zeros((len(R.TOKM), len(R.POST)))
    out["controls"]["unengaged"] = {
        "GO_bars": bar_pattern(z, th),
        "note": "sum|dw| = 0 -> UNDEFINED by the engagement record, "
                "never a FAIL"}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote %s (%d noise levels)" % (args.out, len(levels)))
    return 0


def selftest():
    """Synthetic streams + mini replica in a temp dir; main() end to end
    (both controls), the sha-corruption refusal, and unit checks."""
    res = {}
    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "streams")
        os.makedirs(sd)
        rng = np.random.default_rng(21)
        shas = {}

        def write_stream(path, seed, arm):
            with open(path, "w") as f:
                f.write(json.dumps({"schema": "deixis-toy-stream-v1",
                                    "seed": seed, "T": 60, "arm": arm,
                                    "world_kw": {}, "TOK": R.TOKM,
                                    "POST": R.POST, "length": 60}) + "\n")
                for t in range(60):
                    x = [int(i) for i in rng.choice(len(R.TOKM), 5,
                                                    replace=False)]
                    yv = []
                    for i in range(len(R.POST)):
                        v = 1.0 if rng.random() < 0.3 else (
                            R.PRES if rng.random() < 0.3 else 0.0)
                        if v:
                            yv.append([i, round(float(v), 6)])
                    f.write(json.dumps({"t": t, "x": x, "y": yv,
                                        "S": "V1", "A": "V2"}) + "\n")
            shas[os.path.basename(path)] = R.sha256(path)

        # base s1 + indep s1001
        write_stream(os.path.join(sd, "base_s1.jsonl"), 1, "base")
        write_stream(os.path.join(sd, "indep_s1001.jsonl"), 1001, "indep")
        with open(os.path.join(sd, "manifest_base.json"), "w") as f:
            json.dump({"schema": "deixis-streams-manifest-v1", "arm": "base",
                       "seeds": [1], "length": 60,
                       "streams": [{"seed": 1, "arm": "base",
                                    "file": "base_s1.jsonl",
                                    "sha256": shas["base_s1.jsonl"]}]}, f)
        with open(os.path.join(sd, "manifest_indep.json"), "w") as f:
            json.dump({"schema": "deixis-streams-manifest-v1",
                       "arm": "indep", "seeds": [1001], "length": 60,
                       "streams": [{"seed": 1001, "arm": "indep",
                                    "file": "indep_s1001.jsonl",
                                    "sha256": shas["indep_s1001.jsonl"]}]},
                      f)
        # thresholds + mini replica (every rule at centre 6, seed 1)
        thp = {"thresholds": {"P": [0.40, 0.20], "E": [0.45, 0.20]}}
        cal = os.path.join(td, "calibration.json")
        with open(cal, "w") as f:
            json.dump(thp, f)
        th = R.load_thresholds(cal)
        rows1, sha1 = R.read_stream(os.path.join(sd, "base_s1.jsonl"))
        rep = {"schema": "deixis-replica-v1", "thresholds": th,
               "thresholds_sha256": R.sha256(cal), "rules": {},
               "centre_default": 6}
        for rule in R.RULES:
            W = R.train_m(rows1, rule, 6 / 400.0, 400)
            rep["rules"][rule] = {
                "centre": 6,
                "weights": {"1": [[round(float(v), 6) for v in row]
                                  for row in W]},
                "sha": {"1": sha1}}
        rp = os.path.join(td, "replica.json")
        with open(rp, "w") as f:
            json.dump(rep, f)
        # end to end
        oj = os.path.join(td, "audit_check.json")
        rc = main(["--streams", sd, "--thresholds", cal, "--replica", rp,
                   "--noise", "moderate", "--seeds", "1", "--out", oj])
        assert rc == 0 and os.path.exists(oj)
        got = json.load(open(oj))
        cell = got["matrix"]["moderate"]["covpre|covpre"]
        assert cell["verdict"] in ("PASS", "FAIL") and cell["sha_all_match"]
        assert got["matrix"]["moderate"]["covpre|GO_bars"] in (0, 1)
        ind = got["controls"]["independent_streams_covpre"]
        assert ind["verdict"] == "UNDEFINED" and ind["sha_differ_all"]
        ug = got["controls"]["unengaged"]["GO_bars"]
        # unengaged = every weight 0: only S0's absolute must-fire fails;
        # the remaining GO bars are relative/null-free comparisons that
        # hold trivially at 0 — which is exactly why the engagement
        # record, not the bar verdict, owns the reading of an unengaged
        # arm (the ported arithmetic: bars(ZERO_NULL) behaves the same).
        assert set(ug) == {"S0", "S1", "S2a", "S2b", "S3", "S4", "S5",
                           "MARK_W"}
        assert ug["S0"] is False
        assert all(ug[k] for k in ("S1", "S2a", "S2b", "S3", "S4", "S5",
                                   "MARK_W"))
        assert got["controls"]["unengaged"]["note"]
        assert "lam_min" in got["unit_conversion"]["moderate"]["covpre"]
        res["main_ok"] = True
        res["selftest_cell"] = {k: cell[k] for k in
                                ("verdict", "A1", "A2", "need", "rho_min")}
        # thresholds drift refusal
        rep["thresholds_sha256"] = "0" * 64
        with open(rp, "w") as f:
            json.dump(rep, f)
        try:
            main(["--streams", sd, "--thresholds", cal, "--replica", rp,
                  "--noise", "moderate", "--seeds", "1", "--out", oj])
            raise AssertionError("threshold-drift refusal did not fire")
        except ValueError:
            res["th_drift_refusal"] = True
        # stream sha corruption refusal (packet test line 99): flip one byte
        # in the stream after the manifest recorded it
        rep["thresholds_sha256"] = R.sha256(cal)
        with open(rp, "w") as f:
            json.dump(rep, f)
        with open(os.path.join(sd, "base_s1.jsonl"), "a") as f:
            f.write("\n")
        try:
            main(["--streams", sd, "--thresholds", cal, "--replica", rp,
                  "--noise", "moderate", "--seeds", "1", "--out", oj])
            raise AssertionError("manifest sha refusal did not fire")
        except ValueError:
            res["manifest_refusal"] = True
        # unit checks: spearman identity, route side, audit UNDEFINED path
        W = np.zeros((len(R.TOKM), len(R.POST)))
        assert spearman(W, W, th) == 1.0
        assert side(0.5, "fire", 0.4) and side(0.1, "silent", 0.2)
        res["unit"] = True
        res["audit_undef"] = audit([{"sha_spk": "a", "sha_rep": "b",
                                     "pat_spk": {}, "pat_rep": {},
                                     "bad": []}])["verdict"] == "UNDEFINED"
    res["ok"] = True
    return res


if __name__ == "__main__":
    sys.exit(main())
