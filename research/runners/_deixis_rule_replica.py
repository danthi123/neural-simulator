"""_deixis_rule_replica — step-0a module 4 (0a(ii)): the rate replica on RECORDED streams.

Ports, arithmetic UNCHANGED:
  * `s1-scratch/fix/replica3.py`: the vocabulary constants, train(), bars(),
    shuffle_labels(), shuffle_scene(), nulls() — the 30-form replica.
  * `s1-scratch/fix2/replica4_markers.py`: train_m() and score() (per-target
    thresholds: pool routes vs theta_P, entity routes vs theta_E) over the
    35-token set (30 forms + the five CLASS-W markers), plus the marker
    read-outs MARK.  Streams carry all 35 tokens (module 1 embodies them), so
    every training run here goes through train_m's shape; bars() indexes only
    base rows and works unchanged on a 35-row W.
  * `s1-scratch/fix2/stream_audit.py`: read_stream() — the replica's stream
    reader: sha256 over the bytes, header vocabulary asserted against this
    module's TOK/POST ("stream vocabulary differs from the replica's").

What is NOT copied: world()/world_m() (module 1 writes the streams; the
replica NEVER regenerates a world — it only shuffles loaded rows in memory for
the label/scene nulls, which is a derived transform of recorded data).

INTERFACE CONTRACTS (the packet freezes paths; these docstrings freeze shapes):
  * streams dir (module 1 writes): one manifest_<arm>.json per arm with
    {"schema": "deixis-streams-manifest-v1", "arm", "seeds", "length",
     "streams": [{"seed", "arm", "file", "sha256"}]} and the JSONL stream
    files beside it: header line {"schema", "seed", "T"/"length", "arm",
    "world_kw"/"world parameters", "TOK", "POST", ...} then rows
    {"t", "x" (token ids), "y": [[id, drive]...], "S", "A"}.
  * calibration.json (module 3 writes): must carry
    "thresholds": {"P": [fire, silent], "E": [fire, silent]} as FRACTIONS of
    W_MAX = 20 (the calibration rule's engine-unit ratios; prototype
    P 8/4 = [0.40, 0.20], E 9/4 = [0.45, 0.20]).  This module refuses to run
    without them — no default threshold is invented.
  * replica.json (this module writes): {"schema": "deixis-replica-v1",
    "thresholds", "thresholds_source", "grid", "seeds", "rate_seeds",
    "arm_note", "manifests" {arm: {"length", "entries"}},
    "streams" {seed: {"arm": "base", "file", "sha256"}}, "rules" {rule:
    {"grid_cells" {lt: {"all_go", "bar_passes", "median_min_fire"}},
     "centre" lt (most seeds passing every GO bar; ties: most bar-seed
     passes; then larger median weakest must-fire route — packet lines
     85-87), "weights" {seed: W as nested lists} at centre on the base
     stream (every route's weight per seed), "scores" {seed: score() go
     dict + marker routes + bars_nf + min/max routes} at centre, "sha"
     {seed: stream sha256 read}}, "rate_seed_scores" {rule: {seed: go-dict
     on the base stream at that arm's centre, seeds 1-60}}, "null_arms"
    {rule: {label|scene|voice: {seed: go-dict on that arm's stream}}},
    "k0" {rule: go-dict + route extremes for the all-zero weights} }.

CLI (packet command 116): --streams $OUT/streams --thresholds
$OUT/calibration.json --grid 1,2,3,4,6,9,12,18 --seeds 1-15
--rate-seeds 1-60 --out $OUT/replica.json.  Host class: the packet runs it
under memcap on one pool node.  --selftest builds a synthetic stream in a
temp dir, trains, scores, and checks determinism + threshold refusal.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.lab import attributable_to  # noqa: E402  (gate: attribution discipline)

# ── replica3.py constants (unchanged) ─────────────────────────────────────────
VOICES = ["V1", "V2", "V3", "V4", "V5"]
PERS = VOICES + ["B1", "SELF"]
OBJ = ["O1", "O2", "O3"]
ENT = PERS + OBJ
POOLS = ["SPK", "ADR", "GIV", "LEFT", "NEWEST"]
NPOOL = len(POOLS)
DEC = [3, 4]
GIVTOK = ["ANA", "DEM", "POSS_G", "ORD1", "ORD2"]
TOK = (["SHS", "SHA", "POSS_S", "POSS_A", "PLUR", "OOI1", "PREV", "CN",
        "IDIO", "FILL"] + GIVTOK + ["N_" + p for p in PERS]
       + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)])
ti = {t: i for i, t in enumerate(TOK)}
POST = POOLS + ENT
pi = {p: i for i, p in enumerate(POST)}
PRES = 0.2
NAMES = ["N_" + p for p in PERS] + ["CN"]
NONREF = ["FILL"] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)]
TARGET = {"SHS": ["SPK"], "SHA": ["ADR"], "POSS_S": ["SPK"],
          "POSS_A": ["ADR"], "PLUR": ["SPK", "ADR"], "ANA": ["GIV"],
          "DEM": ["GIV"], "POSS_G": ["GIV"], "ORD1": ["GIV"],
          "ORD2": ["GIV"]}
TH_FIRE, TH_SILENT = 0.40, 0.20
KEYS = ["S0", "S1", "S2a", "S2b", "S3", "S4", "S5"]
KEYS_NF = ["S0", "S1", "S2a", "S2b_nf", "S3_nf", "S4_nf", "S5_nf"]
RULES = ["cov", "covpre", "hebb", "hebbwta", "bcm"]

# ── replica4_markers.py constants (unchanged) ─────────────────────────────────
MARK = ["ART", "ASP", "IND", "AGR", "DEF"]
TOKM = TOK + MARK
tim = {t: i for i, t in enumerate(TOKM)}
CLASS_W_MARK = ["ART", "ASP", "IND"]
HOLD_COST = ["AGR", "DEF"]


def train(rows, rule, lam, tau, alpha=0.005, warm=200, ckpts=(), kap=1.0):
    """replica3.train, unchanged: rows are (x, y, meta) triples."""
    W = np.zeros((len(TOK), len(POST)))
    ybar = np.mean([r[1] for r in rows[:warm]], axis=0)
    th = ybar ** 2
    out = {}
    for t, (x, y, _) in enumerate(rows, 1):
        if rule == "cov":
            W += lam * np.outer(x, y - ybar)
            W *= (1 - 1.0 / tau)
        elif rule == "covpre":
            W += lam * (np.outer(x, y - ybar) - kap * x[:, None] * W)
        elif rule == "hebb":
            W += lam * np.outer(x, y)
            W *= (1 - 1.0 / tau)
        elif rule == "hebbwta":
            yy = y.copy()
            act = [k for k in range(NPOOL) if y[k] > 0]
            if len(act) > 1:
                drive = [y[k] + float(x @ W[:, k]) + 1e-6 * k for k in act]
                keep = act[int(np.argmax(drive))]
                for k in act:
                    if k != keep:
                        yy[k] = 0.0
            W += lam * np.outer(x, yy)
            W *= (1 - 1.0 / tau)
        elif rule == "bcm":
            W += lam * np.outer(x, y * (y - th))
            W *= (1 - 1.0 / tau)
        np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * y
        th = (1 - alpha) * th + alpha * y * y
        if t in ckpts:
            out[t] = W.copy()
    out[len(rows)] = W.copy()
    return out


def train_m(rows, rule, lam, tau, alpha=0.005, warm=200, kap=1.0):
    """replica4_markers.train_m, unchanged: rows are (x, y) pairs over TOKM."""
    W = np.zeros((len(TOKM), len(POST)))
    ybar = np.mean([r[1] for r in rows[:warm]], axis=0)
    th = ybar ** 2
    for x, y in rows:
        if rule == "cov":
            W += lam * np.outer(x, y - ybar)
            W *= (1 - 1.0 / tau)
        elif rule == "covpre":
            W += lam * (np.outer(x, y - ybar) - kap * x[:, None] * W)
        elif rule == "hebb":
            W += lam * np.outer(x, y)
            W *= (1 - 1.0 / tau)
        elif rule == "hebbwta":
            yy = y.copy()
            act = [k for k in range(NPOOL) if y[k] > 0]
            if len(act) > 1:
                drive = [y[k] + float(x @ W[:, k]) + 1e-6 * k for k in act]
                keep = act[int(np.argmax(drive))]
                for k in act:
                    if k != keep:
                        yy[k] = 0.0
            W += lam * np.outer(x, yy)
            W *= (1 - 1.0 / tau)
        elif rule == "bcm":
            W += lam * np.outer(x, y * (y - th))
            W *= (1 - 1.0 / tau)
        np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * y
        th = (1 - alpha) * th + alpha * y * y
    return W


def bars(W, n95, cls95):
    """replica3.bars, unchanged (indexes base TOK rows; S0 uses module
    TH_FIRE/TH_SILENT — the per-target replacement is score()'s job)."""
    r = lambda t: W[ti[t]]
    S1 = True
    tg = {}
    for t, ks in TARGET.items():
        row = r(t)[:NPOOL]
        for k in ks:
            others = [row[j] for j in range(NPOOL) if POOLS[j] not in ks]
            ok = row[pi[k]] >= 2 * max(others)
            S1 &= ok
            tg[(t, k)] = float(row[pi[k]])
    small = min(tg[("SHS", "SPK")], tg[("SHA", "ADR")])
    S2a = all(max(r(t)[pi[k]] for k in TARGET[t])
              >= 2 * r(t)[NPOOL:].max()
              for t in ["SHS", "SHA", "ANA", "DEM", "ORD1", "ORD2"])
    S2b = True
    S2b_cls = True
    names = {}
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        row = r(n)
        ew = row[NPOOL:]
        b = ew[ENT.index(bearer)]
        oth = np.delete(ew, ENT.index(bearer)).max()
        ok = b >= 2 * max(row[:NPOOL].max(), n95[n]) and b >= 2 * oth
        okc = b >= 2 * max(row[:NPOOL].max(), cls95) and b >= 2 * oth
        names[n] = (round(float(b), 3), ok)
        S2b &= ok
        S2b_cls &= okc
    S3 = all(r(t)[:NPOOL].max() <= max(2 * n95[t], 0.25 * small)
             for t in NONREF)
    S4 = all(r(t)[DEC].max() <= max(2 * n95[t], 0.2 * small) for t in TOK) and \
        all(int(r(t)[:NPOOL].argmax()) not in DEC for t in TOK
            if r(t)[:NPOOL].max() >= 0.5 * small)
    S5 = r("OOI1")[:NPOOL].max() <= max(2 * n95["OOI1"], 0.25 * small) and \
        r("PREV")[pi["ADR"]] <= 0.2 * tg[("SHA", "ADR")]
    fire = [tg[k] for k in tg] + \
        [r("N_SELF")[NPOOL + ENT.index("SELF")]] + \
        [names[n][0] for n in NAMES]
    silent = []
    leakH = []
    for t in TOK:
        row = r(t)
        for j in range(NPOOL):
            if t in TARGET and POOLS[j] in TARGET[t]:
                continue
            if t == "PREV" and POOLS[j] in ("GIV", "ADR"):
                continue
            if t == "IDIO":
                continue
            (leakH if t == "PLUR" else silent).append(row[j])
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]:
        silent.append(r(t)[NPOOL:].max())
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        ew = r(n)[NPOOL:]
        silent.append(np.delete(ew, ENT.index(bearer)).max())
    for t in TARGET:
        leakH.append(r(t)[NPOOL:].max())
    S0_fire = min(fire) >= TH_FIRE
    S0_silent = max(silent) <= TH_SILENT
    S0 = S0_fire and S0_silent
    S2b_nf = all(names[n][0] >= 2 * r(n)[:NPOOL].max()
                 and names[n][0] >= 2 * np.delete(
                     r(n)[NPOOL:], ENT.index("V3" if n == "CN" else n[2:])
                 ).max() for n in NAMES)
    S3_nf = all(r(t)[:NPOOL].max() <= max(TH_SILENT, 0.25 * small)
                for t in NONREF)
    S4_nf = all(r(t)[DEC].max() <= max(TH_SILENT, 0.2 * small) for t in TOK
                if t != "PLUR") and \
        all(int(r(t)[:NPOOL].argmax()) not in DEC for t in TOK
            if r(t)[:NPOOL].max() >= 0.5 * small)
    S5_nf = r("OOI1")[:NPOOL].max() <= max(TH_SILENT, 0.25 * small) and \
        r("PREV")[pi["ADR"]] <= 0.2 * tg[("SHA", "ADR")]
    return dict(S2b_nf=S2b_nf, S3_nf=S3_nf, S4_nf=S4_nf, S5_nf=S5_nf,
                S0=S0, S0_fire=S0_fire, S0_silent=S0_silent, S1=S1, S2a=S2a,
                S2b=S2b, S3=S3, S4=S4, S5=S5, S2b_classmax=S2b_cls,
                min_fire=float(min(fire)), max_silent=float(max(silent)),
                max_leakH=float(max(leakH)), S0_strict=bool(
                    S0 and max(leakH) <= TH_SILENT), tg=tg, names=names,
                leakA=float(r("SHA")[NPOOL + ENT.index("SELF")]),
                prev_adr=float(r("PREV")[pi["ADR"]]
                               / max(tg[("SHA", "ADR")], 1e-9)))


ZERO = {t: 0.0 for t in TOK}


def score(W, thP=(0.40, 0.20), thE=(0.40, 0.20)):
    """replica4_markers.score, unchanged: GO bars on the original routes with
    per-target S0, plus the marker read-outs."""
    Wo = W[:len(TOK)]
    b = bars(Wo, ZERO, 0.0)
    r = lambda t: W[tim[t]]
    fire, silent = [], []
    for t, ks in TARGET.items():
        for k in ks:
            fire.append((r(t)[pi[k]], thP[0]))
    fire.append((r("N_SELF")[NPOOL + ENT.index("SELF")], thE[0]))
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        fire.append((r(n)[NPOOL + ENT.index(bearer)], thE[0]))
        silent += [(v, thE[1]) for j, v in enumerate(r(n)[NPOOL:])
                   if ENT[j] != bearer]
    for t in TOK:
        for j, pool in enumerate(POOLS):
            if t in TARGET and pool in TARGET[t]:
                continue
            if t == "PREV" and pool in ("GIV", "ADR"):
                continue
            if t in ("IDIO", "PLUR"):
                continue
            silent.append((r(t)[j], thP[1]))
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]:
        silent += [(v, thE[1]) for v in r(t)[NPOOL:]]
    S0 = all(v >= th for v, th in fire) and all(v <= th for v, th in silent)
    small = min(r("SHS")[pi["SPK"]], r("SHA")[pi["ADR"]])
    mk = {}
    for t in MARK:
        mk[t] = dict(max_pool=float(r(t)[:NPOOL].max()),
                     top_pool=POOLS[int(r(t)[:NPOOL].argmax())],
                     max_ent=float(r(t)[NPOOL:].max()))
    W_ok = all(mk[t]["max_pool"] <= max(thP[1], 0.25 * small)
               and mk[t]["max_pool"] <= thP[1]
               and mk[t]["max_ent"] <= thE[1] for t in CLASS_W_MARK)
    go = dict(S0=S0, S1=b["S1"], S2a=b["S2a"], S2b=b["S2b_nf"],
              S3=b["S3_nf"], S4=b["S4_nf"], S5=b["S5_nf"], MARK_W=W_ok)
    return go, mk


def shuffle_labels(rows, seed):
    """replica3.shuffle_labels over (x, y) rows: pool columns permuted."""
    rng = np.random.default_rng(seed)
    out = []
    for x, y in rows:
        y2 = y.copy()
        y2[:NPOOL] = y[rng.permutation(NPOOL)]
        out.append((x, y2))
    return out


def shuffle_scene(rows, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    return [(rows[i][0], rows[j][1]) for i, j in enumerate(perm)]


def nulls(rows, rule, lam, tau, seed, n=5, **kw):
    """replica3.nulls over (x, y) rows: per-token null95 from n in-memory
    label-shuffled runs of the RECORDED rows (derived transform, no world)."""
    samp = {t: [] for t in TOK}
    cls_name = []
    for i in range(n):
        Wn = train_m(shuffle_labels(rows, 10000 * seed + i), rule, lam, tau,
                     **kw)
        for t in TOK:
            samp[t].extend(list(Wn[ti[t], :NPOOL]))
        cls_name.append(max(Wn[ti[t], :NPOOL].max() for t in NAMES))
    return ({t: float(np.percentile(v, 95)) for t, v in samp.items()},
            float(np.percentile(cls_name, 95)))


# ── the replica's stream reader (stream_audit.read_stream, unchanged) ─────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_stream(path):
    """(rows, sha256): rows are (x, y) in this module's TOKM/POST order.
    Refuses a stream whose header vocabulary differs from the replica's."""
    sha = sha256(path)
    rows = []
    with open(path) as f:
        head = json.loads(f.readline())
        if head["TOK"] != TOKM or head["POST"] != POST:
            raise ValueError("stream vocabulary differs from the replica's")
        for line in f:
            d = json.loads(line)
            x = np.zeros(len(TOKM))
            y = np.zeros(len(POST))
            x[d["x"]] = 1.0
            for i, v in d["y"]:
                y[i] = v
            rows.append((x, y))
    return rows, sha


def load_thresholds(path):
    """calibration.json -> {"P": [fire, silent], "E": [fire, silent]}.
    Refuses (raises) when absent or malformed: no default is invented."""
    with open(path) as f:
        d = json.load(f)
    th = d["thresholds"]
    for k in ("P", "E"):
        v = th[k]
        if not (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(q, (int, float)) for q in v)):
            raise ValueError("thresholds.%s must be [fire, silent]" % k)
    return {"P": [float(th["P"][0]), float(th["P"][1])],
            "E": [float(th["E"][0]), float(th["E"][1])]}


def _repo_git(repo):
    """True only when git metadata belongs to `repo` ITSELF. An ancestor
    directory's repo (e.g. sim/ on pool nodes, or a developer's dotfiles repo)
    must not satisfy the check - otherwise `git show HEAD:<rel>` verifies
    against the wrong HEAD (or worse, silently passes)."""
    import subprocess
    p = subprocess.run(["git", "-C", repo, "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    top = (p.stdout or "").strip()
    return bool(p.returncode == 0 and top and
                os.path.realpath(top) == os.path.realpath(repo))


def verify_committed_sha(path):
    """Refuse a thresholds file whose bytes differ from the committed one
    (packet module-4 spec line 83). Uncommitted -> refusal. Where the tree
    carries NO git metadata at all (pool revisions are tree copies of a
    checkout), there is no HEAD to compare against: record the file's sha256
    with committed="unverifiable-no-git" and say so loudly - never silently
    treat it as checked (the bar-sanity idiom, packet module 6). Every other
    failure (missing path, outside repo, uncommitted or byte-different with
    git present) still refuses."""
    import subprocess
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(".."):
        raise ValueError("%s is outside the repo" % path)
    probe_ok = _repo_git(REPO)
    if not probe_ok:
        got_sha = sha256(path)
        print("verify_committed_sha: NO git metadata at %s -> recording "
              "sha256=%s.. committed=unverifiable-no-git (NOT checked)"
              % (REPO, got_sha[:16]), flush=True)
        return {"sha256": got_sha, "committed": "unverifiable-no-git"}
    got = subprocess.run(["git", "-C", REPO, "show", "HEAD:" + rel],
                         capture_output=True)
    if got.returncode != 0:
        raise ValueError("%s is not committed at HEAD" % rel)
    committed = hashlib.sha256(got.stdout).hexdigest()
    if committed != sha256(path):
        raise ValueError("%s differs from the committed bytes" % rel)
    return {"sha256": committed, "committed": committed}


def load_manifests(streams_dir):
    """Every manifest_<arm>.json in the dir -> (by_arm, by_key) where
    by_key[(arm, seed)] = {"file": abs path, "sha256": recorded}."""
    by_arm = {}
    by_key = {}
    names = sorted(n for n in os.listdir(streams_dir)
                   if n.startswith("manifest_") and n.endswith(".json"))
    if not names:
        raise FileNotFoundError("no manifest_*.json in %s" % streams_dir)
    for name in names:
        with open(os.path.join(streams_dir, name)) as f:
            m = json.load(f)
        if m.get("schema") != "deixis-streams-manifest-v1":
            raise ValueError("%s: unexpected schema" % name)
        for e in m["streams"]:
            p = os.path.join(streams_dir, e["file"])
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "manifest %s lists missing file %s" % (name, e["file"]))
            if sha256(p) != e["sha256"]:
                raise ValueError(
                    "stream sha256 mismatch vs manifest: %s" % e["file"])
            by_key[(e["arm"], int(e["seed"]))] = {
                "file": p, "sha256": e["sha256"]}
        by_arm[m["arm"]] = {"length": m.get("length"),
                            "entries": len(m["streams"]),
                            "manifest": name}
    return by_arm, by_key


def _rng(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="0a(ii): the rate replica on recorded streams")
    ap.add_argument("--streams",
                    help="dir holding manifest_*.json and the stream files")
    ap.add_argument("--thresholds", help="calibration.json")
    ap.add_argument("--grid", help="comma list, e.g. 1,2,...")
    ap.add_argument("--seeds", help="e.g. 1-15")
    ap.add_argument("--rate-seeds", dest="rate_seeds")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest()))
        return 0
    missing = [n for n, v in (("--streams", args.streams),
                              ("--thresholds", args.thresholds),
                              ("--grid", args.grid),
                              ("--seeds", args.seeds),
                              ("--rate-seeds", args.rate_seeds),
                              ("--out", args.out)) if not v]
    if missing:
        ap.error("the following arguments are required: "
                 + ", ".join(missing))
    grid = [float(g) for g in args.grid.split(",") if g]
    seeds = _rng(args.seeds)
    rate_seeds = _rng(args.rate_seeds)
    th = load_thresholds(args.thresholds)
    th_verify = verify_committed_sha(args.thresholds)
    by_arm, by_key = load_manifests(args.streams)

    out = {"schema": "deixis-replica-v1", "thresholds": th,
           "thresholds_source": os.path.abspath(args.thresholds),
           "thresholds_sha256": sha256(args.thresholds),
           "thresholds_verify": th_verify, "grid": grid,
           "seeds": seeds, "rate_seeds": rate_seeds, "manifests": by_arm,
           "arm_note": ("the replica's rate arm 'hebb' stands for CAUSAL and "
                        "RW-HEBB (the replica has no spike timing, stated); "
                        "'hebbwta' = CAUSAL+WTA; plus BCM, COV, COV-PRE"),
           "streams": {}, "rules": {}, "rate_seed_scores": {},
           "null_arms": {}, "k0": {}}
    cache = {}

    def rows_for(arm, s):
        if (arm, s) not in cache:
            if (arm, s) not in by_key:
                raise FileNotFoundError("no %s stream for seed %d" % (arm, s))
            rows, sha = read_stream(by_key[(arm, s)]["file"])
            cache[(arm, s)] = (rows, sha, by_key[(arm, s)])
        return cache[(arm, s)]

    for s in seeds:
        _, sha, ent = rows_for("base", s)
        out["streams"][str(s)] = {"arm": "base",
                                  "file": os.path.basename(ent["file"]),
                                  "sha256": sha}

    for rule in RULES:
        cells = {}
        for lt in grid:
            allgo, passes, mins = 0, 0, []
            for s in seeds:
                rows, _, _ = rows_for("base", s)
                W = train_m(rows, rule, lt / 400.0, 400)
                go, _mk = score(W, th["P"], th["E"])
                allgo += int(all(go.values()))
                passes += sum(1 for v in go.values() if v)
                mins.append(bars(W[:len(TOK)], ZERO, 0.0)["min_fire"])
            cells["%g" % lt] = {"all_go": allgo, "bar_passes": passes,
                                "median_min_fire": float(np.median(mins))}
        # centre = most seeds passing every GO bar; ties: most bar-seed
        # passes; then larger median weakest must-fire route (packet 85-87)
        centre = max(grid, key=lambda lt: (
            cells["%g" % lt]["all_go"], cells["%g" % lt]["bar_passes"],
            cells["%g" % lt]["median_min_fire"]))
        rec = {"grid_cells": cells, "centre": centre, "weights": {},
               "scores": {}, "sha": {}}
        for s in seeds:
            rows, sha, ent = rows_for("base", s)
            W = train_m(rows, rule, centre / 400.0, 400)
            go, mk = score(W, th["P"], th["E"])
            b = bars(W[:len(TOK)], ZERO, 0.0)
            rec["weights"][str(s)] = [[round(float(v), 6) for v in row]
                                      for row in W]
            rec["scores"][str(s)] = {"go": {k: bool(go[k]) for k in go},
                                     "markers": mk,
                                     "bars_nf": {k: bool(b[k])
                                                 for k in KEYS_NF},
                                     "min_fire": b["min_fire"],
                                     "max_silent": b["max_silent"]}
            rec["sha"][str(s)] = sha
        # the 60-seed rate on seeds 1-60 at this arm's centre
        rate_rec = {}
        for s in rate_seeds:
            rows, sha, _ = rows_for("base", s)
            W = train_m(rows, rule, centre / 400.0, 400)
            go, mk = score(W, th["P"], th["E"])
            rate_rec[str(s)] = {"go": {k: bool(go[k]) for k in go},
                                "sha256": sha, "markers": mk}
        out["rate_seed_scores"][rule] = rate_rec
        # label, scene and voice nulls on seeds 1-15 at this centre
        null_rec = {}
        for arm in ("label", "scene", "voice"):
            arm_rec = {}
            for s in seeds:
                rows, sha, _ = rows_for(arm, s)
                W = train_m(rows, rule, centre / 400.0, 400)
                go, mk = score(W, th["P"], th["E"])
                arm_rec[str(s)] = {"go": {k: bool(go[k]) for k in go},
                                   "sha256": sha, "markers": mk}
            null_rec[arm] = arm_rec
        out["null_arms"][rule] = null_rec
        # K = 0 (all weights zero) at this centre
        W0 = np.zeros((len(TOKM), len(POST)))
        go0, mk0 = score(W0, th["P"], th["E"])
        b0 = bars(W0[:len(TOK)], ZERO, 0.0)
        out["k0"][rule] = {"go": {k: bool(go0[k]) for k in go0},
                           "markers": mk0, "min_fire": b0["min_fire"],
                           "max_silent": b0["max_silent"]}
        out["rules"][rule] = rec
        print("rule %-7s centre %g all_go %d/%d" % (
            rule, centre, cells["%g" % centre]["all_go"], len(seeds)),
            flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote %s (%d rules x %d rate seeds, %d null arms)" % (
        args.out, len(out["rules"]),
        sum(len(v) for v in out["rate_seed_scores"].values()),
        len(out["null_arms"])))
    return 0


def selftest():
    """Synthetic stream in a temp dir: determinism, shapes, refusal paths."""
    res = {}
    with tempfile.TemporaryDirectory() as td:
        sfile = os.path.join(td, "base_s1.jsonl")
        rng = np.random.default_rng(7)
        with open(sfile, "w") as f:
            f.write(json.dumps({"schema": "deixis-toy-stream-v1", "seed": 1,
                                "T": 40, "arm": "base", "world_kw": {},
                                "TOK": TOKM, "POST": POST,
                                "length": 40}) + "\n")
            for t in range(40):
                x = [int(i) for i in rng.choice(len(TOKM), 4, replace=False)]
                yv = []
                for i in range(len(POST)):
                    v = 1.0 if rng.random() < 0.25 else (
                        PRES if rng.random() < 0.4 else 0.0)
                    if v:
                        yv.append([i, round(v, 6)])
                f.write(json.dumps({"t": t, "x": x, "y": yv, "S": "V1",
                                    "A": "V2"}) + "\n")
        rows, sha = read_stream(sfile)
        assert len(rows) == 40 and len(sha) == 64
        res["read_ok"] = True
        # header refusal
        bad = os.path.join(td, "bad.jsonl")
        with open(bad, "w") as f:
            f.write(json.dumps({"schema": "deixis-toy-stream-v1", "seed": 2,
                                "T": 1, "arm": "base", "world_kw": {},
                                "TOK": TOK, "POST": POST, "length": 1})
                    + "\n")
            f.write(json.dumps({"t": 0, "x": [0], "y": [[0, 1.0]],
                                "S": "V1", "A": "V2"}) + "\n")
        try:
            read_stream(bad)
            raise AssertionError("vocabulary refusal did not fire")
        except ValueError:
            res["vocab_refusal"] = True
        # determinism + score shape
        W1 = train_m(rows, "covpre", 6 / 400.0, 400)
        W2 = train_m(rows, "covpre", 6 / 400.0, 400)
        assert np.allclose(W1, W2) and W1.shape == (len(TOKM), len(POST))
        assert float(W1.min()) >= 0.0 and float(W1.max()) <= 1.0
        go, mk = score(W1, (0.40, 0.20), (0.45, 0.20))
        assert set(go) == {"S0", "S1", "S2a", "S2b", "S3", "S4", "S5",
                           "MARK_W"} and set(mk) == set(MARK)
        res["score_keys"] = sorted(go)
        n95, c95 = nulls(rows, "covpre", 6 / 400.0, 400, 1, n=2)
        b = bars(W1, n95, c95)
        assert all(k in b for k in KEYS_NF)
        res["bars_nf"] = {k: bool(b[k]) for k in KEYS_NF}
        # thresholds refusal
        tj = os.path.join(td, "cal.json")
        with open(tj, "w") as f:
            json.dump({"thresholds": {"P": [0.4, 0.2], "E": [0.45, 0.2]}}, f)
        assert load_thresholds(tj)["E"] == [0.45, 0.2]
        with open(tj, "w") as f:
            json.dump({"thresholds": {"P": [0.4, 0.2], "E": 0.45}}, f)
        try:
            load_thresholds(tj)
            raise AssertionError("threshold refusal did not fire")
        except ValueError:
            res["th_refusal"] = True
        # shuffle helpers stay in-range
        sl = shuffle_labels(rows, 3)
        assert len(sl) == 40 and float(sl[0][1][:NPOOL].sum()) == float(
            rows[0][1][:NPOOL].sum())
        res["shuffle_ok"] = True
        # whose is the must-fire signal? base stream (treatment) vs the
        # label-shuffled control — the gate's attribution discipline, and
        # the same subtraction the null arms exist for.
        Ws = train_m(shuffle_labels(rows, 11), "covpre", 6 / 400.0, 400)
        res["attribution"] = attributable_to(
            "selftest min_fire: base vs label shuffle",
            float(bars(W1[:len(TOK)], ZERO, 0.0)["min_fire"]),
            float(bars(Ws[:len(TOK)], ZERO, 0.0)["min_fire"]))
        # packet test line 99: refuse a stream whose sha256 differs from
        # its manifest
        mj = os.path.join(td, "manifest_base.json")
        with open(mj, "w") as f:
            json.dump({"schema": "deixis-streams-manifest-v1", "arm": "base",
                       "seeds": [1], "length": 40,
                       "streams": [{"seed": 1, "arm": "base",
                                    "file": "base_s1.jsonl",
                                    "sha256": "0" * 64}]}, f)
        try:
            load_manifests(td)
            raise AssertionError("manifest sha refusal did not fire")
        except ValueError:
            res["manifest_refusal"] = True
        with open(mj, "w") as f:
            json.dump({"schema": "deixis-streams-manifest-v1", "arm": "base",
                       "seeds": [1], "length": 40,
                       "streams": [{"seed": 1, "arm": "base",
                                    "file": "base_s1.jsonl",
                                    "sha256": sha256(sfile)}]}, f)
        _am, bk = load_manifests(td)
        assert ("base", 1) in bk and bk[("base", 1)]["sha256"] == sha256(
            sfile)
        res["manifest_ok"] = True
        # packet line 83: thresholds must match the committed bytes. Two
        # structural facts of provisioned revisions: no git metadata at all,
        # and research/coordination/ is not shipped. Both are RECORDED
        # (never silently treated as checked - the bar-sanity idiom); with
        # git present + files shipped, the positive verifies and an
        # uncommitted/outside path refuses.
        import subprocess as _sp  # noqa: F401  (kept: probe moved to _repo_git)
        rb = os.path.join(REPO, "research/coordination/LOCAL_LLM_RUNBOOK.md")
        have_git = _repo_git(REPO)
        try:
            v = verify_committed_sha(rb)
        except FileNotFoundError:
            v = {"sha256": None, "committed": "absent-from-revision"}
        if have_git:
            assert v["committed"] not in ("unverifiable-no-git",
                                          "absent-from-revision"), v
            assert len(v["committed"]) == 64
            res["commit_verify_ok"] = True
        else:
            assert v["committed"] in ("unverifiable-no-git",
                                      "absent-from-revision"), v
            res["commit_verify_ok"] = ("recorded-not-checked: %s"
                                       % v["committed"])
        # refusal half: a path outside the repo (the tmp cal file) refuses on
        # BOTH branches - before any git probe - and with git present an
        # uncommitted in-repo file refuses too (packet test line 99/83).
        try:
            verify_committed_sha(os.path.join(td, "cal.json"))
            raise AssertionError("uncommitted refusal did not fire")
        except ValueError:
            res["commit_refusal"] = True
    res["ok"] = True
    return res


if __name__ == "__main__":
    sys.exit(main())
