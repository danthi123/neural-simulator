#!/usr/bin/env python3
"""lb_shard.py — run the load-bearing battery as (faculty x seed) SHARDS across the pool / AWS, then aggregate.

WHY (2026-09-23): the full all-fixes adequate battery ran ~13h PER SEED serially on the GPU (26 faculties x lesion arms
x repeats in one process) and blocked 17 GPU-bound lane jobs behind a CPU-bound orchestration workload. Each
`load_bearing_fraction --only <faculty>` run is small (~0.5-1 GB RSS), so the battery shards into faculty x seed jobs
that fill the mini-PC pool and a CPU AWS instance in parallel. Each shard gets its OWN output directory: the battery
writes shared intermediate arm files named by probe group (intact_a_well.json, ...), so two shards in one directory
would race on them.

  python tools/lb_shard.py jobs  --seeds 42 43 44 --tag allfixes [--root REMOTE_ROOT]   # print one shell job per line
  python tools/lb_shard.py aggregate --tag allfixes [--seeds ...]                        # robust core from shard outputs

The ENV below is the ADEQUATE-probe configuration plus every fix merged on main as of 2026-09-23 (each flag must have
code references on main — gates/finding_mechanism_on_main).
"""
import argparse
import glob
import json
import os
import shlex
import sys

ENV = {
    # fixes (default-OFF mechanisms whose 6-seed GOs make up robust core 23)
    "BRAIN_EPISODIC_STORE_VERIFY": "1", "BRAIN_PMEM_FACILITATION": "1", "BRAIN_PMEM_OP_STABILIZER": "1",
    "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE": "1",
    # adequate drive probes (the 7 verified 2026-09-20 + pmem + open-ended distributional)
    "LB_EPISODIC_DRIVE_PROBE": "1", "LB_SURPRISE_CONFIRM_PROBE": "1", "LB_DISCOURSE_REGISTER_DRIVE_PROBE": "1",
    "LB_CG_DRIVE_PROBE": "1", "LB_NONCONTRADICTION_DRIVE_PROBE": "1", "LB_AFFECT_DRIVE_PROBE": "1",
    "LB_BG_SELECT_DRIVE_PROBE": "1", "LB_PMEM_DRIVE_PROBE": "1", "LB_OPEN_ENDED_DISTRIB_PROBE": "1",
}
OUT_BASE = "research/findings/raw/_load_bearing/_shards"
MEASURABLE_KINDS = ("neural-lesion", "whether-disable", "thin", "mechanism-only")


def faculty_keys():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("SIM_NO_PROVENANCE", "1")
    from research.runners import load_bearing_fraction as lbf  # noqa: E402  (map only, no brain build)
    return [k for k, spec in lbf.FACULTY_LESIONS.items() if spec.get("kind") in MEASURABLE_KINDS]


def shard_out(tag, seed, fac):
    return "%s/%s/s%d/%s/lb.json" % (OUT_BASE, tag, seed, fac)


def cmd_jobs(a):
    envd = dict(ENV)
    for kv in (a.extra_env or []):
        k, _, v = kv.partition("=")
        envd[k] = v
    env = " ".join("%s=%s" % kv for kv in sorted(envd.items()))
    keys = a.faculties or faculty_keys()
    for seed in a.seeds:
        for fac in keys:
            out = shard_out(a.tag, seed, fac)
            prefix = ("cd %s && " % a.root) if a.root else ""
            print("%smkdir -p %s && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 %s "
                  ".venv/bin/python -u -m research.runners.load_bearing_fraction --only %s --seed %d --repeats %d "
                  "--out %s" % (prefix, shlex.quote(os.path.dirname(out)), env, shlex.quote(fac), seed, a.repeats,
                                shlex.quote(out)))


def cmd_aggregate(a):
    rows = {}  # fac -> {seed: row}
    for path in glob.glob("%s/%s/s*/*/lb.json" % (OUT_BASE, a.tag)):
        seed = int(path.split("/")[-3][1:])
        if a.seeds and seed not in a.seeds:
            continue
        try:
            rep = json.load(open(path))
        except Exception:
            continue
        for p in rep.get("per_faculty", []):
            rows.setdefault(p["faculty"], {})[seed] = {
                "kind": p.get("kind"), "verdict": p.get("verdict"), "load_bearing": p.get("load_bearing"),
                "null_clean": p.get("null_control_clean"), "unreliable": bool(rep.get("UNRELIABLE"))}
    seeds = sorted(a.seeds or {s for r in rows.values() for s in r})
    out = {"tag": a.tag, "seeds": seeds, "per_faculty": {}, "per_seed": {}}
    for s in seeds:
        cov = [f for f, r in rows.items() if s in r and r[s]["kind"] in ("neural-lesion", "whether-disable")]
        ex = [f for f in cov if rows[f][s]["verdict"] in ("regressed", "pass", "trace-only")]  # trace-only: LB_SWAP_DRIVE_PROBE
        lb = [f for f in ex if rows[f][s]["load_bearing"] is True]
        out["per_seed"][s] = {"n_coverable_present": len(cov), "n_exercised": len(ex), "n_load_bearing": len(lb),
                              "load_bearing_fraction": (len(lb) / len(ex)) if ex else None}
    robust, union, missing = [], [], []
    for f, r in sorted(rows.items()):
        if not any(v["kind"] in ("neural-lesion", "whether-disable") for v in r.values()):
            continue
        have = [s for s in seeds if s in r]
        n_lb = sum(1 for s in have if r[s]["load_bearing"] is True)
        dirty = [s for s in have if r[s]["null_clean"] is False or r[s]["unreliable"]]
        out["per_faculty"][f] = {"seeds_present": have, "n_load_bearing": n_lb, "dirty_seeds": dirty,
                                 "verdicts": {s: r[s]["verdict"] for s in have}}
        if len(have) < len(seeds):
            missing.append(f)
        if n_lb == len(seeds) and len(have) == len(seeds):
            robust.append(f)
        if n_lb:
            union.append(f)
    out["robust_core"] = robust
    out["robust_core_n"] = len(robust)
    out["union_n"] = len(union)
    out["incomplete_faculties"] = missing
    fracs = [v["load_bearing_fraction"] for v in out["per_seed"].values() if v["load_bearing_fraction"] is not None]
    out["mean_fraction"] = (sum(fracs) / len(fracs)) if fracs else None
    out["sd_fraction"] = (sum((f - out["mean_fraction"]) ** 2 for f in fracs) / len(fracs)) ** 0.5 if fracs else None
    out["mean_fraction_3dp"] = round(out["mean_fraction"], 3) if fracs else None
    out["sd_fraction_3dp"] = round(out["sd_fraction"], 3) if fracs else None
    # BACKEND, read from each shard's own provenance sidecar (not assumed): gates/device_and_cost requires the device.
    backends = set()
    for prov in glob.glob("%s/%s/s*/*/lb.json.prov.json" % (OUT_BASE, a.tag)):
        try:
            backends.add((json.load(open(prov)).get("env") or {}).get("SIM_BACKEND") or "unrecorded")
        except Exception:
            backends.add("unreadable")
    out["backend"] = sorted(backends)[0] if len(backends) == 1 else "mixed:" + ",".join(sorted(backends))
    out["backend_source"] = "per-shard provenance sidecars (lb.json.prov.json env.SIM_BACKEND)"
    dest = "%s/%s/aggregate.json" % (OUT_BASE, a.tag)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("robust_core_n", "union_n", "mean_fraction", "incomplete_faculties")}))
    print("wrote", dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--seeds", type=int, nargs="+", required=True)
    j.add_argument("--tag", required=True)
    j.add_argument("--root", default=None, help="cd here first (e.g. a pool isolated-revision dir)")
    j.add_argument("--repeats", type=int, default=2)
    j.add_argument("--faculties", nargs="*", default=None)
    j.add_argument("--extra-env", nargs="*", default=None, help="KEY=VAL flags added on top of ENV (e.g. a newly merged fix)")
    g = sub.add_parser("aggregate")
    g.add_argument("--tag", required=True)
    g.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    {"jobs": cmd_jobs, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
