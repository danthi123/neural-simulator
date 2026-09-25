#!/usr/bin/env python3
"""lb_shard.py — run the load-bearing battery as (faculty x seed) SHARDS across the pool / AWS, then aggregate.

WHY (2026-09-23): the full all-fixes adequate battery ran ~13h PER SEED serially on the GPU (26 faculties x lesion arms
x repeats in one process) and blocked 17 GPU-bound lane jobs behind a CPU-bound orchestration workload. Each
`load_bearing_fraction --only <faculty>` run is small (~0.5-1 GB RSS), so the battery shards into faculty x seed jobs
that fill the mini-PC pool and a CPU AWS instance in parallel. Each shard gets its OWN output directory: the battery
writes shared intermediate arm files named by probe group (intact_a_well.json, ...), so two shards in one directory
would race on them.

  python tools/lb_shard.py jobs  --seeds 42 43 44 --tag allfixes [--root REMOTE_ROOT]   # print one shell job per line
  python tools/lb_shard.py jobs  --seeds 42 --tag T --no-fixes --probe-set thin            # production default, thin probes
  python tools/lb_shard.py aggregate --tag allfixes [--seeds ...]                       # robust core from shard outputs
  python tools/lb_shard.py aggregate --tag T --pin <sha> [--expect-env BRAIN_X=1 ...]  # verified; flipcand arm (B2c)

The ENV below is the ADEQUATE-probe configuration plus every fix merged on main as of 2026-09-23 (each flag must have
code references on main — gates/finding_mechanism_on_main).
"""
import argparse
import calendar
import fnmatch
import glob
import json
import os
import shlex
import sys
import time

# fixes (mechanisms whose 6-seed GOs make up robust core 23/24). Three of these became production default-ON on
# 2026-09-23 (branch research/flip-validated-fixes); passing them explicitly is then redundant but harmless.
FIX_ENV = {
    "BRAIN_EPISODIC_STORE_VERIFY": "1", "BRAIN_PMEM_FACILITATION": "1", "BRAIN_PMEM_OP_STABILIZER": "1",
    "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE": "1",
}
# probe sets. "adequate" = the 7 drive probes verified 2026-09-20 + pmem + open-ended distributional. "thin" = none:
# the battery's DEFAULT probes (the 2026-09-20 6-seed 0.59 / robust-core-14 measurement). These flags change how the
# metric PROBES, not the brain (they live only in load_bearing_fraction.py).
PROBE_SETS = {
    "adequate": {
        "LB_EPISODIC_DRIVE_PROBE": "1", "LB_SURPRISE_CONFIRM_PROBE": "1", "LB_DISCOURSE_REGISTER_DRIVE_PROBE": "1",
        "LB_CG_DRIVE_PROBE": "1", "LB_NONCONTRADICTION_DRIVE_PROBE": "1", "LB_AFFECT_DRIVE_PROBE": "1",
        "LB_BG_SELECT_DRIVE_PROBE": "1", "LB_PMEM_DRIVE_PROBE": "1", "LB_OPEN_ENDED_DISTRIB_PROBE": "1",
        # NOTE (2026-09-24, review of lane AG-REG / research/lbf-row-registry-hook): LB_SWAP_DRIVE_PROBE and
        # LB_WMB_CONTENT_PROBE were added here on a prior pass of this branch with a comment claiming they were
        # already wired into load_bearing_fraction.measure_faculty (an early-return reusing "wm-binding-advanced",
        # reporting under "wm-binding-recurrence-drive" per "Amendment C"). That wiring, and the two findings docs
        # the comment cited (research/findings/2026-09-23-swap-drives-adequate-probe-*.md,
        # research/findings/2026-09-24-wm-binding-ordinary-content-probe-PREREGISTRATION.md), do not exist anywhere
        # in this lineage -- the real implementation lives on research/swap-drives-adequate-probe /
        # fixround-swap-drives-adequate-probe, which are NOT ancestors of this branch. The two entries were pure
        # no-ops: --probe-set adequate (the default) would set both env vars with no code anywhere reading them,
        # while the comment asserted they drove real probes. DROPPED pending an actual merge of that lineage's
        # wiring into load_bearing_fraction.py -- re-add LB_SWAP_DRIVE_PROBE / LB_WMB_CONTENT_PROBE here only once
        # `grep -rn LB_WMB_CONTENT research/runners/load_bearing_fraction.py` (or the swap-drive equivalent) finds
        # a real reader, not just a comment.
    },
    "thin": {},
}
# the historical default (allfixes / allfixes2): adequate probes + every fix.
ENV = dict(FIX_ENV, **PROBE_SETS["adequate"])


def job_env(probe_set="adequate", fixes=True, extra_env=None):
    """The env dict one shard runs under. `fixes=False` passes NO fix flag, so each mechanism runs at its PRODUCTION
    default (what the owner gets); `probe_set="thin"` passes no LB_* probe flag."""
    envd = dict(FIX_ENV) if fixes else {}
    envd.update(PROBE_SETS[probe_set])
    for kv in (extra_env or []):
        k, _, v = kv.partition("=")
        envd[k] = v
    return envd
OUT_BASE = "research/findings/raw/_load_bearing/_shards"
MEASURABLE_KINDS = ("neural-lesion", "whether-disable", "thin", "mechanism-only")
COVERABLE_KINDS = ("neural-lesion", "whether-disable")
PIN_FILENAME = "PIN.txt"
# B2b Amendment 1.2 (research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md), applied
# generically: a (faculty, seed) cell is a measurement of `pin` only if EVERY sidecar in its shard directory
# records `git_sha` == pin IN FULL, `source_kind` "git_archive", both manifest-verified flags true, and
# env.SIM_BACKEND "numpy". lb_shard.py aggregate never checked this (research/findings/2026-09-25-production-
# default-battery-B2a-FAIL.md: 23/28 coverable faculties' seed-102 cell ran off a dirty local worktree with a
# SHORT git_sha and source_kind null, and the aggregate read clean because it never looked).
PROV_ARM_ALLOWED_BRAIN_KEYS = {"BRAIN_CHAT_SEED"}  # arm runners set this after import; lb.json's own env allows NONE
# B2c (research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md): a FLIP-CANDIDATE
# arm runs with declared BRAIN_* flags in every process's env (`jobs --extra-env`). Under the plain pin rule every such
# cell reads "stray BRAIN_* env key" and is excluded, so a flipcand arm could never be aggregated verified. An
# EXPECTED env (`aggregate --expect-env K=V ...`, or the EXPECT_ENV.txt that `jobs --pin` writes beside PIN.txt) turns
# those keys from forbidden into REQUIRED: every checked sidecar must carry each expected key with exactly the expected
# value (the manipulation check at the provenance level: the flag reached every arm process), and no other BRAIN_* key
# beyond it (+ BRAIN_CHAT_SEED on arm sidecars). No expected env == the B2b Amendment 1.2 rule, unchanged.
EXPECT_ENV_FILENAME = "EXPECT_ENV.txt"


def _pin_file(base, tag):
    return "%s/%s/%s" % (base, tag, PIN_FILENAME)


def _expect_env_file(base, tag):
    return "%s/%s/%s" % (base, tag, EXPECT_ENV_FILENAME)


def parse_expect_env(items):
    """`KEY=VAL` strings -> {KEY: VAL}. Only BRAIN_* keys are accepted: they are the only env class the pin rule
    polices (an LB_* probe flag or a thread count is not a production flag). Raises ValueError on anything else, so a
    typo cannot silently widen or narrow what a verified aggregate admits."""
    out = {}
    for kv in items or ():
        k, sep, v = str(kv).strip().partition("=")
        if not sep or not k.startswith("BRAIN_") or not v:
            raise ValueError("expected-env entry %r is not BRAIN_<NAME>=<value>" % (kv,))
        if k in out and out[k] != v:
            raise ValueError("expected-env key %s given twice with different values" % k)
        out[k] = v
    return out


def _prov_sidecar_fails(prov_path, pin, allow_brain_keys, expect_env=None):
    """Fields of ONE `.prov.json` sidecar that fail the pin rule. Empty list == this sidecar is a clean
    measurement of `pin`. Never raises: an unreadable/missing sidecar is reported as a failure, not skipped.
    `expect_env` ({BRAIN_KEY: value}, default none): each key must be present with exactly that value, and is then
    not a stray key; with none given the rule is exactly B2b Amendment 1.2's."""
    if not os.path.exists(prov_path):
        return ["missing"]
    try:
        pj = json.load(open(prov_path))
    except Exception as e:
        return ["unreadable (%s: %s)" % (type(e).__name__, e)]
    fails = []
    sha = pj.get("git_sha")
    if sha != pin:
        fails.append("git_sha=%r (want %s IN FULL)" % (sha, pin))
    if pj.get("source_kind") != "git_archive":
        fails.append("source_kind=%r (want git_archive)" % pj.get("source_kind"))
    if pj.get("source_manifest_verified_at_start") is not True:
        fails.append("source_manifest_verified_at_start=%r (want true)" % pj.get("source_manifest_verified_at_start"))
    if pj.get("source_manifest_verified_at_exit") is not True:
        fails.append("source_manifest_verified_at_exit=%r (want true)" % pj.get("source_manifest_verified_at_exit"))
    env = pj.get("env") or {}
    if env.get("SIM_BACKEND") != "numpy":
        fails.append("env.SIM_BACKEND=%r (want numpy)" % env.get("SIM_BACKEND"))
    expect_env = expect_env or {}
    stray = sorted(k for k in env if k.startswith("BRAIN_") and k not in allow_brain_keys and k not in expect_env)
    if stray:
        fails.append("stray BRAIN_* env key(s): %s" % ",".join(stray))
    for k, v in sorted(expect_env.items()):
        if k not in env:
            fails.append("expected env %s=%s missing" % (k, v))
        elif env.get(k) != v:
            fails.append("env.%s=%r (want %r)" % (k, env.get(k), v))
    return fails


# covered-by-parent (research/oed-provenance-coverage, closing research/FAILURE_LOG.md's oed-provenance-coverage
# row): filenames a parent load_bearing_fraction.py process may write directly from inside its own measurement
# function, with no sidecar of its own, that are still admissible as a measurement of `pin` PROVIDED the enclosing
# cell's lb.json.prov.json passes the full pin rule, the file's own mtime falls inside that parent process's run
# window (_covered_by_parent_reason), AND (fix round, review PoC 2026-09-25) the file's OWN CONTENT matches the
# cell's own lb.json `per_faculty` entry for that faculty on every field in `_CONTENT_BOUND_FIELDS`
# (_content_matches_parent). The mtime window alone is NOT authorship: a foreign or copied file with an
# in-window mtime sitting next to a clean lb.json.prov.json proves nothing about who wrote it -- only matching
# CONTENT (the same faculty/verdict/load_bearing/diffs/... the parent process itself recorded in lb.json) does.
# Going forward `research.runners.declare_output` closes this at the source (the file gets its own sidecar like
# any other declared output); this allow-list exists only because shards predate that fix and their producing
# code is pinned -- they cannot be cheaply re-run to pick it up. Growing this list is a real claim about a NEW
# parent-written artifact; start narrow and extend only against a confirmed case, never speculatively.
_PARENT_COVERED_ALLOWLIST = ("oed_distributional*.json",)
# Fields checked, verified 2026-09-25 against the real b2a0924 shards (s42/s43/s44/s100/s101/s102's
# open-ended-generation cells): on every real cell, oed_distributional*.json is byte-for-byte the SAME dict as
# lb.json's own `per_faculty` entry for that faculty (a superset of this list also matches, but these are the
# fields that carry the substantive measurement result -- the ones a foreign or merely-timing-coincident file
# cannot be expected to reproduce by accident).
_CONTENT_BOUND_FIELDS = ("faculty", "verdict", "load_bearing", "diffs", "treatment_diffs", "control_diffs",
                         "attributable_fraction", "null_control_clean", "lesion_reproduced")
# Tolerance for filesystem mtime / wall-clock granularity (whole-second `started` strings, coarse mtimes on some
# filesystems) around the parent run's window. A couple of seconds cannot turn an unrelated LATER process into a
# false "covered" -- real reruns are seconds-to-hours apart, never sub-2s -- so this loosens flakiness, not the
# actual property being checked (same-process authorship).
_COVERED_WINDOW_SLACK_S = 2.0


def _parent_run_window(lb_prov_path):
    """(start_epoch, exit_epoch) for the process that wrote the `lb.json` sidecared at `lb_prov_path`, or None if
    either bound cannot be read cleanly -- covered-by-parent must never be granted from a value that failed to
    parse.

    A v1 sidecar (the default -- SIM_PROVENANCE_V2 unset) records only a START time (`started`, whole-second
    resolution -- research/runners/__init__.py's `_record_start`/`_stamp_outputs`); there is no recorded end time,
    so the sidecar FILE's own mtime (the moment `_stamp_outputs` wrote it, at the parent's atexit) is the best
    available stand-in for "when the parent process exited". A v2 sidecar (SIM_PROVENANCE_V2=1) records
    `started_utc_ns`/`ended_utc_ns` directly and those are used instead when both are present.

    TIMEZONE (earned running this against the real B2a shards, 2026-09-25): `started` is formatted via
    `time.strftime(..., time.localtime(_START))` on WHICHEVER machine ran the shard -- a pool/cloud worker's local
    zone need not match the machine running `aggregate`. A pool shard's sidecar read `started="...T21:00:51"`
    (that worker's UTC clock) while re-parsing it with `time.mktime` on this machine (America/New_York, EDT)
    produced a start 4 HOURS AFTER the sidecar's own exit-time mtime -- every real cell failed the plain
    `exit_epoch < start_epoch` sanity check before the window check ever ran. Both plausible readings
    (this-machine-local via `time.mktime`, and UTC via `calendar.timegm` -- the common cloud-instance default) are
    computed and the EARLIER one is used as the lower bound: whichever reading is the true one, the real start is
    then always >= this value, so a genuinely covered file is never pushed outside its own window by a TZ guess."""
    try:
        pj = json.load(open(lb_prov_path))
    except Exception:
        return None
    try:
        if (pj.get("schema") == "sim-run-provenance-v2"
                and isinstance(pj.get("started_utc_ns"), (int, float))
                and isinstance(pj.get("ended_utc_ns"), (int, float))):
            return pj["started_utc_ns"] / 1e9, pj["ended_utc_ns"] / 1e9
        started = pj.get("started")
        if not started:
            return None
        struct = time.strptime(started, "%Y-%m-%dT%H:%M:%S")
        candidates = []
        for to_epoch in (time.mktime, calendar.timegm):
            try:
                candidates.append(to_epoch(struct))
            except Exception:
                pass
        if not candidates:
            return None
        start_epoch = min(candidates)
        exit_epoch = os.path.getmtime(lb_prov_path)
        if exit_epoch < start_epoch:
            return None
        return start_epoch, exit_epoch
    except Exception:
        return None


def _content_matches_parent(candidate_path, lb_per_faculty):
    """True if the JSON object at `candidate_path` carries, on every field in `_CONTENT_BOUND_FIELDS`, the exact
    same value as SOME entry in `lb_per_faculty` (the enclosing cell's own lb.json `per_faculty` list) -- i.e. the
    parent process's own measurement record, not merely a file that happens to sit in the same directory during
    the parent's run window. This is the binding that closes the mtime-coincidence gap (review PoC 2026-09-25): a
    foreign file, or a real file copied byte-for-byte from a DIFFERENT cell, will disagree with THIS cell's own
    lb.json entry on at least one of these fields (its own faculty name, if nothing else) and is rejected here
    even when the window/allow-list checks alone would have admitted it. Never raises: an unreadable or
    non-dict candidate is "no match", not an error -- the caller stays invalid, never crashes the aggregate."""
    try:
        obj = json.load(open(candidate_path))
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    for entry in lb_per_faculty or ():
        if not isinstance(entry, dict):
            continue
        if all(entry.get(f) == obj.get(f) for f in _CONTENT_BOUND_FIELDS):
            return True
    return False


def _covered_by_parent_reason(cell_dir, fn, lb_prov_path, lb_fails, lb_per_faculty=()):
    """None (not eligible -- stays invalid) or a human-readable reason string `fn` (which has NO `.prov.json` of
    its own) counts as 'covered-by-parent': written by the SAME process that wrote `lb.json` in `cell_dir`. ALL
    must hold: `lb_fails` (lb.json.prov.json's own pin-rule check) is empty -- an unclean parent proves nothing
    about what it wrote; `fn` matches `_PARENT_COVERED_ALLOWLIST`; the file's mtime falls inside the parent run's
    recorded window (with `_COVERED_WINDOW_SLACK_S` slack); the file's mtime is at or before
    `lb.json.prov.json`'s own write time (ruling out an unrelated LATER process reusing the same directory); AND
    `fn`'s own content matches some entry of `lb_per_faculty` (this cell's own lb.json `per_faculty` list) on
    every field in `_CONTENT_BOUND_FIELDS` (`_content_matches_parent`) -- the mtime window is a NECESSARY but not
    SUFFICIENT condition; a foreign or copied-from-elsewhere file with a coincidentally in-window mtime must still
    fail here."""
    if lb_fails:
        return None
    if not any(fnmatch.fnmatch(fn, pat) for pat in _PARENT_COVERED_ALLOWLIST):
        return None
    window = _parent_run_window(lb_prov_path)
    if window is None:
        return None
    start_epoch, exit_epoch = window
    try:
        file_mtime = os.path.getmtime(os.path.join(cell_dir, fn))
        sidecar_mtime = os.path.getmtime(lb_prov_path)
    except OSError:
        return None
    slack = _COVERED_WINDOW_SLACK_S
    if not (start_epoch - slack <= file_mtime <= exit_epoch + slack):
        return None
    if not (file_mtime <= sidecar_mtime + slack):
        return None
    if not _content_matches_parent(os.path.join(cell_dir, fn), lb_per_faculty):
        return None
    return ("covered-by-parent: no own sidecar, but written by the same process as lb.json.prov.json "
            "(mtime %.0f in parent window [%.0f, %.0f]), content matches this cell's own lb.json entry"
            % (file_mtime, start_epoch, exit_epoch))


def cell_prov_fails(cell_dir, pin, lb_per_faculty=(), expect_env=None):
    """B2b Amendment 1.2's validity rule for the shard directory `cell_dir` (one faculty x one seed), against
    `pin`. `lb_per_faculty` is this cell's own lb.json `per_faculty` list (the caller already parsed lb.json to
    iterate it -- passed through rather than re-read here), used only for `_covered_by_parent_reason`'s content
    binding. Returns (fails, covered). `fails` is {sidecar_name: [fail strings]}; an empty dict means every checked
    sidecar is a clean measurement of `pin`. Checks `lb.json.prov.json` (no BRAIN_* key allowed) and every OTHER
    non-`.prov.json` file's own `<name>.prov.json` sidecar (BRAIN_CHAT_SEED allowed -- `main()` sets it after
    import). `covered` is {filename: reason} for any file admitted with NO sidecar of its own via
    `_covered_by_parent_reason` -- reported, never silent, and never a reason to treat the cell as MORE trustworthy
    than its sidecars actually show (a covered file rides entirely on lb.json.prov.json's own clean verdict AND
    its own content matching that same lb.json's `per_faculty` entry). `expect_env` (B2c, default none) is applied
    to EVERY checked sidecar, lb.json's and each arm's alike (see `_prov_sidecar_fails`)."""
    out = {}
    covered = {}
    lb_prov_path = os.path.join(cell_dir, "lb.json.prov.json")
    lb_fails = _prov_sidecar_fails(lb_prov_path, pin, allow_brain_keys=set(), expect_env=expect_env)
    if lb_fails:
        out["lb.json.prov.json"] = lb_fails
    try:
        names = sorted(os.listdir(cell_dir))
    except OSError:
        names = []
    for fn in names:
        if fn in ("lb.json", "lb.json.prov.json") or fn.endswith(".prov.json"):
            continue
        own_sidecar = os.path.join(cell_dir, fn + ".prov.json")
        if not os.path.exists(own_sidecar):
            reason = _covered_by_parent_reason(cell_dir, fn, lb_prov_path, lb_fails, lb_per_faculty)
            if reason is not None:
                covered[fn] = reason
                continue
            out[fn + ".prov.json"] = ["missing"]
            continue
        f = _prov_sidecar_fails(own_sidecar, pin, allow_brain_keys=PROV_ARM_ALLOWED_BRAIN_KEYS, expect_env=expect_env)
        if f:
            out[fn + ".prov.json"] = f
    return out, covered


def faculty_keys():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("SIM_NO_PROVENANCE", "1")
    from research.runners import load_bearing_fraction as lbf  # noqa: E402  (map only, no brain build)
    return [k for k, spec in lbf.FACULTY_LESIONS.items() if spec.get("kind") in MEASURABLE_KINDS]


def shard_out(tag, seed, fac):
    return "%s/%s/s%d/%s/lb.json" % (OUT_BASE, tag, seed, fac)


def cmd_jobs(a):
    envd = job_env(a.probe_set, fixes=not a.no_fixes, extra_env=a.extra_env)
    env =" ".join("%s=%s" % kv for kv in sorted(envd.items()))
    keys = a.faculties or faculty_keys()
    for seed in a.seeds:
        for fac in keys:
            out = shard_out(a.tag, seed, fac)
            prefix = ("cd %s && " % a.root) if a.root else ""
            print("%smkdir -p %s && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 %s "
                  ".venv/bin/python -u -m research.runners.load_bearing_fraction --only %s --seed %d --repeats %d "
                  "--out %s" % (prefix, shlex.quote(os.path.dirname(out)), env, shlex.quote(fac), seed, a.repeats,
                                shlex.quote(out)))
    pin_arg = getattr(a, "pin", None)  # getattr: callers building a bare Namespace (e.g. tests) may predate this flag
    if pin_arg:
        # Written NEXT TO the job list so a tag cannot be aggregated unverified by accident: `aggregate` reads
        # this file as its default `--pin` whenever the flag itself is omitted (2026-09-25, closing the B2a gap
        # where a battery's pin lived only in a prose pre-registration `lb_shard.py aggregate` never read).
        pf = _pin_file(OUT_BASE, a.tag)
        os.makedirs(os.path.dirname(pf), exist_ok=True)
        with open(pf, "w") as fh:
            fh.write(pin_arg.strip() + "\n")
        print("# pin recorded for tag %r: %s -> %s" % (a.tag, pin_arg, pf), file=sys.stderr)
        # B2c: every BRAIN_* key the job lines carry is part of the tag's registered contract, recorded beside the pin
        # so a flip-candidate arm aggregates verified by default (a --no-fixes base arm carries none and keeps the
        # plain rule). A stale EXPECT_ENV.txt from an earlier `jobs --pin` of the same tag is removed when this call
        # declares none.
        ef = _expect_env_file(OUT_BASE, a.tag)
        expected = {k: v for k, v in envd.items() if k.startswith("BRAIN_")}
        if expected:
            with open(ef, "w") as fh:
                fh.write("".join("%s=%s\n" % kv for kv in sorted(expected.items())))
            print("# expected BRAIN_* env recorded for tag %r: %s -> %s"
                  % (a.tag, " ".join("%s=%s" % kv for kv in sorted(expected.items())), ef), file=sys.stderr)
        elif os.path.exists(ef):
            os.remove(ef)


def cmd_aggregate(a):
    base = getattr(a, "base", None) or OUT_BASE
    pin_arg = getattr(a, "pin", None)
    pin, pin_source = (pin_arg.strip() if pin_arg else None), ("--pin" if pin_arg else None)
    pin_file = _pin_file(base, a.tag)
    if not pin and os.path.exists(pin_file):
        try:
            recorded = open(pin_file).read().strip()
        except OSError:
            recorded = ""
        if recorded:
            pin, pin_source = recorded, "file:%s" % pin_file
    # B2c expected env: `--expect-env` wins (an empty `--expect-env` explicitly declares none); otherwise the
    # EXPECT_ENV.txt `jobs --pin` recorded for this tag; otherwise none (the plain B2b Amendment 1.2 rule).
    expect_arg = getattr(a, "expect_env", None)
    expect_env, expect_source = {}, None
    if expect_arg is not None:
        expect_env, expect_source = parse_expect_env(expect_arg), "--expect-env"
    else:
        ef = _expect_env_file(base, a.tag)
        if os.path.exists(ef):
            try:
                expect_env = parse_expect_env([ln for ln in open(ef).read().split() if ln.strip()])
            except (OSError, ValueError) as e:
                raise SystemExit("⛔ unreadable %s (%s) -- refusing to aggregate with a guessed expected env" % (ef, e))
            expect_source = "file:%s" % ef

    rows = {}  # fac -> {seed: row}
    invalid_cells = {}  # "s<seed>/<fac>" -> {sidecar: [fail strings]}, EXCLUDED from `rows` -- never counted, never 0
    covered_cells = {}  # "s<seed>/<fac>" -> {filename: reason}, INCLUDED in `rows` -- reported, never silent
    n_checked = 0
    for path in glob.glob("%s/%s/s*/*/lb.json" % (base, a.tag)):
        seed = int(path.split("/")[-3][1:])
        if a.seeds and seed not in a.seeds:
            continue
        try:
            rep = json.load(open(path))
        except Exception:
            continue
        cell_dir = os.path.dirname(path)
        for p in rep.get("per_faculty", []):
            fac, kind = p["faculty"], p.get("kind")
            if pin and kind in COVERABLE_KINDS:
                n_checked += 1
                fails, covered = cell_prov_fails(cell_dir, pin, rep.get("per_faculty", []), expect_env=expect_env)
                if fails:
                    invalid_cells["s%d/%s" % (seed, fac)] = fails
                    continue  # NOT a measurement of `pin` -- excluded, reported below, never scored 0
                if covered:
                    covered_cells["s%d/%s" % (seed, fac)] = covered
            rows.setdefault(fac, {})[seed] = {
                "kind": kind, "verdict": p.get("verdict"), "load_bearing": p.get("load_bearing"),
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
    # BACKEND / HOST / LTM state / per-row env, read from each shard's own provenance sidecar (not assumed):
    # gates/device_and_cost requires the device; a mixed-host or mixed-LTM battery (2026-09-24 plan step S08 --
    # e.g. the flip-defaults s102 shards, some local, some pool) must say so rather than silently averaging over it.
    # research/runners/__init__.py's env filter was extended (this same lane's commit) to also capture BRAIN_/LB_
    # prefixes, so `env` below is now the shard's REAL per-row env, not just the SIM_/GAP-family subset it used to be.
    backends, hosts, ltm_modes = set(), set(), set()
    per_shard = {}
    for prov in glob.glob("%s/%s/s*/*/lb.json.prov.json" % (base, a.tag)):
        seed_dir = prov.split("/")[-3]
        fac_dir = prov.split("/")[-2]
        try:
            pj = json.load(open(prov))
            penv = pj.get("env") or {}
            backend = penv.get("SIM_BACKEND") or "unrecorded"
            host = pj.get("host") or "unrecorded"
            # LTM STATE: this harness's own boot-time KB is not the data-lake LTM (that is S04's BRAIN_DATA_ROOT /
            # DA-tag-capture concern); a shard only ran against the LTM if it explicitly carried BRAIN_DATA_ROOT.
            # n_facts itself is not something this generic sidecar can know (it would require the shard to have
            # QUERIED the data lake) -- reported honestly as unmeasured, never guessed at zero.
            ltm_on = bool(penv.get("BRAIN_DATA_ROOT"))
            row_env = {k: v for k, v in sorted(penv.items()) if k.startswith(("BRAIN_", "LB_"))}
        except Exception:
            backend, host, ltm_on, row_env = "unreadable", "unreadable", None, {}
        backends.add(backend)
        hosts.add(host)
        ltm_modes.add(ltm_on)
        per_shard["%s/%s" % (seed_dir, fac_dir)] = {
            "backend": backend, "host": host, "ltm_on": ltm_on, "n_facts": None,  # see note above
            "env": row_env,
        }
    out["backend"] = sorted(backends)[0] if len(backends) == 1 else "mixed:" + ",".join(sorted(backends))
    out["backend_source"] = "per-shard provenance sidecars (lb.json.prov.json env.SIM_BACKEND)"
    out["host"] = sorted(hosts)[0] if len(hosts) == 1 else "mixed:" + ",".join(sorted(hosts))
    out["ltm_mode"] = ("on" if ltm_modes == {True} else "off" if ltm_modes == {False} else
                        "mixed" if len(ltm_modes) > 1 else "unrecorded")
    out["ltm_n_facts_note"] = ("not measured by this generic sidecar -- BRAIN_DATA_ROOT presence/absence (ltm_mode) "
                                "is recorded; a per-shard fact COUNT needs the shard to query the data lake itself")
    out["per_shard_prov"] = per_shard

    # PROVENANCE GATE (2026-09-25, closes research/FAILURE_LOG.md's B2a row): with a pin, every coverable cell is
    # checked against it and a failing cell is EXCLUDED above (never counted, never scored 0) and reported here
    # with its failing fields. Without a pin (none passed, none recorded for this tag), nothing was checked --
    # that is reported loudly rather than silently, exactly the silence that let B2a's 23 off-pin cells through.
    n_invalid = len(invalid_cells)
    n_covered = len(covered_cells)
    if pin:
        out["provenance"] = {
            "status": "verified", "pin": pin, "pin_source": pin_source, "rule": "B2b Amendment 1.2",
            "n_cells_checked": n_checked, "n_valid": n_checked - n_invalid, "n_invalid": n_invalid,
            "invalid_cells": invalid_cells,
            # covered-by-parent (research/oed-provenance-coverage): cells counted as VALID above where at least
            # one file had no sidecar of its own but was admitted via cell_prov_fails's allow-list + run-window
            # check (see tools/lb_shard.py's _covered_by_parent_reason). Named here, not folded silently into
            # n_valid with no trace -- a covered cell's validity rests entirely on lb.json.prov.json's own clean
            # verdict, and that distinction must survive into the artifact.
            "n_covered_by_parent": n_covered, "covered_by_parent_cells": covered_cells,
        }
        if expect_env:
            # B2c: named in the artifact, never folded in silently -- a verified flip-candidate aggregate must say
            # which BRAIN_* flags it REQUIRED on every sidecar (keys absent when none was expected: B2a/B2b unchanged).
            out["provenance"]["rule"] = "B2b Amendment 1.2 + B2c expected env"
            out["provenance"]["expect_env"] = dict(sorted(expect_env.items()))
            out["provenance"]["expect_env_source"] = expect_source
    else:
        out["provenance"] = {
            "status": "unverified", "pin": None, "pin_source": None,
            "warning": ("no --pin given and no recorded pin at %s -- per-cell provenance was NOT checked; a cell "
                        "may have run off the registered revision without detection (2026-09-25 B2a)." % pin_file),
        }

    dest = getattr(a, "out", None) or ("%s/%s/aggregate.json" % (base, a.tag))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("robust_core_n", "union_n", "mean_fraction", "incomplete_faculties")}))
    if pin:
        print("provenance: pin=%s (%s) n_checked=%d n_valid=%d n_invalid=%d n_covered_by_parent=%d"
              % (pin, pin_source, n_checked, n_checked - n_invalid, n_invalid, n_covered))
        if expect_env:
            print("provenance: expected env (%s): %s"
                  % (expect_source, " ".join("%s=%s" % kv for kv in sorted(expect_env.items()))))
        for k in sorted(invalid_cells):
            print("  INVALID %s: %s" % (k, "; ".join("%s[%s]" % (sc, ", ".join(fs))
                                                       for sc, fs in sorted(invalid_cells[k].items()))))
        for k in sorted(covered_cells):
            print("  COVERED-BY-PARENT %s: %s" % (k, "; ".join("%s[%s]" % (fn, reason)
                                                    for fn, reason in sorted(covered_cells[k].items()))))
    else:
        print("⛔ provenance: unverified -- %s" % out["provenance"]["warning"], file=sys.stderr)
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
    j.add_argument("--probe-set", choices=sorted(PROBE_SETS), default="adequate",
                   help="adequate (default) = the verified LB_* drive probes; thin = none (the battery's default probes)")
    j.add_argument("--no-fixes", action="store_true",
                   help="pass NO fix flag: each mechanism runs at its production default")
    j.add_argument("--pin", default=None,
                   help="record this tag's registered git_archive revision (full SHA) as its default pin, written "
                        "to <OUT_BASE>/<tag>/%s -- `aggregate` reads it automatically when --pin is omitted, so a "
                        "tag cannot be aggregated unverified by accident" % PIN_FILENAME)
    g = sub.add_parser("aggregate")
    g.add_argument("--tag", required=True)
    g.add_argument("--seeds", type=int, nargs="*", default=None)
    g.add_argument("--pin", default=None,
                   help="full git SHA this tag is pinned to (B2b Amendment 1.2). A coverable cell whose sidecars "
                        "don't match IN FULL (git_sha, source_kind=git_archive, both manifest-verified flags, "
                        "env.SIM_BACKEND=numpy, no stray BRAIN_* key) is excluded and reported, never counted, "
                        "never scored 0. Defaults to the pin recorded by `jobs --pin` for this tag, if any; with "
                        "neither, prints a loud warning and marks the aggregate provenance 'unverified'.")
    g.add_argument("--expect-env", nargs="*", default=None, metavar="BRAIN_KEY=VAL",
                   help="B2c: BRAIN_* flags this tag's job lines carry (a flip-candidate arm). With --pin, every "
                        "checked sidecar must hold each one with exactly that value and no other BRAIN_* key "
                        "(+ BRAIN_CHAT_SEED on arm sidecars). Defaults to the %s `jobs --pin` recorded for this "
                        "tag, if any; an empty --expect-env explicitly declares none." % EXPECT_ENV_FILENAME)
    g.add_argument("--base", default=OUT_BASE,
                   help="root directory holding <tag>/s<seed>/<faculty>/lb.json (default: %s). Override to "
                        "aggregate a shard tree checked out elsewhere (e.g. the primary checkout) without cd-ing "
                        "there." % OUT_BASE)
    g.add_argument("--out", default=None,
                   help="destination aggregate.json path (default: <base>/<tag>/aggregate.json)")
    a = ap.parse_args()
    {"jobs": cmd_jobs, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
