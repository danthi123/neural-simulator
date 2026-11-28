"""Pick (the measure-first step-0 bars) and verdict of the mix-night screen. Rule: brain/research/2026-11-27-mix-night-prereg.md
(design: brain/research/2026-11-27-sleep-drift-recipe-check.md section 4). The pass rule is
brain.experiments.mix_night_screen.score, imported, not copied. Added here only: the one-word digest line (PASS if any
candidate PASSes, KILL if every candidate is KILL, else INCONCLUSIVE; a KILL-UNREAD, a KILL whose A0 window shows no
readable drift, is not a KILL and makes the digest INCONCLUSIVE, pre-data review 2 N1); the per-candidate statuses, the placebo read and the
full score are carried in the verdict file and are the registered output."""
import math
import re
import shutil
from pathlib import Path

STEP0_MAX_MIN = 45.0     # the note: "if a night + reading exceeds 45 min, drop to nights 14-15 and drop A4 and say so before launch"
CAP_MULT, CAP_ADD_GB = 1.3, 1   # run_node.sh: cap = ceil(1.3 x measured RSS) + 1 GB
ALL_ARMS = "A0r A0L A0 A0e A0k A1 A2 A3 A4 A5s A5f"
STEP0_ARMS = ("A0r", "A2")    # run_node.sh measure: the cap is sized from max(RSS A0r, RSS A2), A2 = the worst-case arm
SEEDS, TIME_SEED = (319, 320, 321), 319   # every node measures its own; the 45-min bar is read on pool40 (s319)


def parse_time_v(text):
    """(max RSS in kB, wall seconds) from a GNU /usr/bin/time -v report; None for a field it lacks."""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text or "")
    w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text or "")
    wall = None
    if w:
        parts = [float(x) for x in w.group(1).split(":")]
        wall = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return (int(m.group(1)) if m else None), wall


def cap_gb(rss_kb):
    return math.ceil(CAP_MULT * rss_kb / 1048576) + CAP_ADD_GB


def reports(texts, arm):
    """[(rss kB, wall s)] of every /usr/bin/time -v report of this arm (time_<arm>.<epoch>.txt, one per attempt)."""
    return [parse_time_v(t) for f, t in sorted((texts or {}).items()) if re.fullmatch(rf"time_{arm}\.\d+\.txt", f)]


def rss_by_arm(texts):
    """{arm: max RSS kB over its reports}: a resumed attempt (tiny RSS) never lowers the measured peak."""
    arms = {re.fullmatch(r"time_(\w+)\.\d+\.txt", f).group(1) for f in (texts or {}) if re.fullmatch(r"time_\w+\.\d+\.txt", f)}
    out = {A: max((r for r, _ in reports(texts, A) if r is not None), default=None) for A in arms}
    return {A: r for A, r in sorted(out.items()) if r is not None}


def det_ok(texts):
    """True / False from step 0's detcheck.txt (its last line is `rc <exit code>`; 0 = the A0r/A0 n13 pair agrees)."""
    t = (texts or {}).get("detcheck.txt")
    m = re.findall(r"^rc (\d+)$", t or "", re.M)
    return None if not m else m[-1] == "0"


def nondeterministic_cuda_units(units):
    """Step-0 units run on cuda without the S3 switch (deterministic is not True): the pre-S3 units, never read."""
    return sorted(f for f, u in (units or {}).items()
                  if isinstance(u, dict) and str(u.get("device", "")).startswith("cuda") and u.get("deterministic") is not True)


def pick(dev, ctx):
    mem = int(ctx.get("memory_gb") or 0)
    info, caps = {"step0": {}}, {}
    for S in SEEDS:
        b = dev.get(f"step0_s{S}") or {}
        units, texts = b.get("units", {}), b.get("texts", {})
        nondet = nondeterministic_cuda_units(units)   # Amendment S3: a cuda unit without the deterministic switch is void
        if nondet:
            return {**info, "stop": f"step 0 s{S}: cuda units without --deterministic {nondet} (Amendment S3: discarded; "
                                    "rerun step 0 with the S3 spec at a new commit, --reset)"}
        d = det_ok(texts)                  # first: a miss ends step 0 before A2, so A2's unit is absent by design
        if d is False:
            return {**info, "stop": f"step 0 s{S}: determinism pair MISS (A0r vs A0 n13); "
                                    + ("under Amendment S3's deterministic cuda settings: the screen does not launch "
                                       "(prereg section 5: no second fallback, the 1e-3 tolerance is not relaxed)"
                                       if any(u.get("device") == "cuda" for u in units.values()) else
                                       "registered fallback: rerun step 0 at THREADS=1 on every node, by hand (prereg "
                                       "section 5)")}
        miss = [u for u in (f"u_A0r_s{S}_n13.json", f"u_A0_s{S}_n13.json", f"u_A2_s{S}_n13.json") if u not in units]
        if miss or d is None:
            return {**info, "stop": f"step 0 s{S} left no {miss or 'detcheck.txt (determinism unread)'}"}
        rss = rss_by_arm(texts)
        if any(A not in rss for A in STEP0_ARMS):
            return {**info, "stop": f"step 0 s{S} has no /usr/bin/time -v report (RSS) for "
                                    f"{[A for A in STEP0_ARMS if A not in rss]}"}
        peak = max(rss[A] for A in STEP0_ARMS)
        caps[S] = cap_gb(peak)
        info["step0"][S] = {"rss_by_arm_kb": rss, "cap_gb": caps[S]}
        if mem and caps[S] > mem:
            return {**info, "stop": f"step 0 s{S} RSS {peak} kB -> cap {caps[S]} GB > the six-phase memcap {mem} GB"}
    b = dev[f"step0_s{TIME_SEED}"]
    walls = [w for _, w in reports(b.get("texts"), "A0r") if w is not None]
    mt = b.get("mtime") or {}
    t_s, t_a = mt.get(f"u_start_s{TIME_SEED}_n12.json"), mt.get(f"u_A0r_s{TIME_SEED}_n13.json")
    # "one night + its reading": A0r unit written minus start unit written; the longest A0r process wall (which also holds
    # model load and the n12 start reading) only when the mtimes are absent, which can only HOLD more often, never less.
    if t_s is not None and t_a is not None and t_a >= t_s:
        mins = (t_a - t_s) / 60.0
    elif walls:
        mins = max(walls) / 60.0
    else:
        return {**info, "stop": f"step 0 s{TIME_SEED}: no A0r wall time and no unit mtimes"}
    info.update(step0_min=round(mins, 1), memcap=caps, six_memcap_gb=mem)
    if mins > STEP0_MAX_MIN:
        return {**info, "stop": f"step 0: night + reading {mins:.1f} min > {STEP0_MAX_MIN:g}; the note says drop to nights "
                                "14-15 and A4 and say so before launch (needs a re-registered arm table)"}
    return {**info, "params": ALL_ARMS}


def verdict(six, pick, ctx):
    from brain.experiments import mix_night_screen as M
    want = (ctx.get("vars") or {}).get("lives", [])
    lives = [str(Path(ctx["repo"]) / p) for p in want if (Path(ctx["repo"]) / p).exists()]
    rss = {s: rss_by_arm(b.get("texts")) for s, b in six.items()}   # per seed, per arm peak RSS (kB): the run's memory record
    if len(lives) < len(want):        # requires_local should have refused at launch; never score a K-sane-unread screen
        return {"verdict": "ERROR", "lives_read": f"{len(lives)} of {len(want)}", "rss_kb": rss,
                "summary": f"life JSONs missing on the driver ({len(lives)} of {len(want)}): K-sane unread, not scored"}
    d = Path(ctx["res"]) / "units"
    shutil.rmtree(d, ignore_errors=True)          # only this run's units: no same-commit leftovers from a --reset
    d.mkdir(parents=True)
    for b in six.values():
        for f, u in b["units"].items():
            M.atomic_write(str(d / f), u)
    v = M.score(str(d), lives)
    st = {X: c["status"] for X, c in v["candidates"].items()}
    passed = [X for X, s in st.items() if s == "PASS"]
    word = "PASS" if passed else ("KILL" if st and all(s == "KILL" for s in st.values()) else "INCONCLUSIVE")
    return {"verdict": word, "candidates": st, "placebo": v["placebo"]["status"], "score": v,
            "lives_read": f"{len(lives)} of {len(want)}", "rss_kb": rss,
            "summary": f"candidates {st}; placebo {v['placebo']['status']}" + (f"; PASS {passed}" if passed else "")}
