# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md
# Expected-fail: none (read helper; changes no brain behaviour)
"""Pre-flight for the test2 read: 18 lives present, complete, same code/AMI/device, frozen hashes = freeze commit.
Read-only. Usage (from the wt-mixday repo root, .venv python):
  python $P/preflight.py <batch-root> [--smoke]      # batch-root = _scratch/wt-test2/_scratch/brainv2/aws
Finds life<ARM>_s<S>.json.json under every batch dir (arm B/KP/PR, seeds 42 43 44 100 101 102). Prints a table and
PROBLEMS; exit 1 on any problem. --smoke: look for smoke<ARM>_s7.json.json, days 2, report only the code_commit/hash lines.
No scoring, no prereg rule is applied here; the frozen scorers apply the registered UNDEFINED rules."""
import glob, hashlib, json, os, subprocess, sys

FREEZE = "d965bc716b2bf600f450dc6d6dd748db66993d29"
FREEZE_WT = "/home/dant123/Projects/sim/_scratch/wt-test2"
AMI_FILE = FREEZE_WT + "/_scratch/stage2/test2/ami"
ARMS = {"B": {}, "KP": {"keep_bank": "prior"}, "PR": {"replay_priority": "loss_a"}}
SEEDS = (42, 43, 44, 100, 101, 102)
smoke = "--smoke" in sys.argv
root = [a for a in sys.argv[1:] if not a.startswith("--")][0]


def freeze_sha(path):
    b = subprocess.run(["git", "-C", FREEZE_WT, "show", f"{FREEZE}:{path}"], capture_output=True).stdout
    return hashlib.sha256(b).hexdigest() if b else None


ami = open(AMI_FILE).read().split()[0] if os.path.exists(AMI_FILE) else None
problems, rows, found = [], [], {}
for arm in ARMS:
    for s in ((7,) if smoke else SEEDS):
        pat = f"smoke{arm}_s{s}.json.json" if smoke else f"life{arm}_s{s}.json.json"
        hits = sorted(glob.glob(os.path.join(root, "*", pat)))
        if not hits:
            problems.append(f"{arm} s{s}: no {pat} under {root}/*/")
            continue
        if len(hits) > 1:
            problems.append(f"{arm} s{s}: {len(hits)} copies {hits}; use the newest complete one on purpose")
        found[(arm, s)] = hits[-1]
codes, amis = set(), set()
# item 8: torch / torch_cuda / gpu_name / tf32 flags are not compared by anything else before the scorers; a difference across
# lives is reported here. (The scorers' own seed_status UNDEFINED rule, life_costs.PROV_CHECKED, still decides per arm-vs-B seed.)
provs = {}
for (arm, s), f in sorted(found.items()):
    L = json.load(open(f))
    pr = L.get("provenance", {})
    n = len(L.get("nights", []))
    want_days = 2 if smoke else 15
    bad = []
    if L.get("seed") != s: bad.append(f"seed {L.get('seed')}")
    if L.get("split") != "test2": bad.append(f"split {L.get('split')}")
    if L.get("days") != want_days: bad.append(f"days {L.get('days')}")
    if n != want_days: bad.append(f"nights recorded {n} (a section-10 kill or a partial file)")
    if L.get("device") != "cuda": bad.append(f"device {L.get('device')}")
    kw = L.get("brain_kw", {})
    for k, v in ARMS[arm].items():
        if kw.get(k) != v: bad.append(f"brain_kw {k}={kw.get(k)} want {v}")
    if arm == "B" and (set(kw) - {"sleep_seed"}): bad.append(f"B has extra brain_kw {sorted(kw)}")
    if arm == "KP" and set(kw) - {"sleep_seed", "keep_bank"}: bad.append(f"KP extra brain_kw {sorted(kw)}")
    if arm == "PR" and set(kw) - {"sleep_seed", "replay_priority"}: bad.append(f"PR extra brain_kw {sorted(kw)}")
    if pr.get("brain_tree_dirty"): bad.append("brain_tree_dirty")
    if not pr.get("brain_code_clean_env"): bad.append("brain_code_clean_env not set")
    if pr.get("cuda_precision") != "cuda fp16": bad.append(f"precision {pr.get('cuda_precision')}")
    codes.add(pr.get("code_commit")); amis.add(pr.get("aws_ami_id"))
    for k in ("torch", "torch_cuda", "gpu_name", "tf32_matmul", "tf32_cudnn"):
        provs.setdefault(k, {}).setdefault(repr(pr.get(k)), []).append(f"{arm}{s}")
        if pr.get(k) is None: bad.append(f"provenance {k} unrecorded")
    for p, h in (pr.get("frozen_sha256") or {}).items():
        e = freeze_sha(p)
        if e is None: bad.append(f"frozen file {p} not in freeze commit")
        elif e != h: bad.append(f"frozen sha differs: {p}")
    if not pr.get("frozen_sha256"): bad.append("no frozen_sha256")
    rows.append((arm, s, n, pr.get("code_commit", "")[:9], pr.get("aws_ami_id"), pr.get("gpu_name"), "; ".join(bad) or "ok"))
    problems += [f"{arm} s{s}: {b}" for b in bad]
print("arm seed nights code_commit ami gpu status")
for r in rows: print(*r, sep=" | ")
if len(codes) > 1: problems.append(f"code_commit differs across lives: {sorted(map(str, codes))}")
if len(amis) > 1: problems.append(f"aws_ami_id differs across lives: {sorted(map(str, amis))}")
if ami and amis and amis != {ami}: problems.append(f"AMI {sorted(map(str, amis))} != pinned {ami}")
fc = FREEZE
if codes and codes != {fc}: problems.append(f"code_commit {sorted(map(str, codes))} != freeze commit {fc[:9]} (T4 launches from the freeze worktree)")
for k, d in provs.items():
    if len(d) > 1:
        problems.append(f"{k} differs across lives: " + "; ".join(f"{v} -> {','.join(ls)}" for v, ls in sorted(d.items())))
print("provenance values across lives:", {k: sorted(d) for k, d in provs.items()})
# item 7: the launch tool changed twice after the freeze (prereg Amendment T4 addendum 1 and 2). It runs on the host, is not in
# FROZEN_FILES, and no life records which version launched it. Print the hash now present in the launcher tree and compare to the
# recorded ones; which lives launched under which version comes from launch_plan.log timestamps (T4 addendum 2 text).
LAUNCH = {"4d94c7ab": "freeze version (hang)", "8ba819ab": "T4 addendum 1 (< /dev/null; still hung)", "2b83d117": "T4 addendum 2 (final fix)"}
lp = FREEZE_WT + "/tools/aws_brain_batch.sh"
lh = hashlib.sha256(open(lp, "rb").read()).hexdigest() if os.path.exists(lp) else None
print(f"launcher tools/aws_brain_batch.sh sha256 now: {lh} ({LAUNCH.get((lh or '')[:8], 'UNKNOWN version')})")
print("  recorded: freeze 4d94c7ab..., addendum1 8ba819ab7851509ac1a9718a0209a79adf4d4479200364b6f63b311a7dc52b5a, addendum2 2b83d117ae4381bb4832e77aca984b077ed087ba43152a06d8b67fd85d970722")
print("  launched under: KP42-KP101 earlier versions (KP42 the freeze one); later lives addendum 2 (T4 addendum 2). Not recorded in any life JSON.")
if lh and lh[:8] not in LAUNCH: problems.append(f"launcher tool hash {lh[:8]} is not one recorded in T4 (replaced again?)")
print("PROBLEMS:" if problems else "no problems")
for p in problems: print(" -", p)
sys.exit(1 if problems else 0)
