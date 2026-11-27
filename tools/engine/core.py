"""Experiment engine core (see tools/engine/README in the module docstring of specs/*.yaml for the spec format).

Why a small runner and not Snakemake / task-spooler: the work is a *data-dependent* DAG (the six-seed jobs and the next spec only
exist after a python pick/verdict function has read the earlier results), the nodes are three ssh boxes with different RAM,
and the repo already has the node conventions (memcap, ~/brain-lane layout). Snakemake would need dynamic DAG hacks plus
a cluster profile; task-spooler is per-host and knows neither RAM fit nor chaining. We reuse what exists: tools/memcap.sh on the
node, rsync as in cascade/ship.sh, detached nohup jobs as in run_node.sh. This file only adds spec, state, chaining and digest.
Never runs a job on the workstation (the Local backend exists for tests only and needs --backend local), except the opt-in
kind 'w3090' node (--allow-w3090): the workstation GPU, only while idle, never preempted.
"""
import argparse
import datetime
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RES = "brain/experiments/results"
POOL = {"pool40": 23, "pool41": 15, "pool42": 15}   # GB RAM per node
W3090_GB = 23                                       # RAM budget for the workstation GPU node (kind w3090, opt-in)
FIT_SLACK = 2                                       # a job's memcap needs ram >= memcap + 2 (20 on 23, 13 on 15 GB)
REQUIRED = ("name", "prereg", "module", "cmd", "memory_gb", "seeds")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def jwrite(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------- spec
class Spec:
    """One registered experiment. YAML keys: name question prereg module pick verdict cmd model dev_seed dev_grid six_params
    seeds memory_gb hosts ladder{max_len,on{VERDICT: spec}}. `cmd` tokens: {seed} {model} {params} {out} {device} (cpu on pool nodes, cuda on w3090)."""

    def __init__(self, d, path, repo):
        miss = [k for k in REQUIRED if k not in d]
        if miss:
            raise ValueError(f"spec {path}: missing {miss}")
        self.__dict__.update(dict(dict(question=d["name"], model="", dev_seed=None, dev_grid={}, six_params="", hosts=[],
                                    pick=None, verdict="verdict", ladder={}), **d))
        self.path, self.repo = Path(path), Path(repo)
        m = Path(self.module)
        m = m if m.is_absolute() else (self.path.parent / m if (self.path.parent / m).exists() else self.repo / m)
        sp = importlib.util.spec_from_file_location(f"engine_spec_{self.name}", m)
        self.mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(self.mod)
        self.pick_fn = getattr(self.mod, self.pick) if self.pick else None
        self.verdict_fn = getattr(self.mod, self.verdict)

    @classmethod
    def find(cls, name, repo=REPO):
        p = Path(name)
        if not p.exists():
            p = Path(repo) / "tools/engine/specs" / f"{name}.yaml"
        return cls(yaml.safe_load(p.read_text()), p, repo)

    def next_for(self, verdict):
        on = (self.ladder or {}).get("next", {})
        return on.get(verdict) or on.get(verdict.split("-")[0])   # exact, else base ("GO-SHY" -> "GO")

    def render(self, seed, params, out, device="cpu"):
        t = self.cmd
        for k, v in (("{seed}", seed), ("{model}", "$BRAIN_MODEL_DIR/" + self.model if self.model else ""),
                     ("{params}", params), ("{out}", out), ("{device}", device)):
            t = t.replace(k, str(v))
        return t


# ---------------------------------------------------------------- nodes
class Node:
    """A pool node reached by ssh (kind ssh), a directory acting as one (kind local, tests only), or this workstation's GPU
    (kind w3090, opt-in: commands run locally with the real HOME)."""

    def __init__(self, name, ram_gb, root=None, py=None, memcap=True, kind=None):
        self.name, self.ram_gb, self.root, self.py, self.memcap = name, ram_gb, root, py, memcap
        self.kind = kind or ("local" if root else "ssh")
        self._home = str(root) if root else (str(Path.home()) if self.kind == "w3090" else None)

    @property
    def local(self):
        return self.kind != "ssh"

    def usable(self):
        return gpu_idle() if self.kind == "w3090" else True

    @property
    def home(self):
        if self._home is None:
            self._home = self.sh("echo $HOME")[1].strip()
        return self._home

    def _ssh(self):
        return "ssh -o IdentitiesOnly=yes -o BatchMode=yes"

    def sh(self, script, timeout=900):
        if self.local:
            cmd, env = ["bash", "-s"], ({**os.environ, "HOME": str(self.root)} if self.root else dict(os.environ))
        else:
            cmd, env = shlex.split(self._ssh()) + [self.name, "bash -s"], {**os.environ, "SSH_AUTH_SOCK": ""}
        try:
            r = subprocess.run(cmd, input=script, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 255, "timeout"
        return r.returncode, r.stdout + r.stderr

    def put(self, src, dest):
        src, dest = Path(src), f"{self.home}/{dest}"
        if self.local:
            if src.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(src, dest, ignore=lambda d, n: [x for x in n if x == "__pycache__" or (
                    x == "results" and d.endswith("experiments"))])
            else:
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            return
        self.sh(f"mkdir -p $(dirname {shlex.quote(dest)})")
        args = ["rsync", "-a", "-e", self._ssh()]
        if src.is_dir():
            args += ["--delete", "--exclude", "experiments/results", "--exclude", "__pycache__", f"{src}/", f"{self.name}:{dest}/"]
        else:
            args += [str(src), f"{self.name}:{dest}"]
        subprocess.run(args, check=True, env={**os.environ, "SSH_AUTH_SOCK": ""})

    def get(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if self.local:
            shutil.copy2(remote, local)
        else:
            subprocess.run(["rsync", "-a", "-e", self._ssh(), f"{self.name}:{remote}", str(local)], check=True,
                           env={**os.environ, "SSH_AUTH_SOCK": ""})


def gpu_idle():
    """True only if nvidia-smi shows < 2 GB used and no compute apps (never preempt anyone)."""
    try:
        q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, apps = q("--query-gpu=memory.used"), q("--query-compute-apps=pid")
        if used.returncode or apps.returncode:
            return False
        return max(int(x) for x in used.stdout.split()) < 2048 and not apps.stdout.strip()
    except Exception:
        return False


def aws_node(*a, **k):
    raise NotImplementedError("TODO AWS lane: busy until test2 ends; wrap tools/aws_brain_batch.sh as a Node with the same "
                              "sh/put/get contract (launch, rsync, pull) and register it under a kind 'aws' in make_nodes.")


def make_nodes(backend, names=None, local_root=None, allow_w3090=False):
    names = names or list(POOL)
    if backend == "local":
        nodes = [Node(n, POOL.get(n, 23), root=Path(local_root) / n, py=sys.executable, memcap=False) for n in names]
    else:
        nodes = [Node(n, POOL[n]) for n in names]
    if allow_w3090:                                   # last, so pool nodes are preferred
        nodes.append(Node("w3090", W3090_GB, py=sys.executable, kind="w3090"))
    return nodes


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, spec_name, nodes, repo=REPO, poll=30, state_path=None, dry=False, log=print):
        self.repo, self.nodes, self.poll, self.dry, self.log = Path(repo), nodes, poll, dry, log
        self.root = Spec.find(spec_name, repo)
        self.res = self.repo / RES
        self.state_path = Path(state_path or self.res / f"ENGINE_STATE_{self.root.name}.json")
        self.shipped, self.fails = set(), {}
        ws = socket.gethostname()
        for n in nodes:                              # never run on the workstation (w3090 is the explicit opt-in exception)
            if n.kind == "ssh" and n.name.split(".")[0] == ws.split(".")[0]:
                raise SystemExit(f"REFUSED: node {n.name} is this workstation ({ws}); the engine never runs jobs here")

    # -- state
    def load(self, reset=False):
        if reset and self.state_path.exists():
            self.state_path.unlink()
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"root": self.root.name, "commit": None, "tree": None, "current": self.root.name, "chain": [], "variants": {},
                "specs": {}, "finished": None}

    def save(self):
        jwrite(self.state_path, self.st)

    # -- git: ship only a clean brain/; state is keyed on the brain/ tree (results/ excluded), HEAD kept as provenance
    def commit(self):
        git = lambda *a: subprocess.run(["git", "-C", str(self.repo), *a], capture_output=True, text=True).stdout.strip()
        dirty = git("status", "--porcelain", "--", "brain", f":(exclude){RES}")
        if dirty:
            raise SystemExit("REFUSED: brain/ is dirty (commit first; results/ excluded):\n" + dirty[:600])
        idx = tempfile.mktemp(prefix="engine_index_")
        env = {**os.environ, "GIT_INDEX_FILE": idx}
        g = lambda *a: subprocess.run(["git", "-C", str(self.repo), *a], capture_output=True, text=True, env=env).stdout.strip()
        try:
            g("read-tree", "--prefix=brain/", "HEAD:brain")
            g("rm", "--cached", "-r", "-q", "--ignore-unmatch", RES)
            tree = g("write-tree")
        finally:
            Path(idx).unlink(missing_ok=True)
        return git("rev-parse", "HEAD"), tree

    def ensure_shipped(self, node):
        c = self.st["tree"][:12]
        d = f"brain-lane/engine/repo_{c}"
        if (node.name, c) in self.shipped:
            return d
        if node.sh(f"test -f {node.home}/{d}/.shipped")[0] != 0:
            node.put(self.repo / "brain", f"{d}/brain")
            for f in ("tools/memcap.sh", "tools/mem_ok.sh"):
                if (self.repo / f).exists():
                    node.put(self.repo / f, f"{d}/{f}")
            node.sh(f"echo {self.st['commit']} > {node.home}/{d}/COMMIT; touch {node.home}/{d}/.shipped")
        self.shipped.add((node.name, c))
        return d

    # -- jobs
    def job_script(self, node, spec, job, repo_dir, jd):
        out = f"{node.home}/{jd}/out.json"
        cmd = spec.render(job["seed"], job["params"], out, "cuda" if node.kind == "w3090" else "cpu")
        run = f"bash -c {shlex.quote(cmd)}"
        if node.memcap:
            run = f"bash tools/memcap.sh {spec.memory_gb} -- {run}"
        py = f"PY={shlex.quote(node.py)}" if node.py else \
            'PY=$HOME/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=$HOME/derisk-pool/sim/.venv/bin/python'
        w = node.kind == "w3090"
        guard = "" if node.local else f'[ "$(hostname)" != {shlex.quote(socket.gethostname())} ] || exit 96\n'
        model = "${BRAIN_MODEL_DIR:-$HOME/brain-lane/models}" if w else "$HOME/brain-lane/models"
        return f"""{guard}[ -f {node.home}/{jd}/pid ] && kill -0 $(cat {node.home}/{jd}/pid) 2>/dev/null && exit 0
echo $$ > {node.home}/{jd}/pid
cd {node.home}/{repo_dir} || exit 97
export COMMIT=$(cat COMMIT) HOST=$(hostname) OMP_NUM_THREADS=$(nproc) BRAIN_MODEL_DIR={model}{" BRAIN_DEVICE=cuda" if w else ""}
export PYTHONPATH=$PWD:$HOME/brain-lane/pylib
{py}; export PY
{run} > {node.home}/{jd}/log 2>&1
rc=$?; [ $rc = 0 ] && [ ! -s {out} ] && rc=98
echo $rc > {node.home}/{jd}/rc
"""

    def launch(self, node, spec, job):
        repo_dir = self.ensure_shipped(node)
        jd = f"brain-lane/engine/jobs/{self.root.name}-{self.st['tree'][:8]}/{job['key'].replace('/', '_')}_a{job['attempts']}"
        script = self.job_script(node, spec, job, repo_dir, jd)
        job.update(status="launching", node=node.name, jd=jd, attempts=job["attempts"] + 1, started=now(), t0=time.time())
        self.save()                                  # recorded BEFORE node.sh: a crash here is resumed, not duplicated
        rc, _ = node.sh(f"mkdir -p {node.home}/{jd} && printf %s {shlex.quote(script)} > {node.home}/{jd}/job.sh && "
                        f"(nohup setsid bash {node.home}/{jd}/job.sh >/dev/null 2>&1 </dev/null &)")
        if rc != 0:                                  # same jd is reused next time; job_script exits if that pid is alive
            job.update(status="pending", node=None, attempts=job["attempts"] - 1)
            self.save()
            return False
        job["status"] = "running"
        self.save()
        return True

    def status(self, node, job):
        d = f"{node.home}/{job['jd']}"
        rc, out = node.sh(f"if [ -f {d}/rc ]; then echo rc=$(cat {d}/rc); elif kill -0 $(cat {d}/pid) 2>/dev/null; "
                          f"then echo running; elif [ -f {d}/rc ]; then echo rc=$(cat {d}/rc); "   # rc landed between the two checks
                          f"elif [ ! -s {d}/pid ] && [ {time.time() - job.get('t0', 0):.0f} -lt 300 ]; then echo running; "
                          f"else echo dead; fi")
        if rc == 0 and out.strip():
            self.fails.pop(job["jd"], None)
            return out.strip().splitlines()[-1]
        n = self.fails[job["jd"]] = self.fails.get(job["jd"], 0) + 1   # ssh blip: keep waiting, but not forever
        if n >= 20:
            self.log(f"{job['key']} on {node.name}: {n} consecutive ssh failures, marking dead")
            return "dead"
        return "running"

    def finish_job(self, node, spec, job, s, dest):
        ok = s == "rc=0"
        if ok:
            for i in range(3):                       # a pull failure is transient: retry, never consume the attempt
                try:
                    node.get(f"{node.home}/{job['jd']}/out.json", dest)
                    break
                except Exception as e:
                    self.log(f"{job['key']}: pull try {i + 1}/3 failed: {e}")
                    time.sleep(min(self.poll, 2))
            else:
                return                               # stays running; next poll sees rc=0 and pulls again
            try:
                node.get(f"{node.home}/{job['jd']}/log", str(dest) + ".log")
            except Exception:
                pass
            try:
                json.loads(Path(dest).read_text())
            except Exception as e:
                ok, s = False, f"parse failed: {e}"
        if ok:
            job.update(status="done", finished=now())
        else:
            try:
                node.get(f"{node.home}/{job['jd']}/log", str(dest) + f".fail{job['attempts']}.log")
            except Exception:
                pass
            job.update(status="pending" if job["attempts"] < 2 else "failed", node=None, error=s)   # retry once
        self.save()

    def run_jobs(self, spec, ss, phase, items):
        outdir = self.res / spec.name / phase
        jobs = []
        for name, seed, params in items:
            j = ss["jobs"].setdefault(f"{phase}/{name}", {"key": f"{phase}/{name}", "seed": seed, "params": params,
                                                           "status": "pending", "attempts": 0, "node": None, "jd": None})
            if j["status"] == "done" and not (outdir / f"{name}.json").exists():
                j["status"] = "pending"          # result file lost locally: redo
            jobs.append((j, outdir / f"{name}.json"))
        fits = [n for n in self.nodes if n.ram_gb >= spec.memory_gb + FIT_SLACK and (not spec.hosts or n.name in spec.hosts)]
        if not fits:
            raise SystemExit(f"no node fits {spec.name}: needs memcap {spec.memory_gb} GB + {FIT_SLACK}, hosts={spec.hosts}")
        by = {n.name: n for n in self.nodes}
        for j, _ in jobs:                         # crash/reboot: reattach to running/launching jobs, else rerun
            if j["status"] in ("running", "launching"):
                if j["node"] not in by or (j["status"] == "launching" and self.status(by[j["node"]], j) == "dead"):
                    j.update(status="pending", attempts=j["attempts"] - (j["status"] == "launching"))
                else:
                    j["status"] = "running"
        while any(j["status"] in ("pending", "running") for j, _ in jobs):
            for j, dest in jobs:
                if j["status"] == "running":
                    s = self.status(by[j["node"]], j)
                    if s != "running":
                        self.log(f"[{spec.name}] {j['key']} on {j['node']}: {s}")
                        self.finish_job(by[j["node"]], spec, j, s, dest)
            busy = {j["node"] for j, _ in jobs if j["status"] == "running"}
            for j, _ in jobs:
                free = [n for n in fits if n.name not in busy and n.usable()] if j["status"] == "pending" else []
                if not free:
                    continue
                n = free[0]
                avail = n.sh("awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo")[1].strip()
                if avail.isdigit() and int(avail) < spec.memory_gb:
                    continue                      # node busy with something else: wait
                if self.launch(n, spec, j):
                    busy.add(n.name)
                    self.log(f"[{spec.name}] {j['key']} -> {n.name} (attempt {j['attempts']})")
            if any(j["status"] == "failed" for j, _ in jobs) and not any(j["status"] in ("running",) for j, _ in jobs):
                break
            if any(j["status"] in ("pending", "running") for j, _ in jobs):
                time.sleep(self.poll)
        bad = [j["key"] for j, _ in jobs if j["status"] == "failed"]
        return {j["key"].split("/", 1)[1]: json.loads(d.read_text()) for j, d in jobs if j["status"] == "done"}, bad

    # -- phases
    def step(self, spec):
        ss = self.st["specs"].setdefault(spec.name, {"phase": "dev", "jobs": {}, "pick": None, "verdict": None})
        ctx = {"repo": str(self.repo), "spec": spec.name, "res": str(self.res / spec.name)}
        if spec.name not in self.st["chain"]:
            self.st["chain"].append(spec.name)
            self.st["variants"].setdefault(spec.question, []).append(spec.name)
        if ss["phase"] == "dev":
            dev, bad = self.run_jobs(spec, ss, "dev", [(n, spec.dev_seed, p) for n, p in spec.dev_grid.items()]) \
                if spec.dev_grid else ({}, [])
            if bad:
                return self.conclude(spec, ss, {"verdict": "ERROR", "summary": f"dev jobs failed twice: {bad}"})
            ss["phase"] = "pick"
            self.save()
        if ss["phase"] == "pick":
            dev = {k.split("/", 1)[1]: json.loads((self.res / spec.name / "dev" / f"{k.split('/', 1)[1]}.json").read_text())
                   for k, j in sorted(ss["jobs"].items()) if k.startswith("dev/") and j["status"] == "done"}   # never stale files
            ss["pick"] = spec.pick_fn(dev, ctx) if spec.pick_fn else {"params": spec.six_params}
            if ss["pick"] is None:
                return self.conclude(spec, ss, {"verdict": "NEGATIVE-DEV", "summary": "no dev candidate met the pick rule; six seeds not run"})
            ss["phase"] = "six"
            self.save()
        if ss["phase"] == "six":
            six, bad = self.run_jobs(spec, ss, "six", [(f"s{s}", s, ss["pick"]["params"]) for s in spec.seeds])
            if bad:
                return self.conclude(spec, ss, {"verdict": "ERROR", "summary": f"six-seed jobs failed twice: {bad}"})
            v = spec.verdict_fn({int(k[1:]): r for k, r in six.items()}, ss["pick"], ctx)
            return self.conclude(spec, ss, v)

    def conclude(self, spec, ss, v):
        ss.update(phase="done", verdict=v)
        v = {**v, "spec": spec.name, "question": spec.question, "prereg": spec.prereg, "pick": ss["pick"],
             "commit": self.st["commit"], "jobs": {k: j["status"] for k, j in ss["jobs"].items()}, "when": now()}
        jwrite(self.res / spec.name / "verdict.json", v)
        nxt = spec.next_for(v["verdict"]) if v["verdict"] != "ERROR" else None
        n = len(self.st["variants"][spec.question])
        mx = (spec.ladder or {}).get("max_len", 1)
        stop = None
        if nxt and n >= mx:
            stop, nxt = f"ladder limit {mx} reached", None
        elif nxt and nxt in self.st["chain"]:
            stop, nxt = f"{nxt} already tried", None
        line = (f"- {now()} | question '{spec.question}' | variant {n} of max {mx} ({spec.name}) | {v['verdict']} | "
                f"{v.get('summary', '')} | commit {v['commit'][:8]} | next: {nxt or 'none' + (' (' + stop + ')' if stop else '')}\n")
        dg = self.res / "ENGINE_DIGEST.md"
        if not dg.exists():
            dg.write_text("# Engine digest (one line per finished spec; written by tools/engine, no AI)\n\n")
        with dg.open("a") as f:
            f.write(line)
        self.log(line.strip())
        self.st["current"], self.st["finished"] = nxt, (None if nxt else v["verdict"])
        self.save()

    # -- entry points
    def plan(self):
        spec, seen = self.root, []
        while spec and spec.name not in seen:
            seen.append(spec.name)
            fits = [n.name for n in self.nodes if n.ram_gb >= spec.memory_gb + FIT_SLACK and (not spec.hosts or n.name in spec.hosts)]
            self.log(f"== {spec.name} (question '{spec.question}'), prereg {spec.prereg}\n   memcap {spec.memory_gb} GB -> nodes {fits or 'NONE FIT'}; "
                     f"one job per node; ladder {spec.ladder or 'none'}")
            for n, p in spec.dev_grid.items():
                self.log(f"   dev   {n}: {spec.render(spec.dev_seed, p, '<out>')}")
            self.log(f"   pick  {spec.pick or 'none'} -> six seeds {spec.seeds}: {spec.render('<seed>', '<params from pick>', '<out>')}")
            self.log(f"   verdict {spec.verdict}; results -> {RES}/{spec.name}/, digest -> {RES}/ENGINE_DIGEST.md")
            nxt = [v for v in (spec.ladder or {}).get("next", {}).values() if v]
            spec = Spec.find(nxt[0], self.repo) if nxt and (self.repo / "tools/engine/specs" / f"{nxt[0]}.yaml").exists() else None

    def run(self, reset=False):
        if self.dry:
            return self.plan()
        self.st = self.load(reset)
        head, tree = self.commit()
        if self.st.get("tree") and self.st["tree"] != tree:
            raise SystemExit(f"state was made at brain/ tree {self.st['tree'][:8]} (commit {self.st['commit'][:8]}), now {tree[:8]}: "
                             "use --reset or check out that brain/")
        if not self.st.get("tree"):
            self.st["commit"], self.st["tree"] = head, tree
        while self.st["current"]:
            self.step(Spec.find(self.st["current"], self.repo))
        return self.st["finished"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.engine")
    ap.add_argument("spec")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="forget the state file and start the chain again")
    ap.add_argument("--backend", default="ssh", choices=["ssh", "local"], help="local = tests only")
    ap.add_argument("--local-root")
    ap.add_argument("--nodes", help="comma list, default pool40,pool41,pool42")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--allow-w3090", action="store_true", help="also use this workstation's GPU, only while idle (<2 GB, no compute apps)")
    a = ap.parse_args(argv)
    nodes = make_nodes(a.backend, a.nodes.split(",") if a.nodes else None, a.local_root, a.allow_w3090)
    print("final verdict:", Engine(a.spec, nodes, poll=a.poll, dry=a.dry_run).run(a.reset))
