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
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from tools.verdict import UNDEFINED, Verdict

REPO = Path(__file__).resolve().parents[2]
RES = "brain/experiments/results"
POOL = {"pool40": 23, "pool41": 15, "pool42": 15}   # GB RAM per node
W3090_GB = 23                                       # RAM budget for the workstation GPU node (kind w3090, opt-in)
FIT_SLACK = 2                                       # a job's memcap needs ram >= memcap + 2 (20 on 23, 13 on 15 GB)
REQUIRED = ("name", "prereg", "module", "cmd", "memory_gb", "seeds")
# Where per-seed data lives on the pool (spec key `placement: "@kr"`). KR night snapshots, life JSONs and trunks were shipped
# one seed per node by brain/experiments/results/2026-11-25-routing-screen/ship.sh (and routing-ceiling/ship.sh).
PLACEMENTS = {"kr": {319: "pool40", 320: "pool41", 321: "pool42"}}
OOM_RC = ("rc=137",)                                # SIGKILL: the cgroup OOM kill under tools/memcap.sh
STATUSES = ("REGISTERED", "DRAFT", "TODO", "SMOKE")   # TODO never runs; DRAFT runs only with --allow-draft
# Repo files shipped beside brain/ (the job side of the engine: memcap, the unit bundler, the smoke job). No yaml needed there.
SHIP_NEED = ("tools/memcap.sh", "tools/mem_ok.sh")    # a memcapped node refuses to ship without these
NODES_JSON = "tools/engine/nodes.json"              # optional node inventory (ram_gb read); else `free -g` on the node
KIND_IMPORTS = ("torch", "rwkv")                    # the node kind check: its python must import these (cached per run)
SHIP_TOOLS = ("tools/memcap.sh", "tools/mem_ok.sh", "tools/engine/__init__.py", "tools/engine/bundle.py",
              "tools/engine/smoke_job.py")


class StopAfterLaunch(Exception):
    """Raised by an Engine made with stop_after_launch=True once every job of a phase is launched (smoke: driver crash)."""


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
    seeds memory_gb hosts ladder{max_len,on{VERDICT: spec}}, plus
      status     REGISTERED | DRAFT | TODO | SMOKE (absent = REGISTERED). TODO never runs; DRAFT needs --allow-draft.
      placement  {seed: node or [nodes]} or "@kr" (PLACEMENTS): a job of that seed (dev or six) runs only there; seeds not
                 listed run on any fitting node. This is where per-seed data lives, not a preference.
      requires   paths relative to the node's $HOME that must exist on the node before a job of that seed launches there
                 (checked once per node and seed; a pinned seed whose data is missing stops the run with the list).
      vars       {name: value} and seed_vars {seed: {name: value}}: extra {name} tokens for cmd and requires.
      requires_local  repo-relative files that must exist on the driver (e.g. the registered prereg) before a run.
      smoke      {params, seed?, memory_gb?}: the REAL smoke stage run before dev/six (one unit of cmd on the real model, e.g.
                 one night / few steps) under /usr/bin/time -v; its peak RSS sets memcap = ceil(1.3 x peak + 1) GB for the
                 later jobs (an explicit dev_memory_gb or pick memcap still wins). Required (with {model} in cmd) unless
                 status SMOKE; a smoke failure stops the chain (ERROR-SMOKE) with its log tail in the digest.
      dev_memory_gb   memcap of the dev jobs (default memory_gb), e.g. a measure-first step 0 with a wide cap; an int, or
                      {seed: gb} (a per-node measure: 21 on the 23 GB node, 11 on the 15 GB ones).
    dev_grid values are a params string (run on dev_seed) or {params, seed} (a per-seed dev job, e.g. step 0 on every node).
    A pick may return memcap {seed: gb}: the six-seed job of that seed runs at that cap (<= memory_gb, which sizes the fit).
    `cmd` tokens: {seed} {model} {params} {out} {device} (cpu on pool nodes, cuda on w3090) {job} (the job name: a dev grid
    key or s<seed>) {commit} (12 hex of the state's commit, the one in the shipped COMMIT file) {tree} (12 hex of the brain/
    tree) + vars. {commit}/{job} give a harness that resumes its own units a stable directory across attempts, and a fresh
    one after --reset at a new commit (units never mix commits)."""

    def __init__(self, d, path, repo):
        miss = [k for k in REQUIRED if k not in d]
        if miss:
            raise ValueError(f"spec {path}: missing {miss}")
        self.__dict__.update(dict(dict(question=d["name"], model="", dev_seed=None, dev_grid={}, six_params="", hosts=[],
                                    pick=None, verdict="verdict", ladder={}, status="REGISTERED", placement={},
                                    requires=[], vars={}, seed_vars={}, requires_local=[], dev_memory_gb=None,
                                    smoke=None), **d))
        self.normalize()
        self.path, self.repo = Path(path), Path(repo)
        m = Path(self.module)
        m = m if m.is_absolute() else (self.path.parent / m if (self.path.parent / m).exists() else self.repo / m)
        sp = importlib.util.spec_from_file_location(f"engine_spec_{self.name}", m)
        self.mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(self.mod)
        self.pick_fn = getattr(self.mod, self.pick) if self.pick else None
        self.verdict_fn = getattr(self.mod, self.verdict)

    def normalize(self):
        if self.status not in STATUSES:
            raise ValueError(f"spec {self.name}: status {self.status!r} not in {STATUSES}")
        pl = self.placement or {}
        if isinstance(pl, str):
            if not pl.startswith("@") or pl[1:] not in PLACEMENTS:
                raise ValueError(f"spec {self.name}: placement {pl!r} is not one of {['@' + k for k in PLACEMENTS]}")
            pl = PLACEMENTS[pl[1:]]
        self.placement = {int(k): ([v] if isinstance(v, str) else list(v)) for k, v in pl.items()}
        self.seed_vars = {int(k): dict(v) for k, v in (self.seed_vars or {}).items()}
        self.vars, self.requires = dict(self.vars or {}), list(self.requires or [])

    def apply(self, over):
        self.__dict__.update(over)
        self.normalize()
        return self

    def mem(self, phase, seed=None, pick=None, measured=None):
        if phase == "smoke":
            return int((self.smoke or {}).get("memory_gb", self.memory_gb))
        if phase == "dev" and self.dev_memory_gb:
            d = self.dev_memory_gb
            return d.get(seed, d.get(str(seed), self.memory_gb)) if isinstance(d, dict) else d
        mc = (pick or {}).get("memcap") if phase == "six" else None
        if isinstance(mc, dict) and str(seed) in {str(k) for k in mc}:
            return min(int({str(k): v for k, v in mc.items()}[str(seed)]), self.memory_gb)
        return measured or self.memory_gb

    def smoke_seed(self):
        sm = self.smoke or {}
        return sm.get("seed", self.dev_seed if self.dev_seed is not None else self.seeds[0])

    def dev_items(self):
        """[(name, seed, params)] of the dev grid."""
        return [(n, p.get("seed", self.dev_seed), p["params"]) if isinstance(p, dict) else (n, self.dev_seed, p)
                for n, p in (self.dev_grid or {}).items()]

    def fill(self, text, seed, tree="", commit="", job=""):
        """{seed} {tree} {commit} {job} and the vars / seed_vars tokens (seed_vars win)."""
        try:
            sv = self.seed_vars.get(int(seed), {})
        except (TypeError, ValueError):
            sv = {}
        for k, v in list({**self.vars, **sv}.items()) + [("seed", seed), ("tree", tree), ("commit", commit), ("job", job)]:
            text = text.replace("{" + k + "}", str(v))
        return text

    def todo(self):
        """Why this spec cannot run yet (status TODO, or a TODO left in its cmd, grid, params, vars or requires), else None."""
        if self.status == "TODO":
            return "status TODO"
        blob = json.dumps([self.cmd, self.dev_grid, self.six_params, self.vars, self.seed_vars, self.requires, self.prereg,
                           self.requires_local], default=str)
        if "TODO" in blob:
            return "a TODO is left in cmd/dev_grid/six_params/vars/requires/prereg"
        miss = [f for f in self.requires_local if not (self.repo / f).exists()]
        return f"requires_local missing on the driver: {miss}" if miss else None

    @classmethod
    def find(cls, name, repo=REPO):
        p = Path(name)
        if not p.exists():
            p = Path(repo) / "tools/engine/specs" / f"{name}.yaml"
        return cls(yaml.safe_load(p.read_text()), p, repo)

    def next_for(self, verdict):
        on = (self.ladder or {}).get("next", {})
        return on.get(verdict) or on.get(verdict.split("-")[0])   # exact, else base ("GO-SHY" -> "GO")

    def render(self, seed, params, out, device="cpu", tree="", commit="", job=""):
        t = self.cmd
        for k, v in (("{params}", params), ("{model}", "$BRAIN_MODEL_DIR/" + self.model if self.model else ""),
                     ("{out}", out), ("{device}", device)):
            t = t.replace(k, str(v))
        return self.fill(t, seed, tree, commit, job)


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


def timev(text):
    """{peak_rss_gb, wall_s} from a GNU `time -v` report, else None."""
    r = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text)
    if not r:
        return None
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(w.group(1).split(":")))) if w else None
    return {"peak_rss_gb": int(r.group(1)) / 1048576, "wall_s": wall}


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
    def __init__(self, spec_name, nodes, repo=REPO, poll=30, state_path=None, dry=False, log=print, allow_draft=False,
                 overrides=None, stop_after_launch=False, launch_checks=True):
        self.repo, self.nodes, self.poll, self.dry, self.log = Path(repo), nodes, poll, dry, log
        self.allow_draft, self.overrides, self.stop_after_launch = allow_draft, dict(overrides or {}), stop_after_launch
        self.root = self.spec(spec_name)
        self.res = self.repo / RES
        self.state_path = Path(state_path or self.res / f"ENGINE_STATE_{self.root.name}.json")
        self.shipped, self.fails, self.req, self.kinds, self.rams = set(), {}, {}, {}, {}
        self.launch_checks, self.imports = launch_checks, KIND_IMPORTS
        nj = self.repo / NODES_JSON
        info = json.loads(nj.read_text()) if nj.exists() else {}
        self.node_info = {n["name"]: n for n in info} if isinstance(info, list) else info   # [{name, ram_gb, ...}] or {name: {}}
        ws = socket.gethostname()
        for n in nodes:                              # never run on the workstation (w3090 is the explicit opt-in exception)
            if n.kind == "ssh" and n.name.split(".")[0] == ws.split(".")[0]:
                raise SystemExit(f"REFUSED: node {n.name} is this workstation ({ws}); the engine never runs jobs here")

    def spec(self, name):
        s = Spec.find(name, self.repo)
        return s.apply(self.overrides[s.name]) if s.name in self.overrides else s

    def check_runnable(self, spec):
        why = spec.todo()
        if why:
            raise SystemExit(f"REFUSED: spec {spec.name} is not ready to run ({why}); --dry-run shows the plan")
        if spec.status == "DRAFT" and not self.allow_draft:
            raise SystemExit(f"REFUSED: spec {spec.name} is DRAFT (prereg {spec.prereg} pending pre-data review); "
                             "pass --allow-draft only for a registered dev/smoke step")
        if self.checks(spec) and (not spec.model or "{model}" not in spec.cmd):
            raise SystemExit(f"REFUSED: spec {spec.name} must name its model explicitly (`model:` set and a {{model}} token in cmd)")
        if self.checks(spec) and not (spec.smoke or {}).get("params"):
            raise SystemExit(f"REFUSED: spec {spec.name} has no `smoke: {{params: ...}}` (the real smoke stage before dev/six)")

    def checks(self, spec):
        return self.launch_checks and spec.status != "SMOKE"

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

    def ship_key(self):
        """brain/ tree + a hash of the shipped tools files: a changed bundler or memcap ships a new repo dir."""
        import hashlib
        h = hashlib.sha1()
        for f in SHIP_TOOLS:
            if (self.repo / f).exists():
                h.update(f.encode() + b"\0" + (self.repo / f).read_bytes())
        return f"{self.st['tree'][:12]}_{self.st['commit'][:8]}_{h.hexdigest()[:8]}"   # one COMMIT per shipped dir

    def ensure_shipped(self, node):
        c = self.ship_key()
        d = f"brain-lane/engine/repo_{c}"
        if (node.name, c) in self.shipped:
            return d
        miss = [f for f in SHIP_NEED if not (self.repo / f).exists()]
        if node.memcap and miss:
            raise SystemExit(f"REFUSED: {miss} missing on the driver; every ship carries memcap.sh and mem_ok.sh")
        if node.sh(f"test -f {node.home}/{d}/.shipped")[0] != 0:
            node.put(self.repo / "brain", f"{d}/brain")
            for f in SHIP_TOOLS:
                if (self.repo / f).exists():
                    node.put(self.repo / f, f"{d}/{f}")
            node.sh(f"echo {self.st['commit']} > {node.home}/{d}/COMMIT; touch {node.home}/{d}/.shipped")
        self.shipped.add((node.name, c))
        return d

    # -- jobs
    def job_script(self, node, spec, job, repo_dir, jd):
        out = f"{node.home}/{jd}/out.json"
        phase, name = job["key"].split("/", 1)
        cmd = spec.render(job["seed"], job["params"], out, "cuda" if node.kind == "w3090" else "cpu", self.st["tree"][:12],
                          self.st["commit"][:12], name)
        run = f"bash -c {shlex.quote(cmd)}"
        if phase == "smoke":                         # GNU time; ENGINE_TIME is a test shim on the local backend only
            tv = '"${ENGINE_TIME:-/usr/bin/time}"' if node.local else "/usr/bin/time"
            run = f"{tv} -v -o {node.home}/{jd}/time.txt {run}"
        if node.memcap:
            run = f"bash tools/memcap.sh {self.mem(spec, phase, job['seed'])} -- {run}"
        py = self.py_line(node)
        w = node.kind == "w3090"
        guard = "" if node.local else f'[ "$(hostname)" != {shlex.quote(socket.gethostname())} ] || exit 96\n'
        model = "${BRAIN_MODEL_DIR:-$HOME/brain-lane/models}" if w else "$HOME/brain-lane/models"
        hostflag = {"ssh": " POOL_NODE=1", "w3090": " BRAIN_ALLOW_W3090=1"}.get(node.kind, " ENGINE_LOCAL_TEST=1")
        return f"""{guard}[ -f {node.home}/{jd}/pid ] && kill -0 $(cat {node.home}/{jd}/pid) 2>/dev/null && exit 0
echo $$ > {node.home}/{jd}/pid
cd {node.home}/{repo_dir} || exit 97
export COMMIT=$(cat COMMIT) HOST=$(hostname) OMP_NUM_THREADS=$(nproc) BRAIN_MODEL_DIR={model}{" BRAIN_DEVICE=cuda" if w else ""}{hostflag}
export PYTHONPATH=$PWD:$HOME/brain-lane/pylib
{py}; export PY
{run} > {node.home}/{jd}/log 2>&1
rc=$?; [ $rc = 0 ] && [ ! -s {out} ] && rc=98
echo $rc > {node.home}/{jd}/rc
"""

    @staticmethod
    def py_line(node):
        return f"PY={shlex.quote(node.py)}" if node.py else \
            'PY=$HOME/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=$HOME/derisk-pool/sim/.venv/bin/python'

    def launch(self, node, spec, job):
        repo_dir = self.ensure_shipped(node)
        jd = f"brain-lane/engine/jobs/{self.root.name}-{self.st['tree'][:8]}/{job['key'].replace('/', '_')}_a{job['attempts']}"
        script = self.job_script(node, spec, job, repo_dir, jd)
        job.update(status="launching", node=node.name, jd=jd, attempts=job["attempts"] + 1, started=now(), t0=time.time())
        self.save()                                  # recorded BEFORE node.sh: a crash here is resumed, not duplicated
        d = f"{node.home}/{jd}"                      # a reused jd (--reset at the same tree) must not report the old rc
        rc, _ = node.sh(f"mkdir -p {d} && {{ [ -f {d}/pid ] && kill -0 $(cat {d}/pid) 2>/dev/null || rm -f {d}/rc {d}/out.json; }} && "
                        f"printf %s {shlex.quote(script)} > {node.home}/{jd}/job.sh && "
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
        ok, phase = s == "rc=0", job["key"].split("/", 1)[0]
        if phase == "smoke":
            try:
                node.get(f"{node.home}/{job['jd']}/time.txt", str(dest) + ".time")
            except Exception:
                pass
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
            for f, suf in (("log", "log"), ("out.json", "json")):   # out.json: a partial bundle, if the job wrote one
                try:
                    node.get(f"{node.home}/{job['jd']}/{f}", str(dest) + f".fail{job['attempts']}.{suf}")
                except Exception:
                    pass
            oom = s in OOM_RC
            if oom:                                  # a memcap (OOM) kill repeats at the same cap: never retried
                s = f"{s} (memcap {self.mem(spec, job['key'].split('/', 1)[0], job['seed'])} GB: OOM kill or kill -9; not retried)"
            again = job["attempts"] < 2 and not oom and phase != "smoke"      # retry once; a smoke never
            job.update(status="pending" if again else "failed", node=None, error=s)
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
        fit = {j["key"]: self.fits(spec, phase, j["seed"]) for j, _ in jobs}
        by = {n.name: n for n in self.nodes}
        for j, _ in jobs:                         # placement and data, once, before anything launches
            if j["status"] == "pending":
                self.preflight(spec, j["seed"], fit[j["key"]])
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
                mem = self.mem(spec, phase, j["seed"])
                free = [n for n in self.allowed(spec, j["seed"], fit[j["key"]]) if n.name not in busy and n.usable()
                        and self.req_ok(n, spec, j["seed"]) is True] if j["status"] == "pending" else []
                if not free:
                    continue
                n = free[0]
                avail = n.sh("awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo")[1].strip()
                if avail.isdigit() and int(avail) < mem:
                    continue                      # node busy with something else: wait
                if self.launch(n, spec, j):
                    busy.add(n.name)
                    self.log(f"[{spec.name}] {j['key']} -> {n.name} (attempt {j['attempts']})")
            if self.stop_after_launch and jobs and all(j["status"] == "running" for j, _ in jobs):
                raise StopAfterLaunch(spec.name)
            if any(j["status"] == "failed" for j, _ in jobs) and not any(j["status"] in ("running",) for j, _ in jobs):
                break
            if any(j["status"] in ("pending", "running") for j, _ in jobs):
                time.sleep(self.poll)
        bad = [j["key"] + (f" [{j['error']}]" if j.get("error") else "") for j, _ in jobs if j["status"] == "failed"]
        return {j["key"].split("/", 1)[1]: json.loads(d.read_text()) for j, d in jobs if j["status"] == "done"}, bad

    # -- placement and data
    def mem(self, spec, phase, seed):
        ss = (getattr(self, "st", None) or {}).get("specs", {}).get(spec.name) or {}
        return spec.mem(phase, seed, ss.get("pick"), (ss.get("smoke") or {}).get("memcap_gb"))

    def ram(self, node):
        """GB RAM of a node: tools/engine/nodes.json, else `free -g` total on an ssh node, else its default."""
        if node.name not in self.rams:
            info, r = self.node_info.get(node.name, {}), None
            if "ram_gb" in info:
                r = int(info["ram_gb"])
            elif node.kind == "ssh":
                o = node.sh("free -g | awk '/^Mem:/{print $2}'", timeout=60)[1].strip()
                r = int(o) if o.isdigit() else None
            self.rams[node.name] = r or node.ram_gb
        return self.rams[node.name]

    def kind_ok(self, node):
        """The node kind check: its python imports torch and rwkv (one cheap probe per node and run)."""
        if node.name not in self.kinds:
            rc, out = node.sh(f"export PYTHONPATH=$HOME/brain-lane/pylib; {self.py_line(node)}; \"$PY\" -c 'import {', '.join(self.imports)}' && echo KIND-OK", timeout=120)
            self.kinds[node.name] = rc == 0 and "KIND-OK" in out
            if not self.kinds[node.name]:
                self.log(f"{node.name}: kind check failed (python cannot import {self.imports}): {out.strip()[-300:]}")
        return self.kinds[node.name]

    def fits(self, spec, phase, seed):
        mem = self.mem(spec, phase, seed)
        f = [n for n in self.nodes if self.ram(n) >= mem + FIT_SLACK and (not spec.hosts or n.name in spec.hosts)]
        if not f:
            raise SystemExit(f"no node fits {spec.name} {phase}: needs memcap {mem} GB + {FIT_SLACK}, hosts={spec.hosts}")
        return f

    def allowed(self, spec, seed, fits):
        pl = spec.placement.get(int(seed)) if seed is not None else None
        return [n for n in fits if (pl is None or n.name in pl) and (not self.checks(spec) or self.kind_ok(n))]

    def req_ok(self, node, spec, seed):
        """True: every `requires` path exists on the node for this seed; a list: the missing paths; None: ssh failed (retry)."""
        model = [f"brain-lane/models/{spec.model}"] if self.checks(spec) and spec.model else []   # must exist on the node
        if not spec.requires and not model:
            return True
        key = (node.name, spec.name, seed)
        if key not in self.req:
            paths = [spec.fill(p, seed, self.st["tree"][:12], self.st["commit"][:12]) for p in spec.requires] + model
            rc, out = node.sh("".join(f"[ -e {node.home}/{shlex.quote(p)} ] || echo MISSING:{shlex.quote(p)}\n" for p in paths)
                              + "echo REQ-DONE")
            if rc != 0 or "REQ-DONE" not in out:
                return None
            miss = [x.split(":", 1)[1] for x in out.splitlines() if x.startswith("MISSING:")]
            self.req[key] = miss or True
        return self.req[key]

    def preflight(self, spec, seed, fits):
        pl = spec.placement.get(int(seed)) if seed is not None else None
        ok = self.allowed(spec, seed, fits)
        if not ok:
            raise SystemExit(f"[{spec.name}] seed {seed} is placed on {pl}, but no such node is in use and fits "
                             f"or passes the kind check (nodes {[n.name for n in self.nodes]}, fitting {[n.name for n in fits]}, hosts {spec.hosts})")
        res = {n.name: self.req_ok(n, spec, seed) for n in ok}
        if all(isinstance(r, list) for r in res.values()):
            raise SystemExit(f"[{spec.name}] seed {seed}: required data missing on every allowed node: "
                             + "; ".join(f"{k}: {v}" for k, v in res.items()))

    def preflight_all(self, spec):
        """Placement and `requires` of EVERY seed of the spec (dev seed at the dev cap, six seeds at the six cap), before the
        first dev job launches: a six-seed node missing its data must not be found only after hours of dev work."""
        sm = [("smoke", [spec.smoke_seed()])] if self.checks(spec) and spec.smoke else []
        for phase, seeds in sm + [("dev", sorted({s for _, s, _ in spec.dev_items()}, key=str)), ("six", spec.seeds)]:
            for s in seeds:
                self.preflight(spec, s, self.fits(spec, phase, s))

    # -- phases
    def step(self, spec):
        ss = self.st["specs"].setdefault(spec.name, {"phase": "smoke" if self.checks(spec) else "dev", "jobs": {},
                                                     "pick": None, "verdict": None})
        if ss["phase"] in ("smoke", "dev", "pick"):
            self.preflight_all(spec)
        ctx = {"repo": str(self.repo), "spec": spec.name, "res": str(self.res / spec.name), "commit": self.st["commit"],
               "tree": self.st["tree"], "vars": dict(spec.vars), "memory_gb": spec.memory_gb}
        if spec.name not in self.st["chain"]:
            self.st["chain"].append(spec.name)
            self.st["variants"].setdefault(spec.question, []).append(spec.name)
        if ss["phase"] == "smoke":
            _, bad = self.run_jobs(spec, ss, "smoke", [("smoke", spec.smoke_seed(), spec.smoke["params"])])
            dest = self.res / spec.name / "smoke" / "smoke.json"
            if bad:
                logs = sorted(dest.parent.glob("smoke.json.fail*.log"))
                tail = " / ".join(logs[-1].read_text(errors="replace").strip().splitlines()[-8:]) if logs else "no log pulled"
                return self.conclude(spec, ss, {"verdict": "ERROR-SMOKE", "summary": f"smoke failed: {bad}; log tail: {tail}"})
            m = timev(Path(str(dest) + ".time").read_text() if Path(str(dest) + ".time").exists() else "")
            if not m:
                return self.conclude(spec, ss, {"verdict": "ERROR-SMOKE", "summary": "smoke ran but left no /usr/bin/time -v report"})
            m["memcap_gb"] = math.ceil(1.3 * m["peak_rss_gb"] + 1)
            ss.update(smoke=m, phase="dev")
            self.save()
            self.log(f"[{spec.name}] smoke: peak RSS {m['peak_rss_gb']:.2f} GB, wall {m['wall_s'] or 0:.0f} s -> memcap {m['memcap_gb']} GB")
            self.fits(spec, "six", spec.seeds[0])     # no node fits the measured cap: stop now, not after dev
        if ss["phase"] == "dev":
            dev, bad = self.run_jobs(spec, ss, "dev", spec.dev_items()) if spec.dev_grid else ({}, [])
            if bad:
                return self.conclude(spec, ss, {"verdict": "ERROR", "summary": f"dev jobs failed twice: {bad}"})
            ss["phase"] = "pick"
            self.save()
        if ss["phase"] == "pick":
            dev = {k.split("/", 1)[1]: json.loads((self.res / spec.name / "dev" / f"{k.split('/', 1)[1]}.json").read_text())
                   for k, j in sorted(ss["jobs"].items()) if k.startswith("dev/") and j["status"] == "done"}   # never stale files
            ss["pick"] = spec.pick_fn(dev, ctx) if spec.pick_fn else {"params": spec.six_params}
            if ss["pick"] is None:
                self.write_pick(spec, ss, "NEGATIVE-DEV")
                return self.conclude(spec, ss, {"verdict": "NEGATIVE-DEV", "summary": "no dev candidate met the pick rule; six seeds not run"})
            if "stop" in ss["pick"]:                  # the prereg's own stop rule on a dev step (e.g. a step-0 timing bar)
                return self.conclude(spec, ss, {"verdict": "HOLD-DEV", "summary": f"pick stopped the chain: {ss['pick']['stop']}"})
            if not self.write_pick(spec, ss, "PICKED")["earned"]:
                return self.conclude(spec, ss, {"verdict": "UNDEFINED-DEV", "summary": "pick preconditions unmet; six seeds not run"})
            ss["phase"] = "six"
            self.save()
        if ss["phase"] == "six":
            ss["six_reached"] = True
            six, bad = self.run_jobs(spec, ss, "six", [(f"s{s}", s, ss["pick"]["params"]) for s in spec.seeds])
            if bad:
                return self.conclude(spec, ss, {"verdict": "ERROR", "summary": f"six-seed jobs failed twice: {bad}"})
            v = spec.verdict_fn({int(k[1:]): r for k, r in six.items()}, ss["pick"], ctx)
            return self.conclude(spec, ss, v)

    # -- preconditions: every verdict.json / pick.json carries what earned it (tools.verdict.Verdict; gate tools/gates/verdict_preconditions.py)
    def collect(self, spec, ss):
        """Parsed results of the done dev/six jobs, read from the local files (never stale ones): {"dev": {name: r}, "six": {seed: r}}."""
        out = {"dev": {}, "six": {}}
        for k, j in sorted(ss["jobs"].items()):
            ph, name = k.split("/", 1)
            f = self.res / spec.name / ph / f"{name}.json"
            if ph in out and j["status"] == "done" and f.exists():
                try:
                    out[ph][name if ph == "dev" else j["seed"]] = json.loads(f.read_text())
                except ValueError:
                    pass
        return out

    def precondition_block(self, spec, ss, stage, v=None):
        """Verdict for the stage ('pick' or 'final'): smoke rc 0 + measured peak, every scheduled dev/six job present and parsed,
        the commit/tree recorded, plus the spec module's optional preconditions(results, ctx) -> [(name, ok, note)]."""
        pv = Verdict(f"engine {spec.name} {stage}")
        res = self.collect(spec, ss)
        pv.require("commit and brain/ tree recorded", bool(self.st.get("commit") and self.st.get("tree")), True)
        sj = ss["jobs"].get("smoke/smoke")
        if self.checks(spec) and spec.smoke:
            pv.require("smoke rc 0", None if sj is None else sj["status"] == "done", True, (sj or {}).get("error") or "")
            pk = (ss.get("smoke") or {}).get("peak_rss_gb")
            pv.require("smoke measured peak RSS", None if pk is None else (isinstance(pk, (int, float)) and pk > 0), True,
                       f"peak_rss_gb={pk}")
        else:
            pv.disabled("real smoke stage", "this spec has no smoke stage (launch checks off, or status SMOKE)")
        pick = ss.get("pick")
        want = {}
        if spec.dev_grid and any(k.startswith("dev/") for k in ss["jobs"]):
            want.update({f"dev/{n}": ("dev", n) for n, _, _ in spec.dev_items()})
        if stage == "final" and ss.get("six_reached"):
            want.update({f"six/s{s}": ("six", f"s{s}") for s in spec.seeds})
        for k, (ph, n) in sorted(want.items()):
            j = ss["jobs"].get(k)
            f = self.res / spec.name / ph / f"{n}.json"
            ok = None if j is None else False
            if j is not None and j["status"] == "done" and f.exists():
                try:
                    json.loads(f.read_text())
                    ok = True
                except ValueError:
                    pass
            pv.require(f"job {k} present and parsed", ok if j is not None else False, True, (j or {}).get("error") or "")
        hook = getattr(spec.mod, "preconditions", None)
        if hook:
            ctx = {"repo": str(self.repo), "spec": spec.name, "res": str(self.res / spec.name), "commit": self.st["commit"],
                   "tree": self.st["tree"], "vars": dict(spec.vars), "memory_gb": spec.memory_gb, "stage": stage}
            try:
                got = list(hook({**res, "pick": pick, "verdict": v}, ctx) or [])
            except Exception as e:
                got = [("spec preconditions hook ran", False, f"{type(e).__name__}: {e}")]
            for g in got:
                pv.require(str(g[0]), g[1], True, str(g[2]) if len(g) > 2 else "")
        return pv

    def earn(self, spec, ss, stage, rec, word, v=None):
        """Attach the block to `rec`; if unearned and the word asserts a result, relabel it UNDEFINED (the raw word kept)."""
        d = self.precondition_block(spec, ss, stage, v).decide(go=True, verbose=False)
        earned = d["status"] != UNDEFINED
        rec.update(preconditions=d["preconditions"], preconditions_status=d["status"], undefined_reasons=d["undefined_reasons"],
                   disabled_processes=d["disabled_processes"], verdict_raw=word)
        rec["verdict"] = word if earned or word.startswith(("ERROR", "HOLD", "UNDEFINED")) else f"UNDEFINED ({word})"
        return earned

    def write_pick(self, spec, ss, word):
        """pick.json beside verdict.json. Written again at conclude: a spec's own pick/verdict may write a file of the same name."""
        rec = {"spec": spec.name, "pick": ss["pick"], "commit": self.st["commit"], "when": now()}
        rec["earned"] = self.earn(spec, ss, "pick", rec, word)
        ss["pick_written"] = True
        jwrite(self.res / spec.name / "pick.json", rec)
        return rec

    def conclude(self, spec, ss, v):
        ss.update(phase="done", verdict=v)
        v = {**v, "spec": spec.name, "question": spec.question, "prereg": spec.prereg, "pick": ss["pick"], "smoke": ss.get("smoke"),
             "commit": self.st["commit"], "jobs": {k: j["status"] for k, j in ss["jobs"].items()}, "when": now()}
        self.earn(spec, ss, "final", v, v["verdict"], v)
        if ss.get("pick_written"):
            self.write_pick(spec, ss, "PICKED" if ss.get("pick") is not None else "NEGATIVE-DEV")
        jwrite(self.res / spec.name / "verdict.json", v)
        nxt = spec.next_for(v["verdict"]) if not v["verdict"].startswith("ERROR") else None
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
            dmem = {s: spec.mem("dev", s) for _, s, _ in spec.dev_items()}
            dfits = {s: [n.name for n in self.nodes if n.ram_gb >= m + FIT_SLACK and (not spec.hosts or n.name in spec.hosts)]
                     for s, m in dmem.items()}
            self.log(f"== {spec.name} (question '{spec.question}'), prereg {spec.prereg}, status {spec.status}"
                     f"{' (NOT RUNNABLE: ' + spec.todo() + ')' if spec.todo() else ''}\n   memcap {spec.memory_gb} GB -> nodes {fits or 'NONE FIT'}; "
                     f"one job per node; ladder {spec.ladder or 'none'}")
            if spec.dev_grid and set(dmem.values()) != {spec.memory_gb}:
                self.log(f"   dev memcap {{seed: GB}} {dmem} -> nodes {dfits}")
            if spec.placement:
                self.log(f"   placement {{seed: nodes}} {spec.placement}")
            for s in sorted({spec.dev_seed, *spec.seeds} - {None}, key=str) if spec.requires else []:
                self.log(f"   requires s{s} (under $HOME): {[spec.fill(p, s, '<tree>', '<commit>') for p in spec.requires]}")
            if spec.smoke:
                s0 = spec.smoke_seed()
                self.log(f"   smoke s{s0} (memcap {spec.mem('smoke')} GB, then ceil(1.3 x peak RSS + 1) GB): "
                         f"{spec.render(s0, spec.smoke.get('params', ''), '<out>', tree='<tree>', commit='<commit>', job='smoke')}")
            for n, s, p in spec.dev_items():
                self.log(f"   dev   {n} (s{s}): {spec.render(s, p, '<out>', tree='<tree>', commit='<commit>', job=n)}")
            six_seeds = [s for s in spec.seeds if s in spec.seed_vars] if spec.seed_vars else ['<seed>']   # seed_vars: per seed
            for s in six_seeds:
                self.log(f"   pick  {spec.pick or 'none'} -> six seeds {spec.seeds}: {spec.render(s, '<params from pick>', '<out>', tree='<tree>', commit='<commit>', job=f's{s}')}")
            self.log(f"   verdict {spec.verdict}; results -> {RES}/{spec.name}/, digest -> {RES}/ENGINE_DIGEST.md")
            nxt = [v for v in (spec.ladder or {}).get("next", {}).values() if v]
            spec = self.spec(nxt[0]) if nxt and (self.repo / "tools/engine/specs" / f"{nxt[0]}.yaml").exists() else None

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
            spec = self.spec(self.st["current"])
            if self.st["specs"].get(spec.name, {}).get("phase") != "done":
                self.check_runnable(spec)
            self.step(spec)
        return self.st["finished"]


# ---------------------------------------------------------------- smoke
SMOKE = "engine_smoke"


def smoke(nodes, repo=REPO, poll=5, log=print, imports="torch,rwkv", sleep_s=20, allow_draft=False):
    """One tiny real job per node (imports only, no model), end to end: ship -> launch -> driver crash right after launch ->
    a fresh driver reattaches (status) -> pull out.json + log -> a third driver resumes a finished chain and launches nothing.
    Seed i is placed on node i (the placement path), every seed requires ~/brain-lane (the requires path). Returns the report."""
    repo, nodes = Path(repo), list(nodes)
    over = {SMOKE: {"seeds": list(range(1, len(nodes) + 1)), "placement": {i + 1: n.name for i, n in enumerate(nodes)},
                    "six_params": f"--imports {imports} --sleep {sleep_s}"}}
    res = repo / RES
    mk = lambda **k: Engine(SMOKE, nodes, repo=repo, poll=poll, log=log, overrides=over, allow_draft=allow_draft, **k)
    e1 = mk(stop_after_launch=True)
    e1.state_path.unlink(missing_ok=True)
    shutil.rmtree(res / SMOKE, ignore_errors=True)
    for n in nodes:                                   # the smoke's own job dirs only (a fresh launch count per run)
        n.sh(f"rm -rf {n.home}/brain-lane/engine/jobs/{SMOKE}-*")
    checks, t0 = {}, time.time()
    try:
        e1.run()
        checks["crash_after_launch"] = False          # never reached StopAfterLaunch
    except StopAfterLaunch:
        jobs = e1.st["specs"][SMOKE]["jobs"]
        checks["crash_after_launch"] = all(j["status"] == "running" and j["attempts"] == 1 for j in jobs.values())
        checks["placement"] = all(j["node"] == nodes[j["seed"] - 1].name for j in jobs.values())
    e2 = mk()
    v2 = e2.run()
    jobs = e2.st["specs"][SMOKE]["jobs"]
    checks["reattached_not_relaunched"] = all(j["attempts"] == 1 and j["status"] == "done" for j in jobs.values())
    checks["pulled"] = all((res / SMOKE / "six" / f"s{i}.json").exists() and (res / SMOKE / "six" / f"s{i}.json.log").exists()
                           for i in over[SMOKE]["seeds"])
    shipped = {}
    for n in nodes:
        rc, out = n.sh(f"cat {n.home}/brain-lane/engine/repo_{e2.ship_key()}/COMMIT && "
                       f"test -f {n.home}/brain-lane/engine/repo_{e2.ship_key()}/.shipped && "
                       f"test -f {n.home}/brain-lane/engine/repo_{e2.ship_key()}/tools/engine/bundle.py && echo SHIP-OK")
        shipped[n.name] = rc == 0 and "SHIP-OK" in out and e2.st["commit"] in out
    checks["shipped"] = all(shipped.values())
    e3, launched = mk(), []
    real = e3.launch
    e3.launch = lambda *a: launched.append(a) or real(*a)
    v3 = e3.run()
    checks["resume_finished_is_noop"] = v3 == v2 and not launched
    checks["verdict_ok"] = v2 == "SMOKE-OK"
    vj = json.loads((res / SMOKE / "verdict.json").read_text())
    rep = {"ok": all(checks.values()), "checks": checks, "shipped": shipped, "verdict": v2, "per_seed": vj.get("per_seed"),
           "nodes": [n.name for n in nodes], "commit": e2.st["commit"], "tree": e2.st["tree"], "wall_s": round(time.time() - t0, 1),
           "when": now(), "driver": socket.gethostname()}
    jwrite(res / SMOKE / "smoke_report.json", rep)
    log(json.dumps({k: rep[k] for k in ("ok", "checks", "verdict", "wall_s")}, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.engine")
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--smoke", action="store_true", help="one tiny real job per node (imports torch, rwkv; no model): "
                    "ship/launch/status/pull/resume end to end; writes results/engine_smoke/smoke_report.json")
    ap.add_argument("--smoke-imports", default="torch,rwkv")
    ap.add_argument("--smoke-sleep", type=float, default=20)
    ap.add_argument("--allow-draft", action="store_true", help="run a spec whose status is DRAFT (never TODO)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="forget the state file and start the chain again")
    ap.add_argument("--backend", default="ssh", choices=["ssh", "local"], help="local = tests only")
    ap.add_argument("--local-root")
    ap.add_argument("--nodes", help="comma list, default pool40,pool41,pool42")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--allow-w3090", action="store_true", help="also use this workstation's GPU, only while idle (<2 GB, no compute apps)")
    a = ap.parse_args(argv)
    nodes = make_nodes(a.backend, a.nodes.split(",") if a.nodes else None, a.local_root, a.allow_w3090)
    if a.smoke:
        rep = smoke(nodes, poll=min(a.poll, 5), imports=a.smoke_imports, sleep_s=a.smoke_sleep)
        print("smoke:", "OK" if rep["ok"] else "FAILED")
        raise SystemExit(0 if rep["ok"] else 1)
    if not a.spec:
        ap.error("spec is required (or --smoke)")
    print("final verdict:", Engine(a.spec, nodes, poll=a.poll, dry=a.dry_run, allow_draft=a.allow_draft).run(a.reset))
