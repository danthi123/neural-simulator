"""audit_dependencies — diff every third-party import in the tree against the requirements files.

WHY (2026-07-16): `scipy` was missing from requirements.txt AND from the box. That did not fail loudly -- it made
`get_sparse_module()` raise ModuleNotFoundError, which `sim/bridge.py:45`'s `except ImportError:` swallowed, falling
back to `import cupy as cp`. Net effect: **SIM_BACKEND=numpy silently ran on the GPU** and the documented CPU path
was dead. The same audit then found `pytest` (imported by 323 test files) in NO manifest -- so a fresh clone
following the documented `pip install -r requirements.txt` could not run a single test, and every "CI passes" claim
on such a box was vacuous.

A dependency list maintained by hand rots exactly like a comment that says "this is inert": nothing checks it.
This makes the check mechanical and repeatable.

    python tools/audit_dependencies.py              # report
    python tools/audit_dependencies.py --strict     # exit 1 if a REQUIRED (non-optional) import is undeclared

Honest limits: it reports IMPORTS, not whether an import is guarded (`try/except ImportError`) or lazy. A guarded
import is genuinely optional -- so the output is a REVIEW list, not a verdict. Verify before adding: on this repo
`psutil`/`hdf5plugin` are guarded in neural-simulator.py (the GUI runs without them), and `torch` is a hard import
in sim/tiny_transformer.py but sim/__init__.py never pulls it in, so the CORE sim does not need it.
"""
import argparse, ast, importlib.util, os, re, sys
from collections import defaultdict

# import-name -> pip distribution name, where they differ
ALIAS = {
    "cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
    "OpenGL": "PyOpenGL", "cupy": "cupy-cuda12x", "cupyx": "cupy-cuda12x", "skimage": "scikit-image",
    "dotenv": "python-dotenv", "llama_index": "llama-index-core", "mpl_toolkits": "matplotlib",
    "sentence_transformers": "sentence-transformers", "serial": "pyserial", "cuda": "cuda-python",
}
# Not pip packages: sibling modules imported via sys.path games (research scripts importing each other), and
# external projects that are deliberately not pip deps.
NOT_PIP = {"soma"}
LOCAL_PKGS = {"sim", "experiment", "viz", "ui", "research", "tools", "webapp", "tests"}
SCAN = ["sim", "experiment", "viz", "ui", "research", "tools", "webapp", "tests", "."]
SKIP_DIRS = (".venv", ".venv-rag", "node_modules", ".git", "__pycache__", "rag_index", "bridges")


def _sibling_modules():
    """Every .py basename in the tree -- a bare `import foo` that matches one is a SIBLING module (sys.path trick),
    not a missing pip package. Without this the report is full of false positives like `step1_onoff_opponent`."""
    names = set()
    for root in SCAN:
        for dp, dn, fn in os.walk(root):
            if any(x in dp for x in SKIP_DIRS):
                continue
            for f in fn:
                if f.endswith(".py"):
                    names.add(f[:-3])
    return names


def collect():
    std = set(sys.stdlib_module_names)
    siblings = _sibling_modules()
    hits = defaultdict(set)
    for root in SCAN:
        for dp, dn, fn in os.walk(root):
            if any(x in dp for x in SKIP_DIRS):
                continue
            if root == "." and dp != ".":
                continue
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(p, encoding="utf-8", errors="replace").read())
                except Exception:
                    continue
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        mods = [a.name.split(".")[0] for a in n.names]
                    elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                        mods = [n.module.split(".")[0]]
                    else:
                        continue
                    for m in mods:
                        if m in std or m in LOCAL_PKGS or m in siblings or m.startswith("_"):
                            continue
                        hits[m].add(p)
    return hits


def declared():
    names = set()
    for rf in ("requirements.txt", "requirements-dev.txt"):
        if not os.path.exists(rf):
            continue
        for line in open(rf, encoding="utf-8"):
            t = line.strip()
            if not t:
                continue
            # a commented-out dep still counts as DOCUMENTED (the optional-extras block lists them deliberately)
            t = t.lstrip("#").strip()
            m = re.match(r"^([A-Za-z0-9_.\-]+)", t)
            if m:
                names.add(m.group(1).lower())
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if an undeclared import is found")
    a = ap.parse_args()

    hits, req = collect(), declared()
    rows, undeclared = [], []
    for m in sorted(hits):
        pkg = ALIAS.get(m, m)
        ok = pkg.lower() in req or m.lower() in req or m in NOT_PIP
        inst = importlib.util.find_spec(m) is not None if _safe(m) else False
        rows.append((m, pkg, ok, inst, len(hits[m])))
        if not ok:
            undeclared.append((m, pkg, inst, sorted(hits[m])[:2]))

    print(f"{'import':<24}{'pip name':<26}{'declared':<10}{'installed':<11}{'files':>6}")
    print("-" * 78)
    for m, pkg, ok, inst, n in rows:
        print(f"{m:<24}{pkg:<26}{'yes' if ok else 'NO':<10}{'yes' if inst else 'no':<11}{n:>6}")

    if undeclared:
        print("\nUNDECLARED (review each: a GUARDED/lazy import is legitimately optional -- declare it as an")
        print("optional extra rather than a hard dep, but do NOT leave it undocumented):")
        for m, pkg, inst, ex in undeclared:
            print(f"  {m:<22} pip:{pkg:<24} installed={'yes' if inst else 'NO':<4} e.g. {ex[0]}")
    else:
        print("\nAll third-party imports are declared.")
    if a.strict and undeclared:
        raise SystemExit(1)


def _safe(m):
    try:
        importlib.util.find_spec(m)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
