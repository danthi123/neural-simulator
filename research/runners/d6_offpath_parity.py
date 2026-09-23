"""D6 OFF-PATH PARITY vs a REFERENCE REVISION (default origin/main): is the D6 branch, with every BRAIN_D6_* flag UNSET,
byte-identical to the code it modifies? (docs/TERMS.md `byte-identical`: shown by an EXACT compare / hash in data, not
by reading the code.)

WHY (2026-09-23 fix round). The branch's `test_off_is_byte_identical` compared `BRAIN_D6_HEBBIAN_STORE` unset vs '0'
on the SAME branch -- both take the identical code path, so it could not fail and said nothing about the pre-change
store path. This tool runs the SAME probe in two source trees -- the branch working tree and a `git archive` of the
reference revision -- in separate processes, and compares their outputs exactly.

Modes (each prints/writes one JSON record):
  store  small OneBrainComposer (D=128, 12-word vocab, seed 42): hear 3 facts (the 2nd inside the D6
         `conversation_write` context when that exists -- a no-op when the flags are unset), then record the sha256 of
         store_conns (exact complex values), the kb, and 3 recall answers. ~10 s.
  chat   one FULL tiny-demo brain through the real /api/brain-chat handler (numpy, stub renderer, no LLM), the D6
         probe's 5 turns with teach "the wolf hunts the deer", recording each turn's full response body (volatile
         timing keys dropped) + the final store_conns sha256. ~15 min, ~6 GB -> run under tools/memcap.sh.
Driver:
  --vs-ref <rev> --mode store|chat --out <json>   archive <rev> into a scratch dir, run the mode in BOTH trees, compare.

Run from the repo root:  .venv/bin/python -m research.runners.d6_offpath_parity --vs-ref origin/main --mode store \
    --out research/findings/raw/_d6_learn_through_use/offpath_parity_store_vs_main.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

VOCAB = sorted({"dog", "chase", "cat", "eat", "fish", "wolf", "hunt", "deer", "fox", "berry", "bird", "worm"})
TURNS = [("teach", "the wolf hunts the deer"), ("d1", "what does the cat eat"), ("d2", "what does the dog chase"),
         ("probe", "what does the wolf hunt"), ("xprobe", "what does the fox eat")]
_VOLATILE = ("ms", "elapsed", "latency", "time", "timing", "wall", "_t", "duration")
# the reference tree = every tracked CODE path of <rev>; the large data dirs are symlinked from this checkout (read-only
# assets the brain build loads -- the branch only ADDS files under research/findings, never edits a loaded one)
DATA_DIRS = ["research/findings", "research/datasets", "research/measurements", "research/packets", "raw", "references",
             "docs", "data"]


def _assert_flags_unset():
    bad = sorted(k for k in os.environ if k.startswith("BRAIN_D6_"))
    if bad:
        raise SystemExit("d6_offpath_parity: BRAIN_D6_* must be UNSET for an off-path parity run, found %s" % bad)


def _sha_store(sc):
    h = hashlib.sha256()
    for p, q, w in sc:
        w = complex(w)
        h.update(("%d,%d,%r,%r;" % (int(p), int(q), w.real, w.imag)).encode())
    return h.hexdigest()


def mode_store():
    _assert_flags_unset()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.one_brain_composer import OneBrainComposer
    try:
        from research.runners.d6_hebbian_store import conversation_write
    except Exception:                                   # the reference revision has no D6 module
        conversation_write = None
    c = OneBrainComposer(seed=42, D=128, vocab=VOCAB, k_max=8, vocab_headroom=2)
    c.hear("dog chase cat")
    if conversation_write is not None:
        with conversation_write(c):
            c.hear("wolf hunt deer")
    else:
        c.hear("wolf hunt deer")
    c.hear("fox eat berry")
    return {"mode": "store", "d6_module_present": conversation_write is not None,
            "store_conns_sha256": _sha_store(c.store_conns), "n_store_conns": len(c.store_conns),
            "kb": [dict(f) for f, _ in c.kb],
            "recall": {"dog/chase": c.query_patient("dog", "chase"), "wolf/hunt": c.query_patient("wolf", "hunt"),
                       "fox/eat": c.query_patient("fox", "eat")}}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in sorted(o.items())
                if not any(k.lower() == t or k.lower().endswith(t) for t in _VOLATILE)}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def mode_chat(seed):
    _assert_flags_unset()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    out = {"mode": "chat", "seed": seed, "turns": {}, "turn_sha256": {}}
    for i, (label, msg) in enumerate(TURNS):
        r = brain_chat(BrainChatRequest(session="d6p", message=msg, brain="tiny-demo", renderer="stub", rich=False,
                                        reset=(i == 0)))
        body = _strip(json.loads(r.body))
        out["turns"][label] = {k: body.get(k) for k in ("answer", "abstained", "recalled_svo")}
        out["turn_sha256"][label] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    chat = S._BRAIN_CHATS.get(("d6p", "tiny-demo", "stub"))
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    out["store_conns_sha256"] = _sha_store(comp.store_conns) if comp is not None else None
    return out


def _run_in_tree(tree, mode, seed, python):
    """Run this file's mode inside `tree` (its own imports), in a fresh process. The script is loaded from THIS file,
    but sys.path[0] is forced to `tree` so every import resolves to that tree's code."""
    code = ("import sys, json, runpy; sys.path.insert(0, %r); sys.path = [p for p in sys.path if p != %r];"
            "g = runpy.run_path(%r, run_name='d6p'); r = g['mode_%s'](%s); print('@@D6P@@' + json.dumps(r, default=str))"
            % (tree, os.path.dirname(os.path.abspath(__file__)), os.path.abspath(__file__), mode,
               ("%d" % seed) if mode == "chat" else ""))
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_D6_")}
    env["PYTHONPATH"] = tree
    p = subprocess.run([python, "-c", code], cwd=tree, env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("@@D6P@@"):
            return json.loads(line[len("@@D6P@@"):])
    raise RuntimeError("parity run failed in %s (rc=%s): %s" % (tree, p.returncode, p.stderr[-3000:]))


def archive_ref(ref, repo, dest):
    sha = subprocess.run(["git", "rev-parse", ref], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    excl = [":(exclude)%s" % d for d in DATA_DIRS if d != "data"]
    tar = subprocess.run(["git", "archive", sha, "--", "."] + excl, cwd=repo, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=tar, check=True)
    for extra in DATA_DIRS:
        src = os.path.join(repo, extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(dest, extra)):
            os.makedirs(os.path.dirname(os.path.join(dest, extra)), exist_ok=True)
            os.symlink(os.path.realpath(src), os.path.join(dest, extra))
    return sha


def compare(ref, mode, seed=42, repo=None, python=None):
    repo = repo or os.getcwd()
    python = python or sys.executable
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    tmp = tempfile.mkdtemp(prefix="d6p_ref_")
    try:
        ref_sha = archive_ref(ref, repo, tmp)
        a = _run_in_tree(os.path.abspath(repo), mode, seed, python)
        b = _run_in_tree(tmp, mode, seed, python)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    keys = ["store_conns_sha256", "kb", "recall"] if mode == "store" else ["turn_sha256", "turns", "store_conns_sha256"]
    diffs = [k for k in keys if a.get(k) != b.get(k)]
    return {"tool": "research.runners.d6_offpath_parity", "mode": mode, "seed": seed, "branch_head": head,
            "branch_worktree_dirty": dirty, "ref": ref, "ref_sha": ref_sha, "branch": a, "reference": b,
            "compared_keys": keys, "diff_keys": diffs, "byte_identical": (diffs == [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["store", "chat"], default="store")
    ap.add_argument("--vs-ref", default=None, help="compare this tree against a git revision (e.g. origin/main)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = compare(a.vs_ref, a.mode, a.seed) if a.vs_ref else (mode_store() if a.mode == "store" else mode_chat(a.seed))
    print(json.dumps({k: v for k, v in res.items() if k not in ("branch", "reference")}, indent=2, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2, default=str)
        print("wrote", a.out)
    return 0 if res.get("byte_identical", True) else 1


if __name__ == "__main__":
    sys.exit(main())
