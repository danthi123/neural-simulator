"""A10 module-level OFF-identity check (fix round, 2026-09-24): `da_mode_drives_chat.observe_turn` from the PRE-PATCH
file (a git blob, default the merge parent on main) vs the POST-PATCH file in this checkout, with
`BRAIN_REWARD_VALUE_AFFERENT` unset and set to "0", over fake chat sessions -- the SHA-256 of the JSON output must
be equal. Each variant runs in its own subprocess (fresh global RNG / module state).

This is the module-level half of (A); the full-handler half is `_reward_value_afferent_derisk.py` (whole brain_chat
responses vs a pre-patch main build). It builds only the small #76 DA substrate, the spiking novelty organ and the
shared salience afferent (no tiny-demo brain, no surprise organ), so it runs locally. It needs git (for the blob).

CONTROLS.
  * determinism: pre-patch run twice -> equal hashes (otherwise equality between pre and post means nothing).
  * sensitivity (negative control): post-patch with the flag ON and the surprise read STUBBED to a fixed drive
    (normalized 0.9 on every turn) -> the hash MUST differ. A comparison that cannot see a change in the patched
    path is not evidence of identity.

Run:  .venv/bin/python -m research.runners._reward_value_afferent_offidentity --pre-ref <sha> \
          --out research/findings/raw/_reward_value_afferent_derisk/v2/offidentity_module.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

MSGS = ["the dog chase the cat", "hello there how are you today", "", "the dog chase the fish",
        "tell me about quantum physics and all its strange wonders", "ok", "what does the dog chase?"]


class _Inner:
    def what_does(self, a, v):
        return "cat" if (a, v) == ("dog", "chase") else None


def _child(variant, pre_path, stub, out):
    """Run observe_turn over two fake sessions and write {sha256, results}."""
    if variant == "pre":
        import importlib.util
        import webapp
        spec = importlib.util.spec_from_file_location("webapp.da_mode_drives_chat", pre_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["webapp.da_mode_drives_chat"] = mod
        spec.loader.exec_module(mod)
        webapp.da_mode_drives_chat = mod
    else:
        import webapp.da_mode_drives_chat as mod
    if stub:
        import webapp.reward_value_afferent_chat as rva
        rva.spiking_reward_value = lambda chat, message, seed: {
            "on": True, "source": "surprise", "drives": True, "normalized": 0.9, "stub": True}
    res = {}
    for sess in ("a", "b"):
        chat = types.SimpleNamespace(inner=_Inner())
        res[sess] = [mod.observe_turn(chat, m) for m in MSGS]
    blob = json.dumps(res, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()
    with open(out, "w") as f:
        json.dump({"variant": variant, "stub": stub, "flag": os.environ.get("BRAIN_REWARD_VALUE_AFFERENT"),
                   "sha256": h, "results": res}, f, sort_keys=True, default=str)
    return 0


def _run(variant, flag, pre_path, stub, scratch, tag):
    out = os.path.join(scratch, "%s.json" % tag)
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env.pop("BRAIN_REWARD_VALUE_AFFERENT", None)
    env.pop("BRAIN_REWARD_VALUE_LESION", None)
    env["BRAIN_CHAT_SEED"] = "7"
    env["SIM_NO_PROVENANCE"] = "1"      # the child's scratch output is not an artifact; this runner's out is
    if flag is not None:
        env["BRAIN_REWARD_VALUE_AFFERENT"] = flag
    cmd = [sys.executable, "-u", "-m", "research.runners._reward_value_afferent_offidentity", "--child", variant,
           "--pre-path", pre_path, "--child-out", out] + (["--stub"] if stub else [])
    p = subprocess.run(cmd, env=env, cwd=_REPO, capture_output=True, text=True)
    if p.returncode != 0:
        return {"tag": tag, "error": (p.stderr or "")[-1500:]}
    d = json.load(open(out))
    return {"tag": tag, "variant": variant, "flag": flag, "stub": stub, "sha256": d["sha256"],
            "confirm_turn_a": d["results"]["a"][0]}


def main(pre_ref, out, scratch):
    os.makedirs(scratch, exist_ok=True)
    pre_path = os.path.join(scratch, "da_mode_drives_chat_pre.py")
    blob = subprocess.run(["git", "show", "%s:webapp/da_mode_drives_chat.py" % pre_ref], cwd=_REPO,
                          capture_output=True, text=True, check=True).stdout
    with open(pre_path, "w") as f:
        f.write(blob)
    pre_sha = hashlib.sha256(blob.encode()).hexdigest()
    post_sha = hashlib.sha256(open(os.path.join(_REPO, "webapp", "da_mode_drives_chat.py")).read().encode()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True).stdout.strip()
    pre_commit = subprocess.run(["git", "rev-parse", pre_ref], cwd=_REPO, capture_output=True, text=True).stdout.strip()

    runs = {
        "pre_unset": _run("pre", None, pre_path, False, scratch, "pre_unset"),
        "pre_unset_repeat": _run("pre", None, pre_path, False, scratch, "pre_unset_repeat"),
        "post_unset": _run("post", None, pre_path, False, scratch, "post_unset"),
        "post_zero": _run("post", "0", pre_path, False, scratch, "post_zero"),
        "post_on_stub": _run("post", "1", pre_path, True, scratch, "post_on_stub"),
    }
    h = {k: r.get("sha256") for k, r in runs.items()}

    from tools.verdict import Verdict
    v = Verdict("A10 module-level OFF identity: observe_turn pre-patch vs post-patch, flag unset / 0")
    for k, r in runs.items():
        v.require("variant %s ran" % k, "error" not in r, expect=True)
    v.require("determinism: pre-patch run twice gives the same hash", h["pre_unset"] == h["pre_unset_repeat"]
              and h["pre_unset"] is not None, expect=True)
    v.require("sensitivity: flag ON with a stubbed drive changes the hash (the comparison can see the patch)",
              h["post_on_stub"] is not None and h["post_on_stub"] != h["post_unset"], expect=True)
    identical = bool(h["pre_unset"] is not None and h["pre_unset"] == h["post_unset"] == h["post_zero"])
    decided = v.decide(go=identical, verbose=True)
    rec = {"runner": "_reward_value_afferent_offidentity", "pre_ref": pre_ref, "pre_commit": pre_commit,
           "head": head, "pre_file_sha256": pre_sha, "post_file_sha256": post_sha, "files_differ": pre_sha != post_sha,
           "messages": MSGS, "hashes": h, "byte_identical_off": identical, "runs": runs,
           "status": decided["status"], "verdict": decided}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True, default=str)
    print(json.dumps({"status": decided["status"], "hashes": h, "out": out}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-ref", default="HEAD^2", help="git ref whose webapp/da_mode_drives_chat.py is pre-patch")
    ap.add_argument("--out", default="research/findings/raw/_reward_value_afferent_derisk/v2/offidentity_module.json")
    ap.add_argument("--scratch", default=".a10_scratch", help="scratch dir for the pre-patch blob + child outputs")
    ap.add_argument("--child", default=None, choices=("pre", "post"), help=argparse.SUPPRESS)
    ap.add_argument("--pre-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        sys.exit(_child(a.child, a.pre_path, a.stub, a.child_out))
    sys.exit(main(a.pre_ref, a.out, os.path.abspath(a.scratch)))
