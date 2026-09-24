"""D6 CHAT-WIRE OBSERVABILITY: is `webapp/d6_hebbian_chat.py` byte-identical off, and does its logic report the
composer's own encode/held state correctly? (research/findings/2026-09-24-d6-chat-wire-PREREGISTRATION.md)

WHAT THIS PROBES. `webapp/d6_hebbian_chat.after_store_d6` is a READ-ONLY reporting hook: it never writes the fact
block (that stays in `research/runners/d6_hebbian_store.hebbian_encode` / `one_brain_composer._store_composite`,
already GO 6/6 on the K1-K7 capability gate — `research/runners/d6_learn_through_use_lb.py`,
`research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`). This probe does NOT re-run
K1-K7 (that gate is unchanged and is run directly: see the module docstring below for the seed-7 command); it
proves two narrower things about the NEW webapp module:
  (1) SELFTEST (no brain): `after_store_d6` returns None when the flag is off; when on, it reports
      `wrote_this_turn: False` on a composer with no fresh encode, reports the encode diagnostic + a held read on
      one with a fresh `_d6_last_encode`, CONSUMES it (the second call afterward reports False again), and never
      raises out of the hook even when `engram_held` itself fails.
  (2) OFFCHECK (needs a tiny-demo brain build): with `BRAIN_D6_HEBBIAN_STORE` unset, a real `/api/brain-chat`
      conversation's replies and the composer's store synapses hash identically between this branch and the
      pinned pre-change tree (the merge-base with origin/main) -- proving the two try/except blocks this branch
      added to `webapp/server.py`'s response assembly are true no-ops off.

Run (no brain, instant):        .venv/bin/python -m research.runners._d6_chat_wire_probe --selftest
Run (needs mem_ok/memcap, a tiny-demo build twice):
  bash tools/mem_ok.sh 12 4 && SIM_BACKEND=numpy OMP_NUM_THREADS=1 bash tools/memcap.sh 12 -- .venv/bin/python -u \
      -m research.runners._d6_chat_wire_probe --offcheck --out research/findings/raw/_d6_chat_wire/offcheck.json

THE K1-K7 SEED-7 SMOKE this probe does not itself run (the gate already exists and is unchanged by this branch):
  bash tools/mem_ok.sh 12 4 && SIM_BACKEND=numpy OMP_NUM_THREADS=1 bash tools/memcap.sh 12 -- .venv/bin/python -u \
      -m research.runners.d6_learn_through_use_lb --seeds 7 --variant capability \
      --arm-dir research/findings/raw/_d6_chat_wire/k1k7_seed7 \
      --json research/findings/raw/_d6_chat_wire/k1k7_seed7_verdict.json
  (seed 7 is a DEV/SMOKE seed, outside {42,43,44,100,101,102}; it never governs a production-default decision.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _merge_base_with_origin_main() -> str:
    r = subprocess.run(["git", "-C", _REPO, "merge-base", "HEAD", "origin/main"], capture_output=True, text=True)
    sha = r.stdout.strip()
    if r.returncode != 0 or not sha:
        raise RuntimeError("merge-base lookup failed: %s" % r.stderr[-500:])
    return sha


# ── selftest (no brain) ─────────────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict:
    sys.path.insert(0, _REPO)
    from webapp import d6_hebbian_chat as W

    class _Inner:
        def __init__(self, composer):
            self.composer = composer

    class _Chat:
        def __init__(self, composer):
            self.inner = _Inner(composer)

    class _Comp:
        pass

    out = {}
    prev = os.environ.pop("BRAIN_D6_HEBBIAN_STORE", None)
    try:
        # (a) flag OFF -> enabled() False, hook returns None, never touches the composer
        assert W.d6_hebbian_enabled() is False, "off by default"
        comp_untouched = _Comp()
        comp_untouched._d6_last_encode = {"block": 0}   # a decoy: must NOT be read when the flag is off
        chat = _Chat(comp_untouched)
        r = W.after_store_d6(chat)
        assert r is None, "hook must be a no-op when BRAIN_D6_HEBBIAN_STORE is unset: got %r" % (r,)
        assert comp_untouched._d6_last_encode == {"block": 0}, "off path must not consume the encode"
        out["off_is_noop"] = True

        # (b) flag ON, no fresh encode -> wrote_this_turn False
        os.environ["BRAIN_D6_HEBBIAN_STORE"] = "1"
        assert W.d6_hebbian_enabled() is True
        comp_idle = _Comp()
        chat2 = _Chat(comp_idle)
        r2 = W.after_store_d6(chat2)
        assert r2 == {"on": True, "wrote_this_turn": False}, r2
        out["on_no_write_reports_false"] = True

        # (c) flag ON, fresh encode present -> reported, then CONSUMED (a held-read failure is caught, not raised)
        comp_wrote = _Comp()
        comp_wrote._d6_last_encode = {"block": 3, "frozen": False, "n_saturated": 16, "mean_abs_w": 0.97}
        chat3 = _Chat(comp_wrote)
        r3 = W.after_store_d6(chat3)
        assert r3["on"] is True and r3["wrote_this_turn"] is True, r3
        assert r3["encode"] == {"block": 3, "frozen": False, "n_saturated": 16, "mean_abs_w": 0.97}, r3
        assert "held" in r3 and "error" in r3["held"], "engram_held on a fake composer must fail closed: %r" % r3
        assert comp_wrote._d6_last_encode is None, "the encode must be consumed"
        r4 = W.after_store_d6(chat3)
        assert r4 == {"on": True, "wrote_this_turn": False}, "a second call must not replay the consumed encode: %r" % r4
        out["encode_reported_then_consumed"] = True

        # (d) a composer object with no `.inner`/`.composer` at all -> None, not a crash
        class _Bare:
            pass
        r5 = W.after_store_d6(_Bare())
        assert r5 is None, r5
        out["no_composer_is_noop"] = True
    finally:
        if prev is None:
            os.environ.pop("BRAIN_D6_HEBBIAN_STORE", None)
        else:
            os.environ["BRAIN_D6_HEBBIAN_STORE"] = prev
    out["pass"] = True
    return out


# ── offcheck (needs a tiny-demo brain, twice) ───────────────────────────────────────────────────────────────────
_OFFCHECK_TURNS = ["the wolf hunts the deer", "what does the cat eat", "what does the dog chase",
                   "what does the wolf hunt", "what does the fox eat"]


def offcheck_worker(repo: str, out: str) -> None:
    os.chdir(repo)
    sys.path.insert(0, repo)
    for k in ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME",
              "BRAIN_D6_ENGRAM_PRUNE"):
        os.environ.pop(k, None)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = "7"      # a dev seed: this offcheck asserts equality, not a capability verdict
    from webapp import server as S
    replies = []
    for i, msg in enumerate(_OFFCHECK_TURNS):
        r = S.brain_chat(S.BrainChatRequest(session="d6chk", message=msg, brain="tiny-demo", renderer="stub",
                                            rich=False, reset=(i == 0)))
        replies.append(json.loads(r.body))
    chat = S._BRAIN_CHATS.get(("d6chk", "tiny-demo", "stub"))
    comp = chat.inner.composer
    store = [(int(p), int(q), complex(w).real, complex(w).imag) for (p, q, w) in comp.store_conns]
    rj = json.dumps(replies, sort_keys=True, default=str)
    sj = json.dumps(store)
    json.dump({"repo": repo, "replies_sha256": hashlib.sha256(rj.encode()).hexdigest(),
               "store_sha256": hashlib.sha256(sj.encode()).hexdigest(), "n_store_conns": len(store),
               "d6_hebbian_key_present": any("d6_hebbian" in rr for rr in replies),
               "replies": replies}, open(out, "w"), indent=1, default=str)


def offcheck(pinned_sha: str, out: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        pin = os.path.join(td, "pinned")
        os.makedirs(pin)
        arc = subprocess.run(["git", "-C", _REPO, "archive", pinned_sha], capture_output=True)
        if arc.returncode != 0:
            raise RuntimeError("git archive failed: %s" % arc.stderr[-500:])
        subprocess.run(["tar", "-x", "-C", pin], input=arc.stdout, check=True)
        os.makedirs(os.path.join(pin, "data"), exist_ok=True)
        os.symlink(os.path.realpath(os.path.join(_REPO, "data", "corpus")), os.path.join(pin, "data", "corpus"))
        res = {"pinned_sha": pinned_sha, "branch_repo": _REPO}
        for tag, repo in (("pinned", pin), ("branch", _REPO)):
            o = os.path.join(td, tag + ".json")
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--offcheck-worker", repo, o],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(o):
                raise RuntimeError("offcheck worker %s failed: %s" % (tag, r.stderr[-2000:]))
            res[tag] = json.load(open(o))
            res[tag].pop("replies", None)
        res["replies_identical"] = res["pinned"]["replies_sha256"] == res["branch"]["replies_sha256"]
        res["store_identical"] = res["pinned"]["store_sha256"] == res["branch"]["store_sha256"]
        res["no_key_leaked_off"] = not res["branch"]["d6_hebbian_key_present"]
        res["byte_identical_off"] = bool(res["replies_identical"] and res["store_identical"]
                                         and res["no_key_leaked_off"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--offcheck", action="store_true")
    ap.add_argument("--offcheck-worker", nargs=2)
    ap.add_argument("--pinned-sha", default=None)
    ap.add_argument("--out", default="research/findings/raw/_d6_chat_wire/offcheck.json")
    a = ap.parse_args()
    if a.offcheck_worker:
        return offcheck_worker(*a.offcheck_worker)
    if a.selftest:
        r = selftest()
        print(json.dumps(r, indent=2))
        return 0 if r.get("pass") else 1
    if a.offcheck:
        sha = a.pinned_sha or _merge_base_with_origin_main()
        r = offcheck(sha, a.out)
        return 0 if r["byte_identical_off"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
