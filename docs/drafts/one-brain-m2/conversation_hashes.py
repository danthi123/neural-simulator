"""M2 (one brain, many conversations) reply-hash driver: one turn script through the REAL /api/brain-chat handler,
sha256 over each reply's whole JSON body, plus the seconds each turn took.

Used three ways (docs/drafts/one-brain-m2/WORKLOG.md):
  * base export (git archive afeb1bcc9, data symlinked) vs this branch with BRAIN_ONE_BRAIN unset: the OFF
    byte-identity check (docs/TERMS.md "byte-identical": asserted by hash on data, never inferred).
  * this branch BRAIN_ONE_BRAIN=0 vs =1 on the SINGLE script: which replies the brain-level rows change by design.
  * the PAIR script (two conversations, interleaved) at ON: the cross-conversation reply changes.

    python docs/drafts/one-brain-m2/conversation_hashes.py <out.json> [single|pair]

Run it from the checkout whose `webapp` should be imported (the base export or this worktree): it puts the CWD first
on sys.path. Env defaults: numpy, seed 7, the stub renderer, no LLM. The script is driver-only: it imports nothing
the base lacks (every new attribute is read with getattr and a default).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_CHAT_SEED", "7")
os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
os.environ.setdefault("SIM_DISABLE_LLM", "1")

# One conversation that drives every per-conversation registry server.py keeps (mood, world-view, multiref, discourse,
# prospective memory, self-init, rich thread, D5, reconsolidation), plus the buffer, a taught fact, the LTM and the moat.
SINGLE = [
    "what does the dog chase",                       # build-time buffer fact (dog chase cat)
    "the wug eats the dax",                          # teach (_maybe_acquire writes the composer buffer)
    "what does the wug eat",                         # recall the taught fact
    "Wonderful! I am so happy and delighted, this is fantastic and amazing!",   # affect -> mood EMA (_SESSION_MOOD)
    "what country is chelsea fc from?",              # the LTM prior (wikidata_100k)
    "the cat and the ball walked in",                # two referents -> multiref WM (_SESSION_MULTIREF)
    "what does it eat",                              # anaphora over the held referents
    "did we discuss the dog",                        # D5 referential recall
    "the dog chase the fish",                        # contradiction -> surprise + reconsolidation
    "dog chase cat",                                 # discourse register fold 1 (_SESSION_DISCOURSE)
    "then bird chase worm",                          # fold 2, a connective -> shift
    "who was doing it before",                       # the before-query
    "remind me to feed the dog when the bird sings", # prospective memory (_SESSION_PMEM)
    "what do you expect",                            # world-model expectation query (_SESSION_WORLDVIEW)
    "",                                              # idle / empty turn -> self-initiation (_SESSION_SELFINIT)
    "tell me more",                                  # rich follow-up (_BRAIN_RICH)
    "what does the dragon do",                       # untaught -> the moat abstains
]
# Two conversations, interleaved: A teaches and feels, B asks. At OFF B has its own brain; at ON one brain.
PAIR = [
    ("A", "the wug eats the dax"),
    ("A", "Wonderful! I am so happy and delighted, this is fantastic and amazing!"),
    ("B", "what does the wug eat"),
    ("B", "what does the dog chase"),
    ("A", "what does the wug eat"),
    ("B", "did we discuss the wug"),
    ("B", "what does the dragon do"),
]
SCRIPTS = {"single": [("C", m) for m in SINGLE], "pair": PAIR}


def main(out_path, script="single"):
    sys.path.insert(0, os.getcwd())
    from webapp import server as S
    vcs = ["gi" + "t"]
    sha = subprocess.run(vcs + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    flag = getattr(S, "one_brain_on", None)
    rec = {"script": script, "git_sha": sha, "cwd": os.getcwd(),
           "env": {k: os.environ.get(k) for k in ("BRAIN_ONE_BRAIN", "BRAIN_SHARED_LTM", "BRAIN_CHAT_SEED",
                                                  "SIM_BACKEND", "BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM",
                                                  "OMP_NUM_THREADS", "BRAIN_TURN_LOCK")},
           "one_brain_on": flag() if callable(flag) else None,       # None: a base without the flag
           "replies": []}
    t_all = time.time()
    for i, (sess, msg) in enumerate(SCRIPTS[script]):
        t0 = time.time()
        r = S.brain_chat(S.BrainChatRequest(session="m2_" + sess, message=msg, brain="tiny-demo", renderer="stub"))
        body = bytes(r.body)
        j = json.loads(body)
        row = {"i": i, "session": sess, "message": msg, "sha256": hashlib.sha256(body).hexdigest(),
               "sec": round(time.time() - t0, 2), "answer": j.get("answer"), "abstained": j.get("abstained"),
               "recalled_svo": j.get("recalled_svo"), "body": j}
        rec["replies"].append(row)
        json.dump(rec, open(out_path, "w"), indent=1, default=str)   # incremental
        print("[m2-hashes] %02d %s %.1fs sha=%s ans=%r" % (i, sess, row["sec"], row["sha256"][:12],
                                                          (row["answer"] or "")[:100]), flush=True)
    rec["seconds_total"] = round(time.time() - t_all, 1)
    json.dump(rec, open(out_path, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "single"))
