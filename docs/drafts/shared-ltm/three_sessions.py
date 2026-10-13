"""Three conversations through the REAL /api/brain-chat handler (dev seed 7): per-session RSS + per-turn reply hashes.

Run per arm (run_arm.sh sets the env): BRAIN_SHARED_LTM=0 (each conversation loads its own LTM copy),
BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=1 (own copies, RNG-isolated), BRAIN_SHARED_LTM=1 (one shared store). Sessions
are built back to back (turn 1 of s1, s2, s3), then every later turn interleaves across sessions, so a later
conversation reads the shared store after another one used it.

    python docs/drafts/shared-ltm/three_sessions.py <out.json> [v1|v2]

v1 = the first run's 3 turns. v2 (review, 2026-10-13) puts buffer recalls (the spiking tiny-demo composer) AFTER LTM
recalls, where a reseeded global RNG would show.

rss_kb from /proc is KiB. The reply hash is sha256 over the handler's whole JSON body bytes, per turn.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_CHAT_SEED", "7")
os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
os.environ.setdefault("SIM_DISABLE_LLM", "1")

TURNSETS = {
    "v1": ["what does the dog chase",            # a tiny-demo (buffer) fact
           "what country is chelsea fc from?",   # an LTM fact: (chelsea_fc, country, united_kingom) is only in wikidata_100k
           "what does the dragon do"],           # untaught: buffer and LTM both abstain (the moat)
    "v2": ["what does the dog chase",            # buffer
           "what country is chelsea fc from?",   # LTM (chelsea_fc, country, united_kingom)
           "what does the cat eat",              # buffer, after an LTM recall
           "what sport is chelsea fc in?",       # LTM (chelsea_fc, sport, association_football_club)
           "what does the brain use",            # buffer, after an LTM recall
           "what does the dragon do"],           # untaught: buffer and LTM both abstain
}
TURNS = TURNSETS["v1"]
SESSIONS = ["s1", "s2", "s3"]


def rss():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0]] = int(line.split()[1])
    return out["VmRSS"], out["VmHWM"]


def main(out_path, turnset="v1"):
    import subprocess
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    turns = TURNSETS[turnset]
    vcs = ["gi" + "t"]
    sha = subprocess.run(vcs + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(vcs + ["status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    rec = {"env": {k: os.environ.get(k) for k in ("BRAIN_SHARED_LTM", "BRAIN_LTM_RNG_ISOLATE", "BRAIN_CHAT_SEED",
                                                  "SIM_BACKEND", "BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM",
                                                  "OMP_NUM_THREADS")},
           "git_sha": sha, "git_dirty": dirty, "turnset": turnset,
           "shared_ltm_on": getattr(S, "_shared_ltm_on", lambda: None)(),   # None: a base without the flag
           "ltm_rng_isolate_on": getattr(S, "_ltm_rng_isolate_on", lambda: None)(),
           "turns": turns, "rss_start_kb": rss()[0], "built": {}, "replies": []}
    for ti, msg in enumerate(turns):
        for sess in SESSIONS:
            t0 = time.time()
            r = brain_chat(BrainChatRequest(session=sess, message=msg))
            body = bytes(r.body)
            j = json.loads(body)
            key = S.brain_chat_cache_key(BrainChatRequest(session=sess, message=msg))
            chat = S._BRAIN_CHATS.get(key)
            ltm = getattr(getattr(getattr(chat, "inner", None), "composer", None), "ltm", None)
            lt = getattr(ltm, "last_trace", None)
            row = {"session": sess, "turn": ti, "message": msg, "sha256": hashlib.sha256(body).hexdigest(),
                   "sec": round(time.time() - t0, 1), "source": j.get("source"), "answer": j.get("answer"),
                   "recalled_svo": j.get("recalled_svo"), "abstained": j.get("abstained"),
                   "ltm_type": type(ltm).__name__,
                   "ltm_store_id": id(getattr(ltm, "shared_store", ltm)) if ltm is not None else None,
                   "ltm_last_trace_source_fact": (lt or {}).get("source_fact") if isinstance(lt, dict) else None,
                   "body": j}
            if ti == 0:
                cur, hwm = rss()
                rec["built"][sess] = {"rss_kb": cur, "hwm_kb": hwm}
            rec["replies"].append(row)
            print("[three_sessions] %s t%d %.1fs sha=%s ans=%r svo=%s ltm=%s" % (
                sess, ti, row["sec"], row["sha256"][:12], row["answer"], row["recalled_svo"], row["ltm_type"]),
                flush=True)
    cur, hwm = rss()
    rec["rss_end_kb"], rec["hwm_end_kb"] = cur, hwm
    rec["n_shared_ltm_entries"] = len(getattr(S, "_SHARED_LTM", {}) or {})
    rec["distinct_ltm_store_ids"] = len({r["ltm_store_id"] for r in rec["replies"]})
    json.dump(rec, open(out_path, "w"), indent=1, default=str)
    print("[three_sessions] built rss_kb %s; end rss %d hwm %d; store ids %d -> %s" % (
        {s: v["rss_kb"] for s, v in rec["built"].items()}, cur, hwm, rec["distinct_ltm_store_ids"], out_path),
        flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v1")
