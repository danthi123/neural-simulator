"""Integrated-conversational-state diagnostic driver #2 (re-run against the two 2026-08-25 landings).

REUSES the harness style of research/findings/raw/_integrated_convo_diagnostic/_diag_driver.py (same
turn() helper, same env vars, same production /api/brain-chat entry point via in-process FastAPI
TestClient, one warm brain reused across every turn). Adds phases probing:
  - the compositional chain route (reasoning DERIVE) that is now production-default
    (research/runners/compositional_chain_route.py, BRAIN_CHAIN_ROUTE),
  - the DA-axis cupy-interop fix (webapp/da_mode_drives_chat.py) -- da_drives.reason should no longer be
    "error:...",
  - the chain route's own moat hardening (untaught 2nd hop; ambiguous/conflicting first hop),
  - a live LESION of the chain route (BRAIN_CHAIN_ROUTE=0, toggled mid-session -- no restart needed since
    chain_route_enabled() reads the env var fresh on every call) as the load-bearing proof,
  - the ORIGINAL diagnostic's regression turns (baseline recall, moat abstain, affect lead, topic-swap
    lead, continuous-wander, self-initiation, comprehension-repair).

READ/RUN only -- no change to production behavior.
"""
import os, sys, json, time

os.environ["SIM_BACKEND"] = "cupy"                 # real spiking substrate (D5/episodic/all organs)
os.environ["BRAIN_CHAT_RENDERER"] = "stub"         # reliable mouth (worktree lacks the Qwen priming corpus);
                                                    # faculty couplings wrap the rendered text so they are fully exercised
os.environ["BRAIN_SOURCE_PROVENANCE_HONESTY"] = "1"
os.environ.setdefault("BRAIN_DATA_ROOT", "/home/dant123/Projects/sim-data")

REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-a7f8e02e3fc0476ec"
sys.path.insert(0, REPO); os.chdir(REPO)

from fastapi.testclient import TestClient
from webapp.server import app

SESSION = "diag2"; BRAIN = "tiny-demo"
OUTDIR = os.path.join(REPO, "research/findings/raw/_integrated_convo_diagnostic_2")
JSONL = os.path.join(OUTDIR, "transcript_2026-08-25.jsonl")
open(JSONL, "w").close()  # truncate

ALL_RECS = []


def turn(client, label, message, rich=None, sleep_after=0.0, env_override=None):
    if env_override:
        for k, v in env_override.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    body = {"session": SESSION, "brain": BRAIN, "message": message}
    if rich is not None:
        body["rich"] = rich
    t0 = time.time()
    r = client.post("/api/brain-chat", json=body)
    dt = time.time() - t0
    try:
        data = r.json()
    except Exception:
        data = {"_nonjson": r.text[:800]}
    rec = {"label": label, "message": message, "rich_req": rich, "env_override": env_override,
           "status": r.status_code, "elapsed_s": round(dt, 2), "resp": data}
    ALL_RECS.append(rec)
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    ans = data.get("answer") if isinstance(data, dict) else None
    print(f"\n### [{label}] status={r.status_code} ({dt:.1f}s)", flush=True)
    print(f"  USER: {message!r}", flush=True)
    if r.status_code != 200:
        print(f"  ERROR: {json.dumps(data, default=str)[:600]}", flush=True)
    else:
        print(f"  BOT : {ans!r}", flush=True)
        interesting = {k: data[k] for k in (
            "abstained", "recalled_svo", "derived", "derived_from", "rich", "n_sentences", "renderer",
            "provenance", "affect_drives", "da_drives", "da_encoding", "swap_drives", "wander_drives",
            "hypothesis", "hypothesis_svo", "fluent_hypothesis", "self_initiated", "supporting_facts",
            "source", "verified", "continuous_drives", "ideation_drives", "metacog", "surprise",
            "curiosity", "gnw_bus")
            if isinstance(data, dict) and k in data and data[k] not in (None, "", [], {})}
        if interesting:
            print(f"  META: {json.dumps(interesting, default=str)[:2000]}", flush=True)
    sys.stdout.flush()
    if sleep_after:
        print(f"  ...idle {sleep_after:.0f}s (continuous tick + D5)...", flush=True)
        time.sleep(sleep_after)


print("=== building warm brain (cupy + stub mouth) ===", flush=True)
with TestClient(app, raise_server_exceptions=False) as client:
    time.sleep(2)

    # ---- PHASE A: baseline recall + moat abstain (regression) ----
    turn(client, "A1-baseline-recall", "what does the dog chase?")
    turn(client, "A2-abstain-moat", "what does the dragon breathe?")

    # ---- PHASE B: teach + idle/wander (regression: in-loop learn, D5 continuous engine) ----
    turn(client, "B1-teach", "the wolf hunts the deer")
    turn(client, "B2-recall-immediate", "what does the wolf hunt?", sleep_after=25)
    turn(client, "B3-recall-after-idle1", "what does the wolf hunt?", sleep_after=25)
    turn(client, "B4-recall-after-idle2", "what does the wolf hunt?")

    # ---- PHASE C: affect / DA-rich-vs-flat / swap (regression + DA-axis-live headline) ----
    turn(client, "C1-engaging-positive", "Wow, I absolutely love wolves, they are magnificent! What does the wolf hunt?")
    turn(client, "C2-abstain-after-engaging", "what does the phoenix breathe?")
    turn(client, "C3-flat-curt", "wolf hunt what")
    turn(client, "C4-topic-swap", "what does the dog chase?")

    # ---- PHASE D: reasoning DERIVE (headline) + old explicit-chain phrasing + self-init + comprehension-repair ----
    turn(client, "D1-generate-hypothesis", "what might the wolf chase?")
    turn(client, "D2-teach-chain-hop2", "the deer eats grass")
    turn(client, "D3-reasoning-chain-DERIVE", "what does the wolf's prey eat?")
    turn(client, "D4-reasoning-explicit-old-phrasing", "the wolf hunts the deer and the deer eats grass; so what does what the wolf hunts eat?")
    turn(client, "D5-self-initiation", "")
    turn(client, "D6-self-init-prompt", "say something on your mind")

    # ---- PHASE E: chain-route moat hardening (new) ----
    turn(client, "E1-teach-hop1-only", "the fox hunts the rabbit")
    turn(client, "E2-untaught-2ndhop-ABSTAIN", "what does the fox's prey eat?")
    turn(client, "E3-teach-conflict-a", "the eagle hunts the fish")
    turn(client, "E4-teach-conflict-b", "the eagle hunts the snake")
    turn(client, "E5-teach-resolvable-branch", "the snake eats mice")
    turn(client, "E6-ambiguous-hop1-ABSTAIN", "what does the eagle's prey eat?")

    # ---- PHASE F: live LESION of the chain route (load-bearing proof; no restart needed -- the flag is
    # read fresh from os.environ on every call) ----
    turn(client, "F1-lesion-chain-route-off", "what does the wolf's prey eat?",
         env_override={"BRAIN_CHAIN_ROUTE": "0"})
    turn(client, "F2-restore-chain-route-on", "what does the wolf's prey eat?",
         env_override={"BRAIN_CHAIN_ROUTE": None})

print("\n=== conversation complete ===", flush=True)

with open(os.path.join(OUTDIR, "transcript_2026-08-25.json"), "w") as f:
    json.dump(ALL_RECS, f, indent=1, default=str)
print(f"wrote {len(ALL_RECS)} turns to transcript_2026-08-25.json + .jsonl", flush=True)
