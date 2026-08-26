"""Integrated-conversational-state diagnostic driver (hardened).

Runs a REAL multi-turn conversation through the production /api/brain-chat handler
(FastAPI TestClient, in-process) reusing ONE warm cupy brain with the real Qwen mouth.
Opt-in source-provenance honesty is enabled. Captures full turn-by-turn JSON,
surfaces 500s inline, and appends each turn to a JSONL so a kill never loses data.

READ/RUN only — no change to production behavior.
"""
import os, sys, json, time
os.environ["SIM_BACKEND"] = "cupy"                 # real spiking substrate (D5/episodic/all organs)
os.environ["BRAIN_CHAT_RENDERER"] = "stub"         # reliable mouth (worktree lacks the Qwen priming corpus);
                                                    # faculty couplings wrap the rendered text so they are fully exercised
os.environ["BRAIN_SOURCE_PROVENANCE_HONESTY"] = "1"
os.environ.setdefault("BRAIN_DATA_ROOT", "/home/dant123/Projects/sim-data")
REPO = "/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/wt-convo-diag"
sys.path.insert(0, REPO); os.chdir(REPO)
from fastapi.testclient import TestClient
from webapp.server import app

SESSION = "diag"; BRAIN = "tiny-demo"
JSONL = os.path.join(REPO, "_diag_transcript.jsonl")
open(JSONL, "w").close()  # truncate

def turn(client, label, message, rich=None, sleep_after=0.0):
    body = {"session": SESSION, "brain": BRAIN, "message": message}
    if rich is not None: body["rich"] = rich
    t0 = time.time()
    r = client.post("/api/brain-chat", json=body)
    dt = time.time() - t0
    try:
        data = r.json()
    except Exception:
        data = {"_nonjson": r.text[:800]}
    rec = {"label": label, "message": message, "rich_req": rich,
           "status": r.status_code, "elapsed_s": round(dt, 2), "resp": data}
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    ans = data.get("answer") if isinstance(data, dict) else None
    print(f"\n### [{label}] status={r.status_code} ({dt:.1f}s)")
    print(f"  USER: {message!r}")
    if r.status_code != 200:
        print(f"  ERROR: {json.dumps(data, default=str)[:600]}")
    else:
        print(f"  BOT : {ans!r}")
        interesting = {k: data[k] for k in (
            "abstained","recalled_svo","rich","n_sentences","renderer","provenance",
            "affect_drives","da_drives","swap_drives","da_encoding","hypothesis","hypothesis_svo",
            "fluent_hypothesis","self_initiated","supporting_facts","source","verified",
            "continuous_drives","ideation_drives","metacog","surprise","curiosity","gnw_bus")
            if isinstance(data, dict) and k in data and data[k] not in (None, "", [], {})}
        if interesting:
            print(f"  META: {json.dumps(interesting, default=str)[:1600]}")
    sys.stdout.flush()
    if sleep_after:
        print(f"  ...idle {sleep_after:.0f}s (continuous tick + D5)..."); sys.stdout.flush()
        time.sleep(sleep_after)

print("=== building warm brain (cupy + Qwen) ===", flush=True)
with TestClient(app, raise_server_exceptions=False) as client:
    time.sleep(2)
    # PHASE A: baseline + knowledge core + abstain
    turn(client, "A1-baseline-recall", "what does the dog chase?")
    turn(client, "A2-knowledge-copula", "what is a physicist?")
    turn(client, "A3-knowledge-relational", "what country is chelsea fc from?")
    turn(client, "A4-knowledge-token-oracle", "what does chelsea_fc country")
    turn(client, "A5-knowledge-token2", "what does penicillium instance_of")
    turn(client, "A6-abstain-moat", "what does the dragon breathe?")
    # PHASE B: teach + learn-through-use (D5)
    turn(client, "B1-teach", "the wolf hunts the deer")
    turn(client, "B2-recall-armsD5", "what does the wolf hunt?", sleep_after=25)
    turn(client, "B3-recall-after-idle1", "what does the wolf hunt?", sleep_after=25)
    turn(client, "B4-recall-after-idle2", "what does the wolf hunt?")
    # PHASE C: affect / DA / swap DRIVE the surface
    turn(client, "C1-engaging-positive", "Wow, I absolutely love wolves, they are magnificent! What does the wolf hunt?")
    turn(client, "C2-flat-curt", "wolf hunt what")
    turn(client, "C3-topic-swap", "what does the dog chase?")
    # PHASE D: honesty perceived vs inferred + own-conclusion + self-init
    turn(client, "D1-generate-hypothesis", "what might the wolf chase?")
    turn(client, "D2-teach-chain", "the deer eats grass")
    turn(client, "D3-reasoning-chain", "what does the wolf's prey eat?")
    turn(client, "D4-reasoning-explicit", "the wolf hunts the deer and the deer eats grass; so what does what the wolf hunts eat?")
    turn(client, "D5-self-initiation", "")
    turn(client, "D6-self-init-prompt", "say something on your mind")
print("\n=== conversation complete ===", flush=True)
