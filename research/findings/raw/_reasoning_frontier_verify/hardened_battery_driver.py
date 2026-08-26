"""Reasoning-frontier HARDENING verify driver (2026-08-25, research/reasoning-frontier-hardened branch).
Runs the REAL production /api/brain-chat handler (FastAPI TestClient, in-process) against ONE warm brain,
exercising the naive build's own (a)-(e) regression items PLUS the 12-scenario adversarial battery from
research/findings/2026-08-25-reasoning-route-moat-audit-hardening-spec.md. READ/RUN only.

ANTI-WEDGE: SIM_BACKEND is read from the environment (set by the caller) so a >4min hang on the cupy onebrain
warm can be killed and retried with SIM_BACKEND=numpy without editing this file.
"""
import os, sys, json, time

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_CHAT_RENDERER"] = "stub"     # worktree lacks the Qwen priming corpus
os.environ.setdefault("BRAIN_DATA_ROOT", "/home/dant123/Projects/sim-data")
REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-a6819a28029c12b51"
sys.path.insert(0, REPO); os.chdir(REPO)
from fastapi.testclient import TestClient
from webapp.server import app

SESSION = "reasoning-frontier-hardened"; BRAIN = "tiny-demo"
OUT_DIR = os.path.join(REPO, "research", "findings", "raw", "_reasoning_frontier_verify")
os.makedirs(OUT_DIR, exist_ok=True)
TAG = os.environ.get("VERIFY_TAG", "hardened")
JSONL = os.path.join(OUT_DIR, f"battery_{TAG}.jsonl")
open(JSONL, "w").close()

records = []


def turn(client, label, message, rich=None, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
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
    rec = {"label": label, "message": message, "rich_req": rich, "env_at_call": dict(extra_env or {}),
           "status": r.status_code, "elapsed_s": round(dt, 2), "resp": data}
    records.append(rec)
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
            "abstained", "recalled_svo", "derived", "derived_from", "rich", "n_sentences", "renderer",
            "source", "provenance", "affect_drives", "swap_drives", "supporting_facts")
            if isinstance(data, dict) and k in data and data[k] not in (None, "", [], {})}
        if interesting:
            print(f"  META: {json.dumps(interesting, default=str)[:1400]}")
    sys.stdout.flush()
    return data


print(f"=== building warm brain (SIM_BACKEND={os.environ['SIM_BACKEND']}, stub renderer, tiny-demo) ===", flush=True)
with TestClient(app, raise_server_exceptions=False) as client:
    time.sleep(1)

    # ==================================================================================================
    # (e) / regression -- pre-existing recall + moat abstain still behave (single-fact path)
    # ==================================================================================================
    turn(client, "E1-regression-recall", "what does the dog chase?", rich=False)
    turn(client, "E2-regression-abstain", "what does the dragon breathe?", rich=False)

    # ==================================================================================================
    # (d) / battery #6 -- LEMMATIZATION store/query mismatch: teach inflected, recall on the base form
    # ==================================================================================================
    turn(client, "D1-teach-inflected", "the wolf hunts the deer", rich=False)
    turn(client, "D2-recall-base-form", "what does the wolf hunt?", rich=False)

    # ==================================================================================================
    # battery #7 -- IRREGULAR inflection (under-merge: caught/catch is NOT in _IRREGULAR_VERBS -- expected
    # to FAIL/abstain; this is audit hardening req #7, explicitly OUT OF SCOPE for this task's 4 items)
    # ==================================================================================================
    turn(client, "IRR1-teach-irregular-past", "the fox caught the mouse", rich=False)
    turn(client, "IRR2-recall-base-form", "what did the fox catch?", rich=False)

    # ==================================================================================================
    # battery #8 -- HOMOGRAPH separation (over-merge guard): saw-the-tool (noun patient) vs saw-the-verb
    # (irregular past of see, IS in _IRREGULAR_VERBS) must stay separately recallable
    # ==================================================================================================
    turn(client, "HOMO1-teach-tool-noun", "the carpenter used the saw", rich=False)
    turn(client, "HOMO2-teach-irregular-verb", "the girl saw the bird", rich=False)
    turn(client, "HOMO3-recall-tool-noun", "what did the carpenter use?", rich=False)
    turn(client, "HOMO4-recall-irregular-verb", "what did the girl see?", rich=False)

    # ==================================================================================================
    # (a) / battery #5 -- CHAIN correctness: teach two hops, derive a NEW fact via the possessive-chain route
    # ==================================================================================================
    turn(client, "A1-teach-hop1", "the wolf eats the deer", rich=False)
    turn(client, "A2-teach-hop2", "the deer eats the grass", rich=False)
    turn(client, "A3-derive-chain-single-fact", "what does the wolf's prey eat?", rich=False)
    # same question through the DEFAULT (rich) path -- confirms the production-default response shape too
    turn(client, "A3b-derive-chain-rich-default", "what does the wolf's prey eat?")

    # ==================================================================================================
    # battery #1 -- CONFAB CRUX: a genuinely fresh, never-taught subject must abstain (base moat)
    # ==================================================================================================
    turn(client, "CONFAB1-crux-fresh-subject", "what does the shark eat?", rich=False)

    # ==================================================================================================
    # battery #2 -- CONFAB near-miss: an in-vocab subject with a DIFFERENT relation taught, no eat-fact
    # ==================================================================================================
    turn(client, "CONFAB2-near-miss", "the fox chases the rabbit", rich=False)
    turn(client, "CONFAB2b-near-miss-query", "what does the fox eat?", rich=False)

    # ==================================================================================================
    # (b) / battery -- MOAT: a chain whose 2nd hop was never taught -> honest abstain (unsupported hop)
    # ==================================================================================================
    turn(client, "B1-moat-teach-hop1-only", "the wolverine hunts the badger", rich=False)
    turn(client, "B2-moat-query-unsupported-hop2", "what does the wolverine's prey eat?", rich=False)

    # ==================================================================================================
    # battery #3 -- MOAT-BYPASS conflict: a multi-valued hop MUST abstain the whole chain (audit req #1 /
    # this arc's hardening #1), both as a single hop AND through the chain route
    # ==================================================================================================
    turn(client, "CONFLICT1-teach-a", "the lion eats the antelope", rich=False)
    turn(client, "CONFLICT2-teach-b", "the lion eats the zebra", rich=False)
    turn(client, "CONFLICT3-teach-c", "the zebra eats the grass", rich=False)
    turn(client, "CONFLICT4-single-hop-ambiguous", "what does the lion eat?", rich=False)
    turn(client, "CONFLICT5-chain-ambiguous-single-fact", "what does the lion's prey eat?", rich=False)
    turn(client, "CONFLICT5b-chain-ambiguous-rich-default", "what does the lion's prey eat?")

    # ==================================================================================================
    # battery #4 -- MOAT-BYPASS provenance: a derived answer must be framed GENERATED, not PERCEIVED,
    # with recalled_svo=null / derived=true / derived_from populated, and NOT reported verified as a
    # directly-recalled fact. Requires the optional #129 monitor ON for this one call.
    # ==================================================================================================
    turn(client, "PROV1-teach-hop1", "the eagle hunts the rabbit", rich=False)
    turn(client, "PROV2-teach-hop2", "the rabbit eats the clover", rich=False)
    turn(client, "PROV3-derive-with-provenance-on", "what does the eagle's prey eat?", rich=False,
         extra_env={"BRAIN_SOURCE_PROVENANCE_HONESTY": "1"})
    # a DIRECT (non-derived) recall with the monitor on too, for contrast (should stay PERCEIVED)
    turn(client, "PROV4-direct-recall-with-provenance-on", "what does the wolf hunt?", rich=False,
         extra_env={"BRAIN_SOURCE_PROVENANCE_HONESTY": "1"})
    turn(client, "PROV5-provenance-off-again", "what does the wolf hunt?", rich=False,
         extra_env={"BRAIN_SOURCE_PROVENANCE_HONESTY": None})

    # ==================================================================================================
    # (c) / LOAD-BEARING lesion: the same chain question, route disabled mid-session -> reverts to abstain;
    # restore -> re-confirm the route (not decoration) is what drives the derived answer
    # ==================================================================================================
    turn(client, "C1-lesion-on-reask", "what does the wolf's prey eat?", rich=False,
         extra_env={"BRAIN_CHAIN_ROUTE": "0"})
    turn(client, "C2-lesion-off-reask", "what does the wolf's prey eat?", rich=False,
         extra_env={"BRAIN_CHAIN_ROUTE": None})

    # ==================================================================================================
    # battery #10 -- FALSE-POSITIVE routing: a modifier-laden single-hop question must NOT be detoured
    # into the chain engine (reuses the pre-existing (cat,eat,fish) tiny-demo fact -- no apostrophe present)
    # ==================================================================================================
    turn(client, "FP1-modifier-laden-single-hop", "what does the big hungry cat eat?", rich=False)

    # ==================================================================================================
    # battery #11 -- SHARD-ROUTING lemmatization (noun side, plural agent): OUT OF SCOPE for this task
    # (audit req #7) -- expected to FAIL/abstain; run + report honestly.
    # ==================================================================================================
    turn(client, "SHARD1-plural-agent", "what do cats eat?", rich=False)

    # ==================================================================================================
    # battery #12 -- OVER-RUN / deep chain: a 3-hop shape the 2-hop regex cannot express -> must abstain,
    # not fabricate (reuses wolf/deer/grass)
    # ==================================================================================================
    turn(client, "OVERRUN1-three-hop-shape", "what does the wolf's prey's food eat?", rich=False)

    # ==================================================================================================
    # (e) continued -- framing faculties still behave on the SAME warm brain (regression, default rich path)
    # ==================================================================================================
    turn(client, "E3-affect-lead", "Wow, I absolutely love wolves, they are magnificent! What does the wolf hunt?")
    turn(client, "E4-swap-lead", "what does the dog chase?")

print("\n=== driver complete ===", flush=True)
with open(os.path.join(OUT_DIR, f"battery_{TAG}.json"), "w") as f:
    json.dump(records, f, indent=2, default=str)
print(f"wrote {len(records)} records to {JSONL} and the .json sibling", flush=True)
