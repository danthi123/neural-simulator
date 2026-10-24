"""Why the 125 stub-mouth HURTS (stub_population_exhaustive.json) are rejected: for every HURTS fact, the claim gate's verdict on the stub sentence
with the flag OFF and ON (the stock claim verifier vs B2h), the verifier's own clause trace verdict, and whether the sentence's relation word is
a stub inflection ('countrys', 'sports', 'subclass_ofs', ...).  If stock-claim-gate == B2h on all of them the HURTS come from ROUTING a one-fact turn
through the claim gate (whose substrate role parse returns no agent for the clause), not from the slug normalisation.
usage (cwd = worktree, as a module): python -m research.findings.raw._moat_slug_norm_heldout.stub_hurts_cause <out.json>"""
import collections, json, os, socket, sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())
import research.runners  # noqa: F401,E402  (provenance door)
from research.runners.brain_chat_tui import ChatBrain, StubRenderer           # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser          # noqa: E402

H = "research/findings/raw/_moat_slug_norm_heldout/"
d = json.load(open(H + "stub_population_exhaustive.json"))
chat = ChatBrain.__new__(ChatBrain); chat.inner = BridgeParser(seed=d["parser_seed"]); stub = StubRenderer()
rows = []; verdicts = collections.Counter(); same = 0
for h in d["hurts_rows"]:
    svo = h["svo"]; surf, _ = stub.render_svo(*svo)
    os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
    a0, r0 = chat._verify_claim_set(surf, [svo])
    os.environ["BRAIN_MOAT_SLUG_NORM"] = "1"
    a1, r1 = chat._verify_claim_set(surf, [svo])
    os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
    tr = (r1 or {}).get("trace") or [{}]
    v = tr[0].get("verdict")
    verdicts[v] += 1
    same += (a0 == a1)
    rows.append({"svo": svo, "surface": surf, "stock_claim_gate": a0, "b2h_claim_gate": a1, "clause_verdict": v,
                 "relation_word_in_sentence": surf.split()[2] if len(surf.split()) > 2 else None})
out = {"_doc": "cause of the stub-mouth HURTS (module docstring)", "backend": "numpy", "device": "cpu", "host": socket.gethostname(),
       "script": H + "stub_hurts_cause.py", "argv": sys.argv, "n_hurts": len(rows),
       "stock_claim_gate_equals_b2h_on_all": same == len(rows), "stock_claim_gate_accepts": sum(1 for r in rows if r["stock_claim_gate"]),
       "b2h_claim_gate_accepts": sum(1 for r in rows if r["b2h_claim_gate"]), "clause_verdicts": dict(verdicts), "rows": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
