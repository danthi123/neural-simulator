"""Which verifier stage stops each labelled held-out sentence under B2h (BRAIN_MOAT_SLUG_NORM=1, seed 42)?  Supplementary to
results.json: shows how much of the 0-unsupported-kept result is the lexical stage (unknown content word) versus the later
proposition/structure/guard stages, i.e. how much the held-out corpus actually exercised the guards.
usage (cwd = worktree): stage_analysis.py <corpus.json> <labels.json> <out.json>"""
import collections, json, os, sys
os.environ["SIM_BACKEND"] = "numpy"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique
from research.runners.brain_chat_tui import ChatBrain
from research.runners.brain_conversational_agent import BridgeParser
corpus = json.load(open(sys.argv[1])); labels = json.load(open(sys.argv[2]))
lab = {(e["turn"], tuple(e["svo"]), e["surface"]): e["label"] for e in labels["labels"]}
os.environ["BRAIN_MOAT_SLUG_NORM"] = "1"
chat = ChatBrain.__new__(ChatBrain); chat.inner = BridgeParser(seed=42)
gated = {t["label"]: t["gated"] for t in corpus["turns"]}
stage = collections.defaultdict(collections.Counter); examples = collections.defaultdict(list)
def bucket(why, acc):
    if acc: return "KEPT"
    if acc is None: return "unbuildable verifier (role-permutation collision): falls back to single-triple _verify, flag-independent"
    if why is None: return "?"
    if "unrepresentable" in why: return "lexical: unknown content word"
    if "common noun after an article" in why: return "guard: relation word as common noun after an article"
    if "not entailed" in why: return "proposition: asserted triple not in the gathered set"
    if "negat" in why: return "negation"
    return "structure: " + why[:60]
for u in unique(corpus):
    acc, res = chat._verify_claim_set(u["surface"], gated[u["turn"]])
    b = bucket((res or {}).get("reject_reason"), acc)
    l = lab[(u["turn"], tuple(u["svo"]), u["surface"])]
    stage[l][b] += 1
    if not b.startswith("lexical") and b != "KEPT" and l in "UN": examples[b].append(u["surface"])
out = {"backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/stage_analysis.py", "argv": sys.argv,
       "seed": 42, "stage_by_label": {l: dict(c) for l, c in stage.items()},
       "U_N_rows_stopped_after_the_lexical_stage": dict(examples)}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for l in "FBUN":
    print(l, dict(stage[l]))
for b, ex in examples.items():
    print("\nU/N stopped by", b, len(ex))
    for e in ex: print("   ", e)
