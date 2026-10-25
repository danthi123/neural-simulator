"""Ablation (NOT the bar): B2h with the article/relation-name guard switched off, on the held-out corpus. If sentences labelled U
are kept without the guard, the guard is doing out-of-sample work; if none are, the held-out set did not exercise it.
usage (cwd = worktree): ablate_guard.py <corpus.json> <labels.json> <out.json>"""
import collections, json, os, sys
os.environ["SIM_BACKEND"] = "numpy"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique
from research.runners import _moat_claim_entailment_derisk as M
from research.runners.brain_chat_tui import ChatBrain
from research.runners.brain_conversational_agent import BridgeParser
corpus = json.load(open(sys.argv[1])); labels = json.load(open(sys.argv[2]))
lab = {(e["turn"], tuple(e["svo"]), e["surface"]): e["label"] for e in labels["labels"]}
gated = {t["label"]: t["gated"] for t in corpus["turns"]}
os.environ["BRAIN_MOAT_SLUG_NORM"] = "1"
res = {}
for name, patch in (("b2h", None), ("b2h_without_article_relation_guard", lambda self, prose: None)):
    orig = M.ClaimEntailmentVerifier._article_relation_guard
    if patch: M.ClaimEntailmentVerifier._article_relation_guard = patch
    chat = ChatBrain.__new__(ChatBrain); chat.inner = BridgeParser(seed=42)
    kept = collections.Counter(); leaked = []
    for u in unique(corpus):
        acc, _ = chat._verify_claim_set(u["surface"], gated[u["turn"]])
        if acc:
            l = lab[(u["turn"], tuple(u["svo"]), u["surface"])]; kept[l] += 1
            if l in "UN": leaked.append(u["surface"])
    M.ClaimEntailmentVerifier._article_relation_guard = orig
    res[name] = {"kept_by_label": dict(kept), "unsupported_kept": leaked}
    print(name, dict(kept), leaked)
json.dump({"backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/ablate_guard.py",
           "argv": sys.argv, "seed": 42, "results": res}, open(sys.argv[3], "w"), indent=1)
