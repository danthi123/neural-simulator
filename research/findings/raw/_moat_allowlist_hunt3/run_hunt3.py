"""Round-3 fresh leak hunt (prereg 2026-10-20-moat-allowlist-claim-checker, bar 3): run every committed case through the production
claim gate (ChatBrain._verify_claim_set) and the composer entry (RichAnswerComposer._verify_rendered) for three arms: stock (flags
off), b2h (BRAIN_MOAT_SLUG_NORM=1) and allowlist (BRAIN_MOAT_ALLOWLIST=1), at parser seeds 42, 43, 44, 100, 101, 102 and 1 (one
deterministic build: one replicate per case). Cases and the label adjudication were committed before this script ran.
usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt3.run_hunt3 <out.json>"""
import json, os, socket, sys

import research.runners  # noqa: F401  (provenance door)
from research.runners.brain_chat_tui import ChatBrain
from research.runners.brain_conversational_agent import BridgeParser
from research.runners.rich_answer_composer import RichAnswerComposer

D = "research/findings/raw/_moat_allowlist_hunt3/"
SEEDS = (42, 43, 44, 100, 101, 102, 1)
ARMS = {"stock": {}, "b2h": {"BRAIN_MOAT_SLUG_NORM": "1"}, "allowlist": {"BRAIN_MOAT_ALLOWLIST": "1"}}
FLAGS = ("BRAIN_MOAT_SLUG_NORM", "BRAIN_MOAT_ALLOWLIST", "BRAIN_CLAIM_MOAT")


def main(out):
    cases = json.load(open(D + "cases.json"))
    adj = json.load(open(D + "adjudication.json"))
    label = {c["id"]: ("B" if c["id"] in adj["relabel_to_B"] else c["expect"]) for c in cases}
    chats = {}
    for sd in SEEDS:
        ch = ChatBrain.__new__(ChatBrain)
        ch.inner = BridgeParser(seed=sd)
        chats[sd] = ch
    rows = []
    for c in cases:
        g = c["gathered"]
        row = {"id": c["id"], "class": c["class"], "label": label[c["id"]], "sentence": c["sentence"], "gate": {}, "composer": {}}
        for arm, env in ARMS.items():
            for k in FLAGS:
                os.environ.pop(k, None)
            os.environ.update(env)
            try:
                gate, comp = [], []
                for sd in SEEDS:
                    acc, res = chats[sd]._verify_claim_set(c["sentence"], g)
                    gate.append(acc)
                    rc = RichAnswerComposer.__new__(RichAnswerComposer)
                    rc.chat = chats[sd]
                    try:
                        comp.append(bool(rc._verify_rendered(c["sentence"], c["sentence"], tuple(g[0]), g)))
                    except Exception as e:  # recorded, never a pass
                        comp.append("ERROR:%s" % type(e).__name__)
                row["gate"][arm] = gate
                row["composer"][arm] = comp
            finally:
                for k in FLAGS:
                    os.environ.pop(k, None)
        rows.append(row)
    summ = {}
    for arm in ARMS:
        kept = lambda r, where: any(v is True for v in r[where][arm])
        s = {}
        for lab in ("U", "B", "F"):
            rs = [r for r in rows if r["label"] == lab]
            s[lab] = {"n": len(rs), "kept_gate_any_seed": sum(kept(r, "gate") for r in rs),
                      "kept_composer_any_seed": sum(kept(r, "composer") for r in rs),
                      "kept_ids": sorted(r["id"] for r in rs if kept(r, "gate") or kept(r, "composer"))}
        s["errors"] = sum(1 for r in rows for v in r["composer"][arm] if isinstance(v, str))
        summ[arm] = s
    json.dump({"_doc": __doc__, "host": socket.gethostname(), "argv": sys.argv, "seeds": SEEDS,
               "seeds_note": "parser seeds of one deterministic build: one replicate per case, not draws",
               "summary": summ, "rows": rows}, open(out, "w"), indent=1)
    for arm, s in summ.items():
        print(arm, {k: (v if k == "errors" else {kk: vv for kk, vv in v.items() if kk != "kept_ids"}) for k, v in s.items()})


if __name__ == "__main__":
    main(sys.argv[1])
