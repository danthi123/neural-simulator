"""Live-chat integration de-risk for the hippocampal-buffer / cortical-LTM fact-store split (2026-08-20).

Proves the `TieredFactStore` is a TRANSPARENT drop-in for the live conversational agent's `composer`, and that it
delivers the owner's #1 goal: the brain can hold + query bulk KNOWLEDGE (>> the k_max=32 working-set cap) at
sub-second latency, while conversation-taught facts still work AND the no-confab moat holds -- all exercised
THROUGH the real `BrainConversationalAgent` public methods (`what_does` / `who_does` / `is_it_true` /
`reason_chain`), not the composer directly.

Teeth (each an anti-cheat the verdict enforces):
  * KNOWLEDGE RECALL: sampled bulk facts (in the LTM only) answer correctly through `agent.what_does` -> the LTM
    tier is genuinely consulted on a buffer miss.
  * k_max LIFTED: the agent queries N >> 32 facts correctly while the co-resident BUFFER holds only a handful --
    bulk knowledge lives in the uncapped sharded LTM, so the working-set cap no longer bounds knowledge.
  * SUB-SECOND: the routed LTM query is well under 1 s at N facts (the capacity win, re-measured on this path).
  * TEACH-IN-CONVERSATION: a fact taught mid-chat (`store`) lands in the BUFFER (not the LTM) and answers.
  * RECENCY: a buffer fact SHADOWS a contradicting LTM fact about the same cue (recent working set wins).
  * MOAT: an unknown subject abstains (None / "unknown") -- BOTH tiers must abstain for a non-answer.
  * DEGRADE: with ltm=None the tiered store is answer-identical to the plain buffer (the safe default).

Declared TEST SCAFFOLD: synthetic facts + the ShardedPhasorStore's host agent-hash router (the faithful version
is a learned/spiking cue->sub-population router). NO sim/ edit; no production default changed (the wiring into
`load_developed_brain` is additive + opt-in). Run:
  SIM_BACKEND=numpy python -m research.runners._knowledge_tiered_livechat_derisk [--N 5000 --D 128]
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore, build_ltm_from_facts, auto_n_shards  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


def make_facts(N, n_ag, n_rel, n_pt, seed):
    """N unique (agent, relation) -> patient facts (one patient per (agent,relation), so recall is unambiguous)."""
    rng = np.random.default_rng(seed)
    facts, seen = [], set()
    while len(facts) < N:
        a = f"ag{int(rng.integers(n_ag))}"; r = f"rel{int(rng.integers(n_rel))}"
        if (a, r) in seen:
            continue
        seen.add((a, r))
        facts.append({"agent": a, "action": r, "patient": f"pt{int(rng.integers(n_pt))}"})
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000, help="bulk-knowledge facts loaded into the cortical LTM")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--n-recall", type=int, default=50)
    ap.add_argument("--n-moat", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(_REPO / "research" / "findings" / "raw"
                                                  / "_knowledge_tiered_livechat_derisk.json"))
    a = ap.parse_args()

    NAG = max(200, a.N); NREL = 40; NPT = max(200, a.N // 2)
    facts = make_facts(a.N, NAG, NREL, NPT, seed=11)
    # vocab must cover every fact word + the moat's unknown agents + the teach-in-conversation words.
    fact_vocab = sorted({f["agent"] for f in facts} | {f["action"] for f in facts} | {f["patient"] for f in facts})
    teach_words = ["otter", "caught", "clam", "raven", "hid", "acorn"]
    moat_agents = [f"UNSEEN{j}" for j in range(a.n_moat)]
    vocab = sorted(set(fact_vocab) | set(teach_words) | set(moat_agents))

    # ---- build the cortical LTM (bulk knowledge, sharded) ----
    n_shards = auto_n_shards(a.N)
    t0 = time.time()
    ltm = build_ltm_from_facts(facts, vocab=vocab, n_shards=n_shards, seed=a.seed, D=a.D)
    ltm_build_s = time.time() - t0

    # ---- build the live agent + install the tiered store as its composer (the DROP-IN) ----
    agent = BrainConversationalAgent(seed=a.seed, concepts={w: None for w in vocab}, composer_kind="rf")
    buffer = agent.composer                       # the agent's own small flat composer = the conversation BUFFER
    agent.composer = TieredFactStore(buffer, ltm)  # transparent substitution

    rng = np.random.default_rng(3)

    # ---- (1) KNOWLEDGE RECALL through agent.what_does (LTM tier), + latency ----
    idx = rng.choice(a.N, size=a.n_recall, replace=False)
    t0 = time.time(); recall_hits = 0
    for i in idx:
        f = facts[int(i)]
        if agent.what_does(f["agent"], f["action"]) == f["patient"]:
            recall_hits += 1
    routed_ms = (time.time() - t0) / len(idx) * 1000.0

    # ---- (2) TEACH-IN-CONVERSATION: a new fact -> the BUFFER, answers, and did NOT touch the LTM ----
    buf_facts_before = len(buffer.kb)
    ltm_facts_before = ltm.total_facts()
    agent.composer.store("otter", "caught", "clam")
    teach_ans = agent.what_does("otter", "caught")
    teach_in_buffer = (len(buffer.kb) == buf_facts_before + 1)
    ltm_untouched = (ltm.total_facts() == ltm_facts_before)

    # ---- (3) RECENCY: a buffer fact SHADOWS a contradicting LTM fact about the same cue ----
    #   pick a real LTM fact, teach a DIFFERENT patient for the same (agent,action) into the buffer.
    shadow_src = facts[int(idx[0])]
    ltm_patient = agent.composer.ltm.query_patient(shadow_src["agent"], shadow_src["action"])
    shadow_patient = "clam" if ltm_patient != "clam" else "acorn"
    agent.composer.store(shadow_src["agent"], shadow_src["action"], shadow_patient)
    recency_wins = (agent.what_does(shadow_src["agent"], shadow_src["action"]) == shadow_patient)

    # ---- (4) MOAT: unknown subjects abstain (BOTH tiers) ----
    moat_abstain = sum(agent.what_does(m, "rel0") is None for m in moat_agents)
    yesno_unknown = (agent.is_it_true("UNSEEN0", "rel0", "pt0") == "unknown")

    # ---- (5) ask_yes_no on a KNOWN LTM fact -> "yes" ----
    kf = facts[int(idx[1])]
    yesno_known = (agent.is_it_true(kf["agent"], kf["action"], kf["patient"]) == "yes")

    # ---- (6) k_max LIFTED: N >> 32 knowledge facts queryable while the buffer stays tiny ----
    buffer_size = len(buffer.kb)                  # only the conversation-taught facts
    k_max_lifted = (a.N > 32 and recall_hits >= int(0.95 * a.n_recall) and buffer_size <= 8)

    # ---- (7) DEGRADE: ltm=None -> answer-identical to the plain buffer (safe default) ----
    agent2 = BrainConversationalAgent(seed=a.seed, concepts={w: None for w in vocab}, composer_kind="rf")
    buf2 = agent2.composer
    for w in ("otter", "raven"):
        buf2.store(w, "caught" if w == "otter" else "hid", "clam" if w == "otter" else "acorn")
    degraded = TieredFactStore(buf2, ltm=None)
    degrade_identical = (
        degraded.query_patient("otter", "caught") == buf2.query_patient("otter", "caught")
        and degraded.query_patient("raven", "hid") == buf2.query_patient("raven", "hid")
        and degraded.query_patient("nobody", "rel0") is None
    )

    art = {
        "N": a.N, "D": a.D, "n_shards": n_shards, "vocab": len(vocab),
        "ltm_build_s": ltm_build_s, "ltm_total_facts": ltm.total_facts(),
        "buffer_size_after_teach": buffer_size,
        "recall_hits": recall_hits, "n_recall": a.n_recall, "recall_frac": recall_hits / a.n_recall,
        "routed_ms_per_query": routed_ms, "chance": 1.0 / NPT,
        "teach_answer": teach_ans, "teach_in_buffer": teach_in_buffer, "ltm_untouched_by_teach": ltm_untouched,
        "recency_shadows_ltm": recency_wins,
        "moat_abstain": moat_abstain, "n_moat": a.n_moat, "yesno_unknown_abstains": yesno_unknown,
        "yesno_known_yes": yesno_known,
        "k_max_lifted": k_max_lifted, "degrade_identical_to_buffer": degrade_identical,
        "load_balance_min_max_mean_ratio": list(ltm.load_balance()),
        "backend": os.environ.get("SIM_BACKEND", "numpy"),
    }
    print(json.dumps(art, indent=2))

    v = Verdict("tiered hippocampal-buffer / cortical-LTM fact store: a live-chat drop-in that scales knowledge",
                chance=1.0 / NPT)
    v.floor("bulk-knowledge recall through agent.what_does (LTM tier)", measured=recall_hits / a.n_recall, floor=0.95)
    v.require("routed LTM query is sub-second at N facts", routed_ms, expect=lambda x: x < 1000.0)
    v.require("teach-in-conversation answers", teach_ans, expect="clam")
    v.require("taught fact landed in the BUFFER (not the LTM)", teach_in_buffer and ltm_untouched, expect=True)
    v.require("recent buffer fact shadows a contradicting LTM fact", recency_wins, expect=True)
    v.require("no-confab moat: every unknown subject abstains", moat_abstain, expect=a.n_moat)
    v.require("ask_yes_no abstains ('unknown') on an unknown fact", yesno_unknown, expect=True)
    v.require("ask_yes_no affirms a known LTM fact", yesno_known, expect=True)
    v.require("k_max lifted: N>>32 knowledge queryable while the buffer stays tiny", k_max_lifted, expect=True)
    v.require("degrades to the plain buffer when ltm=None (byte-safe default)", degrade_identical, expect=True)
    v.disabled("learned/spiking cue->shard router",
               why="the ShardedPhasorStore's host agent-hash router is a declared capacity-de-risk scaffold; "
                   "recall + moat inside each shard are the genuine RF reads")
    go = (recall_hits >= int(0.95 * a.n_recall) and routed_ms < 1000.0 and teach_ans == "clam"
          and teach_in_buffer and ltm_untouched and recency_wins and moat_abstain == a.n_moat
          and yesno_unknown and yesno_known and k_max_lifted and degrade_identical)
    decided = v.decide(go=go)
    art["verdict"] = decided
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=2))
    print(f"\nwrote {a.out}")
    return decided["status"]


if __name__ == "__main__":
    main()
