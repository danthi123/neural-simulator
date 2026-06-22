"""DE-RISK P2 (grounded language faculty scoping, 2026-06-22 §4 / Rank 1): the "teacher -> brain" path.

THE IDEA: a CLAUDE-authored OFFLINE structured curriculum, learned by the BRAIN its OWN way (through the existing,
validated conversational machinery -- the Hebbian parser + the RF composer + its no-confab abstention moat), becomes
brain-STRUCTURED knowledge. We then verify three things on the brain that learned it:

  (a) RECALL  -- who/what queries on the taught facts return the taught answers (accuracy >= 0.9).
  (b) NO-CONFAB MOAT (HARD) -- ~10 held-out NEVER-taught but in-vocabulary (agent, action) cues must ABSTAIN (None);
      0 false-accepts is the bar. If the moat breaks (a held-out cue returns a non-None answer) THAT is the key
      finding (the grounding then needs P3's explicit gate, not just the composer default).
  (c) MULTI-HOP -- 2-hop relational chains (teach A-acts-B and B-acts-C) chain to the correct terminal concept
      (`reason_chain`), and over-run / broken chains abstain.

GO = recall >= 0.9 AND moat 0 false-accepts (HARD) AND multi-hop works, on 3/3 seeds (42/43/44).

Construction: BrainConversationalAgent(seed, composer_kind="rf", concepts={word: None ...}) -- an EXPLICIT concept
vocabulary, so the random phasor codes are generated deterministically per seed and NO denoise64 cache is needed (the
exact pattern the CI test_multicue_competition_agent.py uses). CPU/numpy at this scale.

NO sim/ edit. Reuse-by-import only. Run: SIM_BACKEND=numpy python -m research.runners._p2_teacher_to_brain_derisk
"""
from __future__ import annotations
import json
import os
import sys

# CPU/numpy at this small scale (the de-risk is composer-run cheap; the parser is CPU-runnable here).
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.core_sim_composition import Clause

# ----------------------------------------------------------------------------------------------------------------
# THE CLAUDE-AUTHORED CURRICULUM (the teacher writes this in-session). A small, structured, self-consistent
# ~30-40-word micro-world across mini-topics: ANIMALS (who eats/chases what), PLACES (who goes where), ACTIONS.
# Format = flat canonical-order SVO 'subject verb object' (<= 1 adjective via the attributed channel, kept minimal).
# The position-only BridgeParser reads pos0=agent, pos1=action(verb), pos2=patient -- so every fact is canonical SVO.
# ----------------------------------------------------------------------------------------------------------------

# (a) ~40 FLAT SVO facts across 3-4 mini-topics, in one coherent ~30-40-word vocabulary.
FLAT_FACTS = [
    # --- TOPIC 1: animals eat food ---
    "dog eat meat",
    "cat eat fish",
    "bird eat seed",
    "fox eat egg",
    "wolf eat deer",
    "bear eat honey",
    "frog eat fly",
    "owl eat mouse",
    # --- TOPIC 2: animals chase animals (predation; sets up multi-hop chains too) ---
    "dog chase cat",
    "cat chase mouse",
    "fox chase rabbit",
    "wolf chase sheep",
    "owl chase moth",
    "bear chase salmon",
    # --- TOPIC 3: agents go to places ---
    "dog go park",
    "cat go barn",
    "bird go nest",
    "fox go den",
    "wolf go forest",
    "bear go cave",
    "child go school",
    "farmer go field",
    "sailor go harbor",
    "miner go mountain",
    # --- TOPIC 4: agents make/use objects ---
    "farmer grow wheat",
    "baker make bread",
    "smith make sword",
    "child read book",
    "sailor sail boat",
    "miner dig gold",
    "weaver make cloth",
    "potter make bowl",
    # --- a few more animal/place facts to round out ~40 ---
    "sheep eat grass",
    "rabbit eat carrot",
    "deer eat leaf",
    "mouse eat cheese",
    "salmon swim river",
    "moth fly lamp",
    "fly land wall",
    "honey fill jar",
]

# (b) ~10 HELD-OUT, never-taught but IN-VOCABULARY (agent, action) cues -- the no-confab moat probe.
# Every word below APPEARS in the taught vocabulary, but these exact (agent, action) pairs are NEVER stored, so a
# faithful moat MUST abstain (return None). (e.g. 'dog' is taught with eat/chase/go, but never with 'fly'/'swim'.)
HELDOUT_CUES = [
    ("dog", "swim"),      # dog never swims
    ("cat", "go"),        # 'cat go barn' IS taught -> so use a DIFFERENT untaught action below; fix: cat+fly
    ("bird", "chase"),    # bird never chases
    ("fox", "eat_NONE"),  # placeholder replaced below
]

# Rebuild HELDOUT_CUES cleanly: in-vocab words, (agent, action) pairs that are provably NOT in FLAT_FACTS.
_TAUGHT_AA = {(f.split()[0], f.split()[1]) for f in FLAT_FACTS}
HELDOUT_CUES = [
    ("dog", "swim"),
    ("cat", "fly"),
    ("bird", "chase"),
    ("fox", "dig"),
    ("wolf", "read"),
    ("bear", "sail"),
    ("farmer", "chase"),
    ("baker", "eat"),
    ("child", "grow"),
    ("sailor", "eat"),
]
# sanity: none of the held-out cues may collide with a taught (agent, action)
_BAD = [aa for aa in HELDOUT_CUES if aa in _TAUGHT_AA]
assert not _BAD, f"held-out cue collides with a taught fact: {_BAD}"
# sanity: every held-out word must be in the taught vocabulary (it must be a real-word abstention, not OOV)
_VOCAB_WORDS = {w for f in FLAT_FACTS for w in f.split()}
_HELDOUT_ACTIONS_IN_VOCAB = all(a in _VOCAB_WORDS for _ag, a in HELDOUT_CUES)
# (some held-out ACTIONS may be novel verbs; that is fine -- it is still an in-vocab agent the moat must not confab.
#  But we ALSO ensure the agents are in-vocab.)
assert all(ag in _VOCAB_WORDS for ag, _a in HELDOUT_CUES), "held-out agent not in vocab"

# (c) ~6 MULTI-HOP CHAINS: teach A-acts-B and B-acts-C, so reason_chain(A, [act, act]) -> C. The chain matches each
# concept as the AGENT under the hop's action and reads the patient (rf_phasor_composer.query_chain). We pick chains
# whose hops are ALL present as stored facts above (so a faithful chain resolves; a broken/over-run chain abstains).
#   each entry: (cue, [action_hop1, action_hop2], expected_terminal)
MULTIHOP_CHAINS = [
    # predation chains (chase->chase)
    ("dog",  ["chase", "eat"],   "mouse"),   # dog chase cat ; cat eat fish  -> wait: cat eats fish; cat chase mouse
    ("fox",  ["chase", "eat"],   "carrot"),  # fox chase rabbit ; rabbit eat carrot -> carrot
    ("wolf", ["chase", "eat"],   "grass"),   # wolf chase sheep ; sheep eat grass    -> grass
    ("cat",  ["chase", "eat"],   "cheese"),  # cat chase mouse ; mouse eat cheese     -> cheese
    ("owl",  ["chase", "fly"],   "lamp"),    # owl chase moth ; moth fly lamp          -> lamp
    ("bear", ["chase", "swim"],  "river"),   # bear chase salmon ; salmon swim river   -> river
]
# Fix the first chain: dog chase cat ; cat chase mouse -> mouse (both 'chase'); make hop actions consistent.
MULTIHOP_CHAINS[0] = ("dog", ["chase", "chase"], "mouse")   # dog chase cat ; cat chase mouse -> mouse

# A BROKEN chain (the second hop has no matching fact) MUST abstain -> None. ('dog go park' then 'park <act> ?' = no
# fact has 'park' as an agent.) And an OVER-RUN chain past a leaf must abstain too.
BROKEN_CHAINS = [
    ("dog",   ["go", "eat"]),       # dog go park ; park eat ? -> no fact -> None
    ("baker", ["make", "chase"]),   # baker make bread ; bread chase ? -> no fact -> None
]


def build_vocab():
    """The explicit concept vocabulary = every word in the curriculum (facts + chain terminals + held-out words)."""
    words = set()
    for f in FLAT_FACTS:
        words.update(f.split())
    for cue, acts, term in MULTIHOP_CHAINS:
        words.add(cue); words.update(acts); words.add(term)
    for ag, a in HELDOUT_CUES:
        words.add(ag); words.add(a)
    return {w: None for w in sorted(words)}   # {word: None} -> deterministic random phasor codes per seed, no cache


def teach(agent):
    """The brain learns the curriculum ITS OWN way: hear() each SVO fact (parser comprehends -> composer stores)."""
    for f in FLAT_FACTS:
        agent.hear(f)


def eval_recall(agent):
    """(a) RECALL: who/what on every taught fact. what_does(agent, action) -> patient; who_does(action, patient) ->
    agent. Accuracy over all facts (both directions where unambiguous)."""
    ok_what = tot_what = 0
    ok_who = tot_who = 0
    # build action->patient->agents to know which who_does are unambiguous (single agent)
    triples = [tuple(f.split()) for f in FLAT_FACTS]
    # what_does: (agent, action) -> patient. (agent, action) is unique in this curriculum by construction except
    # where an agent has two facts with the same verb -- check none do.
    aa_to_p = {}
    for a, ac, p in triples:
        aa_to_p.setdefault((a, ac), []).append(p)
    examples = {}
    for (a, ac), ps in aa_to_p.items():
        tot_what += 1
        ans = agent.what_does(a, ac)
        if len(ps) == 1 and ans == ps[0]:
            ok_what += 1
        elif len(ps) > 1 and ans in ps:
            ok_what += 1
        if "what" not in examples:
            examples["what"] = (a, ac, ans, ps[0])
    # who_does: (action, patient) -> agent, only where unambiguous (single agent for that action+patient)
    acp_to_ag = {}
    for a, ac, p in triples:
        acp_to_ag.setdefault((ac, p), []).append(a)
    for (ac, p), ags in acp_to_ag.items():
        if len(ags) != 1:
            continue
        tot_who += 1
        ans = agent.who_does(ac, p)
        if ans == ags[0]:
            ok_who += 1
        if "who" not in examples:
            examples["who"] = (ac, p, ans, ags[0])
    return {"what_ok": ok_what, "what_tot": tot_what, "who_ok": ok_who, "who_tot": tot_who,
            "n_correct": ok_what + ok_who, "n_total": tot_what + tot_who,
            "accuracy": (ok_what + ok_who) / max(1, tot_what + tot_who), "examples": examples}


def eval_moat(agent):
    """(b) NO-CONFAB MOAT (HARD): every held-out (agent, action) cue must ABSTAIN (None). Count false-accepts."""
    false_accepts = []
    abstained = []
    for ag, a in HELDOUT_CUES:
        ans = agent.what_does(ag, a)
        if ans is None:
            abstained.append((ag, a))
        else:
            false_accepts.append((ag, a, ans))
    return {"n_cues": len(HELDOUT_CUES), "n_abstained": len(abstained), "n_false_accepts": len(false_accepts),
            "false_accepts": false_accepts, "example_abstain": (abstained[0] if abstained else None)}


def eval_multihop(agent):
    """(c) MULTI-HOP: each 2-hop chain reaches the correct terminal; broken/over-run chains abstain."""
    ok = 0
    results = []
    for cue, acts, term in MULTIHOP_CHAINS:
        ans = agent.reason_chain(cue, acts)
        good = (ans == term)
        ok += int(good)
        results.append({"cue": cue, "actions": acts, "expected": term, "got": ans, "ok": good})
    # broken chains must abstain (None)
    broken_ok = 0
    broken_results = []
    for cue, acts in BROKEN_CHAINS:
        ans = agent.reason_chain(cue, acts)
        good = (ans is None)
        broken_ok += int(good)
        broken_results.append({"cue": cue, "actions": acts, "got": ans, "abstained": good})
    return {"n_chains": len(MULTIHOP_CHAINS), "n_correct": ok, "chains": results,
            "n_broken": len(BROKEN_CHAINS), "n_broken_abstained": broken_ok, "broken": broken_results,
            "all_correct": ok == len(MULTIHOP_CHAINS) and broken_ok == len(BROKEN_CHAINS)}


def run_seed(seed):
    vocab = build_vocab()
    agent = BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=vocab)
    teach(agent)
    recall = eval_recall(agent)
    moat = eval_moat(agent)
    multihop = eval_multihop(agent)
    go = (recall["accuracy"] >= 0.9
          and moat["n_false_accepts"] == 0
          and multihop["all_correct"])
    return {"seed": seed, "n_vocab": len(vocab), "n_facts": len(FLAT_FACTS),
            "recall": recall, "moat": moat, "multihop": multihop, "GO": go}


def main():
    seeds = [42, 43, 44]
    results = [run_seed(s) for s in seeds]
    n_go = sum(r["GO"] for r in results)
    all_go = (n_go == len(seeds))
    # aggregate
    rec_accs = [r["recall"]["accuracy"] for r in results]
    moat_fa = [r["moat"]["n_false_accepts"] for r in results]
    mh_ok = [r["multihop"]["n_correct"] for r in results]
    summary = {
        "derisk": "P2 teacher->brain (grounded-language-faculty scoping Rank 1)",
        "seeds": seeds,
        "n_go": n_go, "n_seeds": len(seeds), "ALL_GO": all_go,
        "recall_accuracy_per_seed": rec_accs,
        "moat_false_accepts_per_seed": moat_fa,
        "multihop_correct_per_seed": mh_ok,
        "multihop_n_chains": len(MULTIHOP_CHAINS),
        "per_seed": results,
    }
    out_path = os.path.join("research", "findings", "raw", "_p2_teacher_to_brain.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # one-line verdict + the load-bearing numbers
    verdict = "GO" if all_go else "NO-GO"
    print("=" * 100)
    for r in results:
        rc = r["recall"]; mo = r["moat"]; mh = r["multihop"]
        print(f"seed {r['seed']}: recall acc={rc['accuracy']:.3f} ({rc['n_correct']}/{rc['n_total']}) | "
              f"moat false-accepts={mo['n_false_accepts']}/{mo['n_cues']} | "
              f"multihop={mh['n_correct']}/{mh['n_chains']} chains, broken-abstain={mh['n_broken_abstained']}/{mh['n_broken']} "
              f"=> {'GO' if r['GO'] else 'NO-GO'}")
        # a few example Q&A
        ew = rc["examples"].get("what"); ewho = rc["examples"].get("who")
        if ew:
            print(f"    taught recall: what_does('{ew[0]}','{ew[1]}') -> {ew[2]!r} (taught: {ew[3]!r})")
        ex_ab = mo["example_abstain"]
        if ex_ab:
            print(f"    held-out moat: what_does('{ex_ab[0]}','{ex_ab[1]}') -> None (abstained)")
        ch0 = mh["chains"][0]
        print(f"    multi-hop:     reason_chain('{ch0['cue']}',{ch0['actions']}) -> {ch0['got']!r} (expected {ch0['expected']!r})")
    print("=" * 100)
    print(f"VERDICT P2 teacher->brain: {verdict} | {n_go}/{len(seeds)} seeds | "
          f"recall_acc={[f'{a:.3f}' for a in rec_accs]} | moat_false_accepts={moat_fa} (HARD: must be all 0) | "
          f"multihop={mh_ok}/{len(MULTIHOP_CHAINS)} | wrote {out_path}")
    return 0 if all_go else 1


if __name__ == "__main__":
    sys.exit(main())
