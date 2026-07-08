"""PP (spatial/directional) relation store: "the owl flies TO the pond" = fly(agent=owl, GOAL=pond) vs "the
owl flies ON the rock" = fly(agent=owl, LOCATION=rock). A 4-role FHRR store (AGENT/VERB/GOAL/LOCATION; each
fact uses the goal OR the location role) recovers the destination and DISTINGUISHES goal from location -- so
the brain can DISCUSS spatial relations. Production = EMERGE-72 C_PPGOAL/C_PPLOC on spikes. Completes the
relational schema breadth (SVO + ditransitive + PP). numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np
from research.runners._realcorpus_svo_compose_probe import _phasors, _role
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi
MATCH_MARGIN = 0.25


class PPStore:
    """A spatial-relation store: AGENT/VERB + one of GOAL/LOCATION per fact (to-goal vs on-location)."""

    def __init__(self, Z, rows, roles):
        self.Z, self.rows = Z, rows
        self.AGENT, self.VERB, self.GOAL, self.LOC = roles
        self.facts = []

    def store(self, s, v, dest, kind):                     # kind in {"goal","loc"}
        role = self.GOAL if kind == "goal" else self.LOC
        self.facts.append(self.AGENT * self.Z[s] + self.VERB * self.Z[v] + role * self.Z[dest])

    def _cleanup(self, est):
        sims = np.array([np.real(np.vdot(self.Z[r], est)) / len(est) for r in self.rows])
        j = int(np.argmax(sims)); return self.rows[j], float(sims[j])

    def answer(self, s, v, kind):                          # where does the <s> <v> to/on? -> the dest of that KIND
        role = self.GOAL if kind == "goal" else self.LOC
        best, best_m = None, MATCH_MARGIN
        for f in self.facts:
            a, ma = self._cleanup(f * np.conj(self.AGENT))
            vb, mv = self._cleanup(f * np.conj(self.VERB))
            if a == s and vb == v and min(ma, mv) > MATCH_MARGIN:
                d, md = self._cleanup(f * np.conj(role))
                if md > best_m:
                    best, best_m = d, md
        return best


def run_seed(seed, stories, K, n_facts=10):
    vocab, gfreq = discover_vocab(stories, K); target = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target or len(w) < MIN_WORD_LEN: continue
        hubs.append(w)
        if len(hubs) >= N_HUB: break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed); rows = list(range(len(vocab)))
    Z = _phasors(codes, rows, seed)
    roles = (_role(rng), _role(rng), _role(rng), _role(rng))    # AGENT/VERB/GOAL/LOC
    idx = list(rows); rng.shuffle(idx)
    # half goal-facts, half location-facts; each (s,v,dest,kind)
    facts = []
    for i in range(0, n_facts * 3, 3):
        kind = "goal" if (i // 3) % 2 == 0 else "loc"
        facts.append((idx[i], idx[i + 1], idx[i + 2], kind))
    store = PPStore(Z, rows, roles)
    for (s, v, d, k) in facts: store.store(s, v, d, k)
    # recover the correct destination for the fact's own kind
    acc = sum(int(store.answer(s, v, k) == d) for (s, v, d, k) in facts) / len(facts)
    # DISCRIMINATION: querying the WRONG kind (goal on a loc-fact, or loc on a goal-fact) must MISS/abstain
    disc = sum(int(store.answer(s, v, "loc" if k == "goal" else "goal") != d) for (s, v, d, k) in facts) / len(facts)
    # MOAT: an unstored (agent, verb) -> abstain
    used = {(s, v) for (s, v, d, k) in facts}; unstored, tries = [], 0
    while len(unstored) < len(facts) and tries < 800:
        s, v = int(rng.choice(rows)), int(rng.choice(rows))
        if (s, v) not in used: unstored.append((s, v))
        tries += 1
    moat = sum(int(store.answer(s, v, "goal") is None and store.answer(s, v, "loc") is None) for (s, v) in unstored) / max(1, len(unstored))
    return {"seed": seed, "n_facts": len(facts), "n_vocab": len(vocab), "chance": 1.0 / len(vocab),
            "answer_acc": acc, "goal_loc_discrim": disc, "moat_abstain": moat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256); ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-facts", type=int, default=10)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[PP relation store] agent-verb-{{goal|location}}: fly-TO vs fly-ON | K={a.K}", flush=True)
    recs = [run_seed(s, stories, a.K, a.n_facts) for s in seeds]
    for r in recs:
        print(f"  [seed {r['seed']}] answer-acc={r['answer_acc']:.3f} goal/loc-discrim={r['goal_loc_discrim']:.3f} "
              f"moat={r['moat_abstain']:.3f} (chance {r['chance']:.3f})", flush=True)
    def m(k): return float(np.mean([r[k] for r in recs]))
    go = all(r["answer_acc"] >= 0.9 and r["goal_loc_discrim"] >= 0.9 and r["moat_abstain"] >= 0.9 for r in recs)
    print(f"\n  AGGREGATE: answer={m('answer_acc'):.3f} goal/loc-discrim={m('goal_loc_discrim'):.3f} moat={m('moat_abstain'):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL'} -- a 4-role FHRR store recovers PP spatial relations AND distinguishes "
          f"GOAL (fly-to) from LOCATION (fly-on), abstaining on the unstored; the brain can DISCUSS spatial relations "
          f"(production = EMERGE-72 C_PPGOAL/C_PPLOC on spikes). Schema breadth: SVO + ditransitive + PP.", flush=True)


if __name__ == "__main__":
    main()
