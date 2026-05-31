"""THROWAWAY cheap-first (CPU/numpy): does the validated bind support a small KNOWLEDGE BASE
of subject/verb/object FACTS + RELATIONAL QUERIES? This is the gate for the next arc (use the
spiking bind toward conversation). A 2-fact SVO KB = ~6 role-filler bindings = exactly the K=6
capacity the firing-rate lever just unlocked in the spiking substrate.

A fact "dog chases cat" = agent (x) dog + action (x) chase + patient (x) cat (3 bindings).
Store N facts. Query types:
  (1) single-fact role query: "who is the agent?" -> unbind agent + cleanup. (the validated bind.)
  (2) RELATIONAL (find-by-cue, read-other-role): "what does dog chase?" = find the fact whose
      agent is dog, return its patient. Two architectures tested:
        (A) SEPARATE facts (a list of bound vectors): iterate, unbind agent, match cue, read patient.
        (B) SUPERPOSED facts (one summed vector): the hard case (the 2026-05-31 multi-hop arc found
            relational chaining over superposition DEGRADES-WITH-FANIN). Tested honestly.
  (3) two-role-match relational: "what is the action between dog and cat?" (match agent AND patient).

Fillers = real substrate concept codes (denoise64, between-cos ~0.70). Roles agent/action/patient
= distinct distributed +-1. Cleanup = nearest concept.

FROZEN: single-fact >= 0.90 AND separate-facts relational (A) >= 0.90 -> RESOLVES (the bind is a
usable relational fact-memory; build the spiking version). Report (B) superposed + control
honestly (do NOT require B). stdlib+numpy + cache; no protected import.

RESULT 2026-05-31 (seeds 42/43/44): RESOLVES. single=1.000, relational-A(separate)=1.000,
two-role=1.000, control(no-false-match)=1.000 -- all multi-seed. relational-B(superposed)=0.475
DEGRADES (expected -- the multi-hop wall; separate-fact storage is the correct architecture).
The bind is a usable structured FACT-MEMORY with cue-based retrieval (find-by-role, read-other-
role); NOT relational reasoning over superposition. Build the spiking version next.
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
SEEDS = [42, 43, 44]
ROLES = ["agent", "action", "patient"]
N_TRIALS = 40


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _center(d["obs__" + w].mean(axis=0)) for w in ws}


def make_roles(D, rng):
    R = {}
    for name in ROLES:
        r = rng.choice([-1.0, 1.0], size=D)
        R[name] = r / np.linalg.norm(r)
    return R


def bind_fact(fact, concepts, roles):
    D = next(iter(concepts.values())).shape[0]
    S = np.zeros(D)
    for role in ROLES:
        S = S + roles[role] * concepts[fact[role]]
    return S


def cleanup(est, concepts, words):
    sims = np.array([concepts[w] @ est for w in words])
    return words[int(np.argmax(sims))]


def main():
    seeds = [s for s in SEEDS if os.path.exists(CACHE % s)]
    if not seeds:
        print("CANNOT-CONCLUDE (no caches)"); return
    print("=== VSA relational query (SVO fact base) cheap-first ===")
    print(f"roles={ROLES}; fillers=real substrate codes; N_TRIALS={N_TRIALS}\n")

    single_acc, relA_acc, relB_acc, rel2_acc, ctrl_acc = [], [], [], [], []
    for seed in seeds:
        words, concepts = load_concepts(seed)
        D = concepts[words[0]].shape[0]
        rng = np.random.default_rng(seed)
        roles = make_roles(D, rng)

        s_ok = rA_ok = rB_ok = r2_ok = ctrl_ok = tot = 0
        for _ in range(N_TRIALS):
            n_facts = 2                                   # 2-fact KB = ~K=6 bindings
            # distinct agents so the cue is unambiguous; distinct everything for clarity
            picks = rng.choice(len(words), size=3 * n_facts, replace=False)
            facts = []
            for f in range(n_facts):
                facts.append({"agent": words[picks[3 * f]], "action": words[picks[3 * f + 1]],
                              "patient": words[picks[3 * f + 2]]})
            bound = [bind_fact(fc, concepts, roles) for fc in facts]

            # (1) single-fact role query on a random fact + role
            qf = rng.integers(n_facts); qrole = ROLES[rng.integers(3)]
            est = bound[qf] * roles[qrole]
            s_ok += int(cleanup(est, concepts, words) == facts[qf][qrole])

            # (2A) SEPARATE relational: "what does <agent> <action>?" -> find fact by agent, read patient
            tf = rng.integers(n_facts); cue_agent = facts[tf]["agent"]
            # find the stored fact whose unbound-agent matches the cue
            best = None
            for f in range(n_facts):
                a = cleanup(bound[f] * roles["agent"], concepts, words)
                if a == cue_agent:
                    best = f; break
            ans = cleanup(bound[best] * roles["patient"], concepts, words) if best is not None else None
            rA_ok += int(ans == facts[tf]["patient"])

            # (2B) SUPERPOSED relational: sum all facts, try to read patient given agent cue (hard)
            total = np.sum(bound, axis=0)
            # naive: unbind patient from the superposition (no way to use the agent cue) -> ambiguous
            ansB = cleanup(total * roles["patient"], concepts, words)
            rB_ok += int(ansB == facts[tf]["patient"])

            # (3) two-role-match relational: "action between <agent> and <patient>?" (separate facts)
            tf2 = rng.integers(n_facts)
            ca, cp = facts[tf2]["agent"], facts[tf2]["patient"]
            best2 = None
            for f in range(n_facts):
                a = cleanup(bound[f] * roles["agent"], concepts, words)
                p = cleanup(bound[f] * roles["patient"], concepts, words)
                if a == ca and p == cp:
                    best2 = f; break
            ans2 = cleanup(bound[best2] * roles["action"], concepts, words) if best2 is not None else None
            r2_ok += int(ans2 == facts[tf2]["action"])

            # CONTROL: relational query with a cue agent NOT in any fact -> should NOT return the target patient
            non = [w for w in words if w not in [fc["agent"] for fc in facts]]
            cue_bad = str(rng.choice(non))
            bestc = None
            for f in range(n_facts):
                if cleanup(bound[f] * roles["agent"], concepts, words) == cue_bad:
                    bestc = f; break
            ctrl_ok += int(bestc is None)        # correct control behavior = no false match
            tot += 1

        single_acc.append(s_ok / tot); relA_acc.append(rA_ok / tot)
        relB_acc.append(rB_ok / tot); rel2_acc.append(r2_ok / tot); ctrl_acc.append(ctrl_ok / tot)
        print(f"  seed {seed}: single={s_ok/tot:.3f}  relational-A(separate)={rA_ok/tot:.3f}  "
              f"relational-B(superposed)={rB_ok/tot:.3f}  two-role={r2_ok/tot:.3f}  "
              f"control-no-false-match={ctrl_ok/tot:.3f}")

    S, A, B, R2, C = (np.mean(x) for x in (single_acc, relA_acc, relB_acc, rel2_acc, ctrl_acc))
    chance = 1.0 / len(load_concepts(seeds[0])[0])
    print(f"\nMEAN: single={S:.3f}  relational-A={A:.3f}  relational-B(superposed)={B:.3f}  "
          f"two-role={R2:.3f}  control={C:.3f}  (chance={chance:.3f})")
    if S >= 0.90 and A >= 0.90:
        print("VERDICT: RESOLVES -- the bind is a usable RELATIONAL FACT-MEMORY (separate facts + "
              "cue-based retrieval); single-fact + relational-A both >= 0.90. Build the spiking version.")
        print(f"  (superposed-B {'works' if B >= 0.90 else 'DEGRADES (expected -- the multi-hop wall; '
              'separate-fact storage is the right architecture)'}.)")
    else:
        print(f"VERDICT: BOUNDARY/needs-work -- single={S:.2f} relational-A={A:.2f}; the bind may need "
              "a fact-ID mechanism or stronger cleanup for relational queries.")


if __name__ == "__main__":
    main()
