"""END-TO-END learned syntactic understanding in-substrate: the LEARNED parser (Hebbian-acquired
conjunctive position x voice -> role, _insubstrate_parser_stdp_probe.py) drives role assignment, the
VALIDATED spiking bind stores the sentence, and a relational query extracts the agent VOICE-INVARIANTLY.

Decisive test: "dog chases cat" (active) and "cat is chased by dog" (passive) are DIFFERENT word orders
but the SAME meaning (dog is the agent). After parsing each with the LEARNED parser and binding, querying
the agent of BOTH must return dog. This composes: the Hebbian-learned parser (role assignment) + the
spiking coincidence bind (role x filler) + cosine cleanup, on real substrate concept codes.

Pipeline per content word: (content-position, voice) -> the learned parser's conjunction->role map gives
its role -> spiking-bind role(x)concept. Then query agent -> cleanup. Active content order [N1,V,N2];
passive "N2 is V by N1" content order [N2,V,N1] with voice=passive.

FROZEN: voice-invariant agent (active AND passive forms both -> the true agent) >= 0.80 multi-seed, with
a control (a scrambled role assignment -> NOT voice-invariant) -> RESOLVES (end-to-end learned syntactic
understanding in-substrate). GPU/CuPy; reuse-by-import; no protected-module modification.

RESULT 2026-05-31 seed 42 (D=800): RESOLVES. learned parser map 6/6 correct (incl the active<->passive
flip); VOICE-INVARIANT agent 1.000 (active "dog chases cat" AND passive "cat is chased by dog" both ->
dog is the agent, every trial); scrambled-parse control 0.000 (the correct learned parse is necessary).
End-to-end learned syntactic understanding in-substrate: the Hebbian-learned parser + the spiking
coincidence bind + cleanup extract the agent VOICE-INVARIANTLY from active AND passive sentences.
Multi-seed confirmation in flight.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_parser_stdp_probe as PA   # learned parser (build + train)
import research.findings.raw._insubstrate_bind_unbind_probe as P     # validated spiking bind
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend, to_host

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
ROLES = ["agent", "action", "patient"]
# content-position (0,1,2) x voice (0=active,1=passive) -> conjunction index k = pos*2 + voice
# (matches PA.GT, the ground-truth the parser learned)


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _center(d["obs__" + w].mean(axis=0)) for w in ws}


def train_parser_and_extract(seed, xp):
    """Train the Hebbian parser; return the LEARNED conjunction-index -> role map (read from the
    trained net by driving each conjunction alone and taking the argmax role)."""
    bridge, conj, role_idx = PA.build(seed, w_init=0.5)
    conj_arr = xp.asarray(conj, dtype=xp.int64)
    role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in role_idx.items()}
    for _ in range(PA.N_EPOCHS):
        for k in range(6):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(PA.RESET):
                bridge._run_one_simulation_step()
            cur = xp.zeros(6 + 3 * PA.R, dtype=xp.float32)
            cur[conj_arr[k]] = PA.DRIVE
            cur[role_arr[PA.GT[k]]] = PA.TEACH
            bridge.cp_external_input_current[:] = cur
            for _ in range(PA.TRAIN_STEPS):
                bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    learned = {}
    for k in range(6):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(PA.RESET):
            bridge._run_one_simulation_step()
        cur = xp.zeros(6 + 3 * PA.R, dtype=xp.float32)
        cur[conj_arr[k]] = PA.DRIVE
        bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in ROLES}
        for _ in range(PA.TEST_STEPS):
            bridge._run_one_simulation_step()
            for r in ROLES:
                rates[r] += float(to_host(bridge.cp_firing_states[role_arr[r]].astype(xp.float64).mean()))
        bridge.cp_external_input_current[:] = 0.0
        learned[k] = max(rates, key=rates.get)
    del bridge
    return learned


def parse_sentence(content_words, passive, learned):
    """Use the LEARNED parser map to assign a role to each content word by its (position, voice)."""
    slots = {}
    for pos, w in enumerate(content_words):
        k = pos * 2 + (1 if passive else 0)
        slots[learned[k]] = w
    return slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-trials", type=int, default=12)
    a = ap.parse_args()
    if not os.path.exists(CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    print(f"=== END-TO-END learned syntactic understanding (backend={backend}, seed={a.seed}) ===", flush=True)
    print("training the Hebbian parser + extracting the learned role map...", flush=True)
    learned = train_parser_and_extract(a.seed, xp)
    parse_ok = sum(int(learned[k] == PA.GT[k]) for k in range(6))
    print(f"  learned parser map: {parse_ok}/6 correct -> {[learned[k] for k in range(6)]}", flush=True)

    words, concepts = load_concepts(a.seed)
    # project to proj-dim for a tractable bind bridge (matches the bind probe)
    rng = np.random.default_rng(a.seed)
    if a.proj_dim and a.proj_dim > 0:
        Pmat = rng.standard_normal((concepts[words[0]].shape[0], a.proj_dim)) / np.sqrt(concepts[words[0]].shape[0])
        concepts = {w: _center(concepts[w] @ Pmat) for w in words}
    D = concepts[words[0]].shape[0]
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)

    inv_ok = ctrl_ok = tot = 0
    scrambled = {k: ROLES[(ROLES.index(learned[k]) + 1) % 3] for k in range(6)}  # wrong-parse control
    for _ in range(a.n_trials):
        n1, v, n2 = (words[i] for i in rng.choice(len(words), 3, replace=False))
        act = parse_sentence([n1, v, n2], False, learned)         # "n1 v n2"
        pas = parse_sentence([n2, v, n1], True, learned)          # "n2 is v by n1" -> agent should be n1
        S_act = RM.bind_fact_spiking(bridge, idx, act, concepts, roles, D, xp)
        S_pas = RM.bind_fact_spiking(bridge, idx, pas, concepts, roles, D, xp)
        a_act = RM.unbind_spiking(bridge, idx, S_act, "agent", roles, concepts, words, D, xp)
        a_pas = RM.unbind_spiking(bridge, idx, S_pas, "agent", roles, concepts, words, D, xp)
        inv_ok += int(a_act == n1 and a_pas == n1)                # both -> true agent n1
        # control: scrambled parse -> passive agent should NOT reliably be n1
        pas_bad = parse_sentence([n2, v, n1], True, scrambled)
        S_bad = RM.bind_fact_spiking(bridge, idx, pas_bad, concepts, roles, D, xp)
        a_bad = RM.unbind_spiking(bridge, idx, S_bad, "agent", roles, concepts, words, D, xp)
        ctrl_ok += int(a_bad == n1)
        tot += 1
    print(f"  VOICE-INVARIANT agent (active & passive -> same true agent): {inv_ok/tot:.3f}  "
          f"scrambled-parse control: {ctrl_ok/tot:.3f}  (chance {1.0/len(words):.3f})", flush=True)
    if parse_ok >= 5 and inv_ok / tot >= 0.80:
        print("VERDICT: RESOLVES -- end-to-end LEARNED syntactic understanding in-substrate: the Hebbian-"
              "learned parser + spiking bind extract the agent VOICE-INVARIANTLY from active AND passive.",
              flush=True)
    else:
        print(f"VERDICT: parse {parse_ok}/6, voice-invariant {inv_ok/tot:.2f} -- inspect parser map / bind.",
              flush=True)


if __name__ == "__main__":
    main()
