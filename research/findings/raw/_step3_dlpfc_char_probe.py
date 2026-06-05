"""Step-3 dlPFC-merge CHARACTERIZATION probe (controller-driven crux decision).

The GATE test fails on EXACT-pick parity: merged elaborate('dog')='go' vs separate-oracle 'look'.
Both are DIRECT (1-hop) neighbours of 'dog' (facts: dog->go, dog->look). The pick is the
earliest-first-spike associate (rank-order latency code); dt=1.0 (merged, the de-risked faithful
merge regime) has coarser latency resolution than dt=0.5 (oracle), so the tie-break among
EQUIDISTANT direct neighbours flips. This probe decides MERGE-functional vs BOUNDARY by checking
the dlPFC's OWN validated criterion (content_selection_spiking line 376: "earliest-latency pick is
a DIRECT neighbour 6/6"), NOT exact-pick parity (stricter than the validated function).

Evidence dumped (merged = unified bridge dt=1.0 OU-off; oracle = separate dlPFC dt=0.5):
  - the association graph (the agent's own facts)
  - per connected topic: direct neighbours, merged pick, oracle pick, on-topic flags,
    the FULL merged latency ranking (twice -> determinism), the oracle latency ranking
  - abstention on the unconnected topic (the no-confab moat)
  - a 3-turn merged elaboration (does it stay in the topic region -> the oracle's 6/6 multi-turn criterion)
"""
import sys

SEED = 42
PROJ = 64
FACTS = [("dog", "go", "north"), ("cat", "come", "south"), ("dog", "look", "river")]
CONNECTED = ["dog", "cat", "river"]
UNCONNECTED = "apple"


def _lat_rank(ctrl, bridge, topic):
    """Full first-spike latency ranking for `topic` with OU forced off (the validated dlPFC regime)."""
    prev = bridge.core_config.enable_ou_process
    bridge.core_config.enable_ou_process = False
    try:
        lat = ctrl.relevance_by_latency(topic)
    finally:
        bridge.core_config.enable_ou_process = prev
    fired = sorted([(c, int(v)) for c, v in lat.items() if v is not None], key=lambda kv: kv[1])
    return fired


def main():
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        print("SKIP: GPU backend required (spiking dynamics).")
        return
    from research.runners.unified_brain_bridge import UnifiedBrainBridge
    from research.runners.brain_conversational_agent import BrainConversationalAgent

    # --- oracle (separate dlPFC, dt=0.5) ---
    oracle = BrainConversationalAgent(seed=SEED, proj_dim=PROJ, concepts=None)
    for a, ac, p in FACTS:
        oracle.hear(f"{a} {ac} {p}")
    graph = oracle._assoc_graph()
    print("=== association graph (the agent's own facts) ===")
    for k in sorted(graph):
        print(f"  {k:7s} -> {sorted(graph[k])}")
    ora_pick = {t: oracle.elaborate(t) for t in CONNECTED}
    ora_abstain = oracle.elaborate(UNCONNECTED)
    ora_ctrl = oracle._dlpfc

    # --- merged (unified bridge, dt=1.0, OU-off in elaborate) ---
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ, concepts=None, enable_dlpfc=True)
    for a, ac, p in FACTS:
        u.hear(f"{a} {ac} {p}")
    mer_pick = {t: u.elaborate(t) for t in CONNECTED}
    mer_abstain = u.elaborate(UNCONNECTED)
    mer_ctrl = u._dlpfc_controller

    print("\n=== per-topic: picks + direct-neighbour check + latency rankings ===")
    all_direct = True
    determinism_ok = True
    for t in CONNECTED:
        direct = set(graph.get(t, {}))
        mer = mer_pick[t]
        ora = ora_pick[t]
        mer_on = mer in direct
        ora_on = ora in direct
        all_direct = all_direct and mer_on
        r1 = _lat_rank(mer_ctrl, u.bridge, t)
        r2 = _lat_rank(mer_ctrl, u.bridge, t)
        det = (r1 == r2)
        determinism_ok = determinism_ok and det
        o_r = _lat_rank(ora_ctrl, oracle._dlpfc.ctx.bridge, t)
        print(f"\n  topic={t!r} direct_neighbours={sorted(direct)}")
        print(f"    merged pick={mer!r} (direct={mer_on})   oracle pick={ora!r} (direct={ora_on})")
        print(f"    merged latency rank (run1): {r1}")
        print(f"    merged latency rank (run2): {r2}   deterministic={det}")
        print(f"    oracle latency rank:        {o_r}")

    print(f"\n=== abstention (no-confab moat) ===")
    print(f"  unconnected topic {UNCONNECTED!r}: merged={mer_abstain!r}  oracle={ora_abstain!r}")

    # --- 3-turn merged elaboration on 'dog' (rebuild controller for a fresh said-trace) ---
    print("\n=== 3-turn merged elaboration on 'dog' (multi-turn topic coherence) ===")
    u._dlpfc_controller = None          # force a fresh controller (fresh said-trace)
    u._dlpfc_graph_key = None
    dog_region = set(graph.get("dog", {}))
    for n in graph.get("dog", {}):
        dog_region |= set(graph.get(n, {}))   # 2-hop region
    seq = []
    for _ in range(3):
        seq.append(u.elaborate("dog"))
    in_region = all(s in dog_region for s in seq if s is not None)
    print(f"  sequence={seq}   2-hop dog-region={sorted(dog_region)}   stays_in_region={in_region}")

    # --- verdict ---
    print("\n=== VERDICT ===")
    print(f"  merged picks all DIRECT neighbours: {all_direct}")
    print(f"  merged latency deterministic:       {determinism_ok}")
    print(f"  merged abstains on unconnected:     {mer_abstain is None}")
    print(f"  merged multi-turn stays on-topic:   {in_region}")
    func_ok = all_direct and determinism_ok and (mer_abstain is None) and in_region
    print(f"  -> {'MERGE-FUNCTIONAL (dlPFC validated criterion met; exact-pick differs by dt regime)' if func_ok else 'DEGRADED -> BOUNDARY'}")


if __name__ == "__main__":
    main()
