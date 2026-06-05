"""B capstone: a scripted end-to-end conversation on ONE UnifiedBrainBridge (parser + composer + dlPFC).

Exercises the FULL conversational API on a single SimulationBridge — comprehend (parser), store + who/what
Q&A + yes/no + negation + generation (composer), dialogue planning (dlPFC) — to demonstrate that all three
conversational regions interoperate on one interacting brain (the one-bridge unification, steps 1+2+3).

Run (GPU/CuPy; uses the default denoise64 concept codes, skips if the cache is absent):
  python -m research.findings.raw._unified_brain_capstone_demo
"""


def main():
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        print("SKIP: GPU backend required (the substrate's spiking dynamics are GPU-bound).")
        return
    from research.runners.unified_brain_bridge import UnifiedBrainBridge

    # D=2048 is the validated production dimension for the correlated denoise64 codes (cos~0.80): the stage-1.5
    # finding showed D=64/800 degrade the composer's recall on these codes while D=2048 recovers it (the
    # dimensional cost is real). The dlPFC works at any D (it uses pattern-size assemblies, not the D-dim codes).
    print("Building ONE UnifiedBrainBridge (parser + composer + dlPFC) at production D=2048 ...")
    try:
        u = UnifiedBrainBridge(seed=42, proj_dim=2048, concepts=None, enable_dlpfc=True)
    except FileNotFoundError:
        print("SKIP: denoise64 concept-code cache absent for seed 42.")
        return

    assert u.parser.bridge is u.bridge and u.composer.bridge is u.bridge and u.dlpfc_bridge is u.bridge
    print(f"  ONE bridge, {u.bridge.core_config.num_neurons} neurons; all three regions share it.\n")

    # Facts carry an AFFIRM polarity tag so yes/no questions have a bound polarity to read.
    facts = [
        ("dog go north", "AFFIRM"),
        ("cat come south", "AFFIRM"),
        ("dog look river", "AFFIRM"),
    ]
    print("=== COMPREHEND + STORE (the parser assigns roles; the composer binds the fact in spikes) ===")
    for sentence, pol in facts:
        roles = u.hear(sentence, polarity=pol)
        print(f"  heard {sentence!r:22s} -> roles {roles}")

    print("\n=== WHO / WHAT QUESTION ANSWERING (spiking unbind + cleanup) ===")
    qs_what = [("dog", "go"), ("cat", "come"), ("dog", "look")]
    for ag, ac in qs_what:
        print(f"  what does {ag} {ac}? -> {u.query_patient(ag, ac)!r}")
    qs_who = [("go", "north"), ("come", "south"), ("look", "river")]
    for ac, pa in qs_who:
        print(f"  who {ac} {pa}?        -> {u.query_agent(ac, pa)!r}")

    print("\n=== ABSTENTION (the no-confab moat: unknown -> None, never a guess) ===")
    print(f"  what does cat go?     -> {u.query_patient('cat', 'go')!r}   (no such fact)")
    print(f"  who go river?         -> {u.query_agent('go', 'river')!r}   (no such fact)")

    print("\n=== YES / NO (the bound polarity tag) ===")
    print(f"  does dog go north?    -> {u.ask_yes_no('dog', 'go', 'north')!r}")
    print(f"  does dog go south?    -> {u.ask_yes_no('dog', 'go', 'south')!r}")

    print("\n=== GENERATION (produce a sentence from a stored fact, decoded from spikes) ===")
    for ag in ("dog", "cat"):
        print(f"  describe {ag}          -> {u.describe(ag)!r}")
    print(f"  describe apple        -> {u.describe('apple')!r}   (nothing stored -> abstain)")

    print("\n=== DIALOGUE PLANNING (the dlPFC: what to bring up next about a topic) ===")
    for topic in ("dog", "cat", "river"):
        print(f"  elaborate on {topic!r:7s}  -> {u.elaborate(topic)!r}")
    print(f"  elaborate on 'apple'  -> {u.elaborate('apple')!r}   (unconnected -> abstain)")

    print("\nAll of the above ran on ONE interacting SimulationBridge. B is structurally complete.")


if __name__ == "__main__":
    main()
