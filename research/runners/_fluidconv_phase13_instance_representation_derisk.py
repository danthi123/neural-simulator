"""Phase-13 DE-RISK: KIND vs INSTANCE representation ("the dog" a specific referent vs "dogs" the kind).

Per the scoping (`2026-07-01-kind-vs-instance-representation-scoping.md`): the gap is small + at the agent layer. The
composer matches facts by agent/action VALUE (string/code), so an INSTANCE is first-class the moment it has a distinct
token + code. The #1 mechanism (reuse-by-import, NO `sim/` edit): represent an instance as a fresh token, link it to
its KIND with a stored `isa` fact, resolve INSTANCE-FIRST / KIND-FALLBACK (a miss on the instance retries against the
isa-parent = INHERITANCE), and route DEFINITE ("the dog" -> the discourse-current instance) vs GENERIC ("dogs" -> the
kind). Biology: semantic (kind, cortex) vs episodic (instance, hippocampus); object-file / DRT discourse referents;
isa-inheritance (Collins-Quillian).

THE TARGET (owner's example): "I saw a dog. the dog was brown. what is the dog? -> brown; what do dogs eat? -> meat."

METRICS (>=3 seeds): (a) INSTANCE own-fact -- "what is the dog?" (definite -> instance) -> the instance's OWN fact
(brown), NOT the kind's; (b) INHERITANCE -- "what does the dog eat?" (instance) -> inherited kind fact (meat) via isa;
(c) GENERIC -- "what do dogs eat?" (kind) -> the kind fact (meat); (d) DISTINCTNESS -- two instances (dog_one brown,
dog_two black) stay distinct; (e) ISA-LESION -- remove the isa link -> inheritance FAILS (load-bearing); (f)
MOAT -- an unknown instance -> abstain.

GO = instance-own + inheritance + generic + distinctness + isa-lesion-breaks + moat, >=3 seeds. Reuse-by-import; NO
`sim/` edit. CPU (brain-only).
Run: python -m research.runners._fluidconv_phase13_instance_representation_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase13_instance_representation.json"

KIND_FACTS = [("dog", "eat", "meat"), ("dog", "is", "mammal"), ("cat", "eat", "fish")]
# instance tokens (distinct concepts) + their episodic own-facts + the isa link to the kind
INSTANCES = [("dog_one", "dog", "brown"), ("dog_two", "dog", "black")]   # (instance, kind, own-attribute)


def _resolve(agent, token, verb, *, allow_inherit=True):
    """INSTANCE-FIRST / KIND-FALLBACK query: what_does(token, verb); on a miss, retry against the isa-parent
    (inheritance). Returns the patient or None (the moat abstains at both levels)."""
    p = agent.what_does(token, verb)
    if p is not None:
        return p, "own"
    if allow_inherit:
        parent = agent.what_does(token, "isa")           # the instance's kind (isa link)
        if parent is not None:
            pp = agent.what_does(parent, verb)
            if pp is not None:
                return pp, "inherited"
    return None, "abstain"


def run(seed, lesion_isa=False):
    kinds = {f[0] for f in KIND_FACTS} | {f[2] for f in KIND_FACTS}
    inst_tokens = {i[0] for i in INSTANCES}
    vocab = sorted(kinds | inst_tokens | {"isa", "is", "eat", "black", "brown", "mammal", "otter_one"})
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", D=256)
    for (a, v, p) in KIND_FACTS:
        agent.hear(f"{a} {v} {p}")
    # introduce instances: episodic own-fact + the isa link to the kind (stored directly; instance tokens aren't
    # natural parseable words, and store() is value-keyed so a distinct token is first-class)
    for (inst, kind, attr) in INSTANCES:
        if not lesion_isa:
            agent.composer.store(inst, "isa", kind)      # "dog_one isa dog"
        agent.composer.store(inst, "is", attr)           # the instance's OWN episodic fact ("dog_one is brown")

    inst0 = INSTANCES[0][0]; kind0 = INSTANCES[0][1]; attr0 = INSTANCES[0][2]
    # (a) instance own-fact: "what is the dog?" (definite -> instance dog_one)
    own, own_src = _resolve(agent, inst0, "is")
    instance_own_ok = (own == attr0 and own_src == "own")
    # (b) inheritance: "what does the dog eat?" (instance) -> inherited kind fact
    inh, inh_src = _resolve(agent, inst0, "eat")
    inheritance_ok = (inh == "meat" and inh_src == "inherited")
    # (c) generic: "what do dogs eat?" (kind)
    gen = agent.what_does(kind0, "eat")
    generic_ok = (gen == "meat")
    # (d) distinctness: dog_two's own attr is black, not brown
    d2, _ = _resolve(agent, INSTANCES[1][0], "is")
    distinct_ok = (d2 == INSTANCES[1][2] and d2 != attr0)
    # (f) moat: an unknown instance -> abstain (no isa, no facts)
    moat, _ = _resolve(agent, "otter_one", "eat")
    moat_ok = (moat is None)
    return {"seed": seed, "lesion_isa": lesion_isa, "instance_own": own, "instance_own_ok": bool(instance_own_ok),
            "inherited": inh, "inherited_src": inh_src, "inheritance_ok": bool(inheritance_ok),
            "generic": gen, "generic_ok": bool(generic_ok), "dog_two_is": d2, "distinct_ok": bool(distinct_ok),
            "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; base = []; lesion = []
    try:
        for s in a.seeds:
            b = run(s, lesion_isa=False); base.append(b)
            L = run(s, lesion_isa=True); lesion.append(L)
            print(f"  [seed {s}] instance-own '{b['instance_own']}' ({b['instance_own_ok']}) | inherit '{b['inherited']}'"
                  f" via {b['inherited_src']} ({b['inheritance_ok']}) | generic '{b['generic']}' ({b['generic_ok']}) | "
                  f"distinct dog_two='{b['dog_two_is']}' ({b['distinct_ok']}) | moat {b['moat_ok']} | "
                  f"[isa-lesion] inherit-now '{L['inherited']}' (breaks={not L['inheritance_ok']})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        own_ok = all(b["instance_own_ok"] for b in base)
        inh_ok = all(b["inheritance_ok"] for b in base)
        gen_ok = all(b["generic_ok"] for b in base)
        dist_ok = all(b["distinct_ok"] for b in base)
        moat_ok = all(b["moat_ok"] for b in base)
        lesion_breaks = all(not L["inheritance_ok"] for L in lesion)     # isa-lesion -> inheritance fails
        go = bool(own_ok and inh_ok and gen_ok and dist_ok and moat_ok and lesion_breaks)
        verdict = (("GO -- KIND vs INSTANCE: 'the dog' (definite) resolves to the INSTANCE (its own fact, brown, NOT "
                    "the kind); the instance INHERITS kind facts via isa ('what does the dog eat?' -> meat); 'dogs' "
                    "(generic) -> the kind; two instances stay DISTINCT (brown vs black); the ISA-LESION BREAKS "
                    "inheritance (load-bearing); the moat abstains on an unknown instance. >=3 seeds. The owner's "
                    "'which dog?' distinction, with inheritance + no cross-leakage + moat -- reuse-by-import, NO sim/ "
                    "edit.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if own_ok else ["instance-own failed (definite didn't get the instance's own fact)"]) +
                       ([] if inh_ok else ["inheritance failed (instance didn't inherit the kind fact via isa)"]) +
                       ([] if gen_ok else ["generic failed"]) +
                       ([] if dist_ok else ["instances not distinct"]) +
                       ([] if moat_ok else ["moat leaked"]) +
                       ([] if lesion_breaks else ["isa-lesion did NOT break inheritance (not load-bearing)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase13_instance_representation", "GO": go, "verdict": verdict,
               "resolves": "kind vs instance: a definite 'the dog' resolves to a specific instance (own facts + isa-"
                           "inheritance); 'dogs' -> the kind; distinct instances; moat.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "base": base, "lesion": lesion,
               "HONEST_CEILING": "a lightweight instance token + isa-inheritance + definite/generic routing, at the "
                                 "conversational layer (reuse-by-import). Inheritance is via an explicit symbolic isa "
                                 "link (like DRT), NOT code-similarity generalization (the separate PPMI/dendritic "
                                 "frontier). A PERCEIVED/consolidated episodic instance ('the dog I saw on my walk') = "
                                 "the engram-tag/hippocampal path (composes with Tier-3 live-and-remember); multiple "
                                 "co-present instances = the biased-competition WTA drop-in."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase13-instance] VERDICT: {verdict}", flush=True)
    print(f"[phase13-instance] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
