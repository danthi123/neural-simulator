"""Phase-14 DE-RISK: instance-rep in a MULTI-TURN conversational flow (discourse-instance tracking).

Phase-13 proved the kind/instance mechanism (own facts + isa-inheritance + definite/generic). This consolidates it
into a CONVERSATIONAL flow: an instance is MINTED when first mentioned ("I saw a dog"), tracked as the discourse-
current referent, attributed across turns ("the dog was brown"), and queried ("what is the dog?" -> the instance;
"what does the dog eat?" -> inherited; "what do dogs eat?" -> the kind). Reuse-by-import (Phase-13 _resolve + the
brain); NO `sim/` edit. Biology: discourse referents / object files (DRT, Kahneman-Treisman); the WM-held referent.

THE FLOW (owner's example, multi-turn):
  you>  i saw a dog            brain> ok. (mint dog_1 isa dog; discourse-current)
  you>  the dog was brown      brain> ok. (store dog_1 is brown)
  you>  what is the dog?       brain> the dog is brown.   (the INSTANCE's own fact)
  you>  what does the dog eat? brain> the dog eats meat.  (INHERITED from the kind via isa)
  you>  what do dogs eat?      brain> dogs eat meat.      (the KIND)
  you>  i saw a cat            brain> ok. (mint cat_1; discourse-current now the cat)
  you>  what is the dog?       brain> the dog is brown.   (dog_1 still distinct + retrievable)

METRICS (>=3 seeds): (a) MINT+ATTRIBUTE+QUERY: the instance's own fact is retrieved via "the dog" (definite); (b)
INHERITANCE: "what does the dog eat?" -> the kind's fact via isa; (c) GENERIC: "what do dogs eat?" -> the kind; (d)
DISTINCT+PERSIST: after minting a second instance, the first is still distinct + retrievable; (e) MOAT: "the wolf"
never introduced -> abstain.

GO = all of the above, >=3 seeds. Reuse-by-import; NO `sim/` edit; CPU.
Run: python -m research.runners._fluidconv_phase14_instance_conversation_derisk --seeds 42 43 44
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
from research.runners._fluidconv_phase13_instance_representation_derisk import _resolve  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase14_instance_conversation.json"
KIND_FACTS = [("dog", "eat", "meat"), ("dog", "is", "mammal"), ("cat", "eat", "fish")]
KINDS = ["dog", "cat", "wolf"]
_V3 = {"eat": "eats", "is": "is", "chase": "chases", "like": "likes"}


class InstanceConversation:
    """A multi-turn agent that MINTS + tracks discourse-instances. Instance tokens are pre-allocated in the composer
    vocab (dog_1, dog_2, cat_1, ...) so they have codes; minting = assign the next free slot + store the isa link."""

    def __init__(self, seed):
        kinds = {f[0] for f in KIND_FACTS} | {f[2] for f in KIND_FACTS} | set(KINDS)
        self.inst_slots = {k: [f"{k}_1", f"{k}_2"] for k in KINDS}       # 2 instance slots per kind
        vocab = sorted(kinds | {t for slots in self.inst_slots.values() for t in slots}
                       | {"isa", "is", "eat", "brown", "black", "mammal"})
        self.agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", D=256)
        for (a, v, p) in KIND_FACTS:
            self.agent.hear(f"{a} {v} {p}")
        self._used = {k: 0 for k in KINDS}          # how many instances of each kind minted
        self._last_inst = {}                        # kind -> the LAST minted instance token (per-kind referent)
        self._cur = None                            # the most-recent discourse instance (for "it")
        self._cur_kind = None

    def _mint(self, kind):
        if self._used[kind] >= len(self.inst_slots[kind]):
            return None
        tok = self.inst_slots[kind][self._used[kind]]; self._used[kind] += 1
        self.agent.composer.store(tok, "isa", kind)                       # the isa link (inherit the kind's facts)
        self._last_inst[kind] = tok
        self._cur, self._cur_kind = tok, kind
        return tok

    def _kind_of(self, toks):
        """Return (kind, is_plural): a singular kind token -> (kind, False); a plural 'dogs' -> (kind, True)."""
        for w in toks:
            if w in KINDS:
                return w, False
            if w.endswith("s") and w[:-1] in KINDS:
                return w[:-1], True
        return None, False

    def turn(self, text):
        t = text.lower().strip().rstrip("?.!")
        toks = t.split()
        kind, is_plural = self._kind_of(toks)
        is_q = ("what" in toks or "?" in text or toks[:1] == ["do"] or "does" in toks)
        # MINT: "i saw a dog" / "a dog" (indefinite) -> introduce an instance
        if kind and not is_q and not is_plural and (" a " in f" {t} " or t.startswith("a ") or "saw" in toks) and "the" not in toks:
            self._mint(kind); return f"ok, a {kind}."
        # ATTRIBUTE: "the dog was brown" / "it is brown" -> store the discourse-current instance's own fact
        if ("is" in toks or "was" in toks) and self._cur is not None and not is_q:
            attr = next((w for w in toks if w in ("brown", "black", "mammal")), None)
            if attr:
                self.agent.composer.store(self._cur, "is", attr); return f"ok, the {self._cur_kind} is {attr}."
        # QUESTION
        if is_q and kind:
            verb = "eat" if ("eat" in toks or "eats" in toks) else ("is" if ("is" in toks or "was" in toks) else None)
            if verb is None:
                return "I don't know."
            inst = self._last_inst.get(kind)                             # the LAST instance of THIS kind
            if not is_plural and inst is not None:                       # definite "the dog" -> that instance
                p, _src = _resolve(self.agent, inst, verb)               # instance-first / kind-fallback (inherit)
                return (f"the {kind} is {p}." if verb == "is" else f"the {kind} {_V3.get(verb, verb)} {p}.") \
                    if p is not None else "I don't know."
            p = self.agent.what_does(kind, verb)                         # generic "dogs" (or no instance) -> the kind
            return (f"{kind}s are {p}." if verb == "is" else f"{kind}s {_V3.get(verb, verb)} {p}.") \
                if p is not None else "I don't know."
        return "ok."


DEMO = ["i saw a dog", "the dog was brown", "what is the dog?", "what does the dog eat?", "what do dogs eat?",
        "i saw a cat", "what is the dog?", "what does the wolf eat?"]


def run(seed):
    c = InstanceConversation(seed)
    tr = [(u, c.turn(u)) for u in DEMO]
    d = {u: r for u, r in tr}
    own_ok = "brown" in d["what is the dog?"]
    inherit_ok = "meat" in d["what does the dog eat?"]
    generic_ok = "meat" in d["what do dogs eat?"]
    persist_ok = "brown" in tr[6][1]                     # after minting the cat, "the dog" still -> brown
    moat_ok = "know" in d["what does the wolf eat?"].lower()
    return {"seed": seed, "transcript": tr, "own_ok": bool(own_ok), "inherit_ok": bool(inherit_ok),
            "generic_ok": bool(generic_ok), "persist_ok": bool(persist_ok), "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per_seed = []
    try:
        for s in a.seeds:
            r = run(s); per_seed.append(r)
            print(f"  [seed {s}] own {r['own_ok']} | inherit {r['inherit_ok']} | generic {r['generic_ok']} | "
                  f"distinct-persist {r['persist_ok']} | moat {r['moat_ok']}", flush=True)
        print("\n  --- transcript (seed 42) ---", flush=True)
        for u, rr in per_seed[0]["transcript"]:
            print(f"    you>   {u}\n    brain> {rr}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        go = all(r["own_ok"] and r["inherit_ok"] and r["generic_ok"] and r["persist_ok"] and r["moat_ok"] for r in per_seed)
        verdict = (("GO -- instance-rep in a MULTI-TURN flow: an instance is minted on mention, attributed across "
                    "turns, and 'the dog' (definite) retrieves ITS own fact (brown) + inherits the kind's ('the dog "
                    "eats meat'); 'dogs' (generic) -> the kind; a second instance keeps the first distinct + "
                    "retrievable; the moat abstains on a never-introduced 'wolf'. >=3 seeds. The owner's 'which dog?' "
                    "distinction works conversationally.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       [k for k, ok in [("own", all(r["own_ok"] for r in per_seed)),
                                        ("inherit", all(r["inherit_ok"] for r in per_seed)),
                                        ("generic", all(r["generic_ok"] for r in per_seed)),
                                        ("distinct-persist", all(r["persist_ok"] for r in per_seed)),
                                        ("moat", all(r["moat_ok"] for r in per_seed))] if not ok]) + " failed"))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase14_instance_conversation", "GO": go, "verdict": verdict,
               "resolves": "instance-rep in a multi-turn conversational flow: mint on mention + discourse tracking + "
                           "definite/generic + isa-inheritance + distinctness + moat.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "a lightweight rule-based mint/attribute/query router over the validated Phase-13 "
                                 "mechanism + a discourse-current instance pointer (an object-file). Multiple "
                                 "SIMULTANEOUS same-kind instances ('the first dog vs the second dog') -> the "
                                 "biased-competition WTA; perceived/consolidated instances -> the engram/hippocampal "
                                 "path (Tier-3). The interrogative/mint parse is a scaffold (the neural parser, Phase-7, "
                                 "is the brain-based path)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase14-instance-conv] VERDICT: {verdict}", flush=True)
    print(f"[phase14-instance-conv] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
