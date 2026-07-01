"""Phase-17 DE-RISK: PERSISTENT grown knowledge -- the brain REMEMBERS learned facts across sessions.

The console (`_fluidconv_chat_repl.py`) learns real facts on demand (Wikidata) + by being taught, but rebuilt from the
curriculum each run, so the GROWN knowledge was LOST on restart. This de-risks the persistence MECHANISM: a learned fact
uses a DETERMINISTIC per-concept code (md5(word) -> seed -> phases, the console's `_ensure_concept`), so it reproduces
across sessions; persistence = save the learned fact-list (JSON) and, on load, re-inject each concept's (identical)
code + re-store the fact -> the KB is rebuilt. This validates that round-trip on a bare agent (no FT generator needed).
Reuse-by-import; NO `sim/` edit; CPU. The owner's "grow THROUGH experiences" -> the brain now remembers.

METRICS (>=3 seeds): (a) ROUND-TRIP -- save learned facts, load into a FRESH same-seed agent, all recalled; (b)
COLD-START CONTROL -- a fresh agent that does NOT load recalls NONE of them (persistence is load-bearing); (c) MOAT --
a never-learned concept abstains after load; (d) DETERMINISTIC-CODE -- the re-injected code == the original (the round-
trip works BECAUSE codes are reproducible, not stored).

GO = round-trip + cold-start-control + moat + deterministic-code, >=3 seeds. Reuse-by-import; NO `sim/` edit; CPU.
Run: python -m research.runners._fluidconv_phase17_persistence_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase17_persistence.json"
# the same deterministic runtime-code injection the console uses (`_ensure_concept`)
def _ensure(agent, w):
    comp = agent.composer
    if w not in comp.concepts:
        s = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
        comp.concepts[w] = np.random.default_rng(s).uniform(0.0, 1.0, comp.D)
        comp.words = sorted(set(comp.words) | {w})


BASE = [("dog", "eat", "meat")]                                   # a base curriculum fact (re-taught each session)
LEARNED = [["elephant", "isa", "mammal"], ["elephant", "has", "trunk"], ["wolf", "eat", "rabbit"],
           ["banana", "is", "yellow"]]                            # the "grown" delta (Wikidata + taught)


def _agent(seed):
    vocab = sorted({"dog", "eat", "meat", "isa", "is", "has"}
                   | {t for f in LEARNED for t in f})
    a = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", D=256)
    for (s, v, p) in BASE:
        a.hear(f"{s} {v} {p}")
    return a


def run(seed, tmp):
    # session 1: learn the delta (deterministic codes) + save the fact-list
    a1 = _agent(seed)
    for (s, v, p) in LEARNED:
        for t in (s, v, p):
            _ensure(a1, t)
        a1.composer.store(s, v, p)
    orig_codes = {t: a1.composer.concepts[t].copy() for f in LEARNED for t in f}
    Path(tmp).write_text(json.dumps({"learned": LEARNED}))

    # session 2 (fresh, same seed): BEFORE load -> cold-start control; then load -> round-trip
    a2 = _agent(seed)
    cold = [f for f in LEARNED if a2.what_does(f[0], f[1]) is not None]   # should recall NONE pre-load
    learned = json.loads(Path(tmp).read_text())["learned"]
    reinjected_match = True
    for f in learned:
        for t in (f[0], f[1], f[2]):
            _ensure(a2, t)
        a2.composer.store(f[0], f[1], f[2])
    for t, code in orig_codes.items():                            # deterministic-code check
        if not np.allclose(a2.composer.concepts[t], code):
            reinjected_match = False
    recalled = sum(1 for f in LEARNED if a2.what_does(f[0], f[1]) == f[2])
    base_ok = (a2.what_does("dog", "eat") == "meat")              # base survives
    moat_ok = (a2.what_does("dragon", "eat") is None) if "dragon" in a2.composer.concepts else True
    # moat on a truly-unknown concept (never in vocab) -> the agent can't even query it; use a known-but-unlearned one
    _ensure(a2, "griffin")
    moat_ok = moat_ok and (a2.what_does("griffin", "eat") is None)
    return {"seed": seed, "cold_start_recalled": len(cold), "roundtrip_recalled": recalled, "n_learned": len(LEARNED),
            "roundtrip_ok": bool(recalled == len(LEARNED)), "cold_start_ok": bool(len(cold) == 0),
            "base_ok": bool(base_ok), "moat_ok": bool(moat_ok), "deterministic_code_ok": bool(reinjected_match)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per_seed = []
    tmp = str(_REPO / "research" / "findings" / "raw" / "_fluidconv_phase17_tmp_state.json")
    try:
        for s in a.seeds:
            r = run(s, tmp); per_seed.append(r)
            print(f"  [seed {s}] cold-start recalled {r['cold_start_recalled']}/{r['n_learned']} ({r['cold_start_ok']}) "
                  f"| after-load {r['roundtrip_recalled']}/{r['n_learned']} ({r['roundtrip_ok']}) | base {r['base_ok']} "
                  f"| moat {r['moat_ok']} | det-code {r['deterministic_code_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        rt = all(r["roundtrip_ok"] for r in per_seed)
        cs = all(r["cold_start_ok"] for r in per_seed)
        moat = all(r["moat_ok"] for r in per_seed)
        det = all(r["deterministic_code_ok"] for r in per_seed)
        base = all(r["base_ok"] for r in per_seed)
        go = bool(rt and cs and moat and det and base)
        verdict = (("GO -- PERSISTENT grown knowledge: learned facts SAVE + re-LOAD into a fresh same-seed brain (all "
                    "recalled), a cold-start brain recalls NONE (persistence load-bearing), the base survives, the moat "
                    "abstains on the unlearned, and the re-injected concept codes are BIT-IDENTICAL to the originals "
                    "(the round-trip works because codes are deterministic md5, not because they're stored). >=3 seeds. "
                    "The brain REMEMBERS what it learned across sessions -- reuse-by-import, NO sim/ edit.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if rt else ["round-trip (not all recalled after load)"]) +
                       ([] if cs else ["cold-start control (a fresh brain already recalled them?!)"]) +
                       ([] if moat else ["moat"]) + ([] if det else ["deterministic-code mismatch"]) +
                       ([] if base else ["base fact lost"])) + " failed"))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase17_persistence", "GO": go, "verdict": verdict,
               "resolves": "persistent grown knowledge: save learned facts + re-store on load (deterministic codes) -> "
                           "the brain remembers across sessions; cold-start recalls none; moat holds.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "saves the learned FACT-LIST + relies on deterministic per-concept codes (md5) to "
                                 "reproduce the codebook -- NOT the raw composer complex-weight tensor (a JSON re-instate, "
                                 "like the Tier-3 persistence). Base-curriculum codes reproduce because the seed is fixed "
                                 "(a lineage fixes it). Instances (dog_1) are session discourse state -> intentionally "
                                 "not persisted. The console wires save_state/load_state around this."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    try:
        os.remove(tmp)
    except OSError:
        pass
    print("\n" + "=" * 108, flush=True)
    print(f"[phase17-persistence] VERDICT: {verdict}", flush=True)
    print(f"[phase17-persistence] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
