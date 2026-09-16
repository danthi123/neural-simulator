"""SOAK / lesion gate for gap #3 residual A1's LEARNED REFERENT-BIAS wire-in flag,
BRAIN_BIASED_COMPETITION_LEARNED_BIAS (default OFF -- research/runners/biased_competition_prod.py).

Mirrors `_biased_competition_flip_soak.py`'s OFF/ON/LESION-arm soak structure, but targets a DIFFERENT flag: that
soak gates whether the biased-competition WTA itself is built (BRAIN_BIASED_COMPETITION, already 6-seed-soaked and
FLIPPED DEFAULT-ON 2026-08-26); THIS flag gates, once the WTA is already active, WHICH mechanism picks the
bias-target for a held pair of referents — the HOST `content_bias_target` lexicon (`ANIMACY`/`VERB_SELECTS`,
`biased_competition_buffer.py`) vs. the LEARNED spiking `SpikingFeatureCompat`
(`_gap3_spiking_feature_compat_derisk.py`, gap #3 residual A1, already 6-seed-GO + CI-pinned:
`tests/test_gap3_spiking_feature_compat.py`). The WTA itself is unconditionally ON in every arm here
(`enable_biased_competition=True`) — this soak isolates the NEW flag.

PER-SEED PROTOCOL. The agent HEARS a synthetic SVO corpus (`_gap3_learned_feature_compat_derisk.make_corpus` — the
mechanism finding's own corpus generator; it uses the host ANIMACY/VERB_SELECTS ONLY to draw plausible sentences,
exactly as the mechanism's own de-risk did — the LEARNED path itself never reads the lexicon) as its OWN
conversational experience (`composer.store`, what `agent.heard_facts()` reads), then holds the (animate, inanimate)
pair {cat, ball} and asks the 'eat'-selects-animate probe ("what does it eat?", host answer 'cat').

  OFF     : flag off -> `maybe_install_learned_referent_bias` is a byte-identical no-op (`agent` untouched,
            `_feat_compat_source` stays `None`) -> `_resolve_biased` answers via the HOST `content_bias_target`
            lexicon, i.e. today's production behavior, unchanged.
  ON      : flag on -> `build_referent_bias_from_experience` LEARNS the concept-animacy / verb-selection
            compatibility map from the >=40 heard facts and installs the resulting spiking `SpikingFeatureCompat`
            -> `_resolve_biased` now answers via the LEARNED SPIKING chooser, not the host lexicon. GO requires
            this to MATCH the host answer ('cat') on the probe (the mechanism's own 6-seed GO: spiking==host 1.00).
  LESION  : flag on, but the heard corpus is PERMUTED first (patient animacy shuffled — the mechanism's OWN
            anti-cheat control; the 2026-07-18 finding measured this collapses the learned animacy/selection
            signs). If the wire-in is genuinely routing the CORPUS-DERIVED computation into the agent's decision
            (not silently degenerating to a fixed host-matching answer regardless of what it learned), the
            PERMUTED arm's resolution must DIVERGE from the clean ON arm — this is the LOAD-BEARING check.

GO(seed) = off_byte_identical (agent untouched, host lexicon answers) AND on_installed (the learned chooser is
           actually installed) AND on_matches_host (it reproduces the host disambiguation) AND lesion_diverges
           (corrupting the learning corpus changes the outcome — the wire-in is not a no-op read-through).

De-risk GO (the mechanism this routes): research/findings/2026-07-18-gap3-A1-learned-feature-compatibility-cheap-
first-GO.md (6-seed GO, spiking==host 1.00, permuted-corpus anti-cheat collapses to 0.00). CI guard:
tests/test_gap3_spiking_feature_compat.py. Flag organ: research/runners/biased_competition_prod.py.

  Run: SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._learned_referent_bias_flip_soak \
          --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.multi_turn_agent import MultiTurnAgent                      # noqa: E402
from research.runners.biased_competition_buffer import ANIMACY, VERB_SELECTS, content_bias_target  # noqa: E402
from research.runners._gap3_learned_feature_compat_derisk import make_corpus       # noqa: E402
from research.runners.biased_competition_prod import (                            # noqa: E402
    BRAIN_BIASED_COMPETITION_LEARNED_BIAS_ENV, learned_bias_enabled, maybe_install_learned_referent_bias)

ALL_CONCEPTS = list(ANIMACY.keys())
FULL_VOCAB = ALL_CONCEPTS + list(VERB_SELECTS.keys())
PROBE_VERB = "eat"                  # selects animate (host content_bias_target(['cat','ball'], 'eat') == 'cat')
PROBE_PAIR = ("cat", "ball")        # (animate, inanimate)
OUT = _REPO / "research" / "findings" / "raw" / "_learned_referent_bias_flip_soak" / "soak_seed42.json"


def _mk_agent(seed):
    """Construct EXACTLY as a wired production build site does post-fix: `enable_biased_competition=True` (the WTA
    gate is the SEPARATE, already-flipped `BRAIN_BIASED_COMPETITION` flag — held fixed ON here) and NO
    `feat_compat_source` passed at construction — the production sites don't pass one either; the new flag installs
    it AFTER construction, from the agent's own heard experience, via `maybe_install_learned_referent_bias`."""
    return MultiTurnAgent(referent_concepts=ALL_CONCEPTS, concepts={w: None for w in FULL_VOCAB}, seed=seed,
                          enable_biased_competition=True)


def _teach(agent, facts):
    """Feed the synthetic SVO corpus into the agent's OWN fact store, exactly as `heard_facts()` reads it back
    (mirrors `tests/test_gap3_spiking_feature_compat.py::test_agent_learns_referent_bias_from_own_experience`)."""
    for ag, v, pt in facts:
        agent.agent.composer.store(ag, v, pt)


def _run_arm(seed, flag_on, permute_corpus=False, n_facts=120):
    os.environ[BRAIN_BIASED_COMPETITION_LEARNED_BIAS_ENV] = "1" if flag_on else "0"
    assert learned_bias_enabled() is flag_on
    facts = make_corpus(seed, n=n_facts, permute=permute_corpus)
    a = _mk_agent(seed)
    _teach(a, facts)
    installed = maybe_install_learned_referent_bias(a, seed=seed)
    a._write_referent(PROBE_PAIR[0])
    a._write_referent(PROBE_PAIR[1])
    held_ok = (a._held_set() == sorted(PROBE_PAIR))
    resolved = a._resolve_biased(PROBE_VERB)
    return {"installed": bool(installed), "feat_compat_is_none": a._feat_compat_source is None,
            "held_ok": held_ok, "resolved": resolved, "n_heard_facts": len(a.heard_facts())}


def run_one(seed):
    t0 = time.time()
    print("\n" + "=" * 112)
    print(f"[learned-bias-soak] seed={seed} — OFF (host lexicon) vs ON (learned spiking compat) vs LESION "
          f"(permuted-corpus, must diverge)", flush=True)
    host = content_bias_target(list(PROBE_PAIR), PROBE_VERB)
    result = {"seed": seed, "host": host}
    try:
        off = _run_arm(seed, flag_on=False)
        on = _run_arm(seed, flag_on=True)
        lesion = _run_arm(seed, flag_on=True, permute_corpus=True)

        off_byte_identical = (off["installed"] is False and off["feat_compat_is_none"] is True
                              and off["held_ok"] and off["resolved"] == host)
        on_installed = (on["installed"] is True and on["feat_compat_is_none"] is False and on["held_ok"])
        on_matches_host = (on["resolved"] == host)
        # LOAD-BEARING: the permuted-corpus arm must NOT silently reproduce the clean ON answer -- either it fails
        # to install a usable map at all (falls back to abstain), or it installs a WRONG map that resolves
        # differently. Either outcome proves the wire-in is actually running the corpus-derived computation.
        lesion_diverges = (lesion["installed"] is False) or (lesion["resolved"] != on["resolved"])

        GO = bool(off_byte_identical and on_installed and on_matches_host and lesion_diverges)
        result.update(dict(GO=GO, off_byte_identical=off_byte_identical, on_installed=on_installed,
                           on_matches_host=on_matches_host, lesion_diverges=lesion_diverges,
                           off=off, on=on, lesion=lesion))
        print(f"[learned-bias-soak] host={host!r} | OFF(installed={off['installed']},resolved={off['resolved']!r}) "
              f"| ON(installed={on['installed']},resolved={on['resolved']!r}) | "
              f"LESION(installed={lesion['installed']},resolved={lesion['resolved']!r})", flush=True)
        print(f"[learned-bias-soak] seed={seed} off_byte_identical={off_byte_identical} on_installed={on_installed} "
              f"on_matches_host={on_matches_host} lesion_diverges={lesion_diverges} => {'GO' if GO else 'NO-GO'}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    finally:
        os.environ[BRAIN_BIASED_COMPETITION_LEARNED_BIAS_ENV] = "0"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    results = {}
    go = []
    for seed in seeds:
        r = run_one(seed)
        results[seed] = r
        go.append(bool(r.get("GO")))
    out_path = Path(args.out)
    if len(seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(seeds)}seed.json"
        print("\n" + "#" * 112)
        print(f"[learned-bias-soak] {len(seeds)}-SEED SOAK: GO {int(sum(go))}/{len(seeds)} seeds={seeds}")
        print("#" * 112)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": seeds, "n_go": int(sum(go)), "go": go,
                                    "results": {str(s): results[s] for s in seeds}}, indent=2, default=str))
    print(f"[learned-bias-soak] wrote {out_path}")
    return 0 if (go and all(go)) else 1


if __name__ == "__main__":
    sys.exit(main())
