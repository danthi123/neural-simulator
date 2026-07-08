"""PRODUCTIVE regular inflection ON SPIKES (the morphology residual surpass): a NOVEL 3sg verb form whose
whole-form lexeme was NEVER stored is produced NEURALLY as spell(STEM) + spell(AFFIX) -- both decoded on
spikes from `language_output` firing -- replacing the host `emerge_v3` string op. Pinker-Ullman procedural
route: the procedural system (Broca + basal ganglia) composes a productive inflection by concatenating a
bound-morpheme AFFIX to a lexically-retrieved STEM (research gate 2026-07-08).

The load-bearing claim: for verbs whose STEM is spellable (BRIDGE-1: run/jump/walk/sleep/play) but whose 3sg
surface (runs/jumps/walks/sleeps/plays) is stored in NO A->W bridge, the 3sg is produced correctly on spikes
as spell(stem) + spell("s"). Anti-cheats: (1) GENUINELY-PRODUCTIVE -- the 3sg form is in no bridge's vocab
(not a stored lexeme); (2) WRONG-AFFIX -- spell a different bound morpheme ("ed"/"ing") -> wrong surface
(the affix identity is load-bearing); (3) AFFIX-ABLATION -- no affix slot -> the bare stem (agrammatic, not
3sg). numpy. NO `sim/` edit. Requires the BRIDGE-1 + affix A->W checkpoints.
"""
from __future__ import annotations
import argparse

from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB as V1, WORD_TO_POOL as P1
from research.runners._realcorpus_train_affix_pool import VOCAB as VA, WORD_TO_POOL as PA
from research.runners._realcorpus_train_breadth_aw3 import VOCAB as V3   # to assert the 3sg is NOT stored there

BRIDGE1 = "bridges/breadth_aw/seed42.simstate.h5"
AFFIX = "bridges/affix_aw/seed42.simstate.h5"

# regular verbs whose STEM is in BRIDGE-1 but whose 3sg surface is stored in NO bridge -> genuinely productive.
NOVEL = [("run", "runs"), ("jump", "jumps"), ("walk", "walks"), ("sleep", "sleeps"), ("play", "plays")]


def run(seed=42, aw_seed=42, affix_seed=42):
    stem_spk = ConceptFrameSpeaker(BRIDGE1, seed=aw_seed, vocab=V1, word_to_pool=P1)   # stems: validated seed-42 bridge
    affix_path = f"bridges/affix_aw/seed{affix_seed}.simstate.h5"                      # affix decode: the new seed-dep component
    affix_spk = ConceptFrameSpeaker(affix_path, seed=affix_seed, vocab=VA, word_to_pool=PA)
    stored = set(V1) | set(V3) | set(VA)                     # everything the A->W bridges store as whole forms

    n_ok = n = 0
    productive = True
    for (stem, expect) in NOVEL:
        assert stem in V1, f"stem {stem} must be spellable (BRIDGE-1)"
        if expect in stored:                                 # the 3sg must NOT be a stored lexeme -> genuinely productive
            productive = False
        stem_sp = affix_sp = None
        stem_sp = stem_spk.spell(stem)                       # ON SPIKES (BRIDGE-1 stem pool -> language_output)
        affix_sp = affix_spk.spell("s")                      # ON SPIKES (affix pool -> language_output)
        surface = f"{stem_sp}{affix_sp}"                     # [stem][affix] concatenation (surface assembly)
        ok = (surface == expect)
        n += 1; n_ok += int(ok)
        print(f"  PRODUCE 3sg({stem}) -> spell('{stem_sp}')+spell('{affix_sp}') = \"{surface}\"  "
              f"{'[exact, stem+affix ON SPIKES, never stored]' if ok else '[got != '+expect+']'}", flush=True)

    # (2) WRONG-AFFIX control: the same stems with the WRONG bound morpheme -> wrong surface (affix is load-bearing)
    wrong = 0
    for (stem, expect) in NOVEL:
        bad = f"{stem_spk.spell(stem)}{affix_spk.spell('ed')}"    # e.g. "runed" -- wrong morpheme
        if bad == expect:
            wrong += 1
    wrong_affix_ok = (wrong == 0)                            # NONE of the wrong-affix forms match the 3sg

    # (3) AFFIX-ABLATION control: no affix -> the bare stem, which is NOT the 3sg surface (agrammatic)
    ablation_ok = all(stem_spk.spell(stem) != expect for (stem, expect) in NOVEL)

    print(f"  [wrong-affix] none of stem+'-ed' match the 3sg: {wrong_affix_ok}", flush=True)
    print(f"  [affix-ablation] the bare stem != the 3sg surface: {ablation_ok}", flush=True)
    return {"n_ok": n_ok, "n": n, "productive": productive,
            "wrong_affix_ok": wrong_affix_ok, "ablation_ok": ablation_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--affix-seed", type=int, default=42, help="the affix A->W bridge seed (the new seed-dependent component)")
    a = ap.parse_args()
    print(f"[productive inflection] a NOVEL 3sg (stored NOWHERE) produced ON SPIKES as spell(stem)+spell('-s') | "
          f"seed={a.seed} affix_seed={a.affix_seed}", flush=True)
    r = run(a.seed, affix_seed=a.affix_seed)
    go = (r["n_ok"] == r["n"] and r["n"] > 0 and r["productive"] and r["wrong_affix_ok"] and r["ablation_ok"])
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- PRODUCTIVE regular 3sg inflection is composed ON SPIKES "
          f"({r['n_ok']}/{r['n']} exact: spell(stem)+spell('-s'), genuinely productive={r['productive']} "
          f"[the 3sg form is a stored lexeme NOWHERE], wrong-affix collapses={r['wrong_affix_ok']}, "
          f"ablation collapses={r['ablation_ok']}); the host `emerge_v3` is replaced by neural affixation.", flush=True)


if __name__ == "__main__":
    main()
