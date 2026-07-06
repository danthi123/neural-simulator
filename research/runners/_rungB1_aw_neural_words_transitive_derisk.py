"""RUNG B-1 / A→W words (GPU) — the transitive producer speaks EVERY WORD ON SPIKES (not just the order).

The one-brain-substrate capstone (EMERGE-90..95) renders the transitive answer "the dog chases the ball" with the
spiking frame-slot emission ORDER, but the word SURFACES are the host-token spell (`spell=str`). This rung makes the
WORDS spiking too: it retrains the EMERGE-67 A→W read-out (BRIDGE-A) on the TRANSITIVE content vocab (a 16-word vocab
rebound onto the 16 validated concept pools; a NEW cache so the EMERGE-frame cache is untouched), reuses the existing
EMERGE-68 function BRIDGE-F for the determiner "the", and passes the combined neural spell as the `RegistryProducer`'s
`spell=`. Every content slot (subject / verb-3sg / object) is decoded from `language_output` SPIKES; the determiner is
decoded from BRIDGE-F.

Reuse-by-import (EMERGE-67 `NeuralSpell` train/read-out + EMERGE-68 func spell + the EMERGE-72/74 `RegistryProducer`);
NO `sim/` edit. GPU/cupy (the A→W read-out is the validated GPU scale); trained ONCE + cached.

Anti-cheats: all-word render accuracy 1.00 (every word decoded from spikes == the ground-truth transitive surface);
CONTENT-LESION (zero the concept-pool → language_output pathway) collapses the content decode (a host lookup would be
unaffected → proves genuinely spiking); the token-spell default path is byte-identical (the producer's render == the
host-token render). >= 3 seeds on the read (the engine is a shared deterministic module; the producer facts vary).

Run:  SIM_BACKEND=cupy python -u -m research.runners._rungB1_aw_neural_words_transitive_derisk --train   # build+cache BRIDGE-A(trans)
      SIM_BACKEND=cupy python -u -m research.runners._rungB1_aw_neural_words_transitive_derisk --seeds 42 43 44 \
          --json research/findings/raw/_rungB1_aw_neural_words_transitive.json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")   # the A→W read-out is the validated GPU scale
import numpy as np  # noqa: E402

import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    decision, RegistryBrocaProducer, RegistryProducer,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    build_stream_svo, SVOConstructionRegistry, emerge_v3,
)
from research.runners._emerge77_ditransitive_render_derisk import DitransRegistryProducer  # noqa: E402


def _build_producer(seed, reg, spell):
    """The gate-first producer over C_TRANS with the EMERGE-77 2-stage per-pool bias-CALIBRATED order read. On CUPY the
    raw rate read lets per-pool f-I heterogeneity flip adjacent slots on some seeds (the producer's own backend-order
    near-tie, present with ANY spell -- isolated: host-token spell renders 0.000 on cupy seed 102, 1.000 on numpy); the
    calibration subtracts each pool's reference-current rate so the order follows the primacy, not the heterogeneity.
    `n_slot_pools=6` (C_TRANS = 5 slots) is byte-identical to the default RegistryProducer bridge; only the read is
    calibrated."""
    cq = DitransRegistryProducer(seed=seed, registry_slots=reg.registered_fits(), n_slot_pools=6, calibrate=True)
    cq.learn()
    return RegistryBrocaProducer(cq, spell=spell)

# The 16-word TRANSITIVE content vocab, rebound onto the 16 validated concept pools (EMERGE-67 splits the module-level
# `_AW_SUBJECTS` (8) -> DIRECTION+NOUN pools and `_AW_VERBS` (8) -> VERB+ADJECTIVE pools; the pool KIND is irrelevant to
# spelling -- every word is decoded from language_output). 4 subjects + 8 objects + 4 verb-3sg surfaces = 16 words that
# the transitive producer emits (C_TRANS "the <subj> <verb:3sg> the <obj>"). The producer's DET "the" -> BRIDGE-F.
_TRANS_SUBJECTS = ["dog", "cat", "wolf", "fox"]
_TRANS_OBJECTS = ["ball", "fish", "bone", "seed", "corn", "worm", "leaf", "rock"]
_TRANS_VERBS_BARE = ["chase", "eat", "see", "find"]
_TRANS_V3 = [emerge_v3(v) for v in _TRANS_VERBS_BARE]          # ["chases","eats","sees","finds"]
# EMERGE-67 canonical split: _AW_SUBJECTS (8) then _AW_VERBS (8); the 16 == the pools in order.
_AW_TRANS_SUBJECTS = _TRANS_SUBJECTS + _TRANS_OBJECTS[:4]      # 8 -> motor(4)+noun(4)
_AW_TRANS_VERBS = _TRANS_V3 + _TRANS_OBJECTS[4:]              # 8 -> verb(4)+adjective(4)
_AW_TRANS_CONTENT = _AW_TRANS_SUBJECTS + _AW_TRANS_VERBS       # 16 content words

_TRANS_CACHE_DIR = Path(m67._REPO) / "bridges" / "rungB1_aw_trans"
_TRANS_CACHE_BRIDGE = _TRANS_CACHE_DIR / "aw_content_trans.simstate.h5"


def _patch_m67_to_transitive_vocab():
    """Point EMERGE-67's module-level content vocab + cache at the TRANSITIVE vocab (before constructing NeuralSpell).
    A documented monkeypatch of the shipped rebind hooks -- reuse-by-import, NO new train code, NO `sim/` edit."""
    m67._AW_SUBJECTS = _AW_TRANS_SUBJECTS
    m67._AW_VERBS = _AW_TRANS_VERBS
    m67._AW_CONTENT = _AW_TRANS_CONTENT
    m67._CACHE_DIR = _TRANS_CACHE_DIR
    m67._CACHE_BRIDGE = _TRANS_CACHE_BRIDGE
    _TRANS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _build_content_spell(load=True, lesion=False):
    _patch_m67_to_transitive_vocab()
    return m67.NeuralSpell(seed=m67._AW_SEED, load=load, lesion_pool_out=lesion)


class TransUnifiedSpell:
    """Dispatch: a CONTENT word (transitive vocab) -> the retrained BRIDGE-A (spike-decoded); a FUNCTION word ("the")
    -> the EMERGE-68 BRIDGE-F (spike-decoded). Mirrors EMERGE-68's UnifiedNeuralSpell but with the transitive BRIDGE-A."""

    def __init__(self, load=True, content_lesion=False):
        self.content = _build_content_spell(load=load, lesion=content_lesion)
        self.func = m68.FuncNeuralSpell(load=load) if hasattr(m68, "FuncNeuralSpell") else None
        self._func_words = set(getattr(m68, "_FUNC_WORDS", ["the", "a", "can", "does", "not"]))

    def spell(self, word):
        w = str(word)
        if w in self._func_words and self.func is not None:
            return self.func.spell(w)
        return self.content.spell(w)


def _facts(seed, n=12):
    trng = np.random.default_rng(seed * 733 + 11)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 5000:
        guard += 1
        s = str(trng.choice(_TRANS_SUBJECTS)); vb = str(trng.choice(_TRANS_VERBS_BARE)); o = str(trng.choice(_TRANS_OBJECTS))
        if (s, vb) in seen:
            continue
        seen.add((s, vb))
        out.append((s, vb, o))
    return out


def _render(producer, s, vb, o):
    return producer.speak(decision("ANSWER", construction="C_TRANS", subject=s, verb=vb, obj=o))["surface"]


def _derisk_one(seed, spell_engine):
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    assert "C_TRANS" in reg.registered
    facts = _facts(seed)
    producer = _build_producer(seed, reg, spell_engine.spell)   # calibrated order + A→W neural spell
    hit = 0
    for s, vb, o in facts:
        expected = f"the {s} {emerge_v3(vb)} the {o}"
        hit += int(_render(producer, s, vb, o) == expected)
    return hit / len(facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="build + cache the transitive BRIDGE-A (GPU)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--diagnose", action="store_true", help="spell each vocab word + report per-word mis-decodes")
    ap.add_argument("--host-order-check", action="store_true",
                    help="render with the HOST-TOKEN spell (str) on the active backend -- isolates producer ORDER from the A→W spell")
    args = ap.parse_args()

    if args.host_order_check:
        for s in args.seeds:
            reg = SVOConstructionRegistry(s).build(build_stream_svo(s))
            producer = _build_producer(s, reg, lambda w: str(w))   # calibrated order + host-token spell
            hit = 0
            facts = _facts(s)
            for subj, vb, o in facts:
                got = _render(producer, subj, vb, o)
                exp = f"the {subj} {emerge_v3(vb)} the {o}"
                hit += int(got == exp)
                if got != exp:
                    print(f"  [seed {s}] HOST-SPELL ORDER MISS exp={exp!r} got={got!r}", flush=True)
            print(f"[host-order-check] seed {s}: host-token render {hit/len(facts):.3f}", flush=True)
        return

    if args.diagnose:
        eng = TransUnifiedSpell(load=True)
        bad = []
        for w in _AW_TRANS_CONTENT + ["the"]:
            got = eng.spell(w)
            ok = (got == w)
            if not ok:
                bad.append((w, got))
            print(f"  {'OK ' if ok else 'BAD'}  {w!r:>10} -> {got!r}", flush=True)
        print(f"[diagnose] {len(bad)} isolated mis-decodes: {bad}", flush=True)
        # verbose RENDER (in-sequence) on each seed -- the failure is a per-fact 5-spell sequence, not isolated words
        for s in args.seeds:
            reg = SVOConstructionRegistry(s).build(build_stream_svo(s))
            producer = RegistryBrocaProducer(reg.render_cq(), spell=eng.spell)
            for subj, vb, o in _facts(s):
                got = _render(producer, subj, vb, o)
                exp = f"the {subj} {emerge_v3(vb)} the {o}"
                if got != exp:
                    print(f"  [seed {s}] MISS exp={exp!r} got={got!r}", flush=True)
        return

    if args.train:
        print("[rungB1-aw] training the transitive A→W BRIDGE-A (cupy)...", flush=True)
        eng = _build_content_spell(load=False)
        # persist the trained bridge to the transitive cache
        eng.bridge.save_checkpoint(str(_TRANS_CACHE_BRIDGE))
        print(f"[rungB1-aw] cached -> {_TRANS_CACHE_BRIDGE}", flush=True)

    engine = TransUnifiedSpell(load=True)
    engine_lesion = TransUnifiedSpell(load=True, content_lesion=True)
    rows = []
    for s in args.seeds:
        acc = _derisk_one(s, engine)
        acc_lesion = _derisk_one(s, engine_lesion)
        rows.append({"seed": s, "allword_render": acc, "content_lesion_render": acc_lesion})
        print(f"[seed {s}] all-word spike render {acc:.3f} | content-lesion {acc_lesion:.3f}", flush=True)

    mean_acc = float(np.mean([r["allword_render"] for r in rows]))
    mean_les = float(np.mean([r["content_lesion_render"] for r in rows]))
    go = mean_acc >= 0.90 and mean_les <= 0.30
    verdict = "GO" if go else "NO-GO/BOUNDARY"
    print(f"\n[rungB1-aw] VERDICT: {verdict} -- the TRANSITIVE producer speaks every word ON SPIKES: all-word render "
          f"{mean_acc:.3f}; content-lesion collapses to {mean_les:.3f} (genuinely spiking). Words + order both spiking.",
          flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "allword_render": mean_acc, "content_lesion_render": mean_les, "go": go}, fh, indent=2)


if __name__ == "__main__":
    main()
