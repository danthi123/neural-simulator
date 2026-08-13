"""Standalone numpy-CPU verify for the D5-episodic production organ (research/runners/d5_episodic_production_organ.py).

Proves, on the numpy substrate (the production test backend), all four load-bearing properties on ONE seed at the
GO config (kt=8), reusing the committed EpisodicDapMemory 6/6-GO mechanism through the production organ glue:

  1. FLAG SEMANTICS (byte-identical escape) — default ON; BRAIN_EPISODIC=0 -> off; BRAIN_EPISODIC_LESION=1 -> lesion.
     And: an organ that never stores builds NO substrate (mem is None) -> the disabled/idle path costs nothing.
  2. INTACT FIRES — note_topic('dog') (spiking BTSP store) then recall('dog') COMPLETES cue-specifically:
     apical_cue >= 0.20, perm ~ 0, nocue ~ 0, in_memory True.  [the genuine spiking recall gate]
  3. LESION COLLAPSES (load-bearing) — recall('dog', lesion=True) reads through the UNFORMED baseline recurrent
     weights -> apical_cue ~ 0, in_memory False. Proves the recall is carried by the BTSP-formed assembly, not glue.
  4. UNSTORED ABSTAINS (honesty floor) — recall('cat') (never stored) -> apical_cue ~ 0, in_memory False -> the
     disclosure is an honest "don't recall", never a confabulation. And discussed() == ['dog'] (spiking decode).

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._verify_d5_episodic_organ --seed 42 \
          --out research/findings/raw/_d5_episodic_organ/verify_s42_numpy.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_d5_episodic_organ/verify_s42_numpy.json")
    args = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")

    import research.runners.d5_episodic_production_organ as D5
    from sim.backend import get_backend
    _, _dev = get_backend()

    t0 = time.time()
    result = {"seed": args.seed, "backend": os.environ.get("SIM_BACKEND"), "checks": {}, "reads": {}}

    # ---- (1) FLAG SEMANTICS -------------------------------------------------------------------------------------
    flags = {}
    for env, expect in (({}, True), ({"BRAIN_EPISODIC": "0"}, False), ({"BRAIN_EPISODIC": "off"}, False),
                        ({"BRAIN_EPISODIC": "1"}, True)):
        saved = os.environ.pop("BRAIN_EPISODIC", None)
        os.environ.update(env)
        got = D5.episodic_enabled()
        os.environ.pop("BRAIN_EPISODIC", None)
        if saved is not None:
            os.environ["BRAIN_EPISODIC"] = saved
        flags[str(env)] = {"got": got, "expect": expect, "ok": got == expect}
    for env, expect in (({}, False), ({"BRAIN_EPISODIC_LESION": "1"}, True), ({"BRAIN_EPISODIC_LESION": "0"}, False)):
        saved = os.environ.pop("BRAIN_EPISODIC_LESION", None)
        os.environ.update(env)
        got = D5.episodic_lesioned()
        os.environ.pop("BRAIN_EPISODIC_LESION", None)
        if saved is not None:
            os.environ["BRAIN_EPISODIC_LESION"] = saved
        flags["lesion:" + str(env)] = {"got": got, "expect": expect, "ok": got == expect}
    flags_ok = all(v["ok"] for v in flags.values())
    result["checks"]["flag_semantics"] = {"ok": flags_ok, "detail": flags}

    # an organ that never stores builds NO substrate -> the idle/off path is free (byte-identical)
    idle = D5.EpisodicRecallOrgan(args.seed, ["cat", "dog"])
    idle_no_substrate = idle.mem is None
    idle_recall = idle.recall("dog")   # nothing stored yet -> honest not-in-memory, no build
    result["checks"]["idle_builds_nothing"] = {
        "ok": bool(idle_no_substrate and idle_recall["in_memory"] is False and idle.mem is None),
        "mem_is_none": idle.mem is None, "idle_recall_in_memory": idle_recall["in_memory"]}
    print(f"[verify] flags_ok={flags_ok} idle_no_substrate={idle_no_substrate} "
          f"(+{time.time()-t0:.1f}s)", flush=True)

    # ---- build the organ + STORE 'dog' (spiking BTSP write) -----------------------------------------------------
    org = D5.get_episodic_organ(("verify", args.seed), args.seed, ["cat", "dog"], verbose=True)
    t_store = time.time()
    wrote = org.note_topic("dog")
    print(f"[verify] note_topic('dog') wrote={wrote} store_took={time.time()-t_store:.1f}s "
          f"assembly_sizes={org.mem.assembly_sizes} n_ca3={org.mem.n_ca3}", flush=True)

    # ---- (2) INTACT FIRES : recall('dog') completes cue-specifically --------------------------------------------
    t_r = time.time()
    dog = org.recall("dog", lesion=False)
    print(f"[verify] recall('dog') intact = {dog} (read {time.time()-t_r:.1f}s)", flush=True)
    result["reads"]["dog_intact"] = dog

    # ---- (3) LESION COLLAPSES : recall('dog', lesion=True) reads baseline weights -------------------------------
    t_l = time.time()
    dog_les = org.recall("dog", lesion=True)
    print(f"[verify] recall('dog') LESION = {dog_les} (read {time.time()-t_l:.1f}s)", flush=True)
    result["reads"]["dog_lesion"] = dog_les

    # ---- (4) UNSTORED ABSTAINS : recall('cat') never stored -----------------------------------------------------
    t_c = time.time()
    cat = org.recall("cat", lesion=False)
    print(f"[verify] recall('cat') unstored = {cat} (read {time.time()-t_c:.1f}s)", flush=True)
    result["reads"]["cat_unstored"] = cat

    discussed, _ = org.discussed(lesion=False)
    result["reads"]["discussed"] = list(discussed)

    # ---- ATTRIBUTION: whose is the recall? intact dAP completion vs the lesioned (baseline-weights) control -------
    # attributable_to = (treatment - control)/treatment: the FRACTION of the completion carried by the BTSP-formed
    # assembly (not the organ glue / feedforward cue). ~1.0 => the recall is load-bearing on the spiking engram.
    from tools.lab import attributable_to
    attr = attributable_to("dog recall completion: dAP intact vs lesioned baseline weights",
                           float(dog["apical_cue"]), float(dog_les["apical_cue"]))
    result["attribution"] = {"dog_recall_attributable_to_formed_assembly": attr}
    print(f"[verify] attribution: {attr} of the dog recall is carried by the BTSP-formed assembly "
          f"(intact {dog['apical_cue']:.3f} vs lesion {dog_les['apical_cue']:.3f})", flush=True)

    # ---- verdicts ------------------------------------------------------------------------------------------------
    COMPLETE_MIN, CTRL_MAX = 0.20, 0.10
    intact_fires = bool(dog["in_memory"] and dog["apical_cue"] >= COMPLETE_MIN
                        and dog["apical_perm"] <= CTRL_MAX and dog["apical_nocue"] <= CTRL_MAX)
    lesion_collapses = bool((not dog_les["in_memory"]) and dog_les["apical_cue"] <= CTRL_MAX
                            and dog_les["apical_cue"] < dog["apical_cue"]
                            and attr is not None and attr >= 0.9)
    unstored_abstains = bool((not cat["in_memory"]) and cat["apical_cue"] <= CTRL_MAX)
    discussed_ok = bool(list(discussed) == ["dog"])
    disclosure_dog = D5.recall_disclosure(dog, content="A dog runs north.")
    disclosure_cat = D5.recall_disclosure(cat)

    result["checks"]["intact_fires"] = {"ok": intact_fires, "apical_cue": dog["apical_cue"],
                                        "perm": dog["apical_perm"], "nocue": dog["apical_nocue"]}
    result["checks"]["lesion_collapses"] = {"ok": lesion_collapses, "intact_cue": dog["apical_cue"],
                                            "lesion_cue": dog_les["apical_cue"]}
    result["checks"]["unstored_abstains"] = {"ok": unstored_abstains, "cat_cue": cat["apical_cue"]}
    result["checks"]["discussed_decode"] = {"ok": discussed_ok, "discussed": list(discussed)}
    result["disclosures"] = {"dog": disclosure_dog, "cat": disclosure_cat}

    all_ok = bool(flags_ok and result["checks"]["idle_builds_nothing"]["ok"]
                  and intact_fires and lesion_collapses and unstored_abstains and discussed_ok)
    result["ALL_OK"] = all_ok
    result["wall_s"] = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n==================== D5-EPISODIC ORGAN VERIFY ====================", flush=True)
    print(f"  (1) flag semantics + idle-builds-nothing : {flags_ok and result['checks']['idle_builds_nothing']['ok']}",
          flush=True)
    print(f"  (2) INTACT FIRES  dog cue={dog['apical_cue']:.3f} perm={dog['apical_perm']:.3f} "
          f"nocue={dog['apical_nocue']:.3f} in_memory={dog['in_memory']} : {intact_fires}", flush=True)
    print(f"  (3) LESION COLLAPSES  intact={dog['apical_cue']:.3f} -> lesion={dog_les['apical_cue']:.3f} : "
          f"{lesion_collapses}", flush=True)
    print(f"  (4) UNSTORED ABSTAINS  cat cue={cat['apical_cue']:.3f} in_memory={cat['in_memory']} : "
          f"{unstored_abstains}", flush=True)
    print(f"      discussed decode = {list(discussed)} : {discussed_ok}", flush=True)
    print(f"      disclosure(dog) = {disclosure_dog}", flush=True)
    print(f"      disclosure(cat) = {disclosure_cat}", flush=True)
    print(f"  ALL_OK = {all_ok}   (wall {result['wall_s']:.1f}s)  -> {args.out}", flush=True)
    print("=================================================================", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
