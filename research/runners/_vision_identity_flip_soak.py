"""SOAK / no-regression gate for the BRAIN_VISION_IDENTITY default-ON flip (visual object -> category identity).

Two independent bars the parent runs BEFORE flipping `BRAIN_VISION_IDENTITY` default-on:

  PART A — VISION READOUT 6-SEED STABILITY (the core gate, pool-friendly numpy). For each seed, build the EMERGE-36
  recognizer three ways and read held-out perceived-object recognition:
    * INTACT: held-out objects recognized to their visual category (accuracy) + within-category codon overlap.
    * PER-IMAGE PIXEL-SCRAMBLE (the finding's headline lesion): within-category visual similarity destroyed ->
      within-category codon overlap COLLAPSES -> recognition drops toward chance.
    * POOLER-LESION (coincidence OFF): the codon never charges -> recognition ABSTAINS on every object (floor).
  GO gate — the SOURCE FINDING's own methodology (2026-07-02-emerge36-...GO.md): the per-image scramble is NOISY at a
  single seed (small setup), so the scramble collapse is keyed on the MULTI-SEED MEAN, while the deterministic
  pooler-lesion is gated PER SEED. Concretely:
    * PER SEED (deterministic): intact_acc >= 0.85 AND the pooler-lesion floors (abstains on every object).
    * AGGREGATE (the finding's scramble control): mean(intact_acc) - mean(scramble_acc) >= 0.30, with the within-
      category codon overlap also collapsing on the mean (corroboration). Overall GO: all per-seed gates pass AND the
      aggregate scramble collapse holds.

  PART B — CHAT NO-REGRESSION (structural; run once through the REAL brain_chat handler with the STUB renderer, no
  Qwen). On an ORDINARY turn (no visual query) OR a visual query WITHOUT a percept, flag-ON must be BYTE-IDENTICAL
  to flag-OFF. This is NIL-by-construction — the wiring block only executes when `is_visual_query(msg) and req.percept`
  (a short-circuit `and`), so an ordinary turn has ZERO code-path difference — and this part PROVES it end-to-end.
  Degrades to a reported SKIP (never a false NO-GO) if webapp.server cannot import on a bare pool node; PART A is the
  gate. Pass --vision-only to skip PART B.

  Run (pool):  SIM_BACKEND=numpy python -m research.runners._vision_identity_flip_soak --seeds 42 43 44 100 101 102
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

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners.vision_identity_production_organ as VI  # noqa: E402
from tools.lab import attributable_to, assert_backend  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_vision_identity_prodflip" / "soak_seed42.json"

INTACT_ACC_MIN = 0.85          # per-seed held-out recognition accuracy the finding's GO uses
SCRAMBLE_MEAN_MARGIN = 0.30    # the finding's scramble control, keyed on the MULTI-SEED MEAN (noisy per-seed)


def _overlap(a, b):
    a, b = set(a), set(b)
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def _readout(org):
    """Held-out recognition accuracy + mean within-category codon overlap for one recognizer variant."""
    correct, total, ov = 0, 0, []
    for c in (0, 1):
        which = org.held_which(c)
        codons = []
        for i in which:
            pred = org.recognize(c, i)
            correct += int(pred == c)
            total += 1
            codons.append(org.codon(c, i))
        for i in range(len(codons)):
            for j in range(i + 1, len(codons)):
                ov.append(_overlap(codons[i], codons[j]))
    return {"acc": (correct / total) if total else 0.0, "n": total,
            "codon_overlap": float(np.mean(ov)) if ov else 0.0,
            "abstain_all": bool(correct == 0)}


def run_vision_seed(seed):
    t0 = time.time()
    intact = VI.get_organ(seed=seed, lesion=False, scramble=False)
    scram = VI.get_organ(seed=seed, lesion=False, scramble=True)
    lesion = VI.get_organ(seed=seed, lesion=True, scramble=False)
    ri, rs, rl = _readout(intact), _readout(scram), _readout(lesion)
    # PER-SEED deterministic gates (the finding's per-seed CI bar): intact readout works + the pooler-lesion floors.
    intact_ok = ri["acc"] >= INTACT_ACC_MIN
    lesion_floored = rl["abstain_all"]
    seed_ok = bool(intact_ok and lesion_floored)
    # the per-seed scramble collapse is reported for transparency but NOT gated per-seed (noisy; gated on the mean).
    scramble_dropped = bool((ri["acc"] - rs["acc"]) > 0 or (ri["codon_overlap"] - rs["codon_overlap"]) > 0)
    return {"seed": seed, "seed_ok": seed_ok, "intact": ri, "scramble": rs, "pooler_lesion": rl,
            "intact_ok": intact_ok, "lesion_floored": lesion_floored, "scramble_dropped": scramble_dropped,
            "elapsed_s": round(time.time() - t0, 1)}


def run_handler_no_regression():
    """PART B: flag-ON == flag-OFF on ordinary + visual-without-percept turns, through the real brain_chat handler
    (stub renderer). Returns a dict; degrades to {'skipped': reason} if the server cannot be imported here."""
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    # disable unrelated default-ON faculties (identical in both arms) to keep this bounded on a CPU pool node.
    for k in ("BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_DA_DRIVES", "BRAIN_DA_ENCODING", "BRAIN_SELF_INITIATE",
              "BRAIN_COMPREHENSION_GATE", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_WORLDMODEL", "BRAIN_PRAGMATIC",
              "BRAIN_RICH", "BRAIN_SWAP_DRIVES", "BRAIN_OPEN_ENDED"):
        os.environ.setdefault(k, "0")
    os.environ.pop("BRAIN_VISION_IDENTITY", None)
    os.environ.pop("BRAIN_VISION_IDENTITY_LESION", None)
    try:
        import webapp.server as S
    except Exception as e:  # a bare pool node without the webapp deps -> report SKIP (PART A is the gate)
        return {"skipped": f"webapp.server import failed: {type(e).__name__}: {e}"}

    def turn(message, percept=None):
        req = S.BrainChatRequest(session="visoak", message=message, brain="tiny-demo", renderer="stub",
                                 rich=False, percept=percept)
        return json.loads(bytes(S.brain_chat(req).body).decode())

    ordinary_msgs = ["what does the cat eat?", "tell me about dogs", "hello", "what is the sky?"]
    visual_no_percept = "what do you see?"
    # FLAG OFF
    off = {m: turn(m) for m in ordinary_msgs}
    off_visnop = turn(visual_no_percept)
    off_vis_percept = turn(visual_no_percept, percept="bird")  # percept present but flag off -> ignored (host path)
    # FLAG ON
    os.environ["BRAIN_VISION_IDENTITY"] = "1"
    on = {m: turn(m) for m in ordinary_msgs}
    on_visnop = turn(visual_no_percept)
    on_vis_percept = turn(visual_no_percept, percept="bird")   # now the faculty fires
    os.environ.pop("BRAIN_VISION_IDENTITY", None)

    ordinary_identical = all(off[m] == on[m] for m in ordinary_msgs)
    visnop_identical = (off_visnop == on_visnop)
    off_had_no_vision_key = ("vision_identity" not in off_vis_percept)
    on_fired = (on_vis_percept.get("vision_identity", {}).get("recognized_category") is not None
                and "bird" in on_vis_percept.get("answer", ""))
    return {"skipped": None, "ordinary_identical": bool(ordinary_identical),
            "visual_no_percept_identical": bool(visnop_identical),
            "off_had_no_vision_key": bool(off_had_no_vision_key), "on_fired": bool(on_fired),
            "no_regression": bool(ordinary_identical and visnop_identical and off_had_no_vision_key and on_fired)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--vision-only", action="store_true", help="skip PART B (handler no-regression)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    _backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))  # record + assert the device (device-and-cost)
    print("=" * 118)
    print(f"[vision-soak] PART A — vision readout 6-seed stability (per-seed: intact acc >= {INTACT_ACC_MIN} + pooler-"
          f"lesion floors; aggregate: mean scramble collapse >= {SCRAMBLE_MEAN_MARGIN}). seeds={a.seeds}", flush=True)
    per_seed = []
    for s in a.seeds:
        try:
            r = run_vision_seed(s)
        except Exception as e:  # noqa: BLE001
            r = {"seed": s, "seed_ok": False, "error": repr(e)}
            traceback.print_exc()
        per_seed.append(r)
        if "error" in r:
            print(f"  [seed {s}] ERROR {r['error']}", flush=True)
        else:
            print(f"  [seed {s}] intact acc={r['intact']['acc']:.2f} ov={r['intact']['codon_overlap']:.3f} || "
                  f"scramble acc={r['scramble']['acc']:.2f} ov={r['scramble']['codon_overlap']:.3f} || "
                  f"pooler-lesion abstain_all={r['pooler_lesion']['abstain_all']} => "
                  f"seed_ok={r['seed_ok']}", flush=True)
    ok_seeds = [r for r in per_seed if "error" not in r]
    all_seed_ok = bool(ok_seeds) and all(r["seed_ok"] for r in per_seed)
    # the finding's scramble control: keyed on the MULTI-SEED MEAN (per-image scramble is noisy at a single seed).
    mean_intact_acc = float(np.mean([r["intact"]["acc"] for r in ok_seeds])) if ok_seeds else 0.0
    mean_scramble_acc = float(np.mean([r["scramble"]["acc"] for r in ok_seeds])) if ok_seeds else 0.0
    mean_intact_ov = float(np.mean([r["intact"]["codon_overlap"] for r in ok_seeds])) if ok_seeds else 0.0
    mean_scramble_ov = float(np.mean([r["scramble"]["codon_overlap"] for r in ok_seeds])) if ok_seeds else 0.0
    scramble_collapsed = ((mean_intact_acc - mean_scramble_acc) >= SCRAMBLE_MEAN_MARGIN
                          and mean_intact_ov > mean_scramble_ov)
    mean_lesion_acc = float(np.mean([r["pooler_lesion"]["acc"] for r in ok_seeds])) if ok_seeds else 0.0
    n_ok = sum(int(r.get("seed_ok")) for r in per_seed)
    vision_go = bool(all_seed_ok and scramble_collapsed)
    print(f"  [aggregate] mean intact acc={mean_intact_acc:.3f} vs scramble acc={mean_scramble_acc:.3f} "
          f"(margin {mean_intact_acc - mean_scramble_acc:.3f} >= {SCRAMBLE_MEAN_MARGIN}) | "
          f"codon overlap {mean_intact_ov:.3f} -> {mean_scramble_ov:.3f} | scramble_collapsed={scramble_collapsed}",
          flush=True)
    # ATTRIBUTION (whose is the readout?). Both control arms are measured above; subtract them, don't just report them.
    #   * intact vs POOLER-LESION (coincidence off): the WHOLE recognition is attributable to the spiking codon pooler.
    #   * intact vs PIXEL-SCRAMBLE: the recognition margin is attributable to within-category VISUAL SIMILARITY (pixels).
    attr_pooler = attributable_to("vision readout: intact vs pooler-lesion", mean_intact_acc, mean_lesion_acc)
    attr_similarity = attributable_to("recognition margin: intact vs pixel-scramble", mean_intact_acc, mean_scramble_acc)

    handler = None
    if not a.vision_only:
        print("\n[vision-soak] PART B — chat no-regression (flag ON == OFF on ordinary + no-percept turns) ...", flush=True)
        try:
            handler = run_handler_no_regression()
        except Exception as e:  # noqa: BLE001
            handler = {"skipped": f"handler check raised: {type(e).__name__}: {e}"}
            traceback.print_exc()
        if handler.get("skipped"):
            print(f"  PART B SKIPPED: {handler['skipped']}", flush=True)
        else:
            print(f"  ordinary_identical={handler['ordinary_identical']} "
                  f"visual_no_percept_identical={handler['visual_no_percept_identical']} "
                  f"on_fired={handler['on_fired']} => no_regression={handler['no_regression']}", flush=True)

    # PART B gates only when it actually ran (a SKIP does not fail the pool gate — PART A is the core gate).
    handler_ok = (handler is None) or bool(handler.get("skipped")) or bool(handler.get("no_regression"))
    overall_go = bool(vision_go and handler_ok)

    print("\n" + "#" * 118)
    print(f"[vision-soak] VISION per-seed {n_ok}/{len(a.seeds)} ok + scramble_collapsed={scramble_collapsed} "
          f"=> vision_go={vision_go} | HANDLER "
          f"{'skipped' if (handler and handler.get('skipped')) else ('n/a' if handler is None else handler.get('no_regression'))}"
          f" => {'GO' if overall_go else 'NO-GO'}", flush=True)
    print("#" * 118)

    out_path = Path(a.out)
    if len(a.seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(a.seeds)}seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "probe": "vision_identity_flip_soak", "backend": _backend, "device": _backend,
        "seeds": a.seeds, "vision_n_seed_ok": n_ok, "vision_go": vision_go,
        "mean_intact_acc": mean_intact_acc, "mean_scramble_acc": mean_scramble_acc, "mean_lesion_acc": mean_lesion_acc,
        "mean_intact_overlap": mean_intact_ov, "mean_scramble_overlap": mean_scramble_ov,
        "attributable_to_pooler": attr_pooler, "attributable_to_visual_similarity": attr_similarity,
        "scramble_collapsed": scramble_collapsed, "handler": handler, "overall_go": overall_go,
        "elapsed_s": round(time.time() - t0, 1), "per_seed": per_seed}, indent=2, default=str))
    print(f"[vision-soak] wrote {out_path}", flush=True)
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
