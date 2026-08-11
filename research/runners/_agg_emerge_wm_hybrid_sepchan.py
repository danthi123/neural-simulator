"""Aggregate the FANNED per-seed _emerge_wm_hybrid_sepchan runs into one 6-seed summary + apply the GO gate.
Each per-seed json was produced by `-m research.runners._emerge_wm_hybrid_sepchan_derisk --seeds <s>`; this pools the
single-seed raw dicts (points[0].per_seed[0]) across seeds, re-aggregates with the runner's own agg(), and re-decides.

Usage:
  python -m research.runners._agg_emerge_wm_hybrid_sepchan research/findings/raw/_emerge_wm_hybrid_sepchan/seed_*.json \
      --out research/findings/raw/_emerge_wm_hybrid_sepchan/sepchan_6seed.json
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

from research.runners._emerge_wm_hybrid_sepchan_derisk import agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-seed json paths (globs ok)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = []
    for pat in a.inputs:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    per = []
    seeds = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        gp = d.get("go_point")
        if gp is None or not gp.get("per_seed"):
            print(f"  SKIP {p}: no go_point/per_seed"); continue
        for ps in gp["per_seed"]:
            per.append(ps); seeds.append(ps["seed"])
    if not per:
        raise SystemExit("no per-seed points found")
    far = agg(per)
    chance = far["chance"]
    base = max(far["htm_exact"], far["wm_exact"])

    gen = far["gen_defined"]
    beats_both = far["hybrid_sep_exact"] >= base + 0.20
    above_chance = far["hybrid_sep_exact"] >= chance + 0.30
    subj_preserved = far["hybrid_sep_subj"] >= 0.90
    cls_clean = far["hybrid_sep_cls"] >= 0.90
    wm_chan_lb = far["hybrid_sep_exact"] >= far["lesion_wm_chan_exact"] + 0.20 and far["lesion_wm_chan_exact"] <= far["htm_exact"] + 0.10
    htm_chan_lb = far["hybrid_sep_exact"] >= far["lesion_htm_chan_exact"] + 0.20 and far["lesion_htm_chan_exact"] <= far["wm_exact"] + 0.10
    hold_lb = far["hybrid_sep_exact"] >= far["lesion_hold_exact"] + 0.20
    no_leak = far["subj_shuffle_exact"] <= base + 0.05
    bind_learned = far["conj_untrained_exact"] <= chance + 0.10
    spikes_ok = far["zero_input_ok"] and far["hold_alive"] > 1e-3
    n6 = len(set(seeds)) >= 6
    core = bool(gen and beats_both and above_chance and subj_preserved and cls_clean and wm_chan_lb and htm_chan_lb
                and hold_lb and no_leak and bind_learned and spikes_ok)
    go = bool(core and n6)

    checks = {"n_seeds": len(set(seeds)), "gen_defined": gen, "beats_both(+0.20)": beats_both,
              "above_chance(+0.30)": above_chance, "subject_preserved(>=0.90)": subj_preserved,
              "class_clean(>=0.90)": cls_clean, "wm_channel_load_bearing": wm_chan_lb,
              "htm_channel_load_bearing": htm_chan_lb, "hold_load_bearing": hold_lb, "no_leak": no_leak,
              "conjunction_learned": bind_learned, "spikes_ok": spikes_ok, "CORE": core, "GO(6-seed)": go}

    print(f"\n== 6-SEED AGGREGATE (seeds {sorted(set(seeds))}) ==")
    print(f"{'arm':<22}{'exact':>18}{'subject':>10}{'class':>10}")
    def row(name, ek, sk=None, ck=None):
        e = f"{far[ek]:.3f}"
        if ek + '_min' in far:
            e += f"[min {far[ek+'_min']:.3f}]"
        s = f"{far[sk]:.3f}" if sk else "—"
        c = f"{far[ck]:.3f}" if ck else "—"
        print(f"{name:<22}{e:>18}{s:>10}{c:>10}")
    row("HTM-alone", "htm_exact", "htm_subj", "htm_cls")
    row("WM-alone", "wm_exact", "wm_subj", "wm_cls")
    row("old-fusion (ref)", "old_fusion_exact", "old_fusion_subj", "old_fusion_cls")
    row("HYBRID-SEP", "hybrid_sep_exact", "hybrid_sep_subj", "hybrid_sep_cls")
    row("lesion-WM-chan", "lesion_wm_chan_exact", "lesion_wm_chan_subj", "lesion_wm_chan_cls")
    row("lesion-HTM-chan", "lesion_htm_chan_exact", "lesion_htm_chan_subj", "lesion_htm_chan_cls")
    row("lesion-hold", "lesion_hold_exact")
    row("subj-shuffle", "subj_shuffle_exact", "subj_shuffle_subj")
    row("conj-untrained", "conj_untrained_exact")
    row("n-gram floor", "ngram_floor_exact")
    print(f"{'chance':<22}{chance:>18.3f}")
    print(f"slot_decode_acc={far['slot_decode_acc']:.3f}  cls_chan_acc={far['cls_chan_acc']:.3f}  hold_alive={far['hold_alive']:.4f}  zero_input_ok={far['zero_input_ok']}")
    print("\nGO checks:")
    for k, v in checks.items():
        print(f"  {'OK ' if v is True else ('..' if v is False else '  ')} {k}: {v}")
    verdict = ("6-SEED GO" if go else ("6-SEED CORE-PASS(need 6 seeds)" if core else "6-SEED NEGATIVE/PARTIAL")) + \
              f" — HYBRID-SEP {far['hybrid_sep_exact']:.3f}[min {far.get('hybrid_sep_exact_min', float('nan')):.3f}] " \
              f"(subj {far['hybrid_sep_subj']:.3f} cls {far['hybrid_sep_cls']:.3f}) vs old-fusion {far['old_fusion_exact']:.3f} " \
              f"(subj {far['old_fusion_subj']:.3f}); bar max(HTM {far['htm_exact']:.3f}, WM {far['wm_exact']:.3f})+0.20 = {base+0.20:.3f}."
    print("\n" + verdict)

    out = {"probe": "emerge_wm_hybrid_sepchan_6seed_aggregate", "seeds": sorted(set(seeds)), "go": go, "core": core,
           "verdict": verdict, "checks": checks, "aggregate": far, "inputs": paths}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
