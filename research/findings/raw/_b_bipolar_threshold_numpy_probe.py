"""Option B cheap-first (NUMPY, no GPU): does binarizing the bound vector to a per-dimension SIGN preserve the
VSA unbind?  The research synthesis (2026-06-05-spiking-opponency-literature-synthesis.md) recommends Option B
(bipolar threshold / MAP-B): replace the GRADED `onoff(bon-boff)` opponency with a per-dimension ON/OFF
winner-take-all -> a binary +-1 bound vector (biology's push-pull as a DECISION; cancels the common mode).

This probe ISOLATES the binarization question from the spiking question. It builds the IDEAL numpy bind (the +-1
Hadamard `s = sum_role role (x) filler`, NO spiking read noise) and compares, per role, the unbind+cleanup of:
  - GRADED ideal:   B_graded  = onoff(s)            (the current composer representation, ideal)
  - BIPOLAR ideal:  B_bipolar = onoff(sign(s))      (Option B: collapse to a per-dim sign)
against the true filler. The GRADED ideal MUST recover ~100% (a built-in sanity check that the +-1 reconstruction
matches the composer); the BIPOLAR recovery is the load-bearing number.

GATE (cheap-first): bipolar recovery >= graded recovery (binarization does NOT lose the VSA) across seeds, at the
fact loads the composer actually stores (3 flat SVO + 1 one-attribute). If bipolar holds -> build the spiking
per-dim WTA (the real Option B); if bipolar collapses -> binarization itself kills the VSA -> Option A (FHRR) or
Option D (honest boundary).

  python -u -m research.findings.raw._b_bipolar_threshold_numpy_probe --seeds 42 43 44 --proj-dim 800 \
      --out research/findings/raw/_b_bipolar_threshold_numpy.json
"""
import argparse
import json
import numpy as np

from research.runners.core_sim_composition import CoreSimComposer, onoff

ROLES = ("agent", "action", "patient")


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ideal_signed_bind(comp, fact, roles):
    """The IDEAL numpy +-1 Hadamard bind: s = sum_role role_code (x) filler_code (elementwise +-1 products), NO
    spiking. Unbind is the SAME op (role (x) (role (x) filler) = filler since role (x) role = 1)."""
    s = np.zeros(comp.D)
    for role in roles:
        if role not in fact:
            continue
        s = s + comp.roles[role] * comp._filler_signed(fact[role])
    return s


def eval_seed(seed, proj_dim, n_flat, n_attr):
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    rng = np.random.default_rng(seed)

    def pick(k):
        return [str(x) for x in rng.choice(usable, size=k, replace=False)]

    facts = []
    for _ in range(n_flat):
        a, ac, p = pick(3)
        facts.append(({"agent": a, "action": ac, "patient": p}, ROLES))
    for _ in range(n_attr):
        a, ac, adj, noun = pick(4)
        facts.append(({"agent": a, "action": ac, "patient": noun, "attribute": adj},
                      ("agent", "action", "patient", "attribute")))

    n_total = 0
    n_graded = 0
    n_bipolar = 0
    n_bip_eq_grad = 0
    bipolar_signed_cos = []
    for fact, roles in facts:
        s = ideal_signed_bind(comp, fact, roles)
        Bg = onoff(s)                  # graded ideal
        Bb = onoff(np.sign(s))         # bipolar (Option B)
        bipolar_signed_cos.append(_cos(np.sign(s), s))
        sg = Bg[0] - Bg[1]             # == s
        sb = Bb[0] - Bb[1]             # == sign(s)
        for role in roles:
            if role not in fact:
                continue
            true = fact[role]
            est_g = comp.roles[role] * sg       # ideal unbind: role (x) bound
            est_b = comp.roles[role] * sb
            fg = comp._cleanup(est_g, comp.words)
            fb = comp._cleanup(est_b, comp.words)
            n_total += 1
            n_graded += int(fg == true)
            n_bipolar += int(fb == true)
            n_bip_eq_grad += int(fb == fg)

    return {
        "seed": seed, "n_total": n_total,
        "graded_recovery": n_graded / max(n_total, 1),
        "bipolar_recovery": n_bipolar / max(n_total, 1),
        "bipolar_eq_graded": n_bip_eq_grad / max(n_total, 1),
        "mean_bipolar_signed_cos": float(np.mean(bipolar_signed_cos)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=3)
    ap.add_argument("--n-attr", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_seed = {}
    for s in args.seeds:
        r = eval_seed(s, args.proj_dim, args.n_flat, args.n_attr)
        per_seed[s] = r
        print(f"[B-BIPOLAR] seed {s}: graded={r['graded_recovery']:.3f} bipolar={r['bipolar_recovery']:.3f} "
              f"bip==grad={r['bipolar_eq_graded']:.3f} sign_cos={r['mean_bipolar_signed_cos']:.4f} "
              f"(n={r['n_total']})", flush=True)

    min_graded = min(per_seed[s]["graded_recovery"] for s in args.seeds)
    min_bipolar = min(per_seed[s]["bipolar_recovery"] for s in args.seeds)
    mean_bipolar = float(np.mean([per_seed[s]["bipolar_recovery"] for s in args.seeds]))
    # GATE: graded ideal sanity (~1.0) AND bipolar holds vs graded (no VSA loss from binarization)
    sanity_ok = min_graded >= 0.95
    verdict = "GO" if (sanity_ok and min_bipolar >= 0.95) else (
        "BIPOLAR_LOSSY" if sanity_ok else "RECON_BUG")
    print(f"\n[B-BIPOLAR ROBUST] min_graded={min_graded:.3f} min_bipolar={min_bipolar:.3f} "
          f"mean_bipolar={mean_bipolar:.3f}")
    print(f"[VERDICT] binarize(bound) preserves the VSA unbind -> {verdict} "
          f"(GATE: graded ideal ~1.0 sanity AND bipolar>=0.95; cf. graded onoff(bon-boff) is the current rep)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"per_seed": per_seed, "min_graded": min_graded, "min_bipolar": min_bipolar,
                       "mean_bipolar": mean_bipolar, "verdict": verdict}, fh, indent=2)
        print(f"[B-BIPOLAR] wrote {args.out}")


if __name__ == "__main__":
    main()
