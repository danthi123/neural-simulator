"""Gap #5 PAYOFF 6-SEED VALIDATION — the CA3 bistable-dendrite completion GO config (seed-42: cue 0.257, nocue 0.000,
perm 0.000). Intrinsic dendritic bistability (self-regen plateau + KIR) + asymmetric read (strong apical->soma, weak
back) + high recall_k_thresh (trigger specificity) + structural_sep + selective_inhib, FROZEN recall + OU off, with the
mandatory no-cue + permuted anti-cheats (built into the bistable gate). GO/seed: cue>=0.20 AND cue>=3x perm AND
nocue<=0.10. Plus a NO-ENCODING anti-cheat (encode_drive=0 -> completion must collapse: load-bearing on the learned
attractor). GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

GO = dict(n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
          coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0,
          recall_k_thresh=float(os.environ.get("RECALLK", "110.0")),
          recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False, selective_inhib=True,
          structural_sep=1, plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0,
          apical_gc_read=float(os.environ.get("GCREAD", "5.0")))
SEEDS = [42, 43, 44, 100, 101, 102]
t0 = time.time()
print(f"[gap5 CA3 bistable 6-seed] GO config (asym read, silence-fixed, frozen+OU-off, no-cue+permuted anti-cheats)", flush=True)
ngo = 0
for s in SEEDS:
    r = run(s, **GO)
    ngo += int(r["go"])
    print(f"  [seed {s}] cue={r['held_cue']:.3f} nocue={r['held_nocue']:.3f} perm={r['held_perm']:.3f} "
          f"rest={r['rest_firing']:.3f} -> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
# NO-ENCODING anti-cheat (seed 42): encode_drive=0 -> no learned attractor -> completion must collapse
r0 = run(42, **{**GO, "encode_drive": 0.0})
print(f"  [ANTI-CHEAT no-encoding seed42] cue={r0['held_cue']:.3f} (must be ~0; load-bearing on the learned attractor)", flush=True)
print(f"RESULT: {ngo}/{len(SEEDS)} GO | no-encoding cue={r0['held_cue']:.3f} ({time.time()-t0:.0f}s)", flush=True)
