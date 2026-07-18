"""Gap #5 PAYOFF — full lever combination. The bistable dendrite gives a specific+bistable completion (cue 0.087,
ratio 2.41, nocue 0.015) but boosting magnitude (apical_gc, larger assembly) reintroduces NETWORK self-ignition (the
latched plateaus fire the soma -> recurrent spread). Fix: add structural_sep (zero non-member->member recurrents) so a
boosted soma read cannot spread the completion, letting apical_gc raise cue WITHOUT self-sustain. Sweep structural_sep
x apical_gc x recall_k at frac=0.12, sr=0.15, kir=3, selective_inhib. GO: cue>=0.20, cue>=3x perm, nocue<=0.10. GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

GC = [2.0, 4.0]
RK = [90.0, 130.0]
t0 = time.time()
print(f"[gap5 CA3 full-combo sweep] seed 42 frozen OU-off frac=0.12 sr=0.15 kir=3 sel_inhib STRUCTURAL_SEP | apical_gc={GC} recall_k={RK}", flush=True)
best = []
for gc in GC:
    for rk in RK:
        r = run(42, n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
                coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0, recall_k_thresh=rk,
                recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False,
                selective_inhib=True, structural_sep=True, plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=gc)
        ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
        best.append((r["held_cue"], r["held_perm"], r["held_nocue"], ratio, gc, rk, r["go"]))
        print(f"  gc={gc:.1f} recall_k={rk:.0f}: w={r['w_within']:.0f} cue={r['held_cue']:.3f} nocue={r['held_nocue']:.3f} "
              f"perm={r['held_perm']:.3f} ratio={ratio:.2f} -> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
go = [b for b in best if b[6]]
print(f"RESULT: {len(go)} GO of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
for cue, perm, nocue, ratio, gc, rk, g in sorted(best, key=lambda x: -(x[0] if x[2] <= 0.10 else 0))[:3]:
    print(f"  best: gc={gc} recall_k={rk} cue={cue:.3f} nocue={nocue:.3f} perm={perm:.3f} ratio={ratio:.2f} {'GO' if g else ''}", flush=True)
