"""Gap #5 PAYOFF — magnitude recovery at the SPECIFIC+BISTABLE operating point. The bistable dendrite + high
recall_k_thresh gave specificity (ratio up to 3.36) + silent rest (nocue ~0.006), but cue magnitude fell to ~0.06-0.09
(the high trigger threshold latches few held members). Recover magnitude WITHOUT losing specificity: (a) LARGER assembly
-> more within-assembly partners per held member -> more cross the learned-coincidence trigger; (b) STRONGER apical->soma
coupling (apical_gc) -> a latched plateau drives the soma harder -> higher soma firing (completion is read from soma).
Sweep frac x apical_gc at recall_k=70, self_regen=0.15, kir=3, selective_inhib. GO: cue>=0.20, cue>=3x perm, nocue<=0.10. GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

FRAC = [0.12, 0.18, 0.25]
GC = [1.0, 3.0]
RK = 70.0
t0 = time.time()
print(f"[gap5 CA3 magnitude sweep] seed 42 frozen OU-off recall_k={RK} sr=0.15 kir=3 sel_inhib | frac={FRAC} apical_gc={GC}", flush=True)
best = []
for fr in FRAC:
    for gc in GC:
        r = run(42, n_ca3=2000, ca3_density=0.05, assembly_frac=fr, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
                coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0, recall_k_thresh=RK,
                recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False,
                selective_inhib=True, plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=gc)
        ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
        best.append((r["held_cue"], r["held_perm"], r["held_nocue"], ratio, fr, gc, r["go"]))
        print(f"  frac={fr:.2f} gc={gc:.1f}: w={r['w_within']:.0f} cue={r['held_cue']:.3f} nocue={r['held_nocue']:.3f} "
              f"perm={r['held_perm']:.3f} ratio={ratio:.2f} -> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
go = [b for b in best if b[6]]
print(f"RESULT: {len(go)} GO of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
for cue, perm, nocue, ratio, fr, gc, g in sorted(best, key=lambda x: -(x[0] if x[2] <= 0.10 else 0))[:3]:
    print(f"  best: frac={fr} gc={gc} cue={cue:.3f} nocue={nocue:.3f} perm={perm:.3f} ratio={ratio:.2f} {'GO' if g else ''}", flush=True)
