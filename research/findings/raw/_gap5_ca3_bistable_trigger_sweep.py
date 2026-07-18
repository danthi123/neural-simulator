"""Gap #5 PAYOFF follow-on — the bistable dendrite SOLVED the bistability horn (silent rest, nocue~0.06 WITH a
completing cue). Remaining: TRIGGER-SPECIFICITY — a permuted cue's avalanche still triggers the latch. Fix: decouple
encode (low k_thresh -> strong learned weights) vs RECALL (HIGH recall_k_thresh -> only the strong LEARNED within-
assembly coincidence can trigger a latch; the permuted cue's generic coincidence cannot) + selective_inhib. Sweep
recall_k_thresh x self_regen at kir=3 (the silent-rest band). GO: cue>=0.20, cue>=3x perm, nocue<=0.10. GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

RK = [40.0, 70.0, 110.0]
SR = [0.15, 0.3]
t0 = time.time()
print(f"[gap5 CA3 bistable TRIGGER sweep] seed 42 frozen OU-off kir=3 selective_inhib | recall_k_thresh={RK} self_regen={SR}", flush=True)
best = []
for rk in RK:
    for sr in SR:
        r = run(42, n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
                coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0, recall_k_thresh=rk,
                recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False,
                selective_inhib=True, plateau_self_regen=sr, apical_kir_g=3.0)
        ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
        best.append((r["held_cue"], r["held_perm"], r["held_nocue"], ratio, rk, sr, r["go"]))
        print(f"  recall_k={rk:.0f} sr={sr:.2f}: cue={r['held_cue']:.3f} nocue={r['held_nocue']:.3f} "
              f"perm={r['held_perm']:.3f} ratio={ratio:.2f} -> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
go = [b for b in best if b[6]]
print(f"RESULT: {len(go)} GO of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
if go:
    print(f"  BEST GO: recall_k={go[0][4]} sr={go[0][5]} cue={go[0][0]:.3f} perm={go[0][1]:.3f} nocue={go[0][2]:.3f}", flush=True)
