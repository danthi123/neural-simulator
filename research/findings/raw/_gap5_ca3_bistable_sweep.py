"""Gap #5 PAYOFF — find the CA3 bistable-dendrite completion band (seed 42, frozen + OU off). The bistable dendrite
lets W_rec be sub-critical: a correct cue delivers the coincident within-assembly trigger -> held members LATCH their
plateaus + HOLD (completion); a permuted cue delivers no coincident trigger -> no latch; rest silent (bistable down
state). Sweep (self_regen x apical_kir_g x structural_sep) for the band where cue >= 0.20, cue >= 3x perm, nocue <= 0.10.
GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

SR = [0.05, 0.15, 0.3]
KIR = [3.0, 5.0]
SEP = [False, True]
t0 = time.time()
print(f"[gap5 CA3 bistable sweep] seed 42 frozen OU-off | self_regen={SR} kir={KIR} structural_sep={SEP}", flush=True)
best = []
for sep in SEP:
    for sr in SR:
        for kir in KIR:
            r = run(42, n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
                    coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0,
                    recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False,
                    selective_inhib=True, structural_sep=sep, plateau_self_regen=sr, apical_kir_g=kir)
            ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
            best.append((r["held_cue"], r["held_perm"], r["held_nocue"], ratio, sr, kir, sep, r["go"]))
            print(f"  sep={sep!s:5s} sr={sr:.2f} kir={kir:.1f}: cue={r['held_cue']:.3f} "
                  f"nocue={r['held_nocue']:.3f} perm={r['held_perm']:.3f} ratio={ratio:.2f} "
                  f"-> {'GO' if r['go'] else 'no'} ({time.time()-t0:.0f}s)", flush=True)
print("=== TOP by (cue>=0.20 AND nocue<=0.10), ranked by ratio ===", flush=True)
ok = [b for b in best if b[0] >= 0.20 and b[2] <= 0.10]
for cue, perm, nocue, ratio, sr, kir, sep, go in sorted(ok, key=lambda x: -x[3])[:6]:
    print(f"  sr={sr} kir={kir} sep={sep}: cue={cue:.3f} nocue={nocue:.3f} perm={perm:.3f} ratio={ratio:.2f} {'GO' if go else ''}", flush=True)
print(f"RESULT: {sum(1 for b in best if b[7])} GO of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
