"""Gap #5 — FROZEN cue-SPECIFIC completion via BIOLOGICAL SPARSE CA3 recurrence (Guzman-Jonas 2016, ~2% not 50%).
Dense 50% recurrence lets ANY cue complete the stored assembly (perm >= cue); sparse recurrence means a random cue
mostly does not connect to the held members -> specificity (cue >> perm), while a large-enough assembly keeps enough
within-connections for the correct cue to complete. Grid over (density x assembly_frac x fb_inhib) on seed 42, FROZEN
recall + OU off (the genuine fixed-attractor test), to find the GO working point (cue>=0.20, cue>=3x perm, nocue<=0.10)
before 6-seed validation. GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

DENS = [0.02, 0.04, 0.07]
FRAC = [0.075, 0.12, 0.18]
FBI = [15.0, 35.0]
t0 = time.time()
print(f"[gap5 sparse-specificity sweep] seed 42 FROZEN OU-off | grid dens={DENS} frac={FRAC} fb_inhib={FBI}", flush=True)
best = []
for d in DENS:
    for f in FRAC:
        for fb in FBI:
            r = run(42, n_ca3=2000, ca3_density=d, assembly_frac=f, encode_drive=3000.0, hebb_lr=2.0, no_sync=True,
                    coact_thresh=0.02, lam_dep_wi=0.5, hebb_max=2000.0, ca3_fb_inhib=fb, k_thresh=15.0,
                    recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False, enable_ou=False)
            ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
            go = r["go"]
            best.append((r["held_cue"], r["held_perm"], ratio, d, f, fb, go))
            print(f"  d={d} frac={f} fb={fb}: w={r['w_within']:.0f} cue={r['held_cue']:.3f} "
                  f"nocue={r['held_nocue']:.3f} perm={r['held_perm']:.3f} rest={r['rest_firing']:.3f} "
                  f"ratio={ratio:.2f} -> {'GO' if go else 'no'} ({time.time()-t0:.0f}s)", flush=True)
print("=== TOP by cue/perm ratio (specificity) among cue>=0.15 ===", flush=True)
for cue, perm, ratio, d, f, fb, go in sorted([b for b in best if b[0] >= 0.15], key=lambda x: -x[2])[:6]:
    print(f"  d={d} frac={f} fb={fb}: cue={cue:.3f} perm={perm:.3f} ratio={ratio:.2f} {'GO' if go else ''}", flush=True)
ngo = sum(1 for b in best if b[6])
print(f"RESULT: {ngo} GO configs of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
