"""Gap #5 PAYOFF — push MAGNITUDE via stronger encoding at the specific+bistable point. structural_sep + high recall_k
give bistable (nocue ~0.01) + specific-ish (ratio ~1.9) completion but cue ~0.16 (need >=0.20): the high trigger
threshold + sparse recurrence latch only the most-strongly-driven held members. Stronger WITHIN-assembly weights
(higher hebb_lr, more train_events) -> more coincident drive to each held member -> more cross the trigger -> higher
cue, while structural_sep keeps the permuted cue out (specificity preserved). Sweep hebb_lr x train_events at gc=2.5,
recall_k=110, structural_sep, frac=0.12, sr=0.15, kir=3, sel_inhib. GO: cue>=0.20, cue>=3x perm, nocue<=0.10. GPU.
"""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

LR = [3.0, 5.0]
TE = [200]
t0 = time.time()
print(f"[gap5 CA3 encoding sweep] seed42 frozen OU-off gc=2.5 recall_k=110 struct_sep sel_inhib frac0.12 sr0.15 kir3 | hebb_lr={LR} train_events={TE}", flush=True)
best = []
for lr in LR:
    for te in TE:
        r = run(42, n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, hebb_lr=lr, train_events=te,
                no_sync=True, coact_thresh=0.02, lam_dep_wi=0.9, hebb_max=2000.0, ca3_fb_inhib=30.0, k_thresh=15.0,
                recall_k_thresh=110.0, recall_drive=700, recall_steps=150, bistable=True, nmda_recurrent=False,
                enable_ou=False, selective_inhib=True, structural_sep=True, plateau_self_regen=0.15, apical_kir_g=3.0,
                apical_gc=2.5)
        ratio = r["held_cue"] / (r["held_perm"] + 1e-6)
        best.append((r["held_cue"], r["held_perm"], r["held_nocue"], ratio, lr, te, r["go"]))
        print(f"  hebb_lr={lr:.1f} train_events={te}: w={r['w_within']:.0f} cue={r['held_cue']:.3f} "
              f"nocue={r['held_nocue']:.3f} perm={r['held_perm']:.3f} ratio={ratio:.2f} -> {'GO' if r['go'] else 'no'} "
              f"({time.time()-t0:.0f}s)", flush=True)
go = [b for b in best if b[6]]
print(f"RESULT: {len(go)} GO of {len(best)} ({time.time()-t0:.0f}s)", flush=True)
