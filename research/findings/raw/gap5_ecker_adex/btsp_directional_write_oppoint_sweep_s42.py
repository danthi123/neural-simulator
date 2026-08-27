"""Seed-42 op-point sweep of the BTSP-directional write: encode ONCE, then run the seeded consolidation over a grid of
(plat_tau, elig_tau, eta) measuring dw_fwd / dw_rev only (skip the reads). Analog of the STDP band's op-point sweep.
Answers: does ANY BTSP op-point make dw_fwd > dw_rev (directional) on the overlapping self-driven cascade?"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
import sys, json, time
from pathlib import Path
_REPO = Path("/home/dant123/Projects/sim/.claude/worktrees/agent-a080881cb77a46239")
sys.path.insert(0, str(_REPO))
import numpy as np
from sim.backend import to_host
from research.runners._gap5_ecker_adex_ca3_stdp_band_derisk import build_store, encode, _load_weights
from research.runners._gap5_ecker_replay_learn_through_use_derisk import consolidate_by_btsp_replay

SEED = 42
bkw = dict(m_asm=6, asm_size=80, w_within=60.0, between_init=15.0, within_density=0.5, b_override=120.0,
           a_override=None, ou_sigma=40.0, dt=0.1, stdp_w_max=900.0, stdp_a_plus=0.05, stdp_a_minus=0.06, stdp_tau=20.0)
enc_kw = dict(n_laps=14, enc_step=80, enc_dwell=40, enc_gap=600, cue_pa=9000.0, cue_frac=0.6, dt=0.1)
cons_kw = dict(swr_period=325, cue_pa=9000.0, cue_steps=40, cue_frac=0.6, dt=0.1)
CONSOL_STEPS = 4000

# encode once, cache the learned band
st = build_store(SEED, **bkw); encode(st, SEED, **enc_kw)
w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
fwd0 = w_learned[st["fwd_pos"]].mean(); rev0 = w_learned[st["rev_pos"]].mean()
print(f"encoded band: adj_fwd={fwd0:.1f} adj_rev={rev0:.1f}", flush=True)

grid = []
for plat_tau in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
    for elig_tau in (15.0, 40.0):
        grid.append((plat_tau, elig_tau, 0.02))
# a couple of extra eta at the most-directional plat to check eta-invariance of the SIGN
for eta in (0.005, 0.05):
    grid.append((0.1, 15.0, eta))

rows = []
t0 = time.time()
for plat_tau, elig_tau, eta in grid:
    stc = build_store(SEED, **bkw); _load_weights(stc, w_learned)
    r = consolidate_by_btsp_replay(stc, CONSOL_STEPS, SEED, seed_on=True, elig_tau_ms=elig_tau, plat_tau_ms=plat_tau,
                                   eta=eta, w_min=0.0, w_max=900.0, **cons_kw)
    directional = r["dw_fwd"] > r["dw_rev"]
    rows.append(dict(plat_tau=plat_tau, elig_tau=elig_tau, eta=eta, dw_fwd=round(r["dw_fwd"], 2),
                     dw_rev=round(r["dw_rev"], 2), diff=round(r["dw_fwd"] - r["dw_rev"], 2), directional=directional))
    print(f"plat={plat_tau:>4} elig={elig_tau:>4} eta={eta:<5} -> dw_fwd={r['dw_fwd']:8.2f} dw_rev={r['dw_rev']:8.2f} "
          f"diff={r['dw_fwd']-r['dw_rev']:8.2f} {'DIRECTIONAL' if directional else 'symmetrized'} "
          f"({time.time()-t0:.0f}s)", flush=True)

any_dir = any(x["directional"] for x in rows)
print(f"\n=== ANY directional op-point? {any_dir} ===")
Path("/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/btsp_sweep.json").write_text(
    json.dumps(dict(encoded_fwd=float(fwd0), encoded_rev=float(rev0), any_directional=any_dir, rows=rows), indent=2))
