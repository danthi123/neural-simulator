import sys, numpy as np
from research.runners._d3_event_selfsup_pair_derisk import make_pair_task, INTRO, COREF, PROMOTE, BOUND, RETURN
from research.runners._d3_event_pop_gate_derisk import train_pushpop
from research.runners._d3_event_gated_copy_derisk import _sm, _sig

NAMES = {INTRO: "INTRO", COREF: "COREF", PROMOTE: "PROMOTE", BOUND: "BOUND", RETURN: "RETURN"}
seed = int(sys.argv[1]); K = 6
task = make_pair_task(seed, K=K)
roll = train_pushpop(task, seed=seed, epochs=40, stage_pop_epochs=15, freeze_core_in_phase2=False)
W = roll.W; wg, bg, wp, bp = roll.gates
emb, Wr, Wi, Wc, bc, We, be = W["emb"], W["Wr"], W["Wi"], W["Wc"], W["bc"], W["We"], W["be"]
X, OBJ, EMIT, L, AC, AP, PE, PC = task["test_deeper"]
OPS = task["ops_test"]; ident = task["ident"]
eyeM = np.eye(task["M"], dtype=np.float32)

B = len(L); Lm = int(L.max())
sc = np.zeros((B, K), np.float32); sc[:, ident] = 1.0
sp = np.zeros((B, K), np.float32); sp[:, ident] = 1.0
pat = np.zeros((B, K), np.float32); pat[:, ident] = 1.0
mag = {k: [] for k in NAMES}          # |credit reaching the agent layer|
attn = {k: [] for k in NAMES}         # the (1-r) attenuation factor itself
for t in range(Lm):
    act = L > t
    h = np.tanh(np.concatenate([sc @ emb, sp @ emb, pat @ emb], axis=1) @ Wr.T + X[:, t] @ Wi.T)
    raw = _sm(h @ Wc.T + bc)
    g = _sig(X[:, t] @ wg + bg)[:, None]; r = _sig(X[:, t] @ wp + bp)[:, None]
    npv = g * sc + (1.0 - g) * sp
    nsc = r * sp + (1.0 - r) * raw
    se = _sm((nsc @ emb) @ We.T + be)
    d_le = se - eyeM[EMIT[:, t]]
    e_agent = (d_le @ We) @ emb.T
    d_raw = (1.0 - r) * e_agent        # what actually reaches the agent layer through the pop's convex combination
    m = np.where(act)[0]
    for k in NAMES:
        mm = m[OPS[m, t] == k]
        if len(mm):
            mag[k].append(np.abs(d_raw[mm]).mean())
            attn[k].append(float((1.0 - r)[mm].mean()))
    sc = np.where(act[:, None], nsc, sc); sp = np.where(act[:, None], npv, sp)
    pn = np.zeros((B, K), np.float32); pn[np.arange(B), OBJ[:, t]] = 1.0
    pat = np.where(act[:, None], pn, pat)

parts = []
for k in NAMES:
    if mag[k]:
        parts.append(f"{NAMES[k]}: |credit|={np.mean(mag[k]):.5f} (1-r)={np.mean(attn[k]):.3f}")
print(f"seed {seed}  " + "  ".join(parts), flush=True)
