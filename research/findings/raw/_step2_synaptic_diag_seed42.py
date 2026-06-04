"""Diagnostic for the seed-42 `what` −2 on the Step-2 synaptic no-regression gate.

The Task-3 probe found seed 42 synaptic what=4/6 vs python 6/6 (drop 2); seeds 43/44 are 6/6=6/6.
This per-fact diagnostic, on ONE unified bridge (seed 42, D=2048, synaptic route), reports for each of the
6 facts: the parser's role assignment, the synaptic-route decoded patient + agent, and the python-path
decoded patient + agent — so the mechanism of the two synaptic misses is visible (is the parser mis-routing
a role, or is the spiking unbind margin thin on a particular high-cosine word triple?).

    python -m research.findings.raw._step2_synaptic_diag_seed42
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge

SEED = 42
PROJ_DIM = 2048
N = 6


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)

    # The exact per-seed fact stream the Task-3 probe used (rng seeded by seed+1).
    rng = np.random.default_rng(SEED + 1)
    facts = [tuple(str(x) for x in rng.choice(u.words, size=3, replace=False)) for _ in range(N)]

    # Between-code cosine of the real denoise64 V=16 codes (context for thin-margin misses).
    m = np.stack([u.concepts[w] for w in u.words]); g = m @ m.T
    bc = g[np.triu_indices(len(u.words), 1)]
    print(f"[diag] D={PROJ_DIM} between-cos mean={bc.mean():.3f} max={bc.max():.3f}", flush=True)

    print(f"{'fact':<22}  {'parse(role->word)':<40}  {'SYN what/who':<18}  {'PY what/who':<14}", flush=True)
    syn_what = syn_who = py_what = py_who = 0
    for a, ac, p in facts:
        # parser role assignment (active voice) for this fact
        roles = u.parse(f"{a} {ac} {p}")

        # SYNAPTIC path
        u.kb = []
        u.hear_synaptic(f"{a} {ac} {p}")
        s_pat = u.query_patient(a, ac)
        s_agt = u.query_agent(ac, p)
        s_w = int(s_pat == p); s_h = int(s_agt == a)
        syn_what += s_w; syn_who += s_h

        # PYTHON path
        u.kb = []
        u.store(roles["agent"], roles["action"], roles["patient"])
        p_pat = u.query_patient(a, ac)
        p_agt = u.query_agent(ac, p)
        p_w = int(p_pat == p); p_h = int(p_agt == a)
        py_what += p_w; py_who += p_h

        smark = "" if (s_w and s_h) else "  <-- SYN MISS"
        print(f"{a+' '+ac+' '+p:<22}  {str(roles):<40}  "
              f"pat={s_pat!r:<8} agt={s_agt!r:<8}  pat={p_pat!r:<8} agt={p_agt!r:<8}{smark}", flush=True)

    print(f"\n[diag] SYNAPTIC what={syn_what}/{N} who={syn_who}/{N} | PYTHON what={py_what}/{N} who={py_who}/{N}",
          flush=True)


if __name__ == "__main__":
    main()
