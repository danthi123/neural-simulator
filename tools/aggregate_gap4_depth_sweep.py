"""Aggregate the gap#4 depth-diagnosis sweep -> the capability-grounded table.

The question (owner steer 2026-08-01): does deep credit WIN where a frozen random reservoir FAILS?
For each (n_prop, pool_k) cell it reports, across seeds:
  oracle       - the idealized ceiling (MUST fit, else the cell is epoch-floored and uninformative)
  frozen_res   - a FROZEN random hidden layer + trained readout (the reservoir baseline)
  eprop        - full e-prop (trains the hidden layers)
  deep_share   - (eprop - frozen) / (oracle - frozen): >0 => deep credit adds signal
The capability signature we want: at high n_prop the frozen reservoir DROPS toward chance while eprop
(or a better mechanism) stays up => deep_share rises. If oracle also drops, raise epochs (not a verdict).

Usage: .venv/bin/python tools/aggregate_gap4_depth_sweep.py [dir]
"""
import json, sys, glob, os, statistics as st
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "research/findings/raw/gap4_depth_sweep"
cells = defaultdict(list)  # (n_prop, pool_k) -> list of per-seed dicts
for f in sorted(glob.glob(os.path.join(d, "np*_k*_s*.json"))):
    if "smoke" in f:
        continue
    try:
        j = json.load(open(f))
        c = j.get("config", {})
        ps = j["per_seed"][0]
        cells[(c.get("n_prop"), c.get("pool_k"))].append(dict(
            seed=ps.get("seed"), oracle=ps.get("oracle_inherit"),
            frozen=ps.get("frozen_hidden_inherit"), eprop=ps.get("eprop_inherit_heldout"),
            dcs=ps.get("deep_credit_share"), chance=ps.get("chance"),
            shuffle=ps.get("shuffle_dfa_inherit"), signal=bool(j.get("SIGNAL"))))
    except Exception as e:
        print(f"  (skip {os.path.basename(f)}: {e})")


def m(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return round(st.mean(vals), 3) if vals else None


print(f"{'n_prop':>6} {'pool_k':>6} {'n':>2} {'chance':>6} {'oracle':>6} {'frozen_res':>10} {'eprop':>6} "
      f"{'deep_share':>10} {'GO':>3}  note")
for (np_, pk) in sorted(cells, key=lambda x: (x[0] or 0, x[1] or 0)):
    rows = cells[(np_, pk)]
    orc, fro, epr, dcs, ch = m(rows, 'oracle'), m(rows, 'frozen'), m(rows, 'eprop'), m(rows, 'dcs'), m(rows, 'chance')
    go = sum(r['signal'] for r in rows)
    note = ""
    if orc is not None and ch is not None and orc < ch + 0.12:
        note = "ORACLE FLOORED -> raise epochs (uninformative)"
    elif fro is not None and ch is not None and fro < ch + 0.12:
        note = "reservoir FAILS here <= the capability threshold"
    if dcs is not None and dcs > 0.15 and not note.startswith("ORACLE"):
        note = (note + " | deep credit CONTRIBUTES").strip(" |")
    print(f"{str(np_):>6} {str(pk):>6} {len(rows):>2} {str(ch):>6} {str(orc):>6} {str(fro):>10} "
          f"{str(epr):>6} {str(dcs):>10} {go:>3}  {note}")
print(f"\ncells: {len(cells)}  total result files: {sum(len(v) for v in cells.values())}")
print("READ: a cell is INFORMATIVE only if oracle fits (>chance+0.12). The verdict lives where the "
      "reservoir FAILS but eprop/mechanism holds (deep_share>0).")
