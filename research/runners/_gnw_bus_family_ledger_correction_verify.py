"""LEDGER-ACCURACY VERIFY for the GNW-bus row family (Wall 2 — scaffold retirement as a first-class citizen).

Context. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s 2026-09-02 Check-D seeding landing (`560968f03`) classified
THREE rows (`gnw-bus-shadow`, `gnw-two-organ-bus`, `gnw-three-organ-bus`) as `retire_status: RETIRABLE_NOW
2026-09-02` / `scaffold_retired: NO`, describing the host `if recalled == p` combination as "COMPUTED-then-
OVERRIDDEN, not removed" for all three. That description is STALE for all three rows:

  1. `gnw-bus-shadow` already went through this exact retirement on 2026-08-13 (`6ab22eb55`, finding
     `2026-08-13-gnw-bus-scaffold-retirement-SCOPED.md`, artifact
     `research/findings/raw/_gnw_bus_shadow/scaffold_retire_verify.json`, GO 22/22): `install_bus_gate` wraps
     `chat.gate` with `gate_via_bus`, which runs ONLY `chat.gate_extract` (extraction + side effects) and NEVER
     calls `_substrate_recall` / `_gate_router_combine` on the covered 'route' class. `webapp/gnw_bus_shadow.py`
     has not been touched since that commit (`git log -- webapp/gnw_bus_shadow.py` shows no later commit).
  2. `gnw-two-organ-bus` and `gnw-three-organ-bus` were BUILT with the identical pattern from their own inception
     (`two_organ_gate_via` / `three_organ_gate_via`, each explicitly documented as "mirroring
     gnw_bus_shadow.gate_via_bus's routing EXACTLY") — they never went through a separate "flip-then-override"
     phase the RETIRABLE_NOW status describes, so there was never a residual dead-code follow-on to retire for
     them; the ledger's 2026-09-02 seeding pass mis-classified them by assuming the OLD gnw-bus-shadow pattern
     without checking their actual source.

This runner is the FRESH, live, this-HEAD proof for (2), run alongside re-affirming (1) is unmodified. It builds
the REAL production tiny-demo ChatBrain (numpy-CPU, rf recall — same construction `_build_chat_brain` uses),
installs BOTH `install_two_organ_gate` and `install_three_organ_gate` (mirroring `webapp/server.py brain_chat`'s
own install order: N-organ bus, then two-organ, then three-organ), and calls `two_organ_gate_via` /
`three_organ_gate_via` DIRECTLY (exactly as the official verify runners + the production wrappers do) on:
  (a) a COVERED stored-fact query — the substrate must author it (`host_combination_computed: False`);
  (b) a SELF/identity out-of-scope query — the host router must author it (`host_combination_computed: True`,
      `authored_by: host_router`), the residual the SCOPED finding says is honestly KEPT (not a cheat this row
      owes retirement for — it belongs to the `content-selection` row's own declared residual).

Run (numpy-CPU, fast rf recall path; ~3 minutes — builds 3 co-resident organ bridges: N-organ workspace + the
two-organ SurpriseProductionOrgan + the three-organ ComprehensionProductionOrgan):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_bus_family_ledger_correction_verify \
      --out research/findings/raw/_gnw_bus_family_ledger_correction/verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

from research.runners._gnw_bus_default_flip_verify import _build, STORED, SELF  # noqa: E402
from webapp import gnw_two_organ_bus as g2  # noqa: E402
from webapp import gnw_three_organ_bus as g3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    flags = {
        "two_organ_enabled_default": g2.two_organ_enabled(),
        "three_organ_enabled_default": g3.three_organ_enabled(),
        "organ_discriminates_this_backend": g2._organ_discriminates(),
    }

    chat, _gbs = _build()  # N-organ bus (gnw-bus-shadow) already installed on chat.gate, mirroring production
    installed2 = g2.install_two_organ_gate(chat)
    installed3 = g3.install_three_organ_gate(chat)

    covered_q, covered_want = STORED[0]
    self_q = SELF[0]

    svo2_cov, info2_cov = g2.two_organ_gate_via(chat, covered_q)
    svo3_cov, info3_cov = g3.three_organ_gate_via(chat, covered_q)
    svo3_self, info3_self = g3.three_organ_gate_via(chat, self_q)

    rows = {
        "covered_two_organ": {"q": covered_q, "want": covered_want, "svo": svo2_cov,
                               "authored_by": info2_cov.get("authored_by"),
                               "host_combination_computed": info2_cov.get("host_combination_computed")},
        "covered_three_organ": {"q": covered_q, "want": covered_want, "svo": svo3_cov,
                                 "authored_by": info3_cov.get("authored_by"),
                                 "host_combination_computed": info3_cov.get("host_combination_computed")},
        "self_three_organ": {"q": self_q, "svo": svo3_self,
                              "authored_by": info3_self.get("authored_by"),
                              "host_combination_computed": info3_self.get("host_combination_computed")},
    }

    covered_ok = (list(svo2_cov) == list(covered_want) and info2_cov.get("host_combination_computed") is False
                  and list(svo3_cov) == list(covered_want) and info3_cov.get("host_combination_computed") is False)
    self_kept_ok = (info3_self.get("host_combination_computed") is True
                    and info3_self.get("authored_by") == "host_router")

    result = {
        "runner": "_gnw_bus_family_ledger_correction_verify",
        "purpose": "confirm gnw-two-organ-bus + gnw-three-organ-bus already never-compute the host combination on "
                   "the covered class (same pattern gnw-bus-shadow earned via 6ab22eb55 / the SCOPED finding) -- "
                   "the ledger's RETIRABLE_NOW 2026-09-02 classification for all three rows is stale.",
        "installed": {"two_organ": installed2, "three_organ": installed3},
        "flags": flags,
        "rows": rows,
        "verdict": {
            "covered_class_never_computes_host_combination": covered_ok,
            "out_of_scope_class_correctly_keeps_host_router": self_kept_ok,
            "go": bool(covered_ok and self_kept_ok),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
