"""SOAK / no-regression gate for the DR-3 self-schema AUTHORSHIP (self-vs-heard) DEFAULT-ON flip.

Models `research/runners/_d5_graded_flip_soak.py`. The bar the PARENT runs before flipping BRAIN_SELF_SCHEMA
default-on:

  NO-REGRESSION (the flip-safety property): on an ORDINARY turn (NOT a generated hypothesis — the faculty's
  out-of-scope class, which is the vast majority of chat), flag ON must be BYTE-IDENTICAL to flag OFF: no
  `authorship` key, no answer-text change. The faculty only touches its TRIGGERED turns (a generated
  HYPOTHESIS — a volunteered, self-authored proposition).

  LOAD-BEARING (the faculty actually does something on its triggered turn, and it rides the LIVE spiking read):
    * flag OFF on a hypothesis turn -> byte-identical to the pre-flip host default (no marker, no key).
    * flag ON + INTACT on a hypothesis turn -> the self_schema `author` pool reads 'self' -> an honest own-guess
      MARKER is prepended (the answer demonstrably changes).
    * flag ON + LESIONED on the SAME hypothesis turn -> the author access is severed (schema_access=False), the
      pool goes silent, the read collapses to 'heard', and the MARKER VANISHES -> the answer reverts BYTE-
      IDENTICALLY to the flag-OFF/host-default text (while the recalled/content body is unchanged throughout).
      This is the de-risk's own self-lesion oracle (authorship collapses to chance 6/6) exercised in production.

This soak drives the REAL production organ (`self_schema_production_organ.get_organ / read_author /
authorship_marker`) and a faithful reproduction of the server.py brain_chat `is_hyp` wiring block (kept in exact
correspondence with webapp/server.py — see `_apply_authorship_wiring`), across 6 seeds. numpy-CPU (the DR-3
lane); each seed builds one ~690-neuron bridge (~0.07s) + a handful of reads, so 6 seeds run in a few seconds.

  Run: SIM_BACKEND=numpy python -m research.runners._self_schema_authorship_flip_soak --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim.backend import get_backend  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
import research.runners.self_schema_production_organ as SS  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_self_schema_authorship_prodflip" / "soak_seed42.json"

# A synthetic recalled-content body an ordinary turn carries (the recalled fact the reply already produced) and a
# synthetic fluent HYPOTHESIS body a triggered turn carries. Fixed strings so the byte-identity comparisons are exact.
ORDINARY_ANSWER = "The dog chases the cat."
HYPOTHESIS_ANSWER = "Cats probably enjoy warm places."


def _apply_authorship_wiring(resp: dict, *, is_hyp: bool, organ, enabled: bool, lesion: bool) -> dict:
    """FAITHFUL reproduction of the webapp/server.py brain_chat rich-path `is_hyp` self-schema block (kept in
    exact correspondence — the BEGIN/END faculty block). Mutates + returns `resp`. On an ordinary turn
    (is_hyp=False) it is a no-op (the server block lives inside `if is_hyp:`); on a hypothesis turn it reads the
    author pool and, when enabled and the read is 'self', prepends the own-guess marker + attaches `authorship`."""
    if not is_hyp:
        return resp
    # (server: resp["hypothesis"] = True; resp["hypothesis_svo"] = ...; resp["fluent_hypothesis"] = ...)
    resp["hypothesis"] = True
    # ── BEGIN faculty: DR-3 self-schema AUTHORSHIP (self-vs-heard) — additive, DEFAULT-OFF (BRAIN_SELF_SCHEMA) ──
    try:
        if enabled:
            _ss_read = organ.read_author(authored=True, lesion=lesion)
            resp["authorship"] = _ss_read
            if _ss_read.get("is_self"):
                resp["answer"] = SS.authorship_marker() + resp["answer"]
    except Exception as _sse:   # noqa: BLE001
        resp["authorship"] = {"on": True, "error": f"{type(_sse).__name__}: {_sse}"}
    # ── END faculty: DR-3 self-schema AUTHORSHIP ──
    return resp


def _base_resp(answer: str) -> dict:
    """A minimal stand-in for the server's `resp` (only the fields the faculty can touch matter for byte-identity)."""
    return {"answer": answer, "abstained": False, "recalled_svo": None, "verified": True, "rich": True}


def run_one(seed: int) -> dict:
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[selfschema-soak] seed={seed} — ordinary turn OFF==ON byte-identical; hypothesis turn: marker rises ON, "
          f"VANISHES under lesion (reverts to host default).", flush=True)
    result = {"seed": seed}
    try:
        # a fresh organ per seed (the process singleton is per-process; reset it so each seed builds its own bridge).
        SS._ORGAN = None
        organ = SS.get_organ(seed=seed)
        organ.ensure_built()

        # ── ORDINARY turn (out of scope): is_hyp=False -> the wiring never runs -> OFF == ON byte-identical ──
        ord_off = _apply_authorship_wiring(_base_resp(ORDINARY_ANSWER), is_hyp=False, organ=organ,
                                           enabled=False, lesion=False)
        ord_on = _apply_authorship_wiring(_base_resp(ORDINARY_ANSWER), is_hyp=False, organ=organ,
                                          enabled=True, lesion=False)
        ordinary_byte_identical = bool(json.dumps(ord_off, sort_keys=True) == json.dumps(ord_on, sort_keys=True)
                                       and "authorship" not in ord_on
                                       and ord_on["answer"] == ORDINARY_ANSWER)

        # ── TRIGGERED turn (a generated hypothesis): is_hyp=True ──
        hyp_off = _apply_authorship_wiring(_base_resp(HYPOTHESIS_ANSWER), is_hyp=True, organ=organ,
                                           enabled=False, lesion=False)
        hyp_on = _apply_authorship_wiring(_base_resp(HYPOTHESIS_ANSWER), is_hyp=True, organ=organ,
                                          enabled=True, lesion=False)
        hyp_les = _apply_authorship_wiring(_base_resp(HYPOTHESIS_ANSWER), is_hyp=True, organ=organ,
                                           enabled=True, lesion=True)

        # OFF hypothesis == pre-flip host default: no marker, no authorship key, answer is the raw hypothesis body.
        triggered_off_host_default = bool("authorship" not in hyp_off and hyp_off["answer"] == HYPOTHESIS_ANSWER)
        # ON + INTACT: the author pool reads 'self' -> marker prepended -> answer CHANGES (load-bearing rise).
        marker = SS.authorship_marker()
        on_is_self = bool(hyp_on.get("authorship", {}).get("is_self") is True
                          and hyp_on.get("authorship", {}).get("label") == SS.AUTHOR_SELF)
        triggered_on_marker = bool(on_is_self and hyp_on["answer"] == marker + HYPOTHESIS_ANSWER
                                   and hyp_on["answer"] != hyp_off["answer"])
        # ON + LESIONED: the author read collapses to 'heard' -> NO marker -> answer reverts BYTE-IDENTICALLY to OFF.
        les_is_heard = bool(hyp_les.get("authorship", {}).get("is_self") is False
                            and hyp_les.get("authorship", {}).get("label") == SS.AUTHOR_HEARD)
        lesion_vanishes = bool(les_is_heard and hyp_les["answer"] == hyp_off["answer"]
                               and hyp_les["answer"] == HYPOTHESIS_ANSWER)
        # the recalled/content BODY is byte-identical throughout (strip the marker from the ON answer -> the same body).
        content_body_identical = bool(hyp_on["answer"][len(marker):] == HYPOTHESIS_ANSWER
                                      and hyp_off["answer"] == HYPOTHESIS_ANSWER
                                      and hyp_les["answer"] == HYPOTHESIS_ANSWER)

        # ATTRIBUTION (tools.lab): whose is the marker? The author signal that drives it = the INTACT author-pool
        # rate (treatment); the LESIONED rate (author access severed, schema_access=False) is the control. The
        # fraction NOT present in the control is the share attributable to the LIVE author-pool read — it must be
        # ~1.0 (the lesion drives the read to 0), proving the marker rides the spiking access, not a host residual.
        _intact = float(hyp_on.get("authorship", {}).get("author_rate") or 0.0)
        _lesioned = float(hyp_les.get("authorship", {}).get("author_rate") or 0.0)
        attributable_frac = attributable_to(f"selfschema author-pool marker (seed {seed})", _intact, _lesioned)
        marker_attributable = bool(attributable_frac is not None and attributable_frac >= 0.99)

        load_bearing = bool(triggered_off_host_default and triggered_on_marker and lesion_vanishes
                            and content_body_identical and marker_attributable)
        GO = bool(ordinary_byte_identical and load_bearing)

        result.update(dict(
            GO=GO, ordinary_byte_identical=ordinary_byte_identical, load_bearing=load_bearing,
            triggered_off_host_default=triggered_off_host_default, triggered_on_marker=triggered_on_marker,
            lesion_vanishes=lesion_vanishes, content_body_identical=content_body_identical,
            marker_attributable=marker_attributable,
            attributable_frac=(None if attributable_frac is None else round(float(attributable_frac), 6)),
            author_calib=organ.calib,
            author_rate_intact_self=hyp_on.get("authorship", {}).get("author_rate"),
            author_rate_lesioned=hyp_les.get("authorship", {}).get("author_rate"),
            answers=dict(ordinary_on=ord_on["answer"], hyp_off=hyp_off["answer"],
                         hyp_on=hyp_on["answer"], hyp_lesion=hyp_les["answer"])))
        print(f"[selfschema-soak] ordinary OFF==ON byte-identical = {ordinary_byte_identical}", flush=True)
        print(f"[selfschema-soak] author_rate: intact-self={result['author_rate_intact_self']} "
              f"lesioned={result['author_rate_lesioned']} thr={organ.calib['threshold']:.4f}", flush=True)
        print(f"[selfschema-soak] hyp OFF   : {hyp_off['answer']}", flush=True)
        print(f"[selfschema-soak] hyp ON    : {hyp_on['answer']}", flush=True)
        print(f"[selfschema-soak] hyp LESION: {hyp_les['answer']}  (marker VANISHED = {lesion_vanishes})", flush=True)
        print(f"[selfschema-soak] seed={seed} NO_REGRESSION={ordinary_byte_identical} "
              f"LOAD_BEARING={load_bearing} => {'GO' if GO else 'NO-GO'}", flush=True)
        SS._ORGAN = None
        del organ
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    results = {}; go = []
    for seed in seeds:
        r = run_one(seed)
        results[seed] = r; go.append(bool(r.get("GO")))
    out_path = Path(a.out)
    if len(seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(seeds)}seed.json"
        print("\n" + "#" * 118)
        print(f"[selfschema-soak] {len(seeds)}-SEED SOAK: {int(sum(go))}/{len(seeds)} GO seeds={seeds}")
        print("#" * 118)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": seeds, "n_go": int(sum(go)), "go": go, "backend": backend,
                                    "results": {str(s): results[s] for s in seeds}}, indent=2, default=str))
    print(f"[selfschema-soak] wrote {out_path}")
    return 0 if (go and all(go)) else 1


if __name__ == "__main__":
    sys.exit(main())
