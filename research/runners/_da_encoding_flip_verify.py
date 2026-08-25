"""FLIP verification for da-gated-encoding DEFAULT-ON (2026-08-25), through the REAL brain_chat handler.

The wired verify (`_da_encoding_wired_verify`) sets BRAIN_DA_ENCODING=1 explicitly on its ON arms; this runner isolates
the three properties that only the DEFAULT (unset) path exercises after the flip:
  (1) THE FLIP DRIVES ON THE DEFAULT PATH: with BRAIN_DA_ENCODING UNSET (the new default), teaching the same fact under
      a HIGH-DA turn vs a LOW-DA turn writes g_high > g_low (the coupling is armed BY THE DEFAULT, and the write
      magnitude rides the live self-produced DA); the da_encoding key is present on the default turn.
  (2) LESION SEVERS THE DEFAULT-ON DRIVE: unset + BRAIN_DA_ENCODING_LESION=1 -> g pinned 1.0 on both arms -> the
      high-vs-low differential vanishes (the effect rides the DA read, not a host if-engaged).
  (3) =0 IS BYTE-IDENTICAL-OFF (+ disarms the new consolidation wiring): with BRAIN_DA_ENCODING=0, NO da_encoding key,
      the live composer's encoding_gain_fn is None, AND both the idle-tick pass
      (continuous_engine.consolidate_substrate_homeostasis) and the driver entry point
      (da_encoding_drives_chat.apply_substrate_homeostasis) return None -> store + reply + between-turn consolidation
      are byte-identical to pre-flip HEAD (the =0 branch is the identical code path pre- and post-flip).

Run (numpy-CPU, foreground, ~a few minutes): SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._da_encoding_flip_verify
"""
from __future__ import annotations
import json, logging, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
logging.getLogger().setLevel(logging.ERROR)

FACT_MSG = "dog eat grass"
INDUCE_HIGH, INDUCE_LOW = 1300.0, 100.0
_QUIET = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_SURPRISE": "0", "BRAIN_METACOG": "0",
    "BRAIN_MULTIREF": "0", "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_RECONSOLIDATION": "0",
    "BRAIN_EPISODIC_STORE": "0", "BRAIN_CURIOSITY": "0", "BRAIN_RICH": "0", "BRAIN_GNW_BUS": "0",
    "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
}


def _set(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _teach(session, *, da_encoding_env, lesion, induce):
    """Teach through the REAL handler. da_encoding_env: None -> UNSET (exercise the DEFAULT), else the string value."""
    from webapp.server import brain_chat, BrainChatRequest as Req, _BRAIN_CHATS
    env = dict(_QUIET)
    env["BRAIN_DA_ENCODING"] = da_encoding_env
    env["BRAIN_DA_ENCODING_LESION"] = "1" if lesion else None
    env["BRAIN_DA_DRIVES_INDUCE"] = (str(induce) if induce is not None else None)
    _set(env)
    r = brain_chat(Req(session=session, message=FACT_MSG, brain="tiny-demo", renderer="stub", rich=False))
    resp = json.loads(bytes(r.body).decode("utf-8"))
    chat = _BRAIN_CHATS.get((session, "tiny-demo", "stub"))
    return resp, chat


def _g(resp):
    dae = resp.get("da_encoding") or {}
    return (float(dae["g"]) if dae.get("g") is not None else None), ("da_encoding" in resp)


def main():
    out = {"runner": "research/runners/_da_encoding_flip_verify.py",
           "what": "da-gated-encoding default-ON flip verify through the real brain_chat handler"}

    # (1) DEFAULT (unset) drives.
    hi, _ = _teach("flip_def_hi", da_encoding_env=None, lesion=False, induce=INDUCE_HIGH)
    lo, _ = _teach("flip_def_lo", da_encoding_env=None, lesion=False, induce=INDUCE_LOW)
    g_hi, key_hi = _g(hi); g_lo, key_lo = _g(lo)
    p1 = bool(key_hi and key_lo and g_hi is not None and g_lo is not None and g_hi > g_lo)
    out["1_default_unset_drives"] = {"g_high": g_hi, "g_low": g_lo, "key_high": key_hi, "key_low": key_lo, "PASS": p1}

    # (2) LESION severs the DEFAULT-ON drive.
    lh, _ = _teach("flip_les_hi", da_encoding_env=None, lesion=True, induce=INDUCE_HIGH)
    ll, _ = _teach("flip_les_lo", da_encoding_env=None, lesion=True, induce=INDUCE_LOW)
    g_lh, _ = _g(lh); g_ll, _ = _g(ll)
    p2 = bool(g_lh is not None and g_ll is not None and abs(g_lh - 1.0) < 1e-9 and abs(g_ll - 1.0) < 1e-9)
    out["2_lesion_severs_default"] = {"g_lesion_high": g_lh, "g_lesion_low": g_ll, "PASS": p2}

    # (3) =0 byte-identical-off + disarms the consolidation wiring.
    off, off_chat = _teach("flip_off", da_encoding_env="0", lesion=False, induce=INDUCE_HIGH)
    off_no_key = "da_encoding" not in off
    comp = getattr(getattr(off_chat, "inner", None), "composer", None)
    off_fn_none = (comp is not None and getattr(comp, "encoding_gain_fn", None) is None)
    from webapp import continuous_engine as _CE
    homeo_res = _CE.consolidate_substrate_homeostasis(("flip_off", "tiny-demo", "stub"), off_chat)
    from webapp import da_encoding_drives_chat as _DAE
    direct = _DAE.apply_substrate_homeostasis(off_chat)
    p3 = bool(off_no_key and off_fn_none and homeo_res is None and direct is None)
    out["3_off_byte_identical_and_consolidation_noop"] = {
        "no_da_encoding_key": off_no_key, "encoding_gain_fn_is_None": off_fn_none,
        "idle_tick_consolidation_result": homeo_res, "apply_substrate_homeostasis_result": direct, "PASS": p3}

    go = bool(p1 and p2 and p3)
    # EARN the verdict — the preconditions travel with the result (tools.verdict.Verdict; verdict-preconditions gate).
    from tools.verdict import Verdict
    v = Verdict("da-gated-encoding default-ON flip verify (real brain_chat handler)")
    v.require("(1) DEFAULT (unset) drives: g_high > g_low, da_encoding key present on the default turn", p1,
              expect=True, note=f"g_high={g_hi} > g_low={g_lo}; keys hi={key_hi} lo={key_lo}")
    v.require("(2) LESION severs the default-ON drive: g pinned 1.0 on both arms", p2, expect=True,
              note=f"g_les_high={g_lh} == g_les_low={g_ll} == 1.0")
    v.require("(3) =0 byte-identical-off + both consolidation entry points no-op", p3, expect=True,
              note="no da_encoding key, encoding_gain_fn None, apply_substrate_homeostasis + consolidate_substrate_homeostasis return None")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])
    out["GO"] = go
    out["status"] = decided["status"]
    out["preconditions"] = decided["preconditions"]
    op = os.path.join(_REPO, "research", "findings", "raw", "_da_encoding_wired", "flip_verify.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=2, default=str)

    bar = "=" * 96
    print("\n" + bar)
    print("  DA-GATED-ENCODING DEFAULT-ON FLIP VERIFY (real brain_chat handler, numpy-CPU)")
    print(bar)
    print(f"  (1) DEFAULT unset DRIVES:  g_high={g_hi} > g_low={g_lo}  key_hi={key_hi} key_lo={key_lo}  -> {p1}")
    print(f"  (2) LESION severs default: g_les_high={g_lh} == g_les_low={g_ll} == 1.0  -> {p2}")
    print(f"  (3) =0 byte-identical-off: no_key={off_no_key} fn_None={off_fn_none} "
          f"idle_consol={homeo_res} direct_homeo={direct}  -> {p3}")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'}")
    print(f"  [saved] {op}\n" + bar)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
