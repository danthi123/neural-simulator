"""A10 CAPABILITY-GATE runner (midnight plan S15c, 2026-09-24) -- WIRE-IN de-risk for the reward/value SNc
afferent driven by an existing spiking read, through the REAL production `webapp/server.py::brain_chat` handler
(numpy-CPU, rf recall; seconds not ~180s). Mirrors `_da_encoding_wired_verify.py`'s A/B/C template.

PRE-REGISTERED: research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md (own commit,
before this runner's first evaluation artifact).

WHAT THIS PROVES (GO = A and B and C):
  (A) OFF (`BRAIN_REWARD_VALUE_AFFERENT` unset) -- teaching the confirm/contradict pair through the REAL
      handler: NO `reward_value` key nested under the response's `da_drives`, on either turn -- the coupling is
      provably a no-op when off (the code path is identical to before this module existed: the guard around the
      whole block never fires).
  (B) ON, LOAD-BEARING -- on FRESH sessions, "the dog chase the cat" (CONFIRM, the built-in dog-chase-cat fact)
      vs "the dog chase the fish" (CONTRADICT): `reward_value.source == "surprise"` on both, and CONTRADICT's
      `reward_value.normalized` is strictly greater than CONFIRM's (the surprise-organ prediction-error read
      differentiates the two turns; `surprised` False vs True as a sanity cross-check).
  (C) LESION (`BRAIN_REWARD_VALUE_LESION=1`) -- the SAME confirm/contradict pair reads the organ's OWN
      prediction-edges-zeroed twin: the CONFIRM-vs-CONTRADICT differential in `reward_value.normalized`
      COLLAPSES (< 1e-6), attributable to the live surprise-organ read via `tools.lab.attributable_to`.

SEED 7 ONLY (a dev/calibration seed, per the pre-registration and the midnight plan's own rule). This is a
DE-RISK, not a 6-seed gate verdict; the flag's default-ON decision additionally needs a SOUND independent
review (declared as an open item in the pre-registration -- this runner records only a self-trace).

Run (numpy-CPU, foreground, ~seconds):
  bash tools/mem_ok.sh 2 2 && bash tools/memcap.sh 3 -- .venv/bin/python -u -m \
      research.runners._reward_value_afferent_derisk --out research/findings/raw/_reward_value_afferent_derisk/s7.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path; ~s not ~180s)
os.environ.setdefault("BRAIN_CHAT_SEED", "7")        # a DEV/CALIBRATION seed -- never a gate seed (see prereg)

logging.getLogger().setLevel(logging.ERROR)          # quiet the per-build SIM_BRIDGE chatter (the verdict is JSON)

# quiet the heavy Gate-B organs unrelated to this coupling (they would run identically on every arm, adding cost
# and noise); BRAIN_SURPRISE and BRAIN_DA_DRIVES are LEFT AT THEIR DEFAULT-ON anchors -- this coupling's whole
# point is to drive da-mode-drives-response's afferent off the surprise organ's read, so both must stay live.
_QUIET = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_METACOG": "0", "BRAIN_MULTIREF": "0",
    "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_EPISODIC_STORE": "0", "BRAIN_CURIOSITY": "0", "BRAIN_RICH": "0",
    "BRAIN_GNW_BUS": "0", "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
    "BRAIN_RECONSOLIDATION": "0",
    # RAM-tight box (2026-09-24 midnight plan, real contention from other lanes): the default LTM tier
    # (wikidata_100k) has separately been measured to exceed an 11 GB cap while loading (see
    # research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md); this coupling needs only the
    # built-in dog-chase-cat/fish fact, so the LTM tier is off (BRAIN_LTM_SHIP_DEFAULT's own byte-identical
    # no-LTM escape) -- keeps this de-risk's peak RSS near ~1.1 GB.
    "BRAIN_LTM_SHIP_DEFAULT": "off",
}

SEED = 7


def _set(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _turn(session, message, *, reward_value, lesion):
    """Send ONE turn through the REAL brain_chat handler on a FRESH session; return the parsed response dict."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    env = dict(_QUIET)
    env["BRAIN_DA_DRIVES"] = "1"
    # EXPLICIT-OFF PIN: the OFF arm exports BRAIN_REWARD_VALUE_AFFERENT=0, NOT unset, so a future default-ON
    # flip cannot silently arm the coupling here and break the byte-identical (A) proof (mirrors
    # _da_encoding_wired_verify.py's own EXPLICIT-OFF PIN comment).
    env["BRAIN_REWARD_VALUE_AFFERENT"] = "1" if reward_value else "0"
    env["BRAIN_REWARD_VALUE_LESION"] = "1" if lesion else "0"
    _set(env)
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(bytes(r.body).decode("utf-8"))


CONFIRM_MSG = "the dog chase the cat"     # (dog,chase) already known -> asserted == stored -> CONFIRM (low surprise)
CONTRA_MSG = "the dog chase the fish"     # (dog,chase) already known -> asserted != stored -> CONTRADICT (high surprise)


def main(out_path):
    # ── (A) OFF: no reward_value key on either turn. ──
    off_confirm = _turn("rva_off_confirm", CONFIRM_MSG, reward_value=False, lesion=False)
    off_contra = _turn("rva_off_contra", CONTRA_MSG, reward_value=False, lesion=False)
    off_dd_confirm = off_confirm.get("da_drives") or {}
    off_dd_contra = off_contra.get("da_drives") or {}
    go_a = bool(("reward_value" not in off_dd_confirm) and ("reward_value" not in off_dd_contra)
                and off_dd_confirm.get("on") is True and off_dd_contra.get("on") is True)

    # ── (B) ON, LOAD-BEARING: confirm vs contradict, reward-value driving the afferent. ──
    on_confirm = _turn("rva_on_confirm", CONFIRM_MSG, reward_value=True, lesion=False)
    on_contra = _turn("rva_on_contra", CONTRA_MSG, reward_value=True, lesion=False)
    rv_confirm = (on_confirm.get("da_drives") or {}).get("reward_value") or {}
    rv_contra = (on_contra.get("da_drives") or {}).get("reward_value") or {}
    sources_ok = (rv_confirm.get("source") == "surprise") and (rv_contra.get("source") == "surprise")
    norm_confirm = rv_confirm.get("normalized")
    norm_contra = rv_contra.get("normalized")
    go_b_diff = bool(sources_ok and norm_confirm is not None and norm_contra is not None
                     and float(norm_contra) > float(norm_confirm))
    surprised_sanity = bool((rv_confirm.get("surprised") is False) and (rv_contra.get("surprised") is True))

    # ── (C) LESION: the SAME pair under the organ's own prediction-edges-zeroed twin -- the differential VANISHES. ──
    les_confirm = _turn("rva_les_confirm", CONFIRM_MSG, reward_value=True, lesion=True)
    les_contra = _turn("rva_les_contra", CONTRA_MSG, reward_value=True, lesion=True)
    rv_les_confirm = (les_confirm.get("da_drives") or {}).get("reward_value") or {}
    rv_les_contra = (les_contra.get("da_drives") or {}).get("reward_value") or {}
    norm_les_confirm = rv_les_confirm.get("normalized")
    norm_les_contra = rv_les_contra.get("normalized")
    go_c = bool(norm_les_confirm is not None and norm_les_contra is not None
                and abs(float(norm_les_contra) - float(norm_les_confirm)) < 1e-6)

    go = bool(go_a and go_b_diff and go_c)

    # ── ATTRIBUTION: what fraction of the confirm-vs-contradict differential is owed to the LIVE surprise read? ──
    from tools.lab import attributable_to
    diff_live = (float(norm_contra) - float(norm_confirm)) if (norm_contra is not None and norm_confirm is not None) else 0.0
    diff_lesion = ((float(norm_les_contra) - float(norm_les_confirm))
                   if (norm_les_contra is not None and norm_les_confirm is not None) else 0.0)
    lesion_attribution = attributable_to(
        "confirm-vs-contradict reward_value.normalized differential owed to the LIVE surprise-organ read "
        "(control = BRAIN_REWARD_VALUE_LESION)", diff_live, diff_lesion)

    from tools.verdict import Verdict
    v = Verdict("A10 reward/value SNc afferent driven by the surprise-organ spiking read, default-OFF")
    v.require("(A) OFF: no reward_value key on either turn (byte-identical afferent path)", go_a, expect=True)
    v.require("(B) ON: reward_value.source == 'surprise' on both turns", sources_ok, expect=True)
    v.require("(B) ON, LOAD-BEARING: normalized(contradict) > normalized(confirm)", go_b_diff, expect=True,
              note=f"confirm={norm_confirm} contra={norm_contra}")
    v.require("(B) sanity: surprised flips False(confirm)->True(contradict)", surprised_sanity, expect=True)
    v.require("(C) LESION: the differential VANISHES (<1e-6)", go_c, expect=True,
              note=f"les_confirm={norm_les_confirm} les_contra={norm_les_contra}")
    v.control("reward_value.normalized rides the LIVE surprise read (on) and is severed by the lesion",
              treatment=diff_live, control=diff_lesion, min_separation=0.0,
              note="on: contra>confirm; lesion: contra==confirm")
    v.disabled("heavy Gate-B organs (affect/worldmodel/metacog/multiref/... = 0) in the handler proofs",
               why="disabled ONLY for speed/confound-isolation; they run identically on every flag arm")
    v.disabled("the affect-valence FALLBACK path", why="this de-risk exercises the PRIMARY surprise path only "
               "(a fresh session with an expectation-bearing assertion always resolves it); the fallback is "
               "reported in the module but not load-bearing-tested here (declared, see the pre-registration)")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_reward_value_afferent_derisk",
        "seed": SEED, "seed_kind": "dev-calibration (NOT a 6-seed gate seed)",
        "go": go, "status": decided["status"],
        "coupling": "A10 reward/value SNc afferent <- surprise-organ confirm/violate read, default-OFF "
                    "(BRAIN_REWARD_VALUE_AFFERENT), lesion BRAIN_REWARD_VALUE_LESION",
        "A_off_byte_identical": {
            "no_reward_value_key_confirm": "reward_value" not in off_dd_confirm,
            "no_reward_value_key_contra": "reward_value" not in off_dd_contra,
            "da_drives_confirm": off_dd_confirm, "da_drives_contra": off_dd_contra, "GO": go_a,
        },
        "B_on_load_bearing": {
            "reward_value_confirm": rv_confirm, "reward_value_contra": rv_contra,
            "sources_ok": sources_ok, "normalized_contra_gt_confirm": go_b_diff,
            "surprised_sanity": surprised_sanity,
        },
        "C_lesion_collapses": {
            "reward_value_confirm": rv_les_confirm, "reward_value_contra": rv_les_contra,
            "differential_vanishes": go_c,
        },
        "attribution": {"lesion_attribution_fraction": lesion_attribution,
                         "diff_live": diff_live, "diff_lesion": diff_lesion},
        "verdict": decided,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"go": go, "status": decided["status"], "out": out_path}, indent=2))
    return 0 if go else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_reward_value_afferent_derisk/s7.json")
    args = ap.parse_args()
    sys.exit(main(args.out))
