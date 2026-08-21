"""COMPOSED no-regression check for flipping the four WAVE-0/1 faculties default-ON together (2026-08-21, owner-approved).

Each faculty already has a GO de-risk proving OFF-byte-identical + ON-load-bearing + lesion-severable through the REAL
handler (da-gated-encoding, da-gated-curiosity-threshold, gnw-three-organ-bus, continuous-ideation). The ONE thing a
per-faculty de-risk does not cover is INTERACTION: are the four safe when ALL default-ON at once? This runner answers
that — the composed no-regression — so the flip is justified.

Method: over an OUT-OF-SCOPE panel (confident KNOWN-topic recalls + self — turns NONE of the four faculties should
change: no abstain -> no curiosity crave; high comprehension -> no 3-organ veto; recall output is magnitude-invariant
on rf -> no DA-encoding change; no idle tick -> no ideation lead), compare ALL-FOUR-ON (BRAIN_DA_ENCODING=1
BRAIN_CURIOSITY_DA=1 BRAIN_GNW_3ORGAN=1 BRAIN_CONTINUOUS_IDEATE=1) vs ALL-FOUR-OFF (=0) through the REAL brain_chat
handler. GO iff every out-of-scope turn is BYTE-IDENTICAL (answer/recalled_svo/abstained) — i.e. the four together
change nothing they should not, so flipping their defaults is safe (the =0 escape recovers today's behaviour exactly).

Run: SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._wave4_composed_flip_noregression
"""
from __future__ import annotations
import json, os, sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
import logging; logging.disable(logging.INFO)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT = os.path.join(_REPO, "research", "findings", "raw", "_wave_flip_soak", "composed_noregression.json")
# the flag set to flip together (override with WAVE4_FLAGS=comma,sep to isolate an offender)
_DEFAULT_FLAGS = ["BRAIN_DA_ENCODING", "BRAIN_CURIOSITY_DA", "BRAIN_GNW_3ORGAN", "BRAIN_CONTINUOUS_IDEATE"]
FLAGS = [f.strip() for f in os.environ.get("WAVE4_FLAGS", ",".join(_DEFAULT_FLAGS)).split(",") if f.strip()]
# OUT-OF-SCOPE panel: confident KNOWN-topic recalls + self (no abstain, high comprehension, recall turns, no idle tick).
# Ordered so the two turns the GNW-3organ veto regresses (dog/cat recalls) come FIRST, so a WAVE4_PANEL_N-truncated fast
# run still captures BOTH the divergent turns AND clean recall/self controls. WAVE4_PANEL_N (default = full 8) takes the
# first N (per-turn brain rebuild is the cost; a smaller N is the fast artifact-regen path, same deterministic result).
PANEL = ["what does dog chase?", "what does cat eat?", "what are you", "how do you learn",
         "what does brain use?", "what does brain learn?", "what does brain store?", "what do you use"]
PANEL = PANEL[:max(1, int(os.environ.get("WAVE4_PANEL_N", str(len(PANEL)))))]


def _core(resp):
    return {k: resp.get(k) for k in ("answer", "recalled_svo", "abstained")}


def _reply(msg, session):
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(r.body.decode("utf-8"))


def main():
    out = {"runner": "research/runners/_wave4_composed_flip_noregression.py", "flags": FLAGS, "panel_len": len(PANEL)}
    diverged = []
    for i, msg in enumerate(PANEL):
        for f in FLAGS:
            os.environ[f] = "0"
        off = _reply(msg, f"w4-off-{i}")
        for f in FLAGS:
            os.environ[f] = "1"
        on = _reply(msg, f"w4-on-{i}")
        for f in FLAGS:
            os.environ[f] = "0"
        if _core(off) != _core(on):
            diverged.append({"msg": msg, "off": _core(off), "on": _core(on)})
    out["n_turns"] = len(PANEL)
    out["n_diverged"] = len(diverged)
    out["diverged"] = diverged
    go = (len(diverged) == 0)

    from tools.verdict import Verdict
    v = Verdict("all four WAVE-0/1 faculties default-ON together: byte-identical on out-of-scope turns")
    v.require("no out-of-scope turn diverges with all four ON vs all four OFF", len(diverged), expect=0,
              note=f"{len(PANEL)} out-of-scope turns, {len(diverged)} diverged")
    v.disabled("in-scope behaviour of each faculty (curiosity crave / 3-organ veto / ideation lead / encoding gain)",
               why="each is proven load-bearing + off-byte-identical in its own GO de-risk; this runner isolates the "
                   "COMPOSED no-regression on OUT-OF-SCOPE turns (the one thing a per-faculty de-risk cannot cover)")
    decided = v.decide(go=go)
    out["preconditions"] = decided["preconditions"]
    out["VERDICT"] = "GO" if go else "NO-GO"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fp:
        json.dump(out, fp, indent=1)
    print(json.dumps({k: out[k] for k in ("n_turns", "n_diverged", "VERDICT")}, indent=1), flush=True)
    if diverged:
        print("DIVERGENCES:", json.dumps(diverged[:4], indent=1), flush=True)
    print(f"wrote {OUT} -> {decided['status']}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
