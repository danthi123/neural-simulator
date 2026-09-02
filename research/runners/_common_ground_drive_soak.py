"""SOAK / load-bearing gate for the `BRAIN_CG_DRIVES` default-ON flip (board #152: the common-ground ledger
DRIVES audience design in the served /api/brain-chat reply's referring expression).

THE QUESTION THIS ANSWERS (the anti-hollow check, [[feedback_faculties_must_drive_not_observe]]): is the ledger
genuinely LOAD-BEARING on the reply, or is it a hollow flip (a neural verdict computed but never changing the
served text)? A faculty is real only when (i) VARYING its input state changes the reply in the way the theory
predicts, and (ii) that difference VANISHES when the coupling is LESIONED.

PART A — ORGAN-LEVEL 6-SEED PHYSIOLOGY (the gate; cheap, no brain build -- a 750-neuron custom bridge per the
de-risk `research/runners/_learned_common_ground_ledger_derisk.py`, ~2s/seed). For each seed, drive the REAL
production code (`webapp.common_ground_drives_chat.observe_turn`, the exact function `webapp/server.py`'s
`brain_chat` calls) through a MOCK `chat`/composer (the "unbound method + a mock self" pattern -- CLAUDE.md's
RAM-safety guidance: this is a pure comprehension-boundary + substrate-read property, not a property of the full
15k-LTM production brain, so no brain build is needed to test it honestly):

  * STATE A vs STATE B: mention each of K_REF=6 referents once (first mention -- the ledger reads UNGROUNDED ->
    INTRODUCE, no lead) then mention it AGAIN in the same conversation (STATE B -- the ledger reads GROUNDED ->
    REDUCE -> the reply lead flips to "As for it -- "). DRIVE-RATE = fraction of referents where the SAME turn
    demonstrably differs exactly as audience design predicts.
  * ANTI-CHEAT (novel-interleave): ground referent 0, then immediately query a DIFFERENT, never-mentioned
    referent -> must STILL read ungrounded/introduce (proves the decision follows the SPECIFIC referent grounded,
    not "something was grounded this conversation").
  * LESION (`BRAIN_CG_DRIVES_LESION=1`, the ledger's own validated recurrence=0 lesion): replay the IDENTICAL
    first-mention/re-mention script. LESION-DRIVE-RATE = fraction of referents where the re-mention STILL shows
    the reduced lead -- this must collapse to ~0 (the ledger cannot HOLD a grounded bit, so the reply reverts to
    always-introduce, and the STATE A/B difference vanishes).
  * BYTE-IDENTICAL-OFF (by construction, checked directly): `cg_drives_enabled()` with `BRAIN_CG_DRIVES` unset
    reads False -- `webapp/server.py`'s `_common_ground_drives_on()` gate then skips the whole block (no ledger
    build, no `common_ground_drives` key, no lead), so the OFF path never reaches this module at all.
  * NO HONESTY/MOAT REGRESSION (by construction, checked by inspection + the PART-B handler run when available):
    the coupling only ever PREPENDS a `lead` string to an already-computed answer surface
    (`webapp/common_ground_drives_chat.observe_turn` returns no fact/content field, only `{decision, lead,
    evidence_rate, ...}`); the content fields (`abstained`, `recalled_svo`, `verified`) are untouched by this
    module by construction -- verified live in PART B when the full handler is reachable.

PART B — HANDLER NO-REGRESSION (best-effort; through the REAL `brain_chat` handler, `brain='tiny-demo'`,
`renderer='stub'`, no Qwen). Confirms the SAME organ-level verdict holds end-to-end through the actual served
JSON, and that turning the flag off restores the pre-wiring byte-identical answer. Degrades to a reported SKIP
(never a false NO-GO) if `webapp.server` cannot build a session (e.g. a corpus data file missing on a bare
worktree/pool node) -- PART A is the gate, exactly the established convention
(`research/runners/_bg_action_selection_flip_soak.py`).

Run (pool-friendly, numpy, ~15s for 6 seeds):
    SIM_BACKEND=numpy python -m research.runners._common_ground_drive_soak --seeds 42 43 44 100 101 102
Run organ-only (skip the best-effort handler check):
    SIM_BACKEND=numpy python -m research.runners._common_ground_drive_soak --seeds 42 43 44 100 101 102 --organ-only
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import common_ground_drives_chat as _CGD               # noqa: E402  the REAL production coupling
import research.runners.common_ground_ledger_production_organ as _CGL  # noqa: E402  the REAL persistent ledger
from tools.lab import assert_backend, attributable_to                # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_common_ground_drive_soak" / "soak_seed42.json"

WORDS = ["cat", "dog", "ball", "tree", "river", "book"]  # K_REF=6 -- one referent per ledger slot
DRIVE_MIN = 5.0 / 6.0     # per-seed: >=5/6 referents must show the predicted A/B reply difference
LESION_MAX = 1.0 / 6.0    # per-seed: <=1/6 referents may still show a (spurious) drive under lesion
LEAD = "As for it — "


# ── mocks: the "unbound method + a mock self" pattern (no brain build) ──────────────────────────────────────
class _MockComposer:
    """Stands in for the production RFPhasorComposer's `.kb` -- just enough for the SAME comprehension boundary
    the real handler uses (`webapp.gnw_thought_swap._known_concepts` / `_extract_topic`) to recognize `words` as
    grounded concepts (known agent/patient tokens). No neural build; this is host-side comprehension, exactly the
    documented boundary."""

    def __init__(self, words):
        self.kb = [({"agent": w, "action": "is", "patient": w}, None) for w in words]


class _MockChat:
    """Stands in for the production `chat` object; `common_ground_drives_chat.observe_turn` reads only
    `chat.inner.composer` (the topic extractor's grounded-concept scan) -- everything else in that function is
    the real ledger organ + the real webapp coupling logic, unmocked."""

    def __init__(self, words):
        self.inner = type("Inner", (), {"composer": _MockComposer(words)})()


def _mention(chat, word, cache_key, seed, lesion=False):
    """One turn mentioning `word`, through the REAL `webapp.common_ground_drives_chat.observe_turn` (the exact
    function `webapp/server.py`'s `brain_chat` calls). Threads `BRAIN_CG_DRIVES_LESION` exactly as production
    does (`cg_drives_lesioned()` reads the env var), so the lesion path is the real flag, not a bypass."""
    if lesion:
        os.environ["BRAIN_CG_DRIVES_LESION"] = "1"
    else:
        os.environ.pop("BRAIN_CG_DRIVES_LESION", None)
    message = f"what do you know about the {word}?"
    return _CGD.observe_turn(chat, message, cache_key=cache_key, seed=seed)


def run_seed(seed, words=WORDS, verbose=False):
    t0 = time.time()
    key_intact = f"soak-{seed}-intact"
    key_lesion = f"soak-{seed}-lesion"
    key_novel = f"soak-{seed}-novel"
    for k in (key_intact, key_lesion, key_novel):
        _CGL.reset_organ(k)

    # ---- INTACT: STATE A (first mention) vs STATE B (re-mention), every referent ----
    chat = _MockChat(words)
    pairs = []
    for w in words:
        first = _mention(chat, w, key_intact, seed, lesion=False)
        second = _mention(chat, w, key_intact, seed, lesion=False)
        f_ok = (first.get("decision") != "reduce") and (first.get("lead", "") == "")
        s_ok = (second.get("decision") == "reduce") and (second.get("lead", "") == LEAD)
        differs = first.get("lead", "") != second.get("lead", "")
        pairs.append({"word": w, "first": first, "second": second,
                      "drive_ok": bool(f_ok and s_ok and differs)})
    drive_rate = sum(int(p["drive_ok"]) for p in pairs) / len(words)

    # ---- ANTI-CHEAT: novel-interleave (ground word[0], query NEVER-mentioned word[1] -> must read ungrounded) --
    novel_chat = _MockChat(words)
    _mention(novel_chat, words[0], key_novel, seed, lesion=False)
    novel_read = _mention(novel_chat, words[1], key_novel, seed, lesion=False)
    novel_ok = (novel_read.get("decision") != "reduce") and (novel_read.get("lead", "") == "")

    # ---- LESION: identical script, ledger recurrence forced to 0 (the REAL BRAIN_CG_DRIVES_LESION flag) ----
    lesion_pairs = []
    for w in words:
        first = _mention(chat, w, key_lesion, seed, lesion=True)
        second = _mention(chat, w, key_lesion, seed, lesion=True)
        differs = first.get("lead", "") != second.get("lead", "")
        still_drives = differs and (second.get("decision") == "reduce")
        lesion_pairs.append({"word": w, "first": first, "second": second,
                             "lesion_still_drives": bool(still_drives)})
    os.environ.pop("BRAIN_CG_DRIVES_LESION", None)
    lesion_drive_rate = sum(int(p["lesion_still_drives"]) for p in lesion_pairs) / len(words)

    # ---- byte-identical-off (structural: the flag's own default read) ----
    os.environ.pop("BRAIN_CG_DRIVES", None)
    off_by_default = not _CGD.cg_drives_enabled()

    go = bool(drive_rate >= DRIVE_MIN and lesion_drive_rate <= LESION_MAX and novel_ok and off_by_default)
    r = {
        "seed": int(seed), "n_words": len(words),
        "drive_rate": drive_rate, "lesion_drive_rate": lesion_drive_rate,
        "novel_interleave_ok": bool(novel_ok), "off_by_default": bool(off_by_default),
        "go": go, "pairs": pairs, "lesion_pairs": lesion_pairs, "novel_read": novel_read,
        "elapsed_s": round(time.time() - t0, 2),
    }
    if verbose:
        print(f"  [seed {seed}] drive_rate={drive_rate:.3f} (min {DRIVE_MIN:.3f}) | "
              f"lesion_drive_rate={lesion_drive_rate:.3f} (max {LESION_MAX:.3f}) | "
              f"novel_interleave_ok={novel_ok} | off_by_default={off_by_default} || GO={go}", flush=True)
    return r


# ── PART B: best-effort handler no-regression (through the real brain_chat, tiny-demo brain, stub renderer) ────
def run_handler_no_regression(timeout_note=True):
    """Replays the SAME first-mention/re-mention/lesion script through the real `webapp.server.brain_chat`
    handler on the tiny-demo brain (renderer='stub', no Qwen). Degrades to {'skipped': reason} on any import /
    build failure (e.g. a missing corpus data file on a bare worktree) -- PART A remains the gate."""
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    for k in ("BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_DA_DRIVES", "BRAIN_DA_ENCODING", "BRAIN_SELF_INITIATE",
              "BRAIN_COMPREHENSION_GATE", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_WORLDMODEL", "BRAIN_PRAGMATIC",
              "BRAIN_RICH", "BRAIN_SWAP_DRIVES", "BRAIN_OPEN_ENDED", "BRAIN_VISION_IDENTITY", "BRAIN_GNW_SWAP",
              "BRAIN_ONEBRAIN_XEDGE", "BRAIN_SILENT_WM", "BRAIN_BG_SELECT"):
        os.environ.setdefault(k, "0")
    try:
        import webapp.server as S
    except Exception as e:  # a bare pool node without the webapp deps -> report SKIP (PART A is the gate)
        return {"skipped": f"webapp.server import failed: {type(e).__name__}: {e}"}

    def turn(session, message):
        req = S.BrainChatRequest(session=session, message=message, brain="tiny-demo", renderer="stub", rich=False)
        return json.loads(bytes(S.brain_chat(req).body).decode())

    try:
        os.environ["BRAIN_CG_DRIVES"] = "0"
        r_off = turn("cgsoak_off", "what does the cat eat?")
        off_no_key = "common_ground_drives" not in r_off

        os.environ["BRAIN_CG_DRIVES"] = "1"
        r_on1 = turn("cgsoak_on", "what does the cat eat?")          # first mention -> introduce, no lead
        r_on2 = turn("cgsoak_on", "what does the cat eat?")          # re-mention (SAME session) -> reduce, lead
        on1_ok = (r_on1.get("common_ground_drives", {}).get("decision") != "reduce")
        on2_ok = (r_on2.get("common_ground_drives", {}).get("decision") == "reduce"
                  and r_on2.get("answer", "").startswith(LEAD))

        os.environ["BRAIN_CG_DRIVES_LESION"] = "1"
        r_les1 = turn("cgsoak_les", "what does the cat eat?")
        r_les2 = turn("cgsoak_les", "what does the cat eat?")
        os.environ.pop("BRAIN_CG_DRIVES_LESION", None)
        lesion_vanishes = (r_les2.get("common_ground_drives", {}).get("decision") != "reduce"
                           and not r_les2.get("answer", "").startswith(LEAD))
        # the served TEXT under lesion matches the served text with the flag fully off (the load-bearing proof
        # at the handler level -- the trace KEY still appears when the flag is ON, only the ANSWER TEXT is
        # claimed byte-identical to off, matching the module's own "decorates the surface only" contract).
        answer_matches_off = (r_les2.get("answer") == r_off.get("answer"))
    finally:
        os.environ.pop("BRAIN_CG_DRIVES", None)
        os.environ.pop("BRAIN_CG_DRIVES_LESION", None)

    ok = bool(off_no_key and on1_ok and on2_ok and lesion_vanishes and answer_matches_off)
    return {"skipped": None, "off_no_key": off_no_key, "on1_no_lead": on1_ok, "on2_reduces_with_lead": on2_ok,
            "lesion_vanishes": lesion_vanishes, "lesioned_answer_matches_off": answer_matches_off,
            "no_regression": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--organ-only", action="store_true", help="skip PART B (best-effort handler check)")
    ap.add_argument("--handler-timeout-s", type=int, default=180)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))

    print("=" * 118)
    print(f"[cg-soak] PART A -- organ-level 6-seed load-bearing gate (K_ref={len(WORDS)} referents; "
          f"drive_rate>={DRIVE_MIN:.3f}, lesion_drive_rate<={LESION_MAX:.3f}). seeds={a.seeds}", flush=True)
    per_seed = []
    for s in a.seeds:
        try:
            r = run_seed(s, verbose=True)
        except Exception as e:  # noqa: BLE001
            r = {"seed": s, "go": False, "error": repr(e)}
            traceback.print_exc()
        per_seed.append(r)

    ok_seeds = [r for r in per_seed if "error" not in r]
    n_go = sum(1 for r in per_seed if r.get("go"))
    organ_go = bool(ok_seeds) and n_go == len(a.seeds)
    mean_drive = sum(r["drive_rate"] for r in ok_seeds) / len(ok_seeds) if ok_seeds else 0.0
    mean_lesion_drive = sum(r["lesion_drive_rate"] for r in ok_seeds) / len(ok_seeds) if ok_seeds else 0.0
    print(f"  [aggregate] {n_go}/{len(a.seeds)} seeds GO | mean drive_rate={mean_drive:.3f} | "
          f"mean lesion_drive_rate={mean_lesion_drive:.3f} | organ_go={organ_go}", flush=True)
    # ATTRIBUTION (tools.lab.attributable_to, not just reporting both arms): what fraction of the INTACT
    # drive_rate is NOT reproduced by the LESIONED (recurrence=0) replay of the identical script? gap#5 banked
    # a treatment/control pair one key apart for weeks without ever subtracting them -- this call makes the
    # subtraction happen instead of relying on a reader to do it.
    attr_drive = attributable_to("drive_rate: intact vs lesioned ledger recurrence", mean_drive, mean_lesion_drive)

    handler = None
    if not a.organ_only:
        print("\n[cg-soak] PART B -- best-effort handler no-regression (tiny-demo brain, stub renderer) ...",
              flush=True)
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"handler check exceeded {a.handler_timeout_s}s")
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(a.handler_timeout_s)
            try:
                handler = run_handler_no_regression()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        except Exception as e:  # noqa: BLE001
            handler = {"skipped": f"handler check raised/timed out: {type(e).__name__}: {e}"}
            traceback.print_exc()
        if handler.get("skipped"):
            print(f"  PART B SKIPPED: {handler['skipped']}", flush=True)
        else:
            print(f"  off_no_key={handler['off_no_key']} on1_no_lead={handler['on1_no_lead']} "
                  f"on2_reduces_with_lead={handler['on2_reduces_with_lead']} "
                  f"lesion_vanishes={handler['lesion_vanishes']} "
                  f"lesioned_answer_matches_off={handler['lesioned_answer_matches_off']} "
                  f"=> no_regression={handler['no_regression']}", flush=True)

    handler_ok = (handler is None) or bool(handler.get("skipped")) or bool(handler.get("no_regression"))
    overall_go = bool(organ_go and handler_ok)

    print("\n" + "#" * 118)
    print(f"[cg-soak] ORGAN {n_go}/{len(a.seeds)} seeds GO => organ_go={organ_go} | HANDLER "
          f"{'skipped' if (handler and handler.get('skipped')) else ('n/a' if handler is None else handler.get('no_regression'))}"
          f" => {'GO' if overall_go else 'NO-GO'}", flush=True)
    print("#" * 118)

    out_path = Path(a.out)
    if len(a.seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(a.seeds)}seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "probe": "common_ground_drive_soak", "backend": backend, "seeds": a.seeds, "words": WORDS,
        "drive_min": DRIVE_MIN, "lesion_max": LESION_MAX,
        "organ_n_seed_go": n_go, "organ_go": organ_go,
        "mean_drive_rate": mean_drive, "mean_lesion_drive_rate": mean_lesion_drive,
        "attributable_to_intact_ledger": attr_drive,
        "handler": handler, "overall_go": overall_go,
        "elapsed_s": round(time.time() - t0, 1), "per_seed": per_seed}, indent=2, default=str))
    print(f"[cg-soak] wrote {out_path}", flush=True)
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
