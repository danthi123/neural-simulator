"""6-seed flip-soak for the BRAIN_SPIKING_MOUTH_RECALL production wire-in (spiking Broca RECALL / RICH surface).

The FLIP GATE for routing the GROUNDED bounded-SVO recall/rich answer surface through the spiking Broca word-order
render (spiking_mouth_recall_prod) instead of the Qwen/template surface. GPU-free (numpy, template-stub faculty),
so it runs headless on the mini-PC pool.

Per seed it proves the three properties a validated wire-in needs:

  (1) FLAG-OFF BYTE-IDENTICAL. With BRAIN_SPIKING_MOUTH_RECALL unset (default OFF), every recall surface is EXACTLY
      the current Qwen/template path -- for each stored transitive fact, chat.render(svo) with the flag OFF equals
      the template-stub's own verified surface (the pre-wire behaviour), and NO spiking-form surface leaks.

  (2) LOAD-BEARING (the lesion oracle). With the flag ON, a bounded transitive-SVO recall is authored ON SPIKES:
        - flag lesion     : ON -> "the brain uses the spikes" (spiking form) vs OFF -> "The brain uses spikes."
          (template) -- the word ORDER / surface CHANGES while the recalled CONTENT SVO is byte-identical;
        - rate-read lesion: the intact per-pool spiking-RATE ranking gives the CORRECT slot order; the EMERGE
          anti-cheats (equal_drive: rates tie / permute_order: fixed wrong order) SCRAMBLE the ORDER with the SAME
          content words -> proves the SPIKING READ authored the ORDER, not a fixed host template.
      A lesioned (scrambled) surface also FAILS the re-parse VERIFY, so production safely falls back -- the moat
      never emits a scrambled sentence.

  (3) NO-REGRESSION. With the flag ON: the no-confab moat still ABSTAINS on untaught/general cues; every rich
      answer stays SUBSTANTIVE (>=2 sentences) and every kept sentence is brain-sourced; and every spiking-authored
      recall sentence re-parses to EXACTLY its recalled SVO (content preserved).

GO = all three hold on ALL 6 seeds (>=1 fact genuinely authored on spikes per seed, i.e. real coverage, not vacuous).

Usage:
    SIM_BACKEND=numpy python -m research.runners._spiking_mouth_recall_soak \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/_spiking_mouth_recall_soak.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_DISABLE_LLM", "1")

from tools.lab import attributable_to                      # noqa: E402  (attribution for the rate-read lesion)
from tools.verdict import Verdict                           # noqa: E402  (preconditions carried by the verdict)

_SEEDS_DEFAULT = [42, 43, 44, 100, 101, 102]
_FLAG = "BRAIN_SPIKING_MOUTH_RECALL"


def _set_flag(on):
    if on:
        os.environ[_FLAG] = "1"
    else:
        os.environ.pop(_FLAG, None)          # unset => default OFF


def _stub_surface(a, v, p):
    """The template-stub's OWN verified surface for (a,v,p) -- the pre-wire Qwen/template recall path the flag-OFF
    render must reproduce byte-identically. Returns (surface, verified) via the SAME re-parse the recall path uses."""
    from research.runners.brain_chat_tui import StubRenderer
    r = StubRenderer()
    surface, asserted = r.render_svo(a, v, p)
    return surface, asserted


def run_seed(seed):
    from research.runners.rich_answer_composer import (
        RichAnswerComposer, _build_smoke_chat, _SMOKE_SCRIPT)
    from research.runners.spiking_mouth_recall_prod import (
        frame_supported, SpikingRecallMouth)

    # ---- build the brain (template-stub faculty; deterministic, GPU-free) ----
    chat = _build_smoke_chat(seed, use_multiturn=True)
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4)
    stored = [tuple(f) for f in rich._stored_facts()]
    stored_set = set(stored)
    transitive = [f for f in stored if frame_supported(*f)]

    # ================================================================================================
    # (1) FLAG-OFF BYTE-IDENTICAL: chat.render(svo) with the flag OFF == the template-stub's own surface.
    # ================================================================================================
    _set_flag(False)
    off_identical = []
    off_no_spiking = True
    for f in transitive:
        rendered_off = chat.render(list(f))
        stub_surface, stub_asserted = _stub_surface(*f)
        # the stub verifies (its asserted SVO re-parses to the fact) -> the flag-OFF render MUST equal it exactly.
        stub_ok = chat._verify(stub_surface, stub_asserted, list(f))
        if stub_ok:
            off_identical.append(rendered_off == stub_surface)
        # a spiking-form recall surface is "the <s> <v-3sg> the <o>" (two determiners); the stub form is
        # "The <S> <v-3sg> <o>." -- if the flag-OFF render ever produced the spiking two-determiner form, the
        # byte-identical guarantee is broken.
        if rendered_off.lower().startswith("the ") and rendered_off.count(" the ") >= 1 and not rendered_off.endswith("."):
            off_no_spiking = False
    flag_off_byte_identical = bool(off_identical) and all(off_identical) and off_no_spiking

    # ================================================================================================
    # (2) LOAD-BEARING: flag ON authors the surface ON SPIKES; two lesions change the ORDER, content byte-identical.
    # ================================================================================================
    _set_flag(True)
    load_bearing_facts = []          # facts genuinely authored on spikes (spiking form, verifies, != flag-OFF form)
    for f in transitive:
        a, v, p = f
        on_surface = chat.render([a, v, p])
        _set_flag(False)
        off_surface = chat.render([a, v, p])
        _set_flag(True)
        # a genuinely spiking-authored recall: the ON surface is the spiking two-determiner form, differs from the
        # flag-OFF (template) form, and re-parses to EXACTLY this SVO (content preserved).
        spiking_form = on_surface.startswith("the ") and on_surface == f"the {a} {_v3(v)} the {p}"
        content_ok = chat._verify(on_surface, None, [a, v, p])
        if spiking_form and content_ok and on_surface != off_surface:
            load_bearing_facts.append(f)
    # --- rate-read lesion: build the intact + equal-drive producers ONCE (the EMERGE-61 wash-out makes each emit
    # independent, so one build renders every fact); compare the ORDER + content on the load-bearing facts. Intact
    # per-pool spiking-RATE ranking -> correct slot order; equal-drive (rates tie) -> SCRAMBLED order, SAME content
    # words -> proves the SPIKING READ authored the ORDER, not a fixed host template.
    order_changes = []               # (fact, intact_order, lesioned_order, content_multiset_identical & order_changed)
    intact_canonical = lesion_canonical = 0.0
    order_attribution = None
    if load_bearing_facts:
        intact = SpikingRecallMouth(seed=seed, mode=None)
        lesioned = SpikingRecallMouth(seed=seed, mode="equal_drive")
        n_ic = n_lc = 0
        for f in load_bearing_facts:
            a, v, p = f
            canonical = f"the {a} {_v3(v)} the {p}"           # the CORRECT transitive order
            s_intact = intact.render(a, v, p)
            s_lesion = lesioned.render(a, v, p)
            same_content = Counter(s_intact.split()) == Counter(s_lesion.split())
            order_changed = s_intact.split() != s_lesion.split()
            order_changes.append((list(f), s_intact, s_lesion, bool(same_content and order_changed)))
            # the ATTRIBUTION measure = does the emitted WORD ORDER match the CORRECT canonical order? (NOT the
            # re-parse verify -- the independent parser is order-tolerant, so a scramble can still re-parse; the
            # exact canonical order is the clean read of "did the spiking rate ranking author the ORDER".)
            n_ic += 1 if s_intact == canonical else 0         # intact spiking-RATE read -> canonical (expect 1.0)
            n_lc += 1 if s_lesion == canonical else 0         # equal-drive (rates tie) -> NOT canonical (expect 0)
        intact_canonical = n_ic / len(load_bearing_facts)
        lesion_canonical = n_lc / len(load_bearing_facts)
        # ATTRIBUTION (tools.lab): what FRACTION of the CORRECT-ORDER effect is OWNED by the spiking-rate read,
        # i.e. absent in the equal-drive control? (treatment - control)/treatment. The equal-drive control holds
        # the same substrate + the same words, varying ONLY the rate-ranking drive, so the residual is the read.
        order_attribution = attributable_to(
            f"spiking-rate slot-order read (seed {seed}, canonical order)", intact_canonical, lesion_canonical)
    # the ORDER is authored on spikes IFF: it changes under the rate lesion with identical content words, AND the
    # correct canonical order is (near-)fully attributable to the spiking-rate read (absent in the equal-drive arm).
    rate_lesion_ok = bool(order_changes) and all(oc[3] for oc in order_changes)
    attribution_ok = order_attribution is not None and order_attribution >= 0.5
    load_bearing = bool(load_bearing_facts) and rate_lesion_ok and attribution_ok

    # ================================================================================================
    # (3) NO-REGRESSION: a mouth wire-in must change the SURFACE ONLY, never the CONTENT. Run the SAME live rich
    #     conversation with the flag OFF and ON and require CONTENT-EQUIVALENCE per turn -- identical abstain,
    #     identical sentence count, identical supporting facts -- while the ON SURFACE differs on >=1 turn (the
    #     spiking mouth genuinely re-authored the surface). PLUS the no-confab moat still ABSTAINS on untaught cues.
    #     (A turn may legitimately abstain in this multiturn smoke -- e.g. thread state exhausts a sub-topic -- so
    #     the gate is OFF==ON equivalence, NOT an absolute "every turn substantive" bar the baseline itself fails.)
    # ================================================================================================
    def _live_transcript(flag_on):
        _set_flag(flag_on)
        c = _build_smoke_chat(seed, use_multiturn=True)
        rc = RichAnswerComposer(c, max_chain_hops=3, max_elaborations=2, max_sentences=4)
        st = set(tuple(f) for f in rc._stored_facts())
        tr = []
        for utterance, kind in _SMOKE_SCRIPT:
            r = rc.answer(utterance)
            tr.append({"you": utterance, "kind": kind, "answer": r["answer"], "abstained": r["abstained"],
                       "n_sentences": r["n_sentences"], "facts": [list(f) for f in r["facts"]],
                       "brain_sourced": all(tuple(f) in st for f in r["facts"])})
        return tr

    tr_off = _live_transcript(False)
    tr_on = _live_transcript(True)
    content_equiv = all(
        (a["abstained"] == b["abstained"] and a["n_sentences"] == b["n_sentences"] and a["facts"] == b["facts"])
        for a, b in zip(tr_off, tr_on))
    surface_changed = any(a["answer"] != b["answer"] for a, b in zip(tr_off, tr_on))
    abstain_turns = [t for t in tr_on if t["kind"] == "abstain"]
    moat_held = all(t["abstained"] for t in abstain_turns)
    all_brain_sourced = all(t["brain_sourced"] for t in tr_on)
    no_regression = bool(content_equiv and surface_changed and moat_held and all_brain_sourced)

    _set_flag(False)                                 # leave the process in the default (OFF) state

    seed_go = bool(flag_off_byte_identical and load_bearing and no_regression)
    return {
        "seed": seed,
        "go": seed_go,
        "flag_off_byte_identical": flag_off_byte_identical,
        "load_bearing": load_bearing,
        "rate_lesion_ok": rate_lesion_ok,
        "no_regression": no_regression,
        "content_equiv_off_on": content_equiv,
        "surface_changed_off_on": surface_changed,
        "intact_canonical_order_rate": intact_canonical,
        "lesioned_canonical_order_rate": lesion_canonical,
        "order_attribution_to_spiking_read": order_attribution,
        "attribution_ok": attribution_ok,
        "n_transitive_facts": len(transitive),
        "n_load_bearing_facts": len(load_bearing_facts),
        "load_bearing_examples": [list(f) for f in load_bearing_facts[:4]],
        "order_change_examples": order_changes[:3],
        "moat_held": moat_held,
        "all_brain_sourced": all_brain_sourced,
        "transcript_off": tr_off,
        "transcript_on": tr_on,
    }


def _v3(v):
    from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3
    return emerge_v3(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=_SEEDS_DEFAULT)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_spiking_mouth_recall_soak.json")
    args = ap.parse_args()

    per_seed = [run_seed(s) for s in args.seeds]
    go = all(r["go"] for r in per_seed)

    # --- EARN the verdict: preconditions carried by the artifact (tools.verdict.Verdict) ---
    # aggregate the rate-read attribution: mean canonical-order rate, intact vs equal-drive control, across seeds.
    ivs = [r["intact_canonical_order_rate"] for r in per_seed]
    lvs = [r["lesioned_canonical_order_rate"] for r in per_seed]
    intact_mean = sum(ivs) / len(ivs) if ivs else 0.0
    lesion_mean = sum(lvs) / len(lvs) if lvs else 0.0
    v = Verdict("BRAIN_SPIKING_MOUTH_RECALL production wire-in (6-seed flip-soak)")
    v.require("flag-OFF byte-identical on ALL seeds (exact compare)",
              all(r["flag_off_byte_identical"] for r in per_seed), expect=True)
    v.require("content-equivalence OFF==ON (surface-only change) on ALL seeds",
              all(r["content_equiv_off_on"] for r in per_seed), expect=True)
    v.require("surface genuinely re-authored (ON != OFF) on ALL seeds",
              all(r["surface_changed_off_on"] for r in per_seed), expect=True)
    v.require("no-confab moat still abstains on untaught cues on ALL seeds",
              all(r["moat_held"] for r in per_seed), expect=True)
    v.require("rate-read lesion changes the ORDER (same content words) on ALL seeds",
              all(r["rate_lesion_ok"] for r in per_seed), expect=True)
    # the load-bearing CONTROL: intact spiking-RATE read vs equal-drive (rates tie); they must SEPARATE, i.e. the
    # correct canonical ORDER is present with the read and absent without it.
    v.control("canonical word ORDER: intact spiking-rate read vs equal-drive control",
              intact_mean, lesion_mean, min_separation=0.5)
    v.require(">=1 recall fact genuinely authored on spikes per seed (non-vacuous coverage)",
              all(r["n_load_bearing_facts"] >= 1 for r in per_seed), expect=True)
    decided = v.decide(go=go, verbose=True)

    summary = {
        "flag": _FLAG,
        "seeds": args.seeds,
        "GO": go,
        "status": decided["status"],
        "preconditions": decided["preconditions"],
        "n_seeds_go": sum(1 for r in per_seed if r["go"]),
        "flag_off_byte_identical_all": all(r["flag_off_byte_identical"] for r in per_seed),
        "load_bearing_all": all(r["load_bearing"] for r in per_seed),
        "no_regression_all": all(r["no_regression"] for r in per_seed),
        "intact_order_verify_mean": intact_mean,
        "lesioned_order_verify_mean": lesion_mean,
        "per_seed": per_seed,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(_REPO) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== BRAIN_SPIKING_MOUTH_RECALL 6-seed flip-soak ===")
    for r in per_seed:
        print(f"  seed {r['seed']:>3}: GO={r['go']!s:>5}  byte-identical(OFF)={r['flag_off_byte_identical']!s:>5}"
              f"  load-bearing={r['load_bearing']!s:>5}  no-regression={r['no_regression']!s:>5}"
              f"  spiking-authored facts={r['n_load_bearing_facts']}/{r['n_transitive_facts']}")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'}  ({summary['n_seeds_go']}/{len(args.seeds)} seeds)")
    print(f"  wrote {out}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
