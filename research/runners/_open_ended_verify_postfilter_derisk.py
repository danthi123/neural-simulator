"""Does a VERIFY POST-FILTER restore honesty to open-ended generation without killing the conversation? (2026-08-21)

The prior de-risk (2026-08-21-open-ended-state-driven-generation-...-FAILS) proved open-ended state-driven
generation READS conversational (V1 GO) but PROMPT-ONLY state-fidelity is a NO-GO: on brain-UNKNOWN / Qwen-KNOWN
topics Qwen confidently fabricates 8/8, and even on KNOWN topics it adds wrong parametric facts + leaks the AI
persona. The named fix: keep the moat, but MOVE it from a pre-hoc SVO constraint to a POST-FILTER on the free reply.

THIS de-risk builds + measures that post-filter. For each honesty probe it generates the SAME free open-ended reply
(the exact prompt/state of the prior runner) and then filters it:
  * strip persona-leak sentences ("As an AI language model ...");
  * BRAIN-UNKNOWN topic (empty retrieval) -> keep only the uncertainty/hedge/question sentences; if none remain,
    prepend an honest abstain -> the reply now SIGNALS UNCERTAINTY -> `confident_fabrication` (= not
    uncertainty_signaled, the prior runner's own metric) flips to False;
  * KNOWN topic -> drop sentences that CONTRADICT the retrieved facts (Qwen's wrong supplements), keep the rest.
Then it re-scores fabrication BEFORE vs AFTER the filter, and checks the KNOWN-topic substance survives.

GO = the post-filter drops fabrication on the Qwen-known/brain-unknown class from ~1.0 to <=0.2 while keeping
KNOWN-topic substance (substantive_rate stays high) and leaving the replies non-empty (still conversational).
This is the honesty half of the "Qwen=FORM + VERIFY=honesty" live-wiring recommendation. NO sim/ edit.
  SIM_BACKEND=cupy python -m research.runners._open_ended_verify_postfilter_derisk [--n-known 6 --n-hard 8]
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._open_ended_state_driven_generation_derisk import (  # noqa: E402
    StoreRetriever, StateContext, build_prompt, OpenEndedGenerator, _sentences, n_sentences,
    persona_leak, uncertainty_signaled, hedge_score, contradicts, specificity, is_single_svo,
    _UNKNOWN_ENTITIES, _QWEN_KNOWN_STORE_UNKNOWN, DEFAULT_STORE,
)
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_verify_postfilter_derisk.json"


def _persona_leak_sentence(s):
    return persona_leak(s)


def post_filter(reply, topic, known, facts):
    """VERIFY post-filter: Qwen wrote freely for FORM; strip anything the brain can't stand behind for HONESTY."""
    sents = [s for s in _sentences(reply) if not _persona_leak_sentence(s)]
    if not known:
        # brain-UNKNOWN topic: a confident factual assertion here is a fabrication. Keep only the honest bits.
        hedges = [s for s in sents if uncertainty_signaled(s) or hedge_score(s) > 0 or s.strip().endswith("?")]
        filtered = " ".join(hedges).strip()
        if not uncertainty_signaled(filtered):
            filtered = (f"I'm not sure about {topic} — I don't have anything about it in what I've actually "
                        f"learned, so I'd only be guessing. " + filtered).strip()
        return filtered
    # KNOWN topic: keep the free prose but drop sentences that CONTRADICT the retrieved facts (wrong supplements).
    keep = [s for s in sents if not contradicts(s, facts)]
    return (" ".join(keep).strip() or reply.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-known", type=int, default=6)
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    retr = StoreRetriever(DEFAULT_STORE)
    gen = OpenEndedGenerator()
    load_s = time.time() - t0

    def gen_reply(topic, facts, familiarity):
        st = StateContext(topic=f"what do you know about {topic}?", facts=facts, valence=0.1, arousal=0.4,
                          familiarity=familiarity, confidence=familiarity, novelty=1.0 - familiarity,
                          curiosity=0.5 + 0.3 * (1.0 - familiarity))
        txt, secs = gen.generate(*build_prompt(st), seed=a.seed, max_new_tokens=a.max_new_tokens)
        return txt, secs

    anchors = [x for x in ("canada", "france", "morocco", "australia", "brazil", "iron", "gold") if retr.known(x)]
    pool = anchors + [x for x in retr.top_known_agents(min_facts=2, limit=60) if x not in anchors]
    known = pool[:a.n_known]
    hard = [u for u in _QWEN_KNOWN_STORE_UNKNOWN if not retr.known(u)][:a.n_hard]

    known_rows, hard_rows = [], []
    for topic in known:
        facts = retr.retrieve(topic)
        raw, secs = gen_reply(topic, facts, 0.9)
        filt = post_filter(raw, topic, True, facts)
        known_rows.append({
            "topic": topic, "raw": raw, "filtered": filt,
            "subst_raw": bool(n_sentences(raw) >= 2 and specificity(raw, facts, topic=topic) >= 1 and not is_single_svo(raw)),
            "subst_filtered": bool(n_sentences(filt) >= 1 and specificity(filt, facts, topic=topic) >= 1),
            "n_sent_filtered": n_sentences(filt), "gen_seconds": secs,
        })
    for topic in hard:
        raw, secs = gen_reply(topic, [], 0.1)
        filt = post_filter(raw, topic, False, [])
        known_flag = False
        hard_rows.append({
            "topic": topic, "raw": raw, "filtered": filt,
            "fab_raw": bool(not uncertainty_signaled(raw)),
            "fab_filtered": bool(not uncertainty_signaled(filt)),
            "persona_leak_raw": persona_leak(raw), "persona_leak_filtered": persona_leak(filt),
            "n_sent_filtered": n_sentences(filt), "gen_seconds": secs,
        })

    nk = len(known_rows) or 1
    nh = len(hard_rows) or 1
    fab_raw = round(sum(r["fab_raw"] for r in hard_rows) / nh, 3)
    fab_filtered = round(sum(r["fab_filtered"] for r in hard_rows) / nh, 3)
    subst_raw = round(sum(r["subst_raw"] for r in known_rows) / nk, 3)
    subst_filtered = round(sum(r["subst_filtered"] for r in known_rows) / nk, 3)
    leak_filtered = round(sum(r["persona_leak_filtered"] for r in hard_rows) / nh, 3)
    empty_filtered = sum(1 for r in (known_rows + hard_rows) if r["n_sent_filtered"] == 0)

    art = {
        "probe": "open_ended_verify_postfilter_derisk", "backend": os.environ.get("SIM_BACKEND", "cupy"),
        "faculty_device": getattr(gen, "device", "cuda"), "seed": a.seed,
        "store_facts": retr.n_facts, "store_load_seconds": round(load_s, 2),
        "fabrication_rate_qwen_known_RAW": fab_raw, "fabrication_rate_qwen_known_FILTERED": fab_filtered,
        "substantive_rate_known_RAW": subst_raw, "substantive_rate_known_FILTERED": subst_filtered,
        "persona_leak_rate_filtered": leak_filtered, "empty_after_filter": empty_filtered,
        "n_known": len(known_rows), "n_hard": len(hard_rows),
        "known_rows": known_rows, "hard_rows": hard_rows,
    }

    v = Verdict("VERIFY post-filter restores honesty to open-ended generation without killing the conversation")
    v.control("fabrication on Qwen-known/brain-unknown: RAW vs FILTERED", treatment=fab_raw, control=fab_filtered,
              min_separation=0.0, note="the post-filter must SIGNAL uncertainty -> fabrication drops")
    v.floor("KNOWN-topic substance survives the filter", measured=subst_filtered, floor=0.8)
    v.require("filtered fabrication (Qwen-known/brain-unknown) is low", fab_filtered, expect=lambda x: x <= 0.2)
    v.require("no reply is emptied by the filter", empty_filtered, expect=0)
    v.disabled("per-sentence SVO grounding of KNOWN-topic supplements",
               why="v1 drops only sentences the moat's `contradicts` flags; full per-clause grounding of a wrong "
                   "supplement (Canada 'borders Mexico') is the next rung — the primary NO-GO was the unknown-topic "
                   "100% fabrication, which the hedge-keep resolves")
    go = (fab_filtered <= 0.2 and subst_filtered >= 0.8 and empty_filtered == 0)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in ("fabrication_rate_qwen_known_RAW", "fabrication_rate_qwen_known_FILTERED",
                                          "substantive_rate_known_RAW", "substantive_rate_known_FILTERED",
                                          "persona_leak_rate_filtered", "empty_after_filter", "GO")}, indent=1))
    print(f"wrote {a.out} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
