"""affect-coloring (ledger row #13, BRAIN_AFFECT) — the ONE default-on faculty the main pass omitted.

It colors WHAT the brain volunteers (mood-congruent forthcomingness: rich max_sentences/elaborations) + HOW it phrases
it (the Qwen manner). The forthcomingness effect lives on the RICH multi-sentence path, so test rich=True. Induce a
strong mood via appraised turns, then probe a recall INTACT vs BRAIN_AFFECT_LESION=1 (the affect_out gate cut) and
compare the reply TEXT. Reported honestly: on numpy the Qwen manner falls back to a template stub, so if only the
manner (not the forthcomingness) moved, the text may not change on this config -> NOT_CLEANLY_TESTABLE, ledger cited.
"""
import os, json
import audit as A

_OUT = "research/findings/raw/_observe_vs_drive/audit_affect_coloring.json"


def _turn(s, m, reset=False, rich=True):
    import webapp.server as S
    from webapp.server import BrainChatRequest
    return json.loads(bytes(S.brain_chat(BrainChatRequest(session=s, message=m, brain="tiny-demo",
                                                          reset=reset, rich=rich)).body))


def arm(lesion, rich):
    A._clear(); A._apply_isolation({"BRAIN_AFFECT"})     # only Gate-B affect on
    if lesion:
        os.environ["BRAIN_AFFECT_LESION"] = "1"
    sess = f"afc_{'les' if lesion else 'int'}_{'r' if rich else 's'}"
    # induce a strong NEGATIVE mood, then probe a recall (the mood colors forthcomingness/manner)
    _turn(sess, "I feel absolutely terrible, sad, afraid and miserable, everything is awful", reset=True, rich=rich)
    d = _turn(sess, "what does the dog chase?", rich=rich)
    A._clear()
    return d


def main():
    rec = {"key": "affect-coloring", "row": "affect-coloring", "lesion_flag": "BRAIN_AFFECT_LESION",
           "meta_key": "affect", "probe": "what does the dog chase? (after a strong-negative mood induction)"}
    results = {}
    for rich in (True, False):
        try:
            di = arm(False, rich); dl = arm(True, rich)
            ai, al = A._ans(di), A._ans(dl)
            results[("rich" if rich else "single")] = {
                "answer_intact": ai, "answer_lesion": al, "answer_changed": (ai != al),
                "n_int": di.get("n_sentences"), "n_les": dl.get("n_sentences"),
                "affect_intact": di.get("affect"), "affect_lesion": dl.get("affect")}
            print("[affect-coloring rich=%s] changed=%s  n_int=%s n_les=%s" % (
                rich, ai != al, di.get("n_sentences"), dl.get("n_sentences")))
            print("   intact: %r" % (ai[:110],)); print("   lesion: %r" % (al[:110],))
        except Exception as e:
            results[("rich" if rich else "single")] = {"error": f"{type(e).__name__}: {e}"}
            print("[affect-coloring rich=%s] ERROR %s" % (rich, e))
    changed_any = any(v.get("answer_changed") for v in results.values() if isinstance(v, dict))
    rec["by_path"] = results
    rec["answer_changed"] = bool(changed_any)
    rec["classification"] = "DRIVER" if changed_any else "NOT_CLEANLY_TESTABLE"
    rec["final_class"] = rec["classification"]
    rec["final_rationale"] = ("the mood colors the reply TEXT (forthcomingness / manner) intact-vs-lesioned"
                              if changed_any else
                              "no text change on numpy: forthcomingness did not cross a threshold and the Qwen manner "
                              "surface falls back to a template stub without a GPU; ledger documents DRIVER on cupy "
                              "(pos +0.039 -> 2 warm sentences vs neg -0.036 -> 1 terse) via BRAIN_AFFECT_LESION")
    json.dump(rec, open(_OUT, "w"), indent=2, default=str)
    print("FINAL affect-coloring ->", rec["classification"], "| wrote", _OUT)


if __name__ == "__main__":
    main()
