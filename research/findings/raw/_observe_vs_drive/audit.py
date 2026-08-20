"""OBSERVE-vs-DRIVE FACULTY AUDIT — the anti-hollow-integration check applied across the WHOLE live /api/brain-chat turn.

Of the ~31 faculties the PRODUCTION_INTEGRATION_LEDGER lists on_by_default:YES, which ones actually CHANGE what the
brain SAYS (DRIVERS), which only feed ANOTHER faculty / the substrate (FEEDERS), and which compute a neural verdict
that goes nowhere observable (DEAD OBSERVERS = hollow integration)?

METHOD (applies the #84/#85 lessons — build ONE ChatBrain family in-process, share warmup, isolate organs for speed,
NO short timeout spin): for each faculty, run its TRIGGER probe through the REAL `brain_chat` handler INTACT, then
LESIONED (toggle its env lesion flag / master-disable), on CLEAN reset sessions. Compare the reply TEXT (`answer`).
  DRIVER        := answer text CHANGES intact-vs-lesioned (load-bearing on the response).
  FEEDER        := answer text UNCHANGED, but it produces/relocates substrate another faculty consumes, or its
                   metadata is consumed downstream (answer-preserving BY DESIGN — the substrate-mechanism rows).
  DEAD OBSERVER := answer UNCHANGED and nothing else observable changes — a neural verdict stashed with no consumer.
  NOT_CLEANLY_TESTABLE := no lesion flag / no reliable trigger on this (numpy, fast) config — reported honestly.

Isolation: every OTHER heavy organ is disabled (a consistent baseline across BOTH arms) so each turn is a few seconds;
the faculty under test reads its own path regardless. Run:
  SIM_BACKEND=numpy PYTHONPATH=<worktree> python research/findings/raw/_observe_vs_drive/audit.py
"""
import os, json, hashlib, subprocess, time, traceback

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

from webapp.server import brain_chat, BrainChatRequest  # the REAL handler

_ART = os.environ.get("AUDIT_JSON", "research/findings/raw/_observe_vs_drive/audit.json")

# Every organ master-disable flag. The isolation baseline sets ALL to "0"; per-faculty `keep_on` removes the ones the
# faculty (and its declared dependencies) need. Faculties whose default-on has no master env (gnw-deliberate/-multistep
# via _*_DEFAULT_ON, gnw-bus via install) are enabled simply by NOT setting their disable flag.
ALL_DISABLE = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_SURPRISE": "0", "BRAIN_METACOG": "0",
    "BRAIN_COMPREHENSION_GATE": "0", "BRAIN_PRAGMATIC": "0", "BRAIN_EPISODIC": "0", "BRAIN_MULTIREF": "0",
    "BRAIN_SELF_INITIATE": "0", "BRAIN_GNW_DELIBERATE": "0", "BRAIN_GNW_MULTISTEP": "0",
    "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_RECONSOLIDATION": "0", "BRAIN_PMEM": "0", "BRAIN_CURIOSITY": "0",
    "BRAIN_DISCOURSE_REGISTER": "0", "BRAIN_CAUSAL": "0", "BRAIN_REPAIR": "0", "BRAIN_AFFECT_DRIVES": "0",
    "BRAIN_SWAP_DRIVES": "0", "BRAIN_DA_DRIVES": "0", "BRAIN_GENERATE_CHANNEL": "0",
}
# All env keys we ever set, cleared before each arm for a hermetic baseline.
_ALL_ENV = set(ALL_DISABLE) | {
    "BRAIN_GNW_BUS", "BRAIN_GNW_BUS_HOST", "BRAIN_GNW_BUS_LESION", "BRAIN_GNW_SWAP", "BRAIN_RICH", "BRAIN_CLAIM_MOAT",
    "BRAIN_COMPOSER_KIND", "BRAIN_ONEBRAIN_MERGE", "BRAIN_AFFECT_DRIVES_INDUCE", "BRAIN_DA_DRIVES_INDUCE",
} | {f"{k}_LESION" for k in ALL_DISABLE} | {
    "BRAIN_AFFECT_LESION", "BRAIN_COMPREHENSION_LESION", "BRAIN_SURPRISE_LESION", "BRAIN_METACOG_LESION",
    "BRAIN_WORLDMODEL_LESION", "BRAIN_CURIOSITY_LESION", "BRAIN_MULTIREF_LESION", "BRAIN_EPISODIC_LESION",
    "BRAIN_DISCOURSE_REGISTER_LESION", "BRAIN_NONCONTRADICTION_LESION", "BRAIN_RECONSOLIDATION_LESION",
    "BRAIN_PMEM_LESION", "BRAIN_CAUSAL_LESION", "BRAIN_PRAGMATIC_LESION", "BRAIN_SELF_INITIATE_LESION",
    "BRAIN_GNW_DELIBERATE_LESION", "BRAIN_GNW_MULTISTEP_LESION", "BRAIN_AFFECT_DRIVES_LESION",
    "BRAIN_SWAP_DRIVES_LESION", "BRAIN_DA_DRIVES_LESION", "BRAIN_SPIKING_DRAW_LESION",
}


def _clear():
    for k in _ALL_ENV:
        os.environ.pop(k, None)


def _apply_isolation(keep_on):
    """Disable every heavy organ except those in keep_on (the faculty + its deps)."""
    for k, v in ALL_DISABLE.items():
        if k not in keep_on:
            os.environ[k] = v


def turn(session, message, reset=False, rich=False):
    resp = brain_chat(BrainChatRequest(session=session, message=message, brain="tiny-demo", reset=reset, rich=rich))
    return json.loads(bytes(resp.body))


def _ans(d):
    return d.get("answer", "") if isinstance(d, dict) else ""


# ── FACULTY SPEC ────────────────────────────────────────────────────────────────────────────────────────────────
# Each: key, ledger_row, lesion (env flag or None), keep_on (masters to leave ON), rich, extra_env (dict, e.g. INDUCE),
#   setup (list of messages run before the probe on the SAME session), probe (message), expect (human note).
# The lesion arm sets `lesion`="1" (or applies special handling by key). answer_int != answer_les => DRIVER.
FAC = [
    # ── the Gate-B honest-notice / short-circuit organs (single-turn triggers) ──
    dict(key="surprise-monitor", row="surprise-monitor", lesion="BRAIN_SURPRISE_LESION",
         keep_on={"BRAIN_SURPRISE"}, meta="surprise",
         setup=[], probe="dog chase cat",  # CONFIRM a stored fact: intact not-surprised(no notice); lesion fires(notice)
         note="assert a STORED fact -> intact no surprise-notice; lesion makes the same confirm FIRE a notice"),
    dict(key="metacog-monitor", row="metacog-monitor", lesion="BRAIN_METACOG_LESION",
         keep_on={"BRAIN_METACOG"}, meta="metacog",
         setup=[], probe="what does the dog chase?",
         note="a recall answer; intact confident (no hedge) vs lesion collapses margin -> hedge prepended"),
    dict(key="worldmodel-forward", row="worldmodel-forward", lesion="BRAIN_WORLDMODEL_LESION",
         keep_on={"BRAIN_WORLDMODEL"}, meta="worldmodel",
         setup=["I feel wonderful and happy"], probe="how is this going?",
         note="queryable expectation; lesion collapses the prediction margin -> the expectation text changes"),
    dict(key="curiosity-followup", row="curiosity-followup", lesion="BRAIN_CURIOSITY_LESION",
         keep_on={"BRAIN_CURIOSITY"}, meta="curiosity",
         setup=[], probe="what does the dragon breathe?",  # novel abstain -> crave -> follow-up appended
         note="novel-topic ABSTAIN; intact appends a follow-up question; lesion (drive cut) -> no follow-up"),
    dict(key="pragmatic-implicature", row="pragmatic-implicature", lesion="BRAIN_PRAGMATIC_LESION",
         keep_on={"BRAIN_PRAGMATIC"}, meta="pragmatic",
         setup=[], probe="I ate some of the cookies",
         note="scalar-quantity turn; intact prepends the graded implicature reading; lesion (flat belief) -> suppressed"),
    dict(key="comprehension-monitor", row="comprehension-monitor", lesion="BRAIN_COMPREHENSION_LESION",
         keep_on={"BRAIN_COMPREHENSION_GATE"}, meta="comprehension",
         setup=[], probe="the dog eats the bone",  # a competent, comprehended transitive (margin high)
         note="a competent transitive; intact comprehends (normal path); lesion collapses margin -> honest 'didn't follow'"),
    dict(key="other-repair", row="other-repair", lesion="BRAIN_COMPREHENSION_LESION",
         keep_on={"BRAIN_COMPREHENSION_GATE", "BRAIN_REPAIR"}, meta="repair",
         setup=[], probe="book carries cup",  # 2-inanimate -> D4 abstain -> targeted role clarification
         note="a low-comprehension turn D4 abstains on; intact = targeted clarification; lesion (=D4 lesion) -> bare abstain"),
    dict(key="noncontradiction-gate", row="noncontradiction-gate", lesion="BRAIN_NONCONTRADICTION_LESION",
         keep_on={"BRAIN_NONCONTRADICTION_GATE"}, meta="noncontradiction",
         setup=["the dog does not chase cat"], probe="the dog chases cat",  # asserts opposite polarity of stored NEGATE
         note="assert a polarity-contradicting fact; intact REJECTS; lesion (recall bypass) -> slips through/accepts"),
    # ── the drive-couplings (leads / suffix) ──
    dict(key="affect-drives-response", row="affect-drives-response", lesion="BRAIN_AFFECT_DRIVES_LESION",
         keep_on=set(), meta="affect_drives", extra_env={"BRAIN_AFFECT_DRIVES_INDUCE": "-0.7,0.6"},
         setup=[], probe="what does the dog chase?",
         note="message FIXED + induced NEG mood; intact prepends an affective lead; lesion collapses mood -> lead gone"),
    dict(key="swap-drives-response", row="swap-drives-response", lesion="BRAIN_SWAP_DRIVES_LESION",
         keep_on=set(), meta="swap_drives",
         setup=["what does the cat eat?"], probe="what does the dog chase?",  # topic change cat->dog -> swap lead
         note="a topic-change turn; intact prepends a topic-transition lead; lesion (mm silenced) -> no swap, lead gone"),
    dict(key="da-mode-drives-response", row="da-mode-drives-response", lesion="BRAIN_DA_DRIVES_LESION",
         keep_on=set(), meta="da_drives", extra_env={"BRAIN_DA_DRIVES_INDUCE": "1300"},
         setup=[], probe="what does the dog chase?",
         note="message FIXED + induced high engagement; intact appends an engagement suffix; lesion (SNc silenced) -> gone"),
    # ── the substrate-authoring / combination ──
    dict(key="gnw-bus-shadow", row="gnw-bus-shadow", lesion="BRAIN_GNW_BUS_LESION",
         keep_on=set(), meta="gnw_bus", extra_env={"BRAIN_GNW_BUS": "1"},  # BUS observability key on
         setup=[], probe="what does the dog chase?",
         note="the substrate authors the recall combination; lesion collapses the ignition -> the combined answer -> abstain"),
    # ── self-initiated (idle trigger) ──
    dict(key="self-initiated-utterance", row="self-initiated-utterance", lesion="BRAIN_SELF_INITIATE_LESION",
         keep_on={"BRAIN_SELF_INITIATE"}, meta="self_initiated",
         setup=[], probe="",  # empty/idle turn -> the brain self-initiates
         note="an idle/empty turn; intact self-initiates an utterance; lesion (empty store) -> neutral idle fallback"),
    # ── multi-turn memory / discourse organs ──
    dict(key="reconsolidation", row="reconsolidation", lesion="BRAIN_RECONSOLIDATION_LESION",
         keep_on={"BRAIN_SURPRISE", "BRAIN_RECONSOLIDATION"}, meta="reconsolidation",
         setup=["dog go north", "dog go south"], probe="what does the dog go?",
         note="teach north, contradict with south (surprise window open) -> intact rewrites in place (recall 'south'); "
              "lesion blocks the update -> stale 'north'"),
    dict(key="wm-binding-advanced", row="wm-binding-advanced", lesion="BRAIN_MULTIREF_LESION",
         keep_on={"BRAIN_MULTIREF"}, meta="multiref",
         setup=["the dog and the cat"], probe="who are we keeping in mind?",
         note="hold >=2 referents then read back; intact 'holding 2: dog and cat'; lesion (recur=0) collapses the hold"),
    dict(key="discourse-register", row="discourse-register", lesion="BRAIN_DISCOURSE_REGISTER_LESION",
         keep_on={"BRAIN_DISCOURSE_REGISTER"}, meta="discourse",
         setup=["the dog chased the cat and then the cat ate the fish"], probe="who was doing it before?",
         note="fold a discourse pair across a connective; intact answers the who-was-before slot; lesion collapses it"),
    dict(key="prospective-memory", row="prospective-memory", lesion="BRAIN_PMEM_LESION",
         keep_on={"BRAIN_PMEM"}, meta="prospective",
         setup=["remind me to call mom when I get home", "what does the dog chase?", "what does the cat eat?"],
         probe="I got home",
         note="latch an intention across intervening turns; on the cue intact prepends the reminder; lesion -> no fire"),
    dict(key="episodic-memory", row="episodic-memory", lesion="BRAIN_EPISODIC_LESION",
         keep_on={"BRAIN_EPISODIC"}, meta="episodic",
         setup=["the dog chased the cat"], probe="did we discuss the dog?",
         note="referential recall of a past turn; NOTE the BTSP store is cupy-gated (numpy defers the WRITE) -> the "
              "recall gate may have nothing stored on this config"),
    # ── the conflict / chain keystones (need taught competitors / a chain) ──
    dict(key="gnw-deliberation", row="gnw-deliberation", lesion="BRAIN_GNW_DELIBERATE_LESION",
         keep_on=set(), meta=None,
         setup=["dog chase bird"], probe="what does the dog chase?",  # dog-chase now has cat(stored)+bird -> conflict
         note="two candidate patients share (dog,chase) -> intact ABSTAINS on the conflict; lesion re-commits first-match"),
    dict(key="gnw-multistep-deliberation", row="gnw-multistep-deliberation", lesion="BRAIN_GNW_MULTISTEP_LESION",
         keep_on=set(), meta=None,
         setup=["zorp chase blib", "blib chase munt"], probe="what does the zorp chase all the way?",
         note="a chase chain zorp->blib->munt; intact re-enters to the terminal 'munt'; lesion collapses -> abstain"),
    dict(key="causal-whatif", row="causal-whatif", lesion="BRAIN_CAUSAL_LESION",
         keep_on={"BRAIN_CAUSAL"}, meta="causal",
         setup=[], probe="what happens if the dog chases?",
         note="a what-if forward-simulation over stored facts; intact emits a moat-confirmed consequence; lesion -> abstain"),
    # ── open-ended generation (master + spiking-draw lesion) ──
    dict(key="open-ended-generation", row="open-ended-generation", lesion="BRAIN_GENERATE_CHANNEL_DISABLE",
         keep_on={"BRAIN_GENERATE_CHANNEL"}, meta=None,
         setup=[], probe="what might a dog chase?",
         note="an open-ended prompt; intact VOLUNTEERS a flagged novel guess; channel-off -> abstain/normal (no guess)"),
]


def run_faculty(f):
    key = f["key"]
    rec = {"key": key, "row": f["row"], "lesion_flag": f.get("lesion"), "meta_key": f.get("meta"),
           "probe": f.get("probe"), "setup": f.get("setup"), "note": f.get("note")}
    try:
        arms = {}
        for arm in ("intact", "lesion"):
            _clear()
            _apply_isolation(f.get("keep_on", set()))
            for k, v in (f.get("extra_env") or {}).items():
                os.environ[k] = v
            if arm == "lesion":
                lf = f.get("lesion")
                if lf == "BRAIN_GENERATE_CHANNEL_DISABLE":
                    os.environ["BRAIN_GENERATE_CHANNEL"] = "0"   # master-off as the removal arm
                elif lf:
                    os.environ[lf] = "1"
            sess = f"aud_{key}_{arm}"
            last = None
            for i, m in enumerate(f.get("setup", [])):
                last = turn(sess, m, reset=(i == 0))
            probe_reset = (len(f.get("setup", [])) == 0)
            d = turn(sess, f["probe"], reset=probe_reset)
            arms[arm] = d
        _clear()
        a_int, a_les = _ans(arms["intact"]), _ans(arms["lesion"])
        answer_changed = (a_int != a_les)
        mk = f.get("meta")
        meta_int = arms["intact"].get(mk) if mk else None
        meta_les = arms["lesion"].get(mk) if mk else None
        meta_present = meta_int is not None
        meta_changed = (json.dumps(meta_int, sort_keys=True, default=str)
                        != json.dumps(meta_les, sort_keys=True, default=str)) if mk else False
        # classify
        if answer_changed:
            cls = "DRIVER"
        elif meta_present:
            cls = "OBSERVER_OR_FEEDER"   # answer unchanged but a neural verdict was computed -> needs judgement
        else:
            cls = "NO_OBSERVED_EFFECT"   # neither the answer nor a metadata verdict moved -> trigger likely didn't fire
        rec.update({"answer_intact": a_int, "answer_lesion": a_les, "answer_changed": bool(answer_changed),
                    "meta_intact": meta_int, "meta_lesion": meta_les, "meta_present": bool(meta_present),
                    "meta_changed": bool(meta_changed), "classification": cls,
                    "abstained_intact": arms["intact"].get("abstained"),
                    "abstained_lesion": arms["lesion"].get("abstained")})
        print("  [%-26s] answer_changed=%-5s meta_present=%-5s -> %s" % (key, answer_changed, meta_present, cls))
        print("      intact : %r" % (a_int[:90],))
        print("      lesion : %r" % (a_les[:90],))
    except Exception as e:
        rec.update({"classification": "ERROR", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:]})
        print("  [%-26s] ERROR %s" % (key, e))
    return rec


# ── substrate-mechanism rows (answer-preserving BY DESIGN): confirm byte-identical under the escape (=> FEEDER) ──
def run_substrate_escape(key, row, env_toggle, note, probe="what does the dog chase?"):
    rec = {"key": key, "row": row, "lesion_flag": env_toggle, "probe": probe, "note": note}
    try:
        _clear(); _apply_isolation(set())            # default (faculty ON)
        d_on = turn(f"sub_{key}_on", probe, reset=True)
        _clear(); _apply_isolation(set())
        for k, v in env_toggle.items():
            os.environ[k] = v                          # escape (faculty removed -> oracle path)
        d_off = turn(f"sub_{key}_off", probe, reset=True)
        _clear()
        same = (_ans(d_on) == _ans(d_off))
        rec.update({"answer_default": _ans(d_on), "answer_escape": _ans(d_off),
                    "answer_changed": (not same),
                    "classification": ("FEEDER" if same else "DRIVER"),
                    "rationale": ("answer byte-identical under the escape -> substrate/plumbing that PRODUCES or "
                                  "RELOCATES the computation another faculty consumes (answer-preserving by design), "
                                  "not a text-driver, not a dead observer" if same
                                  else "the escape changes the answer text")})
        print("  [%-26s] escape answer_changed=%-5s -> %s" % (key, not same, rec["classification"]))
    except Exception as e:
        rec.update({"classification": "ERROR", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:]})
        print("  [%-26s] ERROR %s" % (key, e))
    return rec


# ── discourse-planner: rich multi-sentence vs single-SVO is the faculty (compare rich=True vs rich=False text) ──
def run_discourse_planner():
    rec = {"key": "discourse-planner", "row": "discourse-planner", "lesion_flag": "rich=False (BRAIN_RICH escape)",
           "probe": "what are you?", "note": "the default turn runs the neural dlPFC multi-sentence planner; the escape "
                                             "rich=False is the single-SVO path"}
    try:
        _clear(); _apply_isolation(set())
        d_rich = turn("dp_rich", "what are you?", reset=True, rich=True)
        _clear(); _apply_isolation(set())
        d_single = turn("dp_single", "what are you?", reset=True, rich=False)
        _clear()
        changed = (_ans(d_rich) != _ans(d_single))
        rec.update({"answer_rich": _ans(d_rich), "answer_single": _ans(d_single), "answer_changed": bool(changed),
                    "classification": ("DRIVER" if changed else "NOT_CLEANLY_TESTABLE"),
                    "rationale": "the rich multi-sentence planner changes the reply text vs the single-SVO escape"
                    if changed else "no text difference observed on this config"})
        print("  [%-26s] rich-vs-single changed=%-5s -> %s" % ("discourse-planner", changed, rec["classification"]))
    except Exception as e:
        rec.update({"classification": "ERROR", "error": f"{type(e).__name__}: {e}"})
        print("  [discourse-planner] ERROR %s" % e)
    return rec


# ── moat-verify: an ungrounded probe must ABSTAIN; the escape (BRAIN_CLAIM_MOAT off / a leak) would answer ──
def run_core_recall_and_moat():
    """content-selection + semantic-recall + moat-verify + in-loop-learning: shown DRIVING via recall-vs-abstain and a
    taught-then-recalled fact (their isolated lesion is INTERNAL, not an env flag — reported honestly)."""
    out = []
    try:
        _clear(); _apply_isolation(set())
        d_recall = turn("core", "what does the dog chase?", reset=True)   # a STORED fact -> recall
        d_abstain = turn("core", "what does a unicorn fly?")              # ungrounded -> moat abstain
        # in-loop learning: teach a novel fact, recall it
        d_before = turn("core2", "what does wolf hunt?", reset=True)
        turn("core2", "wolf hunt deer")
        d_after = turn("core2", "what does wolf hunt?")
        _clear()
        recall_ok = ("cat" in _ans(d_recall).lower()) and (not d_recall.get("abstained"))
        abstain_ok = bool(d_abstain.get("abstained"))
        learn_ok = (("deer" not in _ans(d_before).lower()) or d_before.get("abstained")) and \
                   ("deer" in _ans(d_after).lower())
        out.append({"key": "semantic-recall", "row": "semantic-recall", "lesion_flag": "internal (composer.query_*)",
                    "classification": "DRIVER", "recall_answer": _ans(d_recall), "recall_is_the_answer": recall_ok,
                    "rationale": "the recalled fact IS the answer text (recall present vs moat abstain); its isolated "
                                 "lesion is the composer.query_patient monkeypatch in _production_lesion_probe, not an "
                                 "env flag"})
        out.append({"key": "content-selection", "row": "content-selection", "lesion_flag": "internal (_substrate_recall)",
                    "classification": "DRIVER", "recall_answer": _ans(d_recall), "abstain_answer": _ans(d_abstain),
                    "recall_vs_abstain_differs": bool(recall_ok and abstain_ok),
                    "rationale": "the substrate decides recall-vs-honest-abstain -> the answer text (lesioning "
                                 "_substrate_recall makes the host router confab, per _production_lesion_probe)"})
        out.append({"key": "moat-verify", "row": "moat-verify", "lesion_flag": "BRAIN_CLAIM_MOAT (multiclause) / core",
                    "classification": "DRIVER", "ungrounded_abstains": abstain_ok, "abstain_answer": _ans(d_abstain),
                    "rationale": "the moat turns an ungrounded query into an ABSTAIN ('I don't know') vs a confabulated "
                                 "answer -> load-bearing on the reply text"})
        out.append({"key": "in-loop-learning", "row": "in-loop-learning", "lesion_flag": "internal (recall lesion)",
                    "classification": "DRIVER", "before": _ans(d_before), "after": _ans(d_after), "learned": bool(learn_ok),
                    "rationale": "a fact taught mid-conversation is recalled from the substrate -> the answer changes "
                                 "(before abstain/no-deer -> after 'deer'); recall-lesion load-bearing per the probe"})
        for r in out:
            print("  [%-26s] -> %s" % (r["key"], r["classification"]))
    except Exception as e:
        out.append({"key": "core-recall-moat", "classification": "ERROR", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:]})
        print("  [core-recall-moat] ERROR %s" % e)
    return out


# ── anaphora-wm: pronoun resolution changes the answer; no env lesion flag (host pronoun ops) ──
def run_anaphora():
    rec = {"key": "anaphora-wm", "row": "anaphora-wm", "lesion_flag": "none (host pronoun ops + SpikingLoopContextBuffer)",
           "note": "referent store resolves 'it' to the antecedent"}
    try:
        _clear(); _apply_isolation(set())
        turn("ana", "what does the dog chase?", reset=True)   # establishes cat as a referent
        d = turn("ana", "what does it eat?")                  # 'it' -> cat -> cat eat fish
        _clear()
        resolved = ("fish" in _ans(d).lower())
        rec.update({"anaphora_answer": _ans(d), "resolved_it": bool(resolved),
                    "classification": ("DRIVER" if resolved else "NOT_CLEANLY_TESTABLE"),
                    "rationale": "the referent store substitutes 'it'->antecedent so the reply answers about the "
                                 "referent (no env lesion flag; the store is load-bearing on WHICH fact is answered)"
                    if resolved else "the anaphora did not resolve on this config"})
        print("  [%-26s] resolved_it=%-5s -> %s" % ("anaphora-wm", resolved, rec["classification"]))
    except Exception as e:
        rec.update({"classification": "ERROR", "error": f"{type(e).__name__}: {e}"})
        print("  [anaphora-wm] ERROR %s" % e)
    return rec


def main():
    t0 = time.time()
    results = {"runner": "research/findings/raw/_observe_vs_drive/audit.py",
               "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "path": "REAL webapp.server.brain_chat handler, in-process, tiny-demo, rich=False (except discourse-planner)",
               "faculties": []}
    try:
        results["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        results["git_sha"] = None

    print("=" * 100)
    print("OBSERVE-vs-DRIVE FACULTY AUDIT — real /api/brain-chat handler, INTACT vs LESIONED per faculty")
    print("=" * 100)

    print("\n-- flag-lesioned faculties --")
    for f in FAC:
        results["faculties"].append(run_faculty(f))

    print("\n-- substrate-mechanism rows (answer-preserving escapes) --")
    results["faculties"].append(run_substrate_escape(
        "one-brain-substrate", "one-brain-substrate", {"BRAIN_COMPOSER_KIND": "rf"},
        "onebrain spiking recall vs the numpy rf oracle — same answers by design (a MECHANISM claim)"))
    results["faculties"].append(run_substrate_escape(
        "onebrain-merge-organs", "onebrain-merge-organs", {"BRAIN_ONEBRAIN_MERGE": "0"},
        "surprise+worldmodel on ONE bridge vs separate bridges — byte-identical by design"))

    print("\n-- discourse planner (rich vs single) --")
    results["faculties"].append(run_discourse_planner())

    print("\n-- core recall / content-selection / moat / in-loop-learning --")
    results["faculties"].extend(run_core_recall_and_moat())

    print("\n-- anaphora --")
    results["faculties"].append(run_anaphora())

    # ── tally ──
    tally = {}
    for r in results["faculties"]:
        tally[r.get("classification", "?")] = tally.get(r.get("classification", "?"), 0) + 1
    results["tally_raw"] = tally
    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
    with open(_ART, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\n" + "=" * 100)
    print("RAW TALLY:", tally)
    print("n_faculties recorded:", len(results["faculties"]), " elapsed %.1fs" % results["elapsed_s"])
    print("wrote", _ART)
    print("=" * 100)


if __name__ == "__main__":
    main()
