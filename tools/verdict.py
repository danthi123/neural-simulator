"""A verdict you have to EARN — preconditions carried by the result, not checked beside it.

WHY THIS EXISTS (2026-07-31). Every gate in `tools/gates/` scans FILES at COMMIT time. Every miss on
2026-07-31 was a RELATIONSHIP at RUN time, and no file-scanner can see one:

  * gap#4 credit-on-expanded — the task stopped being depth-REQUIRED when the forward changed, so a deep
    net succeeding on it said nothing about deep credit. Caught only because that runner happened to
    re-measure the precondition.
  * affect eviction — the runner COMPUTED `arm_valid=False` on 3/3 seeds and printed "NO-GO" anyway. The
    validity field sat one key away from the verdict and nothing connected them.
  * the gap#4 crux — the idealised transport ceiling read 0.148 against a chance of 0.200. If the CEILING
    cannot beat chance, no arm beneath it is interpretable; nothing compared them.
  * sAHP — the power control zeroed a SYNAPTIC transmission gate while the mechanism lived in per-neuron
    Izhikevich parameters. The control could not reach the thing it claimed to control, and "passed".
  * `--sweep-weights` — accepted by argparse, consumed only under `--smoke`, silently inert; the run used a
    default 10-30x larger and produced a void negative.

Each was a plausible NEGATIVE that would have entered the record clean.

THE INVERSION. Today a runner computes validity and then OPTIONALLY mentions it. Here, UNDEFINED is the
DEFAULT and a run must earn anything else:

    v = Verdict("credit on the expanded forward")
    v.floor("held-out vs chance", measured=acc, floor=1.0 / k)
    v.require("task still depth-separating", depth_separating, expect=True)
    v.control("apical lesion", treatment=arm_acc, control=lesion_acc)
    v.reaches("sfa lesion", before=izh_d_before, after=izh_d_after)
    v.knob("stp_tau_d", requested=2000.0, applied=cfg.stp_tau_d)
    v.disabled("short-term plasticity", why="isolation: attractor is the only live dynamic")
    result = v.decide(go=(treatment > control))     # -> GO | NO-GO | UNDEFINED

`decide()` returns **UNDEFINED** if any requirement failed, if any was registered but never measured, or
if NONE was registered at all — an unguarded verdict is itself the defect this module exists to stop.

DELIBERATELY GENERIC. The vocabulary is the small set of relationships every experiment here actually has,
not the shape of any one arc: a floor to beat, a precondition to hold, a control to differ, a manipulation
to land, a knob to take effect, a process switched off. `selftest()` replays all five misses above through
this one API; if a sixth shape appears, add a method rather than a special case.

AND IT IS GATEABLE, which is the point. `to_dict()` emits a `preconditions` block into the artifact, so
`tools/gates/verdict_preconditions.py` can enforce that any artifact carrying a verdict also carries what
earned it. The runtime does the seeing; the artifact carries the evidence; the gate enforces its presence.
"""
from __future__ import annotations

GO, NO_GO, UNDEFINED = "GO", "NO-GO", "UNDEFINED"
_UNMEASURED = object()


class Check:
    """One precondition. `ok` is None until measured — an unmeasured check blocks a verdict."""

    def __init__(self, kind, name, detail, ok, note=""):
        self.kind, self.name, self.detail, self.ok, self.note = kind, name, detail, ok, note

    def to_dict(self):
        return {"kind": self.kind, "name": self.name, "ok": self.ok,
                "detail": self.detail, "note": self.note}

    def __repr__(self):
        mark = "?" if self.ok is None else ("ok" if self.ok else "FAIL")
        return "  [%-4s] %-10s %-38s %s" % (mark, self.kind, self.name[:38], self.detail)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Verdict:
    """Accumulate preconditions, then earn a verdict. UNDEFINED unless every one is measured and holds."""

    def __init__(self, label, chance=None):
        self.label = label
        self.chance = chance
        self.checks = []
        self.disabled_processes = []
        self._decided = None

    # ---------------------------------------------------------------- the six relationships
    def require(self, name, measured, expect=True, note=""):
        """A boolean precondition. `expect` is a value or a predicate.

        EARNED BY gap#4: the task must still be DEPTH-REQUIRED on the forward being tested. When the
        expansion made it shallow, a deep net's success stopped being evidence about deep credit."""
        if measured is _UNMEASURED or measured is None:
            self.checks.append(Check("require", name, "NEVER MEASURED", None, note))
            return self
        ok = bool(expect(measured)) if callable(expect) else (measured == expect)
        self.checks.append(Check("require", name, "measured=%s expect=%s" % (measured, expect), ok, note))
        return self

    def floor(self, name, measured, floor, note=""):
        """The result must EXCEED a floor — chance, a majority-class rate, an untrained baseline.

        EARNED BY the gap#4 crux: the idealised transport ceiling read 0.148 against chance 0.200. A
        result at or below its own floor is UNDEFINED, never a negative; the arms beneath an
        uninterpretable ceiling carry no information at all."""
        m, f = _num(measured), _num(floor)
        if m is None or f is None:
            self.checks.append(Check("floor", name, "NEVER MEASURED", None, note))
            return self
        self.checks.append(Check("floor", name, "%.6g vs floor %.6g" % (m, f), m > f, note))
        return self

    def control(self, name, treatment, control, min_separation=0.0, note=""):
        """A control must DIFFER from its treatment by more than `min_separation`.

        EARNED BY the identical-arm class (10 recorded incidents; `_emerge6` had three arms agreeing to
        sixteen digits). Arms that tie mean the manipulation never happened — the comparison is void, not
        negative."""
        t, c = _num(treatment), _num(control)
        if t is None or c is None:
            self.checks.append(Check("control", name, "NEVER MEASURED", None, note))
            return self
        sep = abs(t - c)
        self.checks.append(Check("control", name, "treatment=%.6g control=%.6g |sep|=%.6g > %.6g"
                                 % (t, c, sep, min_separation), sep > min_separation, note))
        return self

    def reaches(self, name, before, after, note=""):
        """A manipulation must actually CHANGE the mechanism variable it targets.

        EARNED BY the sAHP power control: G6 zeroed a SYNAPTIC transmission gate while the mechanism lived
        in per-neuron `cp_izh_a` / `cp_izh_d_increment`. It could not reach what it claimed to control, and
        reported a pass. A control that does not move its own mechanism's read-out is not a control."""
        b, a = _num(before), _num(after)
        if b is None or a is None:
            self.checks.append(Check("reaches", name, "NEVER MEASURED", None, note))
            return self
        self.checks.append(Check("reaches", name, "before=%.6g after=%.6g moved=%s"
                                 % (b, a, b != a), b != a, note))
        return self

    def knob(self, name, requested, applied, tol=1e-9, note=""):
        """A knob must take effect WHERE IT ACTS — read it back, do not trust that it was passed.

        EARNED BY `--sweep-weights`: accepted by argparse, consumed only under `--smoke`, so the run used a
        default 10-30x larger and produced a void negative. argparse proves a flag EXISTS; that is a
        different claim from 'this flag reached the code path this invocation took'."""
        r, a = _num(requested), _num(applied)
        if r is None or a is None:
            ok = (requested == applied) if (requested is not None and applied is not None) else None
            self.checks.append(Check("knob", name, "requested=%s applied=%s" % (requested, applied), ok, note))
            return self
        self.checks.append(Check("knob", name, "requested=%.6g applied=%.6g" % (r, a),
                                 abs(r - a) <= tol, note))
        return self

    def disabled(self, process, why=""):
        """Declare a biological process this run SWITCHED OFF. Not a pass/fail — a SCOPE that travels.

        EARNED BY the affect ratchet: both runners set `enable_short_term_plasticity = False` inside a
        block that also disabled STDP, reward modulation, Hebbian learning, homeostasis and structural
        plasticity. The recurrent loop therefore had nothing that weakens its own drive, it latched, and
        the latch was read as a property of the MECHANISM rather than of the ISOLATION. Anything measured
        here is a property of the mechanism UNDER THIS ISOLATION, and the verdict must say so."""
        self.disabled_processes.append({"process": process, "why": why})
        return self

    # ---------------------------------------------------------------- earning the verdict
    @property
    def unmet(self):
        return [c for c in self.checks if c.ok is False]

    @property
    def unmeasured(self):
        return [c for c in self.checks if c.ok is None]

    def decide(self, go, verbose=True):
        """Return GO / NO-GO / UNDEFINED. UNDEFINED wins whenever the run has not earned a verdict."""
        reasons = []
        if not self.checks:
            reasons.append("NO preconditions were registered — an unguarded verdict is itself the defect")
        reasons += ["unmet: %s (%s)" % (c.name, c.detail) for c in self.unmet]
        reasons += ["never measured: %s" % c.name for c in self.unmeasured]
        status = UNDEFINED if reasons else (GO if go else NO_GO)
        self._decided = {"label": self.label, "status": status, "go": bool(go) and status == GO,
                         "undefined_reasons": reasons,
                         "preconditions": [c.to_dict() for c in self.checks],
                         "disabled_processes": self.disabled_processes,
                         "chance": self.chance}
        if verbose:
            print("  VERDICT %s" % self.label)
            for c in self.checks:
                print(repr(c))
            for d in self.disabled_processes:
                print("  [scope] DISABLED %-30s %s" % (d["process"], d["why"]))
            print("  => %s" % status)
            for r in reasons:
                print("     ⛔ %s" % r)
            if status == UNDEFINED:
                print("     UNDEFINED is NOT a negative. Reporting one here would fabricate a result from "
                      "an instrument failure.")
        return self._decided

    def to_dict(self):
        if self._decided is None:
            raise RuntimeError("call decide() before to_dict() — a verdict must be earned before it is written")
        return self._decided


def selftest():
    """Replay the five distinct misses of 2026-07-31 through this ONE api. If a single vocabulary catches
    all five shapes, it is generic; if it needed five special cases, it would not be."""
    bad = []

    # 1. gap#4 — task went shallow, so the deep result is not about deep credit.
    v = Verdict("gap4 credit on expanded forward")
    v.require("task still depth-separating", False, expect=True)
    if v.decide(go=True, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT catch a failed precondition (shallow task)")

    # 2. affect — arm crushed; the runner had this value and printed NO-GO anyway.
    v = Verdict("affect eviction")
    v.require("arm not crushed (A5)", False, expect=True)
    if v.decide(go=False, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT convert a crushed-arm NO-GO into UNDEFINED")

    # 3. crux — idealised ceiling BELOW chance.
    v = Verdict("gap4 crux", chance=0.2)
    v.floor("transport ceiling vs chance", 0.148, 0.2)
    if v.decide(go=False, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT catch a ceiling below chance")

    # 4. sAHP — the control could not reach its mechanism (izh params unchanged by a synaptic gate).
    v = Verdict("sAHP eviction")
    v.reaches("sfa lesion moves cp_izh_d", before=100.0, after=100.0)
    if v.decide(go=False, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT catch a control that does not reach its mechanism")

    # 5. inert knob — requested 0.05, the run actually applied the 1.5 default.
    v = Verdict("affect sweep")
    v.knob("gabab_weight", requested=0.05, applied=1.5)
    if v.decide(go=False, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT catch a knob that did not take effect")

    # 6. THE UNGUARDED VERDICT — no preconditions at all must never yield GO.
    v = Verdict("unguarded")
    if v.decide(go=True, verbose=False)["status"] != UNDEFINED:
        bad.append("did NOT refuse an unguarded verdict (no preconditions registered)")

    # 7. NEGATIVE CONTROLS — a fully-earned GO and a fully-earned NO-GO must both come through, else the
    #    class is unusable and gets bypassed, which is worse than not having it.
    v = Verdict("earned go", chance=0.2)
    v.require("depth-separating", True, expect=True)
    v.floor("ceiling vs chance", 0.9, 0.2)
    v.control("lesion", treatment=0.87, control=0.11)
    v.reaches("lesion moves the mechanism", before=1.0, after=0.0)
    v.knob("lr", requested=0.005, applied=0.005)
    got = v.decide(go=True, verbose=False)
    if got["status"] != GO:
        bad.append("FALSE POSITIVE: refused a fully-earned GO (%s)" % got["undefined_reasons"])
    if v.decide(go=False, verbose=False)["status"] != NO_GO:
        bad.append("FALSE POSITIVE: refused a fully-earned NO-GO")

    # 8. `disabled` is SCOPE, not a gate — it must never block a verdict, only travel with it.
    v = Verdict("isolation scope")
    v.require("depth-separating", True, expect=True)
    v.disabled("short-term plasticity", why="isolation")
    got = v.decide(go=True, verbose=False)
    if got["status"] != GO or not got["disabled_processes"]:
        bad.append("`disabled` must record scope without blocking the verdict")
    return bad


if __name__ == "__main__":
    problems = selftest()
    print("\n".join(problems) if problems else
          "verdict selftest PASS — all five 2026-07-31 misses caught by one vocabulary, "
          "earned GO/NO-GO still come through, and `disabled` records scope without blocking.")
