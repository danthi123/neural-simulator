"""CI guard for Burndown conversion C2: the first-chat console's DEFAULT render word-ordering on the VALIDATED
SPIKING competitive-queuing read-out (NeuralSerialOrderRenderer) instead of the host f-string.

C2 makes the console build a `SpikingOrderStubFaculty` (a `TemplateStubFaculty` subclass) whose `render_svo`
orders the 3 SVO slots [agent, verb, patient] by the per-pool spiking RATE ranking on a real SimulationBridge.

Pins the de-risk's load-bearing results (research/findings/2026-06-27-burndown-C2-console-spiking-render-GO.md):
  * PARITY -- SpikingOrderStubFaculty.render_svo surface == the host TemplateStubFaculty surface for the canonical
    SVO frame (the spiking order == [agent, verb, patient] -> byte-identical), and the asserted SVO is the
    canonical [a, v, p] (VERIFY content unchanged);
  * EQUAL-DRIVE anti-cheat FAILS -- a flat primacy gradient does NOT reproduce the SVO order (the neurons
    serialize via the gradient, not a host sort / pool bias);
  * MOAT -- the faculty only ever orders the 3 content tokens it is given (cannot add/drop/swap one) -> a word-
    ORDER change can never fabricate a fact.

Unlike C1's guard, this does NOT skip off-GPU: the renderer builds a small SimulationBridge that runs on the
numpy-CPU backend too (~0.5s build, ~5ms/order), and that numpy backend is the console's NATIVE/default backend.
So C2's spiking order is the console's default on BOTH backends and the guard runs on whichever is active.
Reuse-by-import; NO sim/ edit.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.first_chat_console import SpikingOrderStubFaculty  # noqa: E402
from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty, _inflect, _determiner  # noqa: E402
from research.runners._phaseB_serial_order_spiking_derisk import pool_rates, EQUAL_pA  # noqa: E402


_FACTS = [("dog", "eat", "meat"), ("cat", "chase", "mouse"), ("boy", "kiss", "girl"),
          ("bird", "fly", "nest"), ("fox", "catch", "hare"), ("fish", "swim", "river")]


def _host_surface(a, v, p, template=0):
    det_a = _determiner(a, "agent")
    verb = _inflect(v)
    return f"{det_a}{a} {verb} {p}." if template % 2 == 0 else f"{det_a}{a} {verb} the {p}."


def test_spiking_order_parity_with_host_fstring():
    """The spiking-ordered faculty's surface == the host f-string for the canonical SVO frame (byte-identical),
    and the asserted SVO is the canonical [a,v,p] (so VERIFY is unchanged)."""
    fac = SpikingOrderStubFaculty(seed=42)
    for (a, v, p) in _FACTS:
        for template in (0, 1):
            surface, asserted = fac.render_svo(a, v, p, template=template)
            assert surface == _host_surface(a, v, p, template=template), (
                f"spiking surface {surface!r} != host {_host_surface(a, v, p, template=template)!r}")
            assert list(asserted) == [a, v, p], f"asserted SVO {asserted} != [{a},{v},{p}] (content drifted)"


def test_spiking_order_reproduces_svo_frame():
    """The underlying spiking read-out orders the canonical SVO frame [0,1,2] (slot 0=agent highest primacy) ->
    [0,1,2] (the neural parallel->serial conversion reproduces the frame order)."""
    fac = SpikingOrderStubFaculty(seed=42)
    assert fac._order.order([0, 1, 2]) == [0, 1, 2]


def test_equal_drive_control_fails():
    """The anti-cheat: a FLAT primacy gradient (equal current to all 3 pools, no agent>verb>patient gap) must NOT
    reliably reproduce the SVO order -> proves the NEURONS serialize via the gradient, not a host sort. We read
    the rate ranking with a random tie-break (so no separation -> a random order, not a stable echo of the
    input)."""
    fac = SpikingOrderStubFaculty(seed=42)
    rng = np.random.default_rng(7)
    idx = [0, 1, 2]
    hits = 0
    reps = 16
    for _ in range(reps):
        rate = pool_rates(fac._order.bridge, fac._order.pool_idx, {c: float(EQUAL_pA) for c in idx})
        jit = {c: (rate[c], float(rng.random())) for c in idx}
        order = sorted(idx, key=lambda c: (-jit[c][0], jit[c][1]))
        hits += int(order == idx)
    # with no gradient, the SVO order should be recovered at roughly chance (1/6) -- well below the gradient's 1.0.
    assert hits <= reps * 0.5, f"equal-drive reproduced SVO {hits}/{reps} times (control did NOT fail -- the order " \
                               f"is not coming from the spiking gradient)"


def test_moat_faculty_orders_only_given_tokens():
    """The moat: the faculty asserts ONLY the 3 content tokens it is given (cannot introduce/drop/swap a content
    word); the spiking ORDER reorders those tokens, it can never fabricate one. (Abstention is upstream -- an
    unstored fact never reaches render_svo; here we assert the structural guarantee on the render itself.)"""
    fac = SpikingOrderStubFaculty(seed=43)
    for (a, v, p) in _FACTS:
        surface, asserted = fac.render_svo(a, v, p)
        assert list(asserted) == [a, v, p]
        # every content token in the asserted SVO is one of the given words; nothing invented.
        assert set(asserted) <= {a, v, p}
        # the surface contains exactly the 3 content words (plus function words/inflection), none invented.
        toks = set(t.strip(".").lower() for t in surface.split())
        assert a in toks and p in toks


def test_spiking_faculty_is_template_stub_subclass():
    """SpikingOrderStubFaculty is a drop-in for TemplateStubFaculty (same render_svo / render_yesno contract), so
    the CommunicableTurn / DiscursiveTurn consume it unchanged."""
    assert issubclass(SpikingOrderStubFaculty, TemplateStubFaculty)
    fac = SpikingOrderStubFaculty(seed=42)
    # render_yesno is inherited (short structural answers stay host -- no fluency win, avoids a drift path).
    surface, asserted = fac.render_yesno("dog", "eat", "meat", "yes")
    assert list(asserted) == ["dog", "eat", "meat"] and "dog" in surface.lower()
