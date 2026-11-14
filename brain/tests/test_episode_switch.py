"""Brain(episode=...) (switch, default OFF = "tell"): "lived" stores the user's telling with the brain's own reply."""
import pytest

from brain.brain import Brain
from brain.tests.test_brain_key_switch import KW, StubCortex, _talk


def _told(b):
    return [e.text for e in b.hippocampus.episodes if not getattr(e, "permanent", False)]


def test_default_stores_the_telling_only(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW)
    _talk(b)
    told = [t for t in _told(b) if "Pixel" in t or "Volvo" in t]
    assert told and all("Assistant:" not in t for t in told)


def test_lived_stores_the_exchange(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="lived")
    out = _talk(b)
    told = [t for t in _told(b) if t.startswith("User: My cat") or t.startswith("User: I drive")]
    assert told and all("\n\nAssistant: " in t for t in told)
    assert out[0][0].strip() in told[0]


def test_bad_value(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="both")
