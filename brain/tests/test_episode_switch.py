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


def test_lived_key_indexes_by_the_telling(tmp_path):
    from brain.hippocampus import Hippocampus
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="lived_key")
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW, episode="tell")
    _talk(a), _talk(b)
    ea = [e for e in a.hippocampus.episodes if e.text.startswith("User: My cat")][0]
    eb = [e for e in b.hippocampus.episodes if e.text.startswith("User: My cat")][0]
    assert "\n\nAssistant: " in ea.text and "Assistant:" not in eb.text
    import torch
    assert torch.equal(ea.key, eb.key)                  # same index as the telling alone
    assert not all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)   # richer content


def test_speaker_contrast_changes_content_not_key(tmp_path):
    import torch
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, speaker_contrast=3.0)
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW)
    _talk(a), _talk(b)
    ea = [e for e in a.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    eb = [e for e in b.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert torch.equal(ea.key, eb.key)
    assert not all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)
    assert b.speaker_contrast == 0.0


def test_speaker_contrast_with_sall_keeps_routing_and_persists(tmp_path):
    import torch
    from brain.tests.test_key_bar_switch import Stub15
    kw = dict(KW, key="S_all", key_bar="size")
    a = Brain(str(tmp_path / "a"), cortex=Stub15(), **kw, speaker_contrast=3.0)
    b = Brain(str(tmp_path / "b"), cortex=Stub15(), **kw)
    _talk(a), _talk(b)
    q = "What is my cat called?"
    sa = a.hippocampus.similarities(a.cortex, a.base, q)
    sb = b.hippocampus.similarities(b.cortex, b.base, q)
    assert torch.allclose(sa, sb, atol=1e-5)          # routing unchanged
    ea = [e for e in a.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert getattr(ea, "route_delta", None) is not None
    a.save()
    a.release()
    c = Brain(str(tmp_path / "a"), cortex=Stub15(), **kw, speaker_contrast=3.0)
    ec = [e for e in c.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ec.route_delta, ea.route_delta))
    assert torch.allclose(c.hippocampus.similarities(c.cortex, c.base, q), sb, atol=1e-5)
