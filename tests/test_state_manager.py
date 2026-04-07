"""Tests for the UI state manager."""
import pytest
from ui.state_manager import UIState


def test_set_and_get():
    state = UIState()
    state.set("selected_neurons", {1, 2, 3})
    assert state.selected_neurons == {1, 2, 3}


def test_change_notification():
    state = UIState()
    changes = []
    state.on_change("selected_neurons", lambda f, old, new: changes.append((old, new)))
    state.set("selected_neurons", {5})
    assert len(changes) == 1
    assert changes[0] == (set(), {5})


def test_multiple_subscribers():
    state = UIState()
    a, b = [], []
    state.on_change("is_paused", lambda f, o, n: a.append(n))
    state.on_change("is_paused", lambda f, o, n: b.append(n))
    state.set("is_paused", True)
    assert a == [True]
    assert b == [True]
