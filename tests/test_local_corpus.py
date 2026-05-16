"""Tests for the zero-download local corpus loader (Increment 1)."""
from __future__ import annotations
import socket
import pytest

from research.runners.local_corpus import load_local_corpus


def test_returns_substantial_text():
    txt = load_local_corpus()
    assert isinstance(txt, str)
    assert len(txt) > 200_000  # substantial real English (no network)


def test_deterministic():
    assert load_local_corpus() == load_local_corpus()


def test_no_network_used(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network egress attempted by corpus loader")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    load_local_corpus()  # must complete with zero network


def test_is_real_english_not_shuffled():
    # sanity: common English words present (real sequential structure)
    txt = load_local_corpus().lower()
    for w in (" the ", " and ", " is "):
        assert w in txt
