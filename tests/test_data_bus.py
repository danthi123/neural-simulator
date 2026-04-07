"""Tests for the data bus pub/sub system."""
import pytest
from sim.data_bus import DataBus, DataChannel, create_default_bus


def test_publish_subscribe():
    bus = DataBus()
    bus.create_channel("test", max_history=10)
    received = []
    bus.subscribe("test", lambda data: received.append(data))
    bus.publish("test", {"value": 42})
    assert len(received) == 1
    assert received[0]["value"] == 42


def test_ring_buffer_limit():
    ch = DataChannel("test", max_history=3)
    for i in range(5):
        ch.publish(i)
    assert ch.get_history() == [2, 3, 4]


def test_throttle():
    ch = DataChannel("test", max_history=100, throttle_steps=3)
    for i in range(9):
        ch.publish(i)
    # Only every 3rd publish goes through: indices 2, 5, 8
    assert len(ch.get_history()) == 3


def test_latest():
    bus = DataBus()
    bus.create_channel("test")
    assert bus.latest("test") is None
    bus.publish("test", "first")
    bus.publish("test", "second")
    assert bus.latest("test") == "second"


def test_default_bus_channels():
    bus = create_default_bus()
    assert bus.get_channel("firing_rates") is not None
    assert bus.get_channel("spike_events") is not None
    assert bus.get_channel("weights") is not None
    assert bus.get_channel("nonexistent") is None


def test_get_history_n():
    bus = DataBus()
    bus.create_channel("test", max_history=100)
    for i in range(10):
        bus.publish("test", i)
    assert bus.get_history("test", 3) == [7, 8, 9]
