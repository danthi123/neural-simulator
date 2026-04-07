"""Pub/sub data bus with ring buffers for streaming simulation data to UI."""

import threading
import numpy as np
from collections import deque


class DataChannel:
    """A named data stream with ring buffer history."""
    def __init__(self, name, max_history=1000, throttle_steps=1):
        self.name = name
        self.max_history = max_history
        self.throttle_steps = throttle_steps  # Only accept every Nth publish
        self._buffer = deque(maxlen=max_history)
        self._step_counter = 0
        self._subscribers = []

    def publish(self, data):
        """Publish data to this channel. Respects throttle."""
        self._step_counter += 1
        if self._step_counter % self.throttle_steps != 0:
            return
        self._buffer.append(data)
        for callback in self._subscribers:
            try:
                callback(data)
            except Exception:
                pass

    def subscribe(self, callback):
        """Register a callback for new data."""
        self._subscribers.append(callback)

    def get_history(self, n=None):
        """Get last n items from buffer."""
        if n is None:
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def latest(self):
        """Get most recent item, or None."""
        return self._buffer[-1] if self._buffer else None


class DataBus:
    """Central pub/sub hub for simulation data streams."""

    def __init__(self):
        self._channels = {}

    def create_channel(self, name, max_history=1000, throttle_steps=1):
        """Create a named data channel."""
        ch = DataChannel(name, max_history, throttle_steps)
        self._channels[name] = ch
        return ch

    def publish(self, channel_name, data):
        """Publish data to a channel."""
        ch = self._channels.get(channel_name)
        if ch:
            ch.publish(data)

    def subscribe(self, channel_name, callback):
        """Subscribe to a channel."""
        ch = self._channels.get(channel_name)
        if ch:
            ch.subscribe(callback)

    def get_channel(self, channel_name):
        """Get a channel by name."""
        return self._channels.get(channel_name)

    def get_history(self, channel_name, n=None):
        """Get history from a channel."""
        ch = self._channels.get(channel_name)
        return ch.get_history(n) if ch else []

    def latest(self, channel_name):
        """Get latest data from a channel."""
        ch = self._channels.get(channel_name)
        return ch.latest() if ch else None


def create_default_bus():
    """Create a DataBus with standard simulation channels."""
    bus = DataBus()
    bus.create_channel("firing_rates", max_history=10000, throttle_steps=1)
    bus.create_channel("spike_events", max_history=5000, throttle_steps=1)
    bus.create_channel("weights", max_history=100, throttle_steps=1)
    bus.create_channel("experiment_status", max_history=100, throttle_steps=1)
    bus.create_channel("band_power", max_history=100, throttle_steps=2000)
    bus.create_channel("synchrony", max_history=10000, throttle_steps=1)
    bus.create_channel("neuron_state", max_history=1000, throttle_steps=10)
    bus.create_channel("sim_data_update", max_history=10, throttle_steps=1)
    return bus
