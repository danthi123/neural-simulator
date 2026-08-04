"""Streaming microphone-to-cochlea transduction and shared-bridge A1 wiring.

The architecture boundary is strict:

* ``CochlearTransducer`` is external sensory preprocessing. It converts a
  physical waveform into place-coded auditory-nerve spikes.
* ``build_tonotopic_a1_slice`` declares auditory-nerve input, excitatory A1,
  and inhibitory A1 populations as ordinary ``BrainRegion`` objects on the
  shared substrate.
* ``AuditoryNerveBridgeAdapter`` can drive only the auditory-nerve input
  regions. A1 receives normal synaptic spikes; it never receives waveform
  samples, labels, phonemes, transcripts, or host-computed audio features.

This is a deterministic construction/smoke package. It registers no random
seed, trains nothing, and writes no research evidence by default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

import numpy as np
from scipy import signal
from scipy.io import wavfile

from sim.enums import NeuronType
from sim.regions import BrainRegion, RegionPathway


AUDITORY_NERVE_PREFIX = "auditory_nerve_input_ch_"
A1_EXC_PREFIX = "a1_exc_ch_"
A1_PV_PREFIX = "a1_pv_ch_"


def erb_rate(frequency_hz: np.ndarray | float) -> np.ndarray:
    """Glasberg-Moore ERB-rate approximation for channel placement."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    return 21.4 * np.log10(1.0 + 0.00437 * frequency)


def inverse_erb_rate(rate: np.ndarray | float) -> np.ndarray:
    rate = np.asarray(rate, dtype=np.float64)
    return (np.power(10.0, rate / 21.4) - 1.0) / 0.00437


def erb_spaced_centers(low_hz: float, high_hz: float, channels: int) -> np.ndarray:
    if channels < 3:
        raise ValueError("channels must be at least 3")
    if not (0.0 < low_hz < high_hz):
        raise ValueError("expected 0 < low_hz < high_hz")
    rates = np.linspace(float(erb_rate(low_hz)), float(erb_rate(high_hz)), channels)
    return inverse_erb_rate(rates).astype(np.float32)


@dataclass(frozen=True)
class AuditoryFrontendConfig:
    sample_rate_hz: int = 16_000
    channels: int = 32
    low_hz: float = 80.0
    high_hz: float = 7_600.0
    ihc_lowpass_hz: float = 1_000.0
    ihc_compression_power: float = 1.0 / 3.0
    ihc_saturation: float = 0.12
    adaptation_ms: float = 50.0
    adaptation_strength: float = 2.0
    auditory_nerve_max_rate_hz: float = 300.0
    auditory_nerve_refractory_ms: float = 1.0

    def validate(self) -> None:
        if self.sample_rate_hz < 8_000:
            raise ValueError("sample_rate_hz must be at least 8000")
        if not (0.0 < self.high_hz < 0.49 * self.sample_rate_hz):
            raise ValueError("high_hz must stay below the Nyquist guard")
        if not (0.0 < self.low_hz < self.high_hz):
            raise ValueError("expected 0 < low_hz < high_hz")
        if self.channels < 3:
            raise ValueError("channels must be at least 3")
        if not (0.0 < self.ihc_lowpass_hz < 0.49 * self.sample_rate_hz):
            raise ValueError("ihc_lowpass_hz must stay below Nyquist")
        if not (0.0 < self.ihc_compression_power <= 1.0):
            raise ValueError("ihc_compression_power must be in (0, 1]")
        if self.ihc_saturation <= 0.0 or self.adaptation_ms <= 0.0:
            raise ValueError("hair-cell saturation and adaptation must be positive")
        if self.auditory_nerve_max_rate_hz <= 0.0:
            raise ValueError("auditory_nerve_max_rate_hz must be positive")


@dataclass(frozen=True)
class TonotopicA1SliceConfig:
    auditory_nerve_neurons_per_channel: int = 2
    excitatory_neurons_per_channel: int = 8
    inhibitory_neurons_per_channel: int = 2
    afferent_weight: float = 5.0
    local_exc_to_pv_weight: float = 3.0
    local_pv_to_exc_weight: float = 5.0
    flank_pv_to_exc_weight: float = 3.0
    flank_radius_channels: int = 2
    input_drive_pA: float = 900.0

    def validate(self) -> None:
        counts = (
            self.auditory_nerve_neurons_per_channel,
            self.excitatory_neurons_per_channel,
            self.inhibitory_neurons_per_channel,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("all per-channel neuron counts must be positive")
        if self.flank_radius_channels < 1:
            raise ValueError("flank_radius_channels must be positive")
        weights = (
            self.afferent_weight,
            self.local_exc_to_pv_weight,
            self.local_pv_to_exc_weight,
            self.flank_pv_to_exc_weight,
            self.input_drive_pA,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("A1 weights and input drive must be finite and positive")


@dataclass(frozen=True)
class CochlearFrame:
    center_frequencies_hz: np.ndarray
    hair_cell_drive: np.ndarray
    auditory_nerve_spikes: np.ndarray


@dataclass(frozen=True)
class TonotopicA1Slice:
    regions: tuple[BrainRegion, ...]
    pathways: tuple[RegionPathway, ...]
    center_frequencies_hz: np.ndarray
    auditory_nerve_regions: tuple[str, ...]
    excitatory_a1_regions: tuple[str, ...]
    inhibitory_a1_regions: tuple[str, ...]


@dataclass(frozen=True)
class BridgeDriveSummary:
    simulation_steps: int
    channel_event_counts: tuple[int, ...]


class CochlearTransducer:
    """Stateful peripheral transducer with no semantic information."""

    def __init__(self, config: AuditoryFrontendConfig):
        config.validate()
        self.config = config
        self.center_frequencies_hz = erb_spaced_centers(
            config.low_hz, config.high_hz, config.channels
        )
        self._filters: list[tuple[np.ndarray, np.ndarray]] = []
        self._filter_state: list[np.ndarray] = []
        for center in self.center_frequencies_hz:
            b, a = signal.gammatone(float(center), "iir", fs=config.sample_rate_hz)
            self._filters.append((b, a))
            self._filter_state.append(np.zeros(max(len(a), len(b)) - 1, dtype=np.float64))

        ihc_alpha = 1.0 - np.exp(-2.0 * np.pi * config.ihc_lowpass_hz / config.sample_rate_hz)
        adapt_alpha = 1.0 - np.exp(-1.0 / (config.adaptation_ms * config.sample_rate_hz / 1_000.0))
        self._ihc_b = np.asarray([ihc_alpha], dtype=np.float64)
        self._ihc_a = np.asarray([1.0, -(1.0 - ihc_alpha)], dtype=np.float64)
        self._adapt_b = np.asarray([adapt_alpha], dtype=np.float64)
        self._adapt_a = np.asarray([1.0, -(1.0 - adapt_alpha)], dtype=np.float64)
        self._ihc_state = np.zeros((config.channels, 1), dtype=np.float64)
        self._adapt_state = np.zeros((config.channels, 1), dtype=np.float64)
        self._phase = np.zeros(config.channels, dtype=np.float64)
        self._refractory = np.zeros(config.channels, dtype=np.int32)
        self._refractory_samples = max(
            1,
            int(round(config.auditory_nerve_refractory_ms * config.sample_rate_hz / 1_000.0)),
        )

    def process(self, waveform: np.ndarray) -> CochlearFrame:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("waveform must be mono with shape [samples]")
        if not np.all(np.isfinite(audio)):
            raise ValueError("waveform contains non-finite samples")
        if audio.size == 0:
            empty = np.zeros((self.config.channels, 0), dtype=np.float32)
            return CochlearFrame(self.center_frequencies_hz.copy(), empty, empty.astype(bool))
        if float(np.max(np.abs(audio))) > 1.0 + 1e-6:
            raise ValueError("waveform samples must be normalized to [-1, 1]")

        basilar = np.empty((self.config.channels, audio.size), dtype=np.float64)
        for index, ((b, a), zi) in enumerate(zip(self._filters, self._filter_state)):
            basilar[index], self._filter_state[index] = signal.lfilter(b, a, audio, zi=zi)

        rectified = np.maximum(basilar, 0.0)
        compressed = np.power(rectified, self.config.ihc_compression_power)
        ihc, self._ihc_state = signal.lfilter(
            self._ihc_b, self._ihc_a, compressed, axis=1, zi=self._ihc_state
        )
        adaptation, self._adapt_state = signal.lfilter(
            self._adapt_b, self._adapt_a, ihc, axis=1, zi=self._adapt_state
        )
        denominator = (
            ihc
            + self.config.ihc_saturation
            * (1.0 + self.config.adaptation_strength * adaptation)
        )
        drive = np.divide(ihc, denominator, out=np.zeros_like(ihc), where=denominator > 0.0)
        instantaneous_rate = self.config.auditory_nerve_max_rate_hz * drive

        spikes = np.zeros((self.config.channels, audio.size), dtype=bool)
        for sample in range(audio.size):
            self._refractory = np.maximum(self._refractory - 1, 0)
            self._phase += instantaneous_rate[:, sample] / self.config.sample_rate_hz
            firing = (self._refractory == 0) & (self._phase >= 1.0)
            spikes[:, sample] = firing
            self._phase[firing] -= 1.0
            self._phase = np.minimum(self._phase, 1.0)
            self._refractory[firing] = self._refractory_samples

        return CochlearFrame(
            self.center_frequencies_hz.copy(), drive.astype(np.float32), spikes
        )


def _region_name(prefix: str, channel: int) -> str:
    return f"{prefix}{channel:02d}"


def build_tonotopic_a1_slice(
    frontend_config: AuditoryFrontendConfig | None = None,
    slice_config: TonotopicA1SliceConfig | None = None,
) -> TonotopicA1Slice:
    """Return regions/pathways to merge into a shared bridge before init.

    Channel-specific regions make the place map structural rather than relying
    on random within-region coordinates. All A1 afferents originate in the
    auditory-nerve input regions created here.
    """
    front = frontend_config or AuditoryFrontendConfig()
    a1 = slice_config or TonotopicA1SliceConfig()
    front.validate()
    a1.validate()
    centers = erb_spaced_centers(front.low_hz, front.high_hz, front.channels)
    normalized_places = np.linspace(0.0, 1.0, front.channels)
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
    nerve_names = tuple(_region_name(AUDITORY_NERVE_PREFIX, i) for i in range(front.channels))
    exc_names = tuple(_region_name(A1_EXC_PREFIX, i) for i in range(front.channels))
    pv_names = tuple(_region_name(A1_PV_PREFIX, i) for i in range(front.channels))
    regions: list[BrainRegion] = []
    pathways: list[RegionPathway] = []

    for channel, place in enumerate(normalized_places):
        coordinate = (float(place),)
        regions.extend(
            [
                BrainRegion(
                    name=nerve_names[channel],
                    n_neurons=a1.auditory_nerve_neurons_per_channel,
                    exc_fraction=1.0,
                    internal_density=0.0,
                    weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=rs,
                    coordinate_dim=1,
                    coordinate_center=coordinate,
                ),
                BrainRegion(
                    name=exc_names[channel],
                    n_neurons=a1.excitatory_neurons_per_channel,
                    exc_fraction=1.0,
                    internal_density=0.15,
                    exc_weight_mean=0.3,
                    weight_jitter=0.1,
                    plastic_internal=True,
                    izh_neuron_type=rs,
                    coordinate_dim=1,
                    coordinate_center=coordinate,
                ),
                BrainRegion(
                    name=pv_names[channel],
                    n_neurons=a1.inhibitory_neurons_per_channel,
                    exc_fraction=0.0,
                    internal_density=0.10,
                    inh_weight_mean=0.8,
                    weight_jitter=0.1,
                    plastic_internal=False,
                    izh_neuron_type=fs,
                    coordinate_dim=1,
                    coordinate_center=coordinate,
                ),
            ]
        )
        pathways.extend(
            [
                RegionPathway(
                    nerve_names[channel], exc_names[channel], density=1.0,
                    weight_mean=a1.afferent_weight, weight_jitter=0.0, plastic=False,
                ),
                RegionPathway(
                    nerve_names[channel], pv_names[channel], density=1.0,
                    weight_mean=a1.afferent_weight, weight_jitter=0.0, plastic=False,
                ),
                RegionPathway(
                    exc_names[channel], pv_names[channel], density=0.5,
                    weight_mean=a1.local_exc_to_pv_weight, weight_jitter=0.1, plastic=True,
                ),
                RegionPathway(
                    pv_names[channel], exc_names[channel], density=1.0,
                    weight_mean=a1.local_pv_to_exc_weight, weight_jitter=0.0,
                    plastic=False, receptor="gaba_a",
                ),
            ]
        )
        for offset in range(1, a1.flank_radius_channels + 1):
            for target in (channel - offset, channel + offset):
                if 0 <= target < front.channels:
                    pathways.append(
                        RegionPathway(
                            pv_names[channel], exc_names[target], density=1.0,
                            weight_mean=a1.flank_pv_to_exc_weight,
                            weight_jitter=0.0, plastic=False, receptor="gaba_a",
                        )
                    )

    return TonotopicA1Slice(
        tuple(regions), tuple(pathways), centers, nerve_names, exc_names, pv_names
    )


class AuditoryNerveBridgeAdapter:
    """Drive only shared-bridge auditory-nerve regions from cochlear spikes."""

    def __init__(
        self,
        bridge,
        frontend_config: AuditoryFrontendConfig | None = None,
        slice_config: TonotopicA1SliceConfig | None = None,
    ):
        self.bridge = bridge
        self.frontend_config = frontend_config or AuditoryFrontendConfig()
        self.slice_config = slice_config or TonotopicA1SliceConfig()
        self.frontend_config.validate()
        self.slice_config.validate()
        if getattr(bridge, "region_manager", None) is None:
            raise ValueError("bridge must be initialized with the auditory A1 slice")
        if getattr(bridge, "runtime_state", None) is None:
            raise ValueError("bridge must expose shared simulation time state")
        self._nerve_indices = []
        for channel in range(self.frontend_config.channels):
            name = _region_name(AUDITORY_NERVE_PREFIX, channel)
            try:
                indices = np.asarray(list(bridge.region_manager.indices(name)), dtype=np.int64)
            except (KeyError, ValueError) as error:
                raise ValueError(f"bridge is missing auditory region {name}") from error
            if indices.size != self.slice_config.auditory_nerve_neurons_per_channel:
                raise ValueError(f"auditory region {name} has the wrong size")
            self._nerve_indices.append(indices)
        merged = np.concatenate(self._nerve_indices)
        if np.unique(merged).size != merged.size:
            raise ValueError("auditory-nerve bridge regions overlap")
        dt_ms = float(bridge.core_config.dt_ms)
        samples_per_step = self.frontend_config.sample_rate_hz * dt_ms / 1_000.0
        if not float(samples_per_step).is_integer() or samples_per_step < 1.0:
            raise ValueError("bridge dt_ms must map to a whole positive audio-sample count")
        self._samples_per_step = int(samples_per_step)
        self._pending = np.zeros((self.frontend_config.channels, 0), dtype=bool)

    def drive(self, frame: CochlearFrame) -> BridgeDriveSummary:
        if not isinstance(frame, CochlearFrame):
            raise TypeError("adapter accepts CochlearFrame auditory-nerve spikes only")
        expected_centers = erb_spaced_centers(
            self.frontend_config.low_hz,
            self.frontend_config.high_hz,
            self.frontend_config.channels,
        )
        if not np.array_equal(frame.center_frequencies_hz, expected_centers):
            raise ValueError("cochlear channel centers do not match the bridge A1 slice")
        spikes = np.asarray(frame.auditory_nerve_spikes, dtype=bool)
        if spikes.ndim != 2 or spikes.shape[0] != self.frontend_config.channels:
            raise ValueError("auditory_nerve_spikes must have shape [channels, samples]")
        combined = np.concatenate((self._pending, spikes), axis=1)
        steps = combined.shape[1] // self._samples_per_step
        used = steps * self._samples_per_step
        self._pending = combined[:, used:].copy()
        binned = combined[:, :used].reshape(
            self.frontend_config.channels, steps, self._samples_per_step
        ).sum(axis=2, dtype=np.int64)
        current = self.bridge.cp_external_input_current
        saved = [current[indices].copy() for indices in self._nerve_indices]
        try:
            for step in range(steps):
                for channel, indices in enumerate(self._nerve_indices):
                    current[indices] = (
                        self.slice_config.input_drive_pA if binned[channel, step] > 0 else 0.0
                    )
                self.bridge._run_one_simulation_step()
                self.bridge.runtime_state.current_time_ms += self.bridge.core_config.dt_ms
                self.bridge.runtime_state.current_time_step += 1
        finally:
            for indices, values in zip(self._nerve_indices, saved):
                current[indices] = values
        return BridgeDriveSummary(
            simulation_steps=steps,
            channel_event_counts=tuple(int(value) for value in binned.sum(axis=1)),
        )


def synthesize_tone(
    frequency_hz: float,
    duration_s: float,
    sample_rate_hz: int,
    amplitude: float = 0.20,
) -> np.ndarray:
    samples = int(round(duration_s * sample_rate_hz))
    t = np.arange(samples, dtype=np.float64) / sample_rate_hz
    ramp_samples = min(samples // 2, max(1, int(round(0.005 * sample_rate_hz))))
    window = np.ones(samples, dtype=np.float64)
    if ramp_samples:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, ramp_samples)) ** 2
        window[:ramp_samples] = ramp
        window[-ramp_samples:] = ramp[::-1]
    return (amplitude * window * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def load_wav_mono(path: Path, expected_sample_rate_hz: int) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    if int(sample_rate) != int(expected_sample_rate_hz):
        raise ValueError(f"WAV sample rate is {sample_rate}; expected {expected_sample_rate_hz} Hz")
    source_dtype = audio.dtype
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    if np.issubdtype(source_dtype, np.integer):
        info = np.iinfo(source_dtype)
        audio = audio.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        audio = audio.astype(np.float64)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def capture_microphone_alsa(
    duration_s: float,
    sample_rate_hz: int,
    *,
    device: str = "default",
    command: Sequence[str] | None = None,
) -> np.ndarray:
    """Capture mono PCM through ALSA/arecord without a Python audio dependency."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    argv = list(command) if command is not None else [
        "arecord", "-q", "-D", device, "-t", "raw", "-f", "S16_LE",
        "-r", str(sample_rate_hz), "-c", "1", "-d", str(int(np.ceil(duration_s))),
    ]
    completed = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pcm = np.frombuffer(completed.stdout, dtype="<i2")
    expected = int(round(duration_s * sample_rate_hz))
    if pcm.size < expected:
        raise RuntimeError(f"microphone returned {pcm.size} samples; expected at least {expected}")
    return (pcm[:expected].astype(np.float32) / 32768.0).astype(np.float32)


def construction_smoke(config: AuditoryFrontendConfig | None = None) -> dict:
    cfg = config or AuditoryFrontendConfig()
    centers = erb_spaced_centers(cfg.low_hz, cfg.high_hz, cfg.channels)
    probe_indices = [4, cfg.channels // 2, cfg.channels - 5]
    rows = []
    started = time.perf_counter()
    for target in probe_indices:
        frame = CochlearTransducer(cfg).process(
            synthesize_tone(float(centers[target]), 0.25, cfg.sample_rate_hz)
        )
        counts = frame.auditory_nerve_spikes.sum(axis=1)
        peak = int(np.argmax(counts))
        rows.append(
            {
                "target_channel": target,
                "target_hz": float(centers[target]),
                "peak_channel": peak,
                "peak_hz": float(centers[peak]),
                "auditory_nerve_spikes": int(counts.sum()),
            }
        )
    silence = CochlearTransducer(cfg).process(
        np.zeros(int(0.25 * cfg.sample_rate_hz), dtype=np.float32)
    )
    weak = CochlearTransducer(cfg).process(
        synthesize_tone(1_000.0, 0.10, cfg.sample_rate_hz, amplitude=0.02)
    )
    strong = CochlearTransducer(cfg).process(
        synthesize_tone(1_000.0, 0.10, cfg.sample_rate_hz, amplitude=0.20)
    )
    a1_slice = build_tonotopic_a1_slice(cfg)
    elapsed_s = time.perf_counter() - started
    processed_audio_s = 1.20
    peaks = [row["peak_channel"] for row in rows]
    weak_mean = float(weak.hair_cell_drive.mean())
    strong_mean = float(strong.hair_cell_drive.mean())
    checks = {
        "tone_evokes_auditory_nerve": all(row["auditory_nerve_spikes"] > 0 for row in rows),
        "ordered_cochlear_place": peaks == sorted(peaks),
        "place_error_within_two_channels": max(
            abs(row["target_channel"] - row["peak_channel"]) for row in rows
        ) <= 2,
        "silence_is_quiet": int(silence.auditory_nerve_spikes.sum()) == 0,
        "compression_is_monotonic_and_sublinear": (
            strong_mean > weak_mean > 0.0 and strong_mean / weak_mean < 10.0
        ),
        "shared_a1_slice_is_declarative": len(a1_slice.regions) == 3 * cfg.channels,
        "cpu_real_time_budget": elapsed_s <= processed_audio_s,
    }
    return {
        "kind": "cochlear_construction_smoke_no_seed",
        "claims_excluded": ["A1 functional calibration", "speech recognition", "speech understanding"],
        "config": asdict(cfg),
        "tones": rows,
        "silence_auditory_nerve_spikes": int(silence.auditory_nerve_spikes.sum()),
        "performance": {
            "processed_audio_s": processed_audio_s,
            "wall_s": elapsed_s,
            "real_time_factor": elapsed_s / processed_audio_s,
        },
        "shared_slice": {
            "regions": len(a1_slice.regions),
            "pathways": len(a1_slice.pathways),
            "brain_internal_host_dynamics": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=Path)
    source.add_argument("--microphone-seconds", type=float)
    parser.add_argument("--device", default="default", help="ALSA capture device")
    parser.add_argument("--output", type=Path, help="optional JSON peripheral summary path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = AuditoryFrontendConfig()
    if args.wav is None and args.microphone_seconds is None:
        result = construction_smoke(config)
    else:
        if args.wav is not None:
            audio = load_wav_mono(args.wav, config.sample_rate_hz)
            source = {"kind": "wav", "path": str(args.wav)}
        else:
            audio = capture_microphone_alsa(
                args.microphone_seconds, config.sample_rate_hz, device=args.device
            )
            source = {"kind": "microphone", "device": args.device}
        started = time.perf_counter()
        frame = CochlearTransducer(config).process(audio)
        elapsed = time.perf_counter() - started
        counts = frame.auditory_nerve_spikes.sum(axis=1)
        peak = int(np.argmax(counts)) if np.any(counts) else None
        result = {
            "kind": "peripheral_cochlear_observation_no_seed",
            "source": source,
            "duration_s": audio.size / config.sample_rate_hz,
            "auditory_nerve_spikes": int(counts.sum()),
            "peak_channel": peak,
            "peak_center_hz": None if peak is None else float(frame.center_frequencies_hz[peak]),
            "wall_s": elapsed,
            "real_time_factor": elapsed / max(audio.size / config.sample_rate_hz, 1e-12),
            "claims_excluded": ["A1 activity", "speech recognition", "speech understanding"],
        }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
