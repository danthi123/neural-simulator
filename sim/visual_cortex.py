"""Cluster K v1 — Visual cortex utilities (Hubel-Wiesel 1962, Felleman & Van Essen 1991).

This module provides Gabor receptive-field initialization for V1 simple cells,
phase-pooling for V1 complex cells, and image rendering for the gridworld
agent. Used by the visual-cortex region wiring in g11_bg_runner.py when
--enable-visual-cortex is set.

Design: docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md

Architecture (v1 minimal):
    image (32x32 pixels) -> retina (32*32*2 = 2048 ON/OFF) ->
        cortex_v1_simple (8 orientations × 4 freqs × 16x16 positions = 8192) ->
        cortex_v1_complex (8 × 16x16 = 2048, phase-pooled) ->
        cortex_v2 (512, plastic) ->
        cortex_it (128, plastic) ->
        cortex_X motor planning

For v1 we skip V4 (no color in gridworld). V1 simple weights are fixed
Gabor at init; V2/IT learn via STDP.

Biology source: Kandel 6e Ch 22; Hubel & Wiesel 1962; Felleman & Van Essen
1991; Tanaka 1996 IT.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Standard V1 architecture parameters
N_ORIENTATIONS = 8           # 0°, 22.5°, ..., 157.5°
N_FREQUENCIES = 4            # cycles/pixel: 0.05, 0.1, 0.2, 0.4
V1_POSITIONS_PER_DIM = 16    # 16x16 spatial grid of V1 cells
RETINA_SIZE = 32             # 32x32 pixel image
N_RETINA_CHANNELS = 2        # ON, OFF


def gabor_kernel(
    sigma_x: float,
    sigma_y: float,
    theta: float,
    freq: float,
    phase: float = 0.0,
) -> callable:
    """Return a 2D Gabor receptive field as a callable kernel.

    Args:
        sigma_x, sigma_y: Gaussian envelope std in pixels.
        theta: orientation in radians (0 = vertical bars, π/2 = horizontal).
        freq: spatial frequency in cycles/pixel.
        phase: phase offset (default 0 = symmetric/cosine).

    Returns:
        kernel(dx, dy) -> float in [-1, 1], the response strength
        of a V1 simple cell to a stimulus at relative position (dx, dy)
        from the cell's preferred center.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def kernel(dx: float, dy: float) -> float:
        # Rotate to filter coordinates
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        # Gaussian envelope
        env = math.exp(-(x_rot * x_rot / (sigma_x * sigma_x) +
                          y_rot * y_rot / (sigma_y * sigma_y)) / 2)
        # Cosine carrier (along x_rot direction)
        carrier = math.cos(2 * math.pi * freq * x_rot + phase)
        return env * carrier

    return kernel


def build_v1_simple_weights(
    n_orientations: int = N_ORIENTATIONS,
    n_frequencies: int = N_FREQUENCIES,
    n_positions_per_dim: int = V1_POSITIONS_PER_DIM,
    retina_size: int = RETINA_SIZE,
    receptive_field_radius: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse weights from retina to V1 simple cells.

    Each V1 simple cell at (orient_idx, freq_idx, pos_x, pos_y) receives
    weighted input from retina pixels within receptive_field_radius of
    (pos_x * stride, pos_y * stride) where stride = retina_size /
    n_positions_per_dim.

    Returns:
        pre_indices: shape (n_synapses,), retina neuron indices
        post_indices: shape (n_synapses,), V1 simple cell indices
        weights: shape (n_synapses,), Gabor-filtered weight values

    V1 simple cell index: orient_idx * (n_freqs * n_positions^2) +
                          freq_idx * (n_positions^2) +
                          pos_y * n_positions + pos_x

    Retina neuron index (channel-first): channel * (retina_size^2) +
                                           pixel_y * retina_size + pixel_x
    where channel ∈ {0=ON, 1=OFF}.
    """
    stride = retina_size // n_positions_per_dim  # 32/16 = 2
    sigma_xy_per_freq = [3.0, 2.5, 2.0, 1.5]   # Smaller envelope for higher freq
    freqs = [0.05, 0.10, 0.20, 0.40]
    orient_thetas = [i * math.pi / n_orientations for i in range(n_orientations)]

    pre_idx = []
    post_idx = []
    weights = []

    for orient_i, theta in enumerate(orient_thetas):
        for freq_i, freq in enumerate(freqs[:n_frequencies]):
            sigma = sigma_xy_per_freq[freq_i] if freq_i < len(sigma_xy_per_freq) else 2.0
            kernel = gabor_kernel(sigma, sigma, theta, freq, phase=0.0)
            for pos_y in range(n_positions_per_dim):
                for pos_x in range(n_positions_per_dim):
                    cx = pos_x * stride + stride // 2
                    cy = pos_y * stride + stride // 2
                    v1_idx = (
                        orient_i * (n_frequencies * n_positions_per_dim * n_positions_per_dim)
                        + freq_i * (n_positions_per_dim * n_positions_per_dim)
                        + pos_y * n_positions_per_dim + pos_x
                    )
                    # Sparse: only sample retina pixels within radius
                    for dy in range(-receptive_field_radius, receptive_field_radius + 1):
                        for dx in range(-receptive_field_radius, receptive_field_radius + 1):
                            px = cx + dx
                            py = cy + dy
                            if not (0 <= px < retina_size and 0 <= py < retina_size):
                                continue
                            w = kernel(dx, dy)
                            if abs(w) < 0.01:
                                continue  # Skip near-zero weights
                            # ON channel for positive weights, OFF for negative
                            # (split the bipolar Gabor into ON+OFF responses)
                            if w > 0:
                                channel = 0  # ON
                            else:
                                channel = 1  # OFF
                                w = -w  # Magnitude
                            retina_idx = (
                                channel * (retina_size * retina_size)
                                + py * retina_size + px
                            )
                            pre_idx.append(retina_idx)
                            post_idx.append(v1_idx)
                            weights.append(w)

    return (np.asarray(pre_idx, dtype=np.int64),
            np.asarray(post_idx, dtype=np.int64),
            np.asarray(weights, dtype=np.float32))


def render_gridworld_to_image(
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    grid_size: int = 8,
    image_size: int = 32,
    landmarks: list[Tuple[int, int]] = None,
) -> np.ndarray:
    """Render the gridworld state as an (image_size, image_size, 2) ON/OFF
    image suitable for feeding into the retina.

    Each grid cell is rendered as an `image_size / grid_size`-pixel block.
    The agent is a bright ON-channel block; the goal is a dimmer ON block;
    the rest of the grid is mid-gray (low ON, low OFF).

    Returns shape (2, image_size, image_size) — channel-first.
    Channel 0 = ON (bright), channel 1 = OFF (dark/edges).
    """
    pixels_per_cell = image_size // grid_size
    image = np.zeros((2, image_size, image_size), dtype=np.float32)

    # Agent: bright spot in ON channel
    ax, ay = agent_pos
    a_px = ax * pixels_per_cell + pixels_per_cell // 2
    a_py = ay * pixels_per_cell + pixels_per_cell // 2
    if 0 <= a_px < image_size and 0 <= a_py < image_size:
        image[0, a_py, a_px] = 1.0
        # spread to neighboring pixels for thickness
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (0 <= a_py + dy < image_size and 0 <= a_px + dx < image_size):
                    image[0, a_py + dy, a_px + dx] = max(
                        image[0, a_py + dy, a_px + dx], 0.7
                    )

    # Goal: dimmer ON block
    gx, gy = goal_pos
    g_px = gx * pixels_per_cell + pixels_per_cell // 2
    g_py = gy * pixels_per_cell + pixels_per_cell // 2
    if 0 <= g_px < image_size and 0 <= g_py < image_size:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (0 <= g_py + dy < image_size and 0 <= g_px + dx < image_size):
                    image[0, g_py + dy, g_px + dx] = max(
                        image[0, g_py + dy, g_px + dx], 0.5
                    )

    # Edges: OFF channel along grid lines (rough boundary signal)
    for i in range(0, image_size, pixels_per_cell):
        if i < image_size:
            image[1, i, :] = 0.3
            image[1, :, i] = 0.3

    return image


def image_to_retina_drive(
    image: np.ndarray,
    drive_max_pA: float = 200.0,
) -> np.ndarray:
    """Convert (2, H, W) image to flat (2*H*W,) array of input currents
    for the retina region. Channel 0 = ON, channel 1 = OFF, both flattened.

    Output index = channel * (H * W) + py * W + px, matching the convention
    used by build_v1_simple_weights."""
    flat = image.flatten().astype(np.float32) * drive_max_pA
    return flat
