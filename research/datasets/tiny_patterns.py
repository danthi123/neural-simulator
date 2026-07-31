"""TinyPatternDataset: K-class Poisson rate-vector synthetic dataset.

Each class has a fixed mean rate vector (drawn once from a class-mean RNG).
Examples are sampled as class_mean + Gaussian(0, noise_sigma), clipped to
[rate_min, rate_max]. Labels are balanced across classes.

Saved as a single .npz with X_train, y_train, X_test, y_test, class_means,
and a JSON metadata blob recording the generator seed and hyperparameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TinyPatternDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path):
        data = np.load(path)
        metadata = json.loads(str(data["metadata_json"]))
        return cls(
            X_train=data["X_train"].astype(np.float32),
            y_train=data["y_train"].astype(np.int32),
            X_test=data["X_test"].astype(np.float32),
            y_test=data["y_test"].astype(np.int32),
            metadata=metadata,
        )


def build_dataset(
    out_path,
    *,
    seed,
    K=4,
    n_features=64,
    n_train=200,
    n_test=50,
    noise_sigma=4.0,
    rate_min=1.0,
    rate_max=40.0,
):
    """Generate and save a TinyPatternDataset."""
    rng = np.random.default_rng(seed)

    margin = 5.0
    class_means = rng.uniform(rate_min + margin, rate_max - margin,
                              size=(K, n_features)).astype(np.float32)

    def _sample(n_per_split, split_seed_offset):
        split_rng = np.random.default_rng(seed + split_seed_offset)
        labels = np.tile(np.arange(K, dtype=np.int32),
                         n_per_split // K + 1)[:n_per_split]
        split_rng.shuffle(labels)
        X = np.empty((n_per_split, n_features), dtype=np.float32)
        for i, y in enumerate(labels):
            noise = split_rng.normal(0.0, noise_sigma,
                                     size=n_features).astype(np.float32)
            X[i] = np.clip(class_means[y] + noise, rate_min, rate_max)
        return X, labels

    X_train, y_train = _sample(n_train, split_seed_offset=1)
    X_test, y_test = _sample(n_test, split_seed_offset=2)

    metadata = {
        "seed": int(seed),
        "K": int(K),
        "n_features": int(n_features),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "noise_sigma": float(noise_sigma),
        "rate_min": float(rate_min),
        "rate_max": float(rate_max),
        "class_means_shape": list(class_means.shape),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        class_means=class_means,
        metadata_json=np.array(json.dumps(metadata)),
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build"])
    p.add_argument("--out", default="research/datasets/tiny_patterns.npz")
    p.add_argument("--seed", type=lambda x: int(x, 0), default=0xD47A5E7)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--noise-sigma", type=float, default=4.0)
    args = p.parse_args()

    if args.command == "build":
        build_dataset(args.out, seed=args.seed, K=args.K,
                      n_features=args.n_features,
                      n_train=args.n_train, n_test=args.n_test,
                      noise_sigma=args.noise_sigma)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
