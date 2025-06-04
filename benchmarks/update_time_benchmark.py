#!/usr/bin/env python3
"""Benchmark update_time performance for large networks."""

import time
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.network import CameraNetwork, NetworkConfig


def run_benchmark(num_cameras: int = 5000, steps: int = 1000) -> None:
    """Run benchmark comparing vectorized vs naive update."""
    config = NetworkConfig(num_cameras=num_cameras, num_classes=3, num_objects=5)
    network = CameraNetwork(config)

    # Precompute energy model
    model = network.cameras[0].energy_model

    def slow_update(energies: np.ndarray, delta: float) -> np.ndarray:
        new_e = energies.copy()
        for i in range(len(new_e)):
            new_e[i] = min(new_e[i] + model.harvest(delta), model.capacity)
        return new_e

    # Benchmark slow loop
    energies = network.camera_energies.copy()
    start = time.perf_counter()
    for _ in range(steps):
        energies = slow_update(energies, 1.0)
    slow_time = time.perf_counter() - start

    # Benchmark vectorized network.update_time
    start = time.perf_counter()
    for _ in range(steps):
        network.update_time(1.0)
    fast_time = time.perf_counter() - start

    print(f"Naive loop time: {slow_time:.3f}s")
    print(f"Vectorized time: {fast_time:.3f}s")
    if fast_time > 0:
        print(f"Speedup: {slow_time / fast_time:.2f}x")


if __name__ == "__main__":
    run_benchmark()
