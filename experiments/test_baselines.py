#!/usr/bin/env python3
"""Test baseline algorithms implementation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters
from src.algorithms.baselines.random_selection import RandomSelectionAlgorithm
from src.algorithms.baselines.greedy_energy import GreedyEnergyAlgorithm
from src.algorithms.baselines.round_robin import RoundRobinAlgorithm
from src.algorithms.baselines.coverage_based import CoverageBasedAlgorithm
from src.algorithms.baselines.threshold_based import ThresholdBasedAlgorithm
import numpy as np


def create_test_cameras(n=10):
    """Create test cameras."""
    energy_params = EnergyParameters(
        capacity=1000,
        recharge_rate=10,
        classification_cost=50,
        min_operational=100
    )
    energy_model = EnergyModel(energy_params)
    
    accuracy_params = AccuracyParameters(
        max_accuracy=0.95,
        min_accuracy_ratio=0.3,
        correlation_factor=0.2
    )
    accuracy_model = AccuracyModel(accuracy_params, energy_model)
    
    cameras = []
    for i in range(n):
        position = np.random.uniform(-50, 50, size=3)
        camera = Camera(
            camera_id=i,
            position=position,
            energy_model=energy_model,
            accuracy_model=accuracy_model,
            num_classes=2
        )
        cameras.append(camera)
    
    return cameras


def test_algorithm(algo_class, algo_name, cameras):
    """Test a single algorithm."""
    print(f"\nTesting {algo_name}:")
    
    # Create algorithm instance
    if algo_name == 'coverage':
        algo = algo_class(cameras, min_accuracy_threshold=0.7)
    else:
        algo = algo_class(cameras, min_accuracy_threshold=0.7)
    
    # Test selection
    for i in range(5):
        selected = algo.select_cameras(instance_id=i, current_time=i*10)
        print(f"  Instance {i}: Selected {len(selected)} cameras: {selected}")
        
        if selected:
            # Check collective accuracy
            selected_cameras = [cameras[idx] for idx in selected]
            collective_acc = algo._calculate_collective_accuracy(selected_cameras)
            print(f"    Collective accuracy: {collective_acc:.3f}")


def main():
    """Test all baseline algorithms."""
    print("=" * 60)
    print("BASELINE ALGORITHMS TEST")
    print("=" * 60)
    
    # Create test cameras
    cameras = create_test_cameras(10)
    print(f"Created {len(cameras)} test cameras")
    
    # Test each baseline
    baselines = [
        (RandomSelectionAlgorithm, 'random'),
        (GreedyEnergyAlgorithm, 'greedy'),
        (RoundRobinAlgorithm, 'round_robin'),
        (CoverageBasedAlgorithm, 'coverage'),
        (ThresholdBasedAlgorithm, 'threshold')
    ]
    
    for algo_class, name in baselines:
        try:
            test_algorithm(algo_class, name, cameras)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()