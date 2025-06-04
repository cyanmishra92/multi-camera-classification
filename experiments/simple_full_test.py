#!/usr/bin/env python3
"""Simple test of full experimental setup."""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.network import CameraNetwork, NetworkConfig
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm
from src.algorithms.variable_frequency import VariableFrequencyAlgorithm
from src.algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from src.algorithms.baselines.random_selection import RandomSelectionAlgorithm
from src.algorithms.baselines.greedy_energy import GreedyEnergyAlgorithm
from src.game_theory.utility_functions import UtilityParameters
from src.utils.logger import setup_logging


def create_test_network(num_cameras=10, num_classes=3):
    """Create a test network."""
    config = NetworkConfig(
        num_cameras=num_cameras,
        num_classes=num_classes,
        num_objects=2,
        energy_params=EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100
        ),
        accuracy_params=AccuracyParameters(
            max_accuracy=0.95,
            min_accuracy_ratio=0.3,
            correlation_factor=0.2
        )
    )
    return CameraNetwork(config)


def run_simple_simulation(network, algorithm, duration=1000, frequency=0.1):
    """Run a simple simulation."""
    network.set_algorithm(algorithm)
    
    # Generate events
    events = []
    current_time = 0
    while current_time < duration:
        interval = np.random.exponential(1.0 / frequency)
        current_time += interval
        if current_time < duration:
            events.append(current_time)
    
    print(f"  Generated {len(events)} classification events")
    
    # Run simulation
    results = {
        'successful': 0,
        'total': 0,
        'energy_violations': 0,
        'selected_cameras': []
    }
    
    for i, event_time in enumerate(events):
        # Update network time
        time_delta = event_time - network.current_time if i > 0 else 0
        network.update_time(time_delta)
        
        # Classify
        object_position = np.random.uniform(-50, 50, size=3)
        true_label = np.random.randint(0, 2)
        
        result = network.classify_object(object_position, true_label)
        
        results['total'] += 1
        if result['success']:
            results['successful'] += 1
        results['selected_cameras'].extend(result.get('selected_cameras', []))
    
    # Compute metrics
    accuracy = results['successful'] / results['total'] if results['total'] > 0 else 0
    avg_cameras = len(results['selected_cameras']) / results['total'] if results['total'] > 0 else 0
    
    # Check energy violations
    stats = network.algorithm.get_performance_metrics()
    energy_violations = stats.get('energy_violations', 0)
    
    return {
        'accuracy': accuracy,
        'total_events': results['total'],
        'successful_events': results['successful'],
        'avg_cameras_per_event': avg_cameras,
        'energy_violations': energy_violations
    }


def main():
    """Run simple full test."""
    print("=" * 60)
    print("SIMPLE FULL EXPERIMENTAL TEST")
    print("=" * 60)
    
    setup_logging('WARNING')
    
    # Test parameters
    algorithms = {
        'Fixed': lambda net: FixedFrequencyAlgorithm(net.cameras, 3, 0.7),
        'Variable': lambda net: VariableFrequencyAlgorithm(net.cameras, 3, 0.7, 0.1),
        'Unknown': lambda net: UnknownFrequencyAlgorithm(net.cameras, 3, UtilityParameters(), 0.7),
        'Random': lambda net: RandomSelectionAlgorithm(net.cameras, 0.7),
        'Greedy': lambda net: GreedyEnergyAlgorithm(net.cameras, 0.7)
    }
    
    frequencies = [0.05, 0.1, 0.5]
    duration = 1000
    
    results = []
    
    # Run tests
    for algo_name, algo_factory in algorithms.items():
        print(f"\nTesting {algo_name} algorithm:")
        
        for freq in frequencies:
            print(f"  Frequency {freq}:", end=' ', flush=True)
            
            # Create fresh network
            network = create_test_network()
            algorithm = algo_factory(network)
            
            try:
                start = time.time()
                metrics = run_simple_simulation(network, algorithm, duration, freq)
                runtime = time.time() - start
                
                print(f"Acc={metrics['accuracy']:.3f}, "
                      f"Violations={metrics['energy_violations']}, "
                      f"Time={runtime:.2f}s")
                
                results.append({
                    'algorithm': algo_name,
                    'frequency': freq,
                    'accuracy': metrics['accuracy'],
                    'energy_violations': metrics['energy_violations'],
                    'avg_cameras': metrics['avg_cameras_per_event'],
                    'runtime': runtime
                })
                
            except Exception as e:
                print(f"Failed: {e}")
    
    # Save results
    output_dir = Path(f'simple_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = df.groupby('algorithm')['accuracy'].agg(['mean', 'std'])
    print("\nAccuracy by Algorithm:")
    for algo, row in summary.iterrows():
        print(f"  {algo:10s}: {row['mean']:.3f} ± {row['std']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()