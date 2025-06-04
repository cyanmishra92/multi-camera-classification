#!/usr/bin/env python3
"""
Demo version of research experiments for testing.

Runs a subset of experiments to demonstrate the framework.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import create_network_from_config, run_simulation, load_config
from src.core.network import NetworkConfig
from src.core.energy_model import EnergyParameters
from src.core.accuracy_model import AccuracyParameters
from src.utils.logger import setup_logging
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm
from src.algorithms.variable_frequency import VariableFrequencyAlgorithm
from src.algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from src.algorithms.baselines.random_selection import RandomSelectionAlgorithm
from src.algorithms.baselines.greedy_energy import GreedyEnergyAlgorithm
from src.game_theory.utility_functions import UtilityParameters

# Demo parameters
DEMO_ALGORITHMS = ['fixed', 'variable', 'unknown', 'random', 'greedy']
DEMO_FREQUENCIES = [0.05, 0.1, 0.5]
DEMO_DURATION = 1000
DEMO_RUNS = 2


def run_demo_experiment(algo_name: str, frequency: float, run_id: int) -> Dict:
    """Run a single demo experiment."""
    print(f"  Run {run_id}: {algo_name} @ {frequency} Hz...", end='', flush=True)
    
    # Create configuration
    config_dict = {
        'network': {
            'num_cameras': 10,
            'num_classes': 3,
            'num_objects': 2,
            'energy_params': {
                'capacity': 1000,
                'recharge_rate': 10,
                'classification_cost': 50,
                'min_operational': 100
            },
            'accuracy_params': {
                'max_accuracy': 0.95,
                'min_accuracy_ratio': 0.3,
                'correlation_factor': 0.2
            }
        },
        'simulation': {
            'duration': DEMO_DURATION,
            'time_step': 1.0,
            'classification_frequency': frequency
        }
    }
    
    # Create network
    network = create_network_from_config(config_dict)
    
    # Create and set algorithm
    if algo_name == 'fixed':
        algorithm = FixedFrequencyAlgorithm(
            cameras=network.cameras,
            num_classes=3,
            min_accuracy_threshold=0.7
        )
    elif algo_name == 'variable':
        algorithm = VariableFrequencyAlgorithm(
            cameras=network.cameras,
            num_classes=3,
            min_accuracy_threshold=0.7,
            classification_frequency=frequency
        )
    elif algo_name == 'unknown':
        utility_params = UtilityParameters()
        algorithm = UnknownFrequencyAlgorithm(
            cameras=network.cameras,
            num_classes=3,
            utility_params=utility_params,
            min_accuracy_threshold=0.7
        )
    elif algo_name == 'random':
        algorithm = RandomSelectionAlgorithm(
            cameras=network.cameras,
            min_accuracy_threshold=0.7
        )
    elif algo_name == 'greedy':
        algorithm = GreedyEnergyAlgorithm(
            cameras=network.cameras,
            min_accuracy_threshold=0.7
        )
    
    network.set_algorithm(algorithm)
    
    # Run simulation
    start_time = datetime.now()
    results = run_simulation(
        network=network,
        algorithm_type=algo_name,
        duration=DEMO_DURATION,
        time_step=1.0,
        classification_frequency=frequency
    )
    runtime = (datetime.now() - start_time).total_seconds()
    
    # Extract metrics
    stats = results['network_stats']
    metrics = {
        'algorithm': algo_name,
        'frequency': frequency,
        'run': run_id,
        'overall_accuracy': stats.get('accuracy', 0),
        'recent_accuracy': stats.get('recent_accuracy', 0),
        'energy_violations': stats.get('energy_violations', 0),
        'accuracy_violations': stats.get('accuracy_violations', 0),
        'avg_cameras_per_event': stats.get('avg_cameras_per_classification', 0),
        'total_classifications': stats.get('total_classifications', 0),
        'runtime': runtime
    }
    
    print(f" Done! (Acc: {metrics['overall_accuracy']:.3f})")
    return metrics


def main():
    """Run demo experiments."""
    print("=" * 60)
    print("RESEARCH EXPERIMENTS DEMO")
    print("=" * 60)
    
    setup_logging('WARNING')  # Reduce verbosity
    
    # Create output directory
    output_dir = Path(f'demo_research_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Run experiments
    print("\nRunning experiments...")
    for algo in DEMO_ALGORITHMS:
        print(f"\nAlgorithm: {algo}")
        for freq in DEMO_FREQUENCIES:
            for run in range(DEMO_RUNS):
                try:
                    metrics = run_demo_experiment(algo, freq, run)
                    results.append(metrics)
                except Exception as e:
                    print(f" Failed: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'demo_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Group by algorithm
    summary = df.groupby('algorithm').agg({
        'overall_accuracy': ['mean', 'std'],
        'energy_violations': 'sum',
        'avg_cameras_per_event': 'mean'
    })
    
    print("\nAccuracy Comparison:")
    for algo in DEMO_ALGORITHMS:
        if algo in summary.index:
            acc_mean = summary.loc[algo, ('overall_accuracy', 'mean')]
            acc_std = summary.loc[algo, ('overall_accuracy', 'std')]
            violations = summary.loc[algo, ('energy_violations', 'sum')]
            efficiency = summary.loc[algo, ('avg_cameras_per_event', 'mean')]
            print(f"  {algo:12s}: {acc_mean:.3f} ± {acc_std:.3f} "
                  f"(violations: {violations}, cameras/event: {efficiency:.2f})")
    
    # Best by frequency
    print("\nBest Algorithm by Frequency:")
    for freq in DEMO_FREQUENCIES:
        freq_data = df[df['frequency'] == freq]
        best = freq_data.groupby('algorithm')['overall_accuracy'].mean().idxmax()
        best_acc = freq_data.groupby('algorithm')['overall_accuracy'].mean().max()
        print(f"  {freq:.2f} Hz: {best} ({best_acc:.3f})")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Generate simple plots
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy by algorithm
        algo_acc = df.groupby('algorithm')['overall_accuracy'].mean()
        algo_acc.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Accuracy by Algorithm')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Algorithm')
        
        # Accuracy vs frequency
        for algo in DEMO_ALGORITHMS:
            algo_data = df[df['algorithm'] == algo]
            freq_acc = algo_data.groupby('frequency')['overall_accuracy'].mean()
            ax2.plot(freq_acc.index, freq_acc.values, marker='o', label=algo)
        
        ax2.set_title('Accuracy vs Classification Frequency')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Accuracy')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'demo_results.png', dpi=150)
        print(f"Plot saved to: {output_dir / 'demo_results.png'}")
        
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()