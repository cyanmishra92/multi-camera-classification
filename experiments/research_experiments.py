#!/usr/bin/env python3
"""
Comprehensive experimental framework for research paper evaluation.

This script runs all experiments needed for a research paper submission,
comparing our algorithms against baselines across multiple metrics and scenarios.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import create_network_from_config, run_simulation, load_config
from src.core.network import NetworkConfig
from src.utils.logger import setup_logging
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm
from src.algorithms.variable_frequency import VariableFrequencyAlgorithm
from src.algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from src.algorithms.baselines.random_selection import RandomSelectionAlgorithm
from src.algorithms.baselines.greedy_energy import GreedyEnergyAlgorithm
from src.algorithms.baselines.round_robin import RoundRobinAlgorithm
from src.algorithms.baselines.coverage_based import CoverageBasedAlgorithm
from src.algorithms.baselines.threshold_based import ThresholdBasedAlgorithm

# Algorithm configurations
ALGORITHMS = {
    'fixed': FixedFrequencyAlgorithm,
    'variable': VariableFrequencyAlgorithm,
    'unknown': UnknownFrequencyAlgorithm,
    'random': RandomSelectionAlgorithm,
    'greedy': GreedyEnergyAlgorithm,
    'round_robin': RoundRobinAlgorithm,
    'coverage': CoverageBasedAlgorithm,
    'threshold': ThresholdBasedAlgorithm
}

# Experimental parameters
NETWORK_SIZES = {
    'small': {'cameras': 10, 'classes': 3},
    'medium': {'cameras': 50, 'classes': 5},
    'large': {'cameras': 100, 'classes': 10}
}

FREQUENCIES = [0.01, 0.05, 0.1, 0.5, 1.0]
ENERGY_CAPACITIES = [500, 1000, 2000]
RECHARGE_RATES = [5, 10, 20]
ACCURACY_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
SIMULATION_DURATION = 5000
RUNS_PER_EXPERIMENT = 5


class ExperimentRunner:
    """Manages and runs comprehensive experiments."""
    
    def __init__(self, output_dir: Path):
        """Initialize experiment runner."""
        self.output_dir = output_dir
        self.results = []
        setup_logging('INFO')
        
    def run_single_experiment(self, params: Dict) -> Dict:
        """Run a single experiment with given parameters."""
        # Create network configuration
        config = NetworkConfig(
            num_cameras=params['num_cameras'],
            num_classes=params['num_classes'],
            num_objects=2,  # Binary classification
            energy_params={
                'capacity': params['energy_capacity'],
                'recharge_rate': params['recharge_rate'],
                'classification_cost': 50,
                'min_operational': 100
            },
            accuracy_params={
                'max_accuracy': 0.95,
                'min_accuracy_ratio': 0.3,
                'correlation_factor': 0.2
            }
        )
        
        # Create network
        network = create_network_from_config({'network': config.__dict__})
        
        # Set algorithm
        algo_class = ALGORITHMS[params['algorithm']]
        if params['algorithm'] in ['fixed', 'variable', 'unknown']:
            algorithm = algo_class(
                cameras=network.cameras,
                num_classes=params['num_classes'],
                min_accuracy_threshold=params['accuracy_threshold']
            )
        else:
            algorithm = algo_class(
                cameras=network.cameras,
                min_accuracy_threshold=params['accuracy_threshold']
            )
        
        network.set_algorithm(algorithm)
        
        # Run simulation
        results = run_simulation(
            network=network,
            algorithm_type=params['algorithm'],
            duration=SIMULATION_DURATION,
            time_step=1.0,
            classification_frequency=params['frequency']
        )
        
        # Extract metrics
        metrics = self._extract_metrics(results, params)
        return metrics
    
    def _extract_metrics(self, results: Dict, params: Dict) -> Dict:
        """Extract evaluation metrics from simulation results."""
        stats = results['network_stats']
        perf_history = results['performance_history']
        energy_history = results['energy_history']
        
        # Primary metrics
        metrics = {
            'algorithm': params['algorithm'],
            'network_size': params['network_size'],
            'num_cameras': params['num_cameras'],
            'frequency': params['frequency'],
            'energy_capacity': params['energy_capacity'],
            'recharge_rate': params['recharge_rate'],
            'accuracy_threshold': params['accuracy_threshold'],
            
            # Accuracy metrics
            'overall_accuracy': stats.get('accuracy', 0),
            'recent_accuracy': stats.get('recent_accuracy', 0),
            
            # Energy metrics
            'energy_violations': stats.get('energy_violations', 0),
            'accuracy_violations': stats.get('accuracy_violations', 0),
            'avg_energy': stats.get('avg_energy', 0),
            'min_energy_ever': min([e['min_energy'] for e in energy_history]),
            
            # Efficiency metrics
            'avg_cameras_per_event': stats.get('avg_cameras_per_classification', 0),
            'total_classifications': stats.get('total_classifications', 0),
            
            # Fairness metrics (compute from camera stats)
            'participation_variance': self._compute_participation_variance(results),
            'jains_fairness': self._compute_jains_fairness(results),
            
            # Adaptability (for adaptive algorithms)
            'convergence_time': self._estimate_convergence_time(perf_history),
            
            # Runtime
            'algorithm_runtime': results.get('runtime', 0)
        }
        
        return metrics
    
    def _compute_participation_variance(self, results: Dict) -> float:
        """Compute variance in camera participation."""
        # Extract participation counts from performance history
        participation_counts = {}
        for record in results['performance_history']:
            for cam_id in record['result'].get('selected_cameras', []):
                participation_counts[cam_id] = participation_counts.get(cam_id, 0) + 1
        
        if not participation_counts:
            return 0.0
        
        counts = list(participation_counts.values())
        return np.var(counts)
    
    def _compute_jains_fairness(self, results: Dict) -> float:
        """Compute Jain's fairness index."""
        participation_counts = {}
        for record in results['performance_history']:
            for cam_id in record['result'].get('selected_cameras', []):
                participation_counts[cam_id] = participation_counts.get(cam_id, 0) + 1
        
        if not participation_counts:
            return 1.0
        
        counts = list(participation_counts.values())
        n = len(counts)
        if n == 0:
            return 1.0
        
        sum_x = sum(counts)
        sum_x2 = sum(x**2 for x in counts)
        
        return (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
    
    def _estimate_convergence_time(self, perf_history: List) -> float:
        """Estimate time to convergence based on accuracy stability."""
        if len(perf_history) < 100:
            return 0.0
        
        # Compute rolling accuracy
        window = 50
        accuracies = []
        
        for i in range(window, len(perf_history)):
            window_records = perf_history[i-window:i]
            accuracy = sum(r['result']['success'] for r in window_records) / window
            accuracies.append((perf_history[i]['timestamp'], accuracy))
        
        # Find when variance becomes stable
        for i in range(100, len(accuracies)):
            recent_accuracies = [a[1] for a in accuracies[i-100:i]]
            if np.var(recent_accuracies) < 0.01:  # Stable
                return accuracies[i][0]
        
        return SIMULATION_DURATION  # Never converged
    
    def run_experiment_suite(self):
        """Run complete suite of experiments."""
        experiments = []
        
        # Generate all experiment configurations
        for size_name, size_config in NETWORK_SIZES.items():
            for freq in FREQUENCIES:
                for capacity in ENERGY_CAPACITIES:
                    for recharge in RECHARGE_RATES:
                        for threshold in ACCURACY_THRESHOLDS:
                            for algo in ALGORITHMS.keys():
                                for run in range(RUNS_PER_EXPERIMENT):
                                    experiments.append({
                                        'network_size': size_name,
                                        'num_cameras': size_config['cameras'],
                                        'num_classes': size_config['classes'],
                                        'frequency': freq,
                                        'energy_capacity': capacity,
                                        'recharge_rate': recharge,
                                        'accuracy_threshold': threshold,
                                        'algorithm': algo,
                                        'run': run
                                    })
        
        print(f"Total experiments to run: {len(experiments)}")
        
        # Run experiments (can be parallelized)
        for i, exp_params in enumerate(experiments):
            print(f"Running experiment {i+1}/{len(experiments)}: {exp_params['algorithm']} "
                  f"on {exp_params['network_size']} network...")
            
            try:
                metrics = self.run_single_experiment(exp_params)
                metrics['run'] = exp_params['run']
                self.results.append(metrics)
                
                # Save intermediate results
                if i % 100 == 0:
                    self._save_results()
            
            except Exception as e:
                print(f"Experiment failed: {e}")
                continue
        
        # Save final results
        self._save_results()
        
    def _save_results(self):
        """Save results to file."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'experiment_results.csv', index=False)
        
        # Also save as JSON for flexibility
        with open(self.output_dir / 'experiment_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_analysis(self):
        """Generate analysis and plots from results."""
        # This will be implemented in the visualization module
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for research paper"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=f'research_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Output directory for results'
    )
    parser.add_argument(
        '--subset',
        type=str,
        choices=['small', 'medium', 'large', 'all'],
        default='small',
        help='Network size subset to run'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run experiments
    runner = ExperimentRunner(output_dir)
    runner.run_experiment_suite()
    
    print(f"\\nExperiments complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()