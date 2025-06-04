#!/usr/bin/env python3
"""
Full-scale experimental evaluation for research paper.

This script runs comprehensive experiments comparing all algorithms
across multiple metrics and parameter settings.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from itertools import product
from typing import Dict, List, Tuple
import argparse
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import create_network_from_config, run_simulation
from src.core.network import NetworkConfig
from src.core.energy_model import EnergyParameters
from src.core.accuracy_model import AccuracyParameters
from src.utils.logger import setup_logging
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm
from src.algorithms.variable_frequency import VariableFrequencyAlgorithm
from src.algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from src.algorithms.baselines.random_selection import RandomSelectionAlgorithm
from src.algorithms.baselines.greedy_energy import GreedyEnergyAlgorithm
from src.algorithms.baselines.round_robin import RoundRobinAlgorithm
from src.algorithms.baselines.coverage_based import CoverageBasedAlgorithm
from src.algorithms.baselines.threshold_based import ThresholdBasedAlgorithm
from src.game_theory.utility_functions import UtilityParameters

# Experimental configurations
ALGORITHMS = {
    'Fixed-Freq': 'fixed',
    'Variable-Freq': 'variable', 
    'Unknown-Freq': 'unknown',
    'Random': 'random',
    'Greedy-Energy': 'greedy',
    'Round-Robin': 'round_robin',
    'Coverage': 'coverage',
    'Threshold': 'threshold'
}

# Parameter configurations for comprehensive evaluation
PARAM_CONFIGS = {
    'small_scale': {
        'network_sizes': [10],  # cameras
        'num_classes': [3],
        'frequencies': [0.05, 0.1, 0.5],
        'energy_capacities': [1000],
        'recharge_rates': [10],
        'accuracy_thresholds': [0.7, 0.8],
        'durations': [2000],
        'runs': 3
    },
    'medium_scale': {
        'network_sizes': [10, 30],
        'num_classes': [3, 5],
        'frequencies': [0.01, 0.05, 0.1, 0.5, 1.0],
        'energy_capacities': [500, 1000, 2000],
        'recharge_rates': [5, 10, 20],
        'accuracy_thresholds': [0.6, 0.7, 0.8, 0.9],
        'durations': [5000],
        'runs': 5
    },
    'full_scale': {
        'network_sizes': [10, 50, 100],
        'num_classes': [3, 5, 10],
        'frequencies': [0.01, 0.05, 0.1, 0.5, 1.0],
        'energy_capacities': [500, 1000, 2000],
        'recharge_rates': [5, 10, 20],
        'accuracy_thresholds': [0.6, 0.7, 0.8, 0.9],
        'durations': [10000],
        'runs': 10
    }
}


class FullExperimentRunner:
    """Manages full-scale experimental evaluation."""
    
    def __init__(self, output_dir: Path, scale: str = 'medium_scale'):
        """Initialize experiment runner."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / 'raw_results').mkdir(exist_ok=True)
        (self.output_dir / 'aggregated_results').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        self.scale = scale
        self.config = PARAM_CONFIGS[scale]
        self.results = []
        
        # Setup logging
        setup_logging('INFO', log_file=self.output_dir / 'logs' / 'experiment.log')
        
    def generate_experiment_list(self):
        """Generate list of all experiments to run."""
        experiments = []
        
        # Generate all parameter combinations
        param_combinations = product(
            self.config['network_sizes'],
            self.config['num_classes'],
            self.config['frequencies'],
            self.config['energy_capacities'],
            self.config['recharge_rates'],
            self.config['accuracy_thresholds']
        )
        
        exp_id = 0
        for params in param_combinations:
            n_cams, n_classes, freq, capacity, recharge, threshold = params
            
            # Skip invalid combinations
            if n_classes > n_cams:
                continue
                
            for algo_name, algo_key in ALGORITHMS.items():
                for run in range(self.config['runs']):
                    experiments.append({
                        'exp_id': exp_id,
                        'algorithm': algo_key,
                        'algorithm_name': algo_name,
                        'num_cameras': n_cams,
                        'num_classes': n_classes,
                        'frequency': freq,
                        'energy_capacity': capacity,
                        'recharge_rate': recharge,
                        'accuracy_threshold': threshold,
                        'duration': self.config['durations'][0],
                        'run': run
                    })
                    exp_id += 1
        
        return experiments
    
    def run_single_experiment(self, exp_params: Dict) -> Dict:
        """Run a single experiment."""
        try:
            # Create configuration dictionary
            config = {
                'network': {
                    'num_cameras': exp_params['num_cameras'],
                    'num_classes': exp_params['num_classes'],
                    'num_objects': 2  # Binary classification
                },
                'energy': {
                    'battery_capacity': exp_params['energy_capacity'],
                    'recharge_rate': exp_params['recharge_rate'],
                    'classification_cost': 50,
                    'min_operational': 100
                },
                'accuracy': {
                    'max_accuracy': 0.95,
                    'min_accuracy_ratio': 0.3,
                    'correlation_factor': 0.2
                },
                'game_theory': {
                    'reward_scale': 1.0,
                    'incorrect_penalty': 0.5,
                    'non_participation_penalty': 0.1,
                    'discount_factor': 0.95
                }
            }
            
            # Create network
            network = create_network_from_config(config)
            
            # Create algorithm
            algo_key = exp_params['algorithm']
            if algo_key == 'fixed':
                algorithm = FixedFrequencyAlgorithm(
                    cameras=network.cameras,
                    num_classes=exp_params['num_classes'],
                    min_accuracy_threshold=exp_params['accuracy_threshold'],
                    use_game_theory=True
                )
            elif algo_key == 'variable':
                # Calculate recharge time based on energy parameters
                recharge_time = exp_params['energy_capacity'] / exp_params['recharge_rate']
                algorithm = VariableFrequencyAlgorithm(
                    cameras=network.cameras,
                    num_classes=exp_params['num_classes'],
                    classification_frequency=exp_params['frequency'],
                    recharge_time=recharge_time,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'unknown':
                algorithm = UnknownFrequencyAlgorithm(
                    cameras=network.cameras,
                    num_classes=exp_params['num_classes'],
                    utility_params=UtilityParameters(),
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'random':
                algorithm = RandomSelectionAlgorithm(
                    cameras=network.cameras,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'greedy':
                algorithm = GreedyEnergyAlgorithm(
                    cameras=network.cameras,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'round_robin':
                algorithm = RoundRobinAlgorithm(
                    cameras=network.cameras,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'coverage':
                algorithm = CoverageBasedAlgorithm(
                    cameras=network.cameras,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            elif algo_key == 'threshold':
                algorithm = ThresholdBasedAlgorithm(
                    cameras=network.cameras,
                    min_accuracy_threshold=exp_params['accuracy_threshold']
                )
            
            network.algorithm = algorithm
            
            # Generate classification events
            events = []
            current_time = 0
            while current_time < exp_params['duration']:
                interval = np.random.exponential(1.0 / exp_params['frequency'])
                current_time += interval
                if current_time < exp_params['duration']:
                    events.append(current_time)
            
            # Run simulation
            start_time = time.time()
            results = {
                'successful': 0,
                'total': len(events),
                'selected_cameras': [],
                'performance_history': [],
                'energy_history': []
            }
            
            for i, event_time in enumerate(events):
                # Update network time
                time_delta = event_time - network.current_time if i > 0 else event_time
                network.update_time(time_delta)
                
                # Classify
                object_position = np.random.uniform(-50, 50, size=3)
                true_label = np.random.randint(0, 2)
                
                result = network.classify_object(object_position, true_label)
                
                if result['success']:
                    results['successful'] += 1
                results['selected_cameras'].extend(result.get('selected_cameras', []))
                
                # Record history
                results['performance_history'].append({
                    'timestamp': event_time,
                    'result': result
                })
                
                if i % 100 == 0:
                    results['energy_history'].append({
                        'timestamp': event_time,
                        'avg_energy': np.mean([cam.state.energy for cam in network.cameras]),
                        'min_energy': min([cam.state.energy for cam in network.cameras])
                    })
            
            runtime = time.time() - start_time
            
            # Add network stats
            results['network_stats'] = network.get_network_stats()
            results['algorithm_type'] = algo_key
            results['total_events'] = len(events)
            results['duration'] = exp_params['duration']
            results['classification_frequency'] = exp_params['frequency']
            
            # Extract comprehensive metrics
            metrics = self._extract_metrics(results, exp_params, runtime)
            
            return metrics
            
        except Exception as e:
            print(f"Experiment {exp_params['exp_id']} failed: {e}")
            return None
    
    def _extract_metrics(self, results: Dict, params: Dict, runtime: float) -> Dict:
        """Extract comprehensive metrics from simulation results."""
        stats = results['network_stats']
        perf_history = results['performance_history']
        energy_history = results['energy_history']
        
        # Basic experiment info
        metrics = {
            'exp_id': params['exp_id'],
            'algorithm': params['algorithm'],
            'algorithm_name': params['algorithm_name'],
            'num_cameras': params['num_cameras'],
            'num_classes': params['num_classes'],
            'frequency': params['frequency'],
            'energy_capacity': params['energy_capacity'],
            'recharge_rate': params['recharge_rate'],
            'accuracy_threshold': params['accuracy_threshold'],
            'duration': params['duration'],
            'run': params['run'],
            'runtime': runtime
        }
        
        # Accuracy metrics
        metrics.update({
            'overall_accuracy': stats.get('accuracy', 0),
            'recent_accuracy': stats.get('recent_accuracy', 0),
            'accuracy_std': self._compute_accuracy_std(perf_history),
            'accuracy_improvement': self._compute_accuracy_improvement(perf_history)
        })
        
        # Energy metrics
        metrics.update({
            'energy_violations': stats.get('energy_violations', 0),
            'accuracy_violations': stats.get('accuracy_violations', 0),
            'avg_energy': stats.get('avg_energy', 0),
            'min_energy_ever': min([e['min_energy'] for e in energy_history]) if energy_history else 0,
            'energy_variance': np.var([e['avg_energy'] for e in energy_history]) if energy_history else 0,
            'energy_sustainability': self._compute_sustainability(energy_history)
        })
        
        # Efficiency metrics
        metrics.update({
            'avg_cameras_per_event': stats.get('avg_cameras_per_classification', 0),
            'total_classifications': stats.get('total_classifications', 0),
            'successful_classifications': stats.get('successful_classifications', 0),
            'classification_rate': stats.get('total_classifications', 0) / params['duration'] if params['duration'] > 0 else 0
        })
        
        # Fairness metrics
        participation_stats = self._compute_participation_stats(perf_history, params['num_cameras'])
        metrics.update({
            'jains_fairness': participation_stats['jains_fairness'],
            'participation_variance': participation_stats['variance'],
            'max_participation': participation_stats['max_participation'],
            'min_participation': participation_stats['min_participation'],
            'gini_coefficient': participation_stats['gini']
        })
        
        # Adaptability metrics
        metrics.update({
            'convergence_time': self._estimate_convergence_time(perf_history),
            'stability_score': self._compute_stability_score(perf_history),
            'adaptation_rate': self._compute_adaptation_rate(results)
        })
        
        # Quality metrics
        metrics.update({
            'avg_collective_accuracy': self._compute_avg_collective_accuracy(perf_history),
            'accuracy_consistency': self._compute_accuracy_consistency(perf_history),
            'response_time_avg': np.mean([r.get('response_time', 0) for r in perf_history]) if perf_history else 0
        })
        
        return metrics
    
    def _compute_accuracy_std(self, perf_history: List) -> float:
        """Compute standard deviation of accuracy over time."""
        if len(perf_history) < 2:
            return 0.0
        
        window = 100
        accuracies = []
        for i in range(window, len(perf_history), window):
            window_acc = sum(r['result']['success'] for r in perf_history[i-window:i]) / window
            accuracies.append(window_acc)
        
        return np.std(accuracies) if accuracies else 0.0
    
    def _compute_accuracy_improvement(self, perf_history: List) -> float:
        """Compute accuracy improvement from start to end."""
        if len(perf_history) < 200:
            return 0.0
        
        # First 10% vs last 10%
        n = len(perf_history)
        first_tenth = perf_history[:n//10]
        last_tenth = perf_history[-n//10:]
        
        first_acc = sum(r['result']['success'] for r in first_tenth) / len(first_tenth)
        last_acc = sum(r['result']['success'] for r in last_tenth) / len(last_tenth)
        
        return last_acc - first_acc
    
    def _compute_sustainability(self, energy_history: List) -> float:
        """Compute energy sustainability score."""
        if not energy_history:
            return 0.0
        
        # Percentage of time all cameras had > 20% energy
        threshold = 0.2
        sustainable_count = sum(
            1 for e in energy_history 
            if e['min_energy'] > threshold * 1000  # Assuming capacity=1000
        )
        
        return sustainable_count / len(energy_history)
    
    def _compute_participation_stats(self, perf_history: List, num_cameras: int) -> Dict:
        """Compute comprehensive participation statistics."""
        participation_counts = [0] * num_cameras
        
        for record in perf_history:
            for cam_id in record['result'].get('selected_cameras', []):
                if 0 <= cam_id < num_cameras:
                    participation_counts[cam_id] += 1
        
        total_participations = sum(participation_counts)
        if total_participations == 0:
            return {
                'jains_fairness': 1.0,
                'variance': 0.0,
                'max_participation': 0,
                'min_participation': 0,
                'gini': 0.0
            }
        
        # Jain's fairness index
        sum_x = sum(participation_counts)
        sum_x2 = sum(x**2 for x in participation_counts)
        jains = (sum_x ** 2) / (num_cameras * sum_x2) if sum_x2 > 0 else 1.0
        
        # Gini coefficient
        sorted_counts = sorted(participation_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'jains_fairness': jains,
            'variance': np.var(participation_counts),
            'max_participation': max(participation_counts),
            'min_participation': min(participation_counts),
            'gini': gini
        }
    
    def _estimate_convergence_time(self, perf_history: List) -> float:
        """Estimate time to performance convergence."""
        if len(perf_history) < 200:
            return -1  # Not converged
        
        window = 100
        threshold = 0.02  # 2% variance threshold
        
        for i in range(200, len(perf_history), 50):
            recent = perf_history[i-window:i]
            accuracies = [r['result']['success'] for r in recent]
            
            if np.var(accuracies) < threshold:
                return perf_history[i]['timestamp']
        
        return -1  # Never converged
    
    def _compute_stability_score(self, perf_history: List) -> float:
        """Compute performance stability score."""
        if len(perf_history) < 100:
            return 0.0
        
        # Compute variance of accuracy in sliding windows
        window = 50
        variances = []
        
        for i in range(window, len(perf_history), 10):
            window_data = perf_history[i-window:i]
            accuracies = [r['result']['success'] for r in window_data]
            variances.append(np.var(accuracies))
        
        # Stability is inverse of average variance
        avg_variance = np.mean(variances) if variances else 1.0
        return 1.0 / (1.0 + avg_variance)
    
    def _compute_adaptation_rate(self, results: Dict) -> float:
        """Compute adaptation rate for adaptive algorithms."""
        # This would analyze parameter changes over time
        # For now, return a placeholder
        return 0.0
    
    def _compute_avg_collective_accuracy(self, perf_history: List) -> float:
        """Compute average collective accuracy."""
        collective_accs = [
            r['result'].get('collective_accuracy', 0) 
            for r in perf_history 
            if 'collective_accuracy' in r['result']
        ]
        return np.mean(collective_accs) if collective_accs else 0.0
    
    def _compute_accuracy_consistency(self, perf_history: List) -> float:
        """Compute consistency of accuracy (1 - coefficient of variation)."""
        if len(perf_history) < 10:
            return 0.0
        
        accuracies = [r['result']['success'] for r in perf_history]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        if mean_acc > 0:
            cv = std_acc / mean_acc
            return 1.0 - min(cv, 1.0)
        return 0.0
    
    def run_experiments(self, parallel: bool = True, num_workers: int = None):
        """Run all experiments."""
        experiments = self.generate_experiment_list()
        total_experiments = len(experiments)
        
        print(f"Running {total_experiments} experiments ({self.scale})")
        print(f"Output directory: {self.output_dir}")
        
        # Save experiment list
        with open(self.output_dir / 'experiment_list.json', 'w') as f:
            json.dump(experiments, f, indent=2)
        
        if parallel and total_experiments > 10:
            # Run in parallel
            if num_workers is None:
                num_workers = mp.cpu_count() - 1
            
            print(f"Running experiments in parallel with {num_workers} workers")
            
            with mp.Pool(processes=num_workers) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(self.run_single_experiment, experiments)):
                    if result is not None:
                        results.append(result)
                        self.results.append(result)
                    
                    # Progress update
                    if (i + 1) % 50 == 0:
                        print(f"Progress: {i+1}/{total_experiments} experiments completed")
                        self._save_intermediate_results()
        else:
            # Run sequentially
            print("Running experiments sequentially")
            
            for i, exp in enumerate(experiments):
                print(f"\nExperiment {i+1}/{total_experiments}: {exp['algorithm_name']} "
                      f"(n={exp['num_cameras']}, f={exp['frequency']}, run={exp['run']})")
                
                result = self.run_single_experiment(exp)
                if result is not None:
                    self.results.append(result)
                
                # Save intermediate results
                if (i + 1) % 10 == 0:
                    self._save_intermediate_results()
        
        # Save final results
        self._save_final_results()
        
    def _save_intermediate_results(self):
        """Save intermediate results."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_dir / 'raw_results' / 'intermediate_results.csv', index=False)
    
    def _save_final_results(self):
        """Save final results and generate summary statistics."""
        if not self.results:
            print("No results to save!")
            return
        
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'raw_results' / 'all_results.csv', index=False)
        
        # Save as JSON for flexibility
        with open(self.output_dir / 'raw_results' / 'all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate aggregated results
        self._generate_aggregated_results(df)
        
        # Generate summary report
        self._generate_summary_report(df)
    
    def _generate_aggregated_results(self, df: pd.DataFrame):
        """Generate aggregated results by algorithm and parameters."""
        # Aggregate by algorithm
        algo_summary = df.groupby('algorithm_name').agg({
            'overall_accuracy': ['mean', 'std', 'min', 'max'],
            'energy_violations': ['sum', 'mean'],
            'accuracy_violations': ['sum', 'mean'],
            'avg_cameras_per_event': ['mean', 'std'],
            'jains_fairness': ['mean', 'std'],
            'runtime': ['mean', 'std']
        }).round(4)
        
        algo_summary.to_csv(self.output_dir / 'aggregated_results' / 'algorithm_summary.csv')
        
        # Aggregate by frequency
        freq_summary = df.groupby(['algorithm_name', 'frequency']).agg({
            'overall_accuracy': ['mean', 'std'],
            'energy_violations': 'sum',
            'convergence_time': 'mean'
        }).round(4)
        
        freq_summary.to_csv(self.output_dir / 'aggregated_results' / 'frequency_summary.csv')
        
        # Aggregate by network size
        size_summary = df.groupby(['algorithm_name', 'num_cameras']).agg({
            'overall_accuracy': ['mean', 'std'],
            'runtime': 'mean',
            'jains_fairness': 'mean'
        }).round(4)
        
        size_summary.to_csv(self.output_dir / 'aggregated_results' / 'size_summary.csv')
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate comprehensive summary report."""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULL-SCALE EXPERIMENTAL RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Experiment Scale: {self.scale}\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Successful Experiments: {len(df[df['overall_accuracy'] > 0])}\n")
            f.write(f"Total Runtime: {df['runtime'].sum():.2f} seconds\n\n")
            
            # Best performing algorithms
            f.write("TOP PERFORMING ALGORITHMS (by accuracy):\n")
            f.write("-" * 40 + "\n")
            
            algo_perf = df.groupby('algorithm_name')['overall_accuracy'].agg(['mean', 'std'])
            algo_perf = algo_perf.sort_values('mean', ascending=False)
            
            for algo, row in algo_perf.iterrows():
                f.write(f"{algo:20s}: {row['mean']:.3f} ± {row['std']:.3f}\n")
            
            # Energy efficiency
            f.write("\n\nENERGY EFFICIENCY (violations per experiment):\n")
            f.write("-" * 40 + "\n")
            
            energy_eff = df.groupby('algorithm_name')['energy_violations'].mean()
            energy_eff = energy_eff.sort_values()
            
            for algo, violations in energy_eff.items():
                f.write(f"{algo:20s}: {violations:.2f}\n")
            
            # Fairness
            f.write("\n\nFAIRNESS (Jain's Index):\n")
            f.write("-" * 40 + "\n")
            
            fairness = df.groupby('algorithm_name')['jains_fairness'].mean()
            fairness = fairness.sort_values(ascending=False)
            
            for algo, jains in fairness.items():
                f.write(f"{algo:20s}: {jains:.3f}\n")
            
            # Best by scenario
            f.write("\n\nBEST ALGORITHM BY SCENARIO:\n")
            f.write("-" * 40 + "\n")
            
            # By frequency
            for freq in sorted(df['frequency'].unique()):
                freq_data = df[df['frequency'] == freq]
                best_algo = freq_data.groupby('algorithm_name')['overall_accuracy'].mean().idxmax()
                best_acc = freq_data.groupby('algorithm_name')['overall_accuracy'].mean().max()
                f.write(f"Frequency {freq:4.2f}: {best_algo} ({best_acc:.3f})\n")
            
            # Statistical significance would be computed here
            f.write("\n\nNote: Statistical significance testing pending\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full-scale experiments for research paper"
    )
    parser.add_argument(
        '--scale',
        type=str,
        choices=['small_scale', 'medium_scale', 'full_scale'],
        default='medium_scale',
        help='Scale of experiments to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Run experiments in parallel'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'research_results_{args.scale}_{timestamp}')
    else:
        output_dir = Path(args.output_dir)
    
    # Run experiments
    runner = FullExperimentRunner(output_dir, args.scale)
    runner.run_experiments(parallel=args.parallel, num_workers=args.workers)
    
    print(f"\nExperiments complete! Results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Generate figures: python src/visualization/research_plots.py", 
          f"{output_dir}/raw_results/all_results.csv {output_dir}/figures")
    print("2. Run statistical tests: python experiments/statistical_analysis.py",
          f"{output_dir}/raw_results/all_results.csv")


if __name__ == "__main__":
    main()