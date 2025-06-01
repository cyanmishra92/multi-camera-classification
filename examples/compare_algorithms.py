#!/usr/bin/env python3
"""
Example: Compare performance of different algorithms.

This script runs simulations with all three algorithms and compares their performance.
"""

import sys
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_simulation(algorithm, duration=5000, frequency=0.1):
    """Run a single simulation and return results."""
    output_file = f"results_{algorithm}_comparison.json"
    
    # Run simulation
    result = subprocess.run([
        sys.executable, 'run_simulation.py',
        '--algorithm', algorithm,
        '--duration', str(duration),
        '--frequency', str(frequency),
        '--output', output_file,
        '--log-level', 'WARNING'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {algorithm}: {result.stderr}")
        return None
        
    # Load and return results
    with open(output_file, 'r') as f:
        return json.load(f)


def compare_algorithms():
    """Compare all three algorithms."""
    algorithms = ['fixed', 'variable', 'unknown']
    results = {}
    
    print("Running algorithm comparison...")
    print("-" * 50)
    
    # Run simulations
    for algo in algorithms:
        print(f"Running {algo} algorithm...")
        results[algo] = run_simulation(algo)
        
    # Extract metrics
    metrics = {
        'accuracy': {},
        'energy_efficiency': {},
        'violations': {}
    }
    
    for algo, result in results.items():
        if result:
            stats = result['network_stats']
            metrics['accuracy'][algo] = stats['accuracy']
            metrics['energy_efficiency'][algo] = stats['avg_cameras_per_classification']
            metrics['violations'][algo] = stats['energy_violations'] + stats['accuracy_violations']
            
    # Print comparison
    print("\n" + "=" * 50)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 50)
    
    print("\nAccuracy:")
    for algo, acc in metrics['accuracy'].items():
        print(f"  {algo:>10}: {acc:.3f}")
        
    print("\nEnergy Efficiency (cameras per classification):")
    for algo, eff in metrics['energy_efficiency'].items():
        print(f"  {algo:>10}: {eff:.2f}")
        
    print("\nTotal Violations:")
    for algo, viol in metrics['violations'].items():
        print(f"  {algo:>10}: {viol}")
        
    # Create visualization
    create_comparison_plots(metrics)
    
    return results, metrics


def create_comparison_plots(metrics):
    """Create comparison plots."""
    algorithms = list(metrics['accuracy'].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy plot
    ax = axes[0]
    accuracies = [metrics['accuracy'][algo] for algo in algorithms]
    bars = ax.bar(algorithms, accuracies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Energy efficiency plot
    ax = axes[1]
    efficiencies = [metrics['energy_efficiency'][algo] for algo in algorithms]
    bars = ax.bar(algorithms, efficiencies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Cameras per Classification')
    ax.set_title('Energy Efficiency')
    
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.2f}', ha='center', va='bottom')
    
    # Violations plot
    ax = axes[2]
    violations = [metrics['violations'][algo] for algo in algorithms]
    bars = ax.bar(algorithms, violations, color=['blue', 'green', 'red'])
    ax.set_ylabel('Total Violations')
    ax.set_title('Constraint Violations')
    
    for bar, viol in zip(bars, violations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{viol}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150)
    print("\nPlots saved to: algorithm_comparison.png")
    plt.close()


def analyze_energy_dynamics(results):
    """Analyze energy dynamics over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (algo, result) in enumerate(results.items()):
        if result and 'energy_history' in result:
            ax = axes[idx]
            
            # Extract energy history
            energy_history = result['energy_history'][:100]  # First 100 time steps
            timestamps = [e['timestamp'] for e in energy_history]
            avg_energies = [e['avg_energy'] for e in energy_history]
            min_energies = [e['min_energy'] for e in energy_history]
            max_energies = [e['max_energy'] for e in energy_history]
            
            # Plot
            ax.plot(timestamps, avg_energies, 'b-', label='Average', linewidth=2)
            ax.fill_between(timestamps, min_energies, max_energies, 
                          alpha=0.3, color='blue', label='Min-Max Range')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy Level')
            ax.set_title(f'{algo.capitalize()} Algorithm')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig('energy_dynamics.png', dpi=150)
    print("Energy dynamics saved to: energy_dynamics.png")
    plt.close()


if __name__ == "__main__":
    # Run comparison
    results, metrics = compare_algorithms()
    
    # Analyze energy dynamics
    print("\nAnalyzing energy dynamics...")
    analyze_energy_dynamics(results)
    
    print("\nComparison complete!")