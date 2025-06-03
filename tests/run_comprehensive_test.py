#!/usr/bin/env python3
"""
Comprehensive test script for multi-camera classification system.

This script:
1. Runs simulations with all three algorithms
2. Generates comparative visualizations
3. Saves all results in an organized structure
4. Creates a summary report
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import create_network_from_config, run_simulation, load_config
from src.utils.logger import setup_logging


def create_results_directory():
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"test_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (results_dir / "data").mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    (results_dir / "reports").mkdir(exist_ok=True)
    
    return results_dir


def plot_energy_dynamics(results_dict, output_dir):
    """Plot energy dynamics for all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (algo_name, results) in enumerate(results_dict.items()):
        if idx >= 3:
            break
            
        ax = axes[idx]
        energy_history = results['energy_history']
        
        # Extract data
        timestamps = [e['timestamp'] for e in energy_history]
        avg_energies = [e['avg_energy'] for e in energy_history]
        min_energies = [e['min_energy'] for e in energy_history]
        max_energies = [e['max_energy'] for e in energy_history]
        
        # Plot
        ax.plot(timestamps, avg_energies, 'b-', label='Average', linewidth=2)
        ax.fill_between(timestamps, min_energies, max_energies, alpha=0.3, label='Min-Max Range')
        
        ax.set_title(f'{algo_name.capitalize()} Algorithm - Energy Dynamics')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Combined plot in 4th subplot
    ax = axes[3]
    for algo_name, results in results_dict.items():
        energy_history = results['energy_history']
        timestamps = [e['timestamp'] for e in energy_history]
        avg_energies = [e['avg_energy'] for e in energy_history]
        ax.plot(timestamps, avg_energies, label=algo_name.capitalize(), linewidth=2)
    
    ax.set_title('Energy Comparison - All Algorithms')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Energy Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "energy_dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_metrics(results_dict, output_dir):
    """Plot accuracy metrics comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract metrics
    algorithms = list(results_dict.keys())
    accuracies = [results_dict[algo]['network_stats']['accuracy'] for algo in algorithms]
    energy_violations = [results_dict[algo]['network_stats']['energy_violations'] for algo in algorithms]
    accuracy_violations = [results_dict[algo]['network_stats']['accuracy_violations'] for algo in algorithms]
    
    # Bar plot for accuracy
    x = np.arange(len(algorithms))
    bars = ax1.bar(x, accuracies, color=['blue', 'green', 'red'])
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Overall Accuracy')
    ax1.set_title('Classification Accuracy by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.capitalize() for a in algorithms])
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Violations comparison
    width = 0.35
    x = np.arange(len(algorithms))
    bars1 = ax2.bar(x - width/2, energy_violations, width, label='Energy Violations', color='orange')
    bars2 = ax2.bar(x + width/2, accuracy_violations, width, label='Accuracy Violations', color='purple')
    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Number of Violations')
    ax2.set_title('Constraint Violations by Algorithm')
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.capitalize() for a in algorithms])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "accuracy_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_timeline(results_dict, output_dir):
    """Plot performance metrics over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for algo_name, results in results_dict.items():
        perf_history = results['performance_history']
        
        # Extract classification events
        timestamps = [p['timestamp'] for p in perf_history]
        successes = [1 if p['result']['success'] else 0 for p in perf_history]
        num_cameras = [len(p['result']['selected_cameras']) for p in perf_history]
        
        # Calculate rolling accuracy
        window_size = min(10, len(successes))
        rolling_accuracy = []
        for i in range(len(successes)):
            start = max(0, i - window_size + 1)
            window = successes[start:i+1]
            rolling_accuracy.append(sum(window) / len(window))
        
        # Plot rolling accuracy
        axes[0].plot(timestamps, rolling_accuracy, label=algo_name.capitalize(), linewidth=2)
        
        # Plot number of cameras used
        axes[1].scatter(timestamps, num_cameras, label=algo_name.capitalize(), alpha=0.6, s=20)
    
    axes[0].set_ylabel('Rolling Accuracy (10 events)')
    axes[0].set_title('Classification Accuracy Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Number of Cameras Used')
    axes[1].set_title('Camera Participation Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "performance_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(results_dict, output_dir, config):
    """Generate a comprehensive summary report."""
    report_path = output_dir / "reports" / "test_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-CAMERA CLASSIFICATION SYSTEM - COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        # Configuration summary
        f.write("CONFIGURATION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of Cameras: {config['network']['num_cameras']}\n")
        f.write(f"Number of Classes: {config['network']['num_classes']}\n")
        f.write(f"Battery Capacity: {config['energy']['battery_capacity']}\n")
        f.write(f"Recharge Rate: {config['energy']['recharge_rate']}\n")
        f.write(f"Classification Cost: {config['energy']['classification_cost']}\n")
        f.write(f"Max Accuracy: {config['accuracy']['max_accuracy']}\n\n")
        
        # Results summary for each algorithm
        f.write("ALGORITHM PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n\n")
        
        for algo_name, results in results_dict.items():
            stats = results['network_stats']
            f.write(f"{algo_name.upper()} ALGORITHM:\n")
            f.write(f"  Total Classifications: {stats['total_classifications']}\n")
            f.write(f"  Successful Classifications: {stats['successful_classifications']}\n")
            f.write(f"  Overall Accuracy: {stats['accuracy']:.3f}\n")
            f.write(f"  Energy Violations: {stats['energy_violations']}\n")
            f.write(f"  Accuracy Violations: {stats['accuracy_violations']}\n")
            f.write(f"  Avg Cameras per Classification: {stats['avg_cameras_per_classification']:.2f}\n")
            f.write(f"  Final Avg Energy: {stats['avg_energy']:.1f}\n\n")
        
        # Comparative analysis
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        accuracies = {algo: results['network_stats']['accuracy'] for algo, results in results_dict.items()}
        best_accuracy = max(accuracies.items(), key=lambda x: x[1])
        f.write(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]:.3f})\n")
        
        violations = {algo: results['network_stats']['energy_violations'] + 
                           results['network_stats']['accuracy_violations'] 
                      for algo, results in results_dict.items()}
        least_violations = min(violations.items(), key=lambda x: x[1])
        f.write(f"Least Violations: {least_violations[0]} ({least_violations[1]} total)\n")
        
        avg_participation = {algo: results['network_stats']['avg_cameras_per_classification'] 
                            for algo, results in results_dict.items()}
        most_efficient = min(avg_participation.items(), key=lambda x: x[1])
        f.write(f"Most Energy Efficient: {most_efficient[0]} ({most_efficient[1]:.2f} cameras/event)\n\n")
        
        f.write("TEST COMPLETED SUCCESSFULLY\n")
        f.write("=" * 80 + "\n")


def run_comprehensive_test(duration=1000, frequency=0.1, config_path='configs/default_config.yaml'):
    """Run comprehensive test suite."""
    # Setup
    results_dir = create_results_directory()
    setup_logging('INFO')
    
    print(f"Starting comprehensive test suite...")
    print(f"Results will be saved to: {results_dir}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Save configuration
    with open(results_dir / "data" / "test_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Algorithms to test
    algorithms = ['fixed', 'variable', 'unknown']
    results_dict = {}
    
    # Run simulations
    for algo in algorithms:
        print(f"\nRunning {algo} algorithm simulation...")
        
        # Create fresh network for each algorithm
        network = create_network_from_config(config)
        
        # Run simulation
        results = run_simulation(
            network,
            algo,
            duration,
            time_step=1.0,
            classification_frequency=frequency,
            visualize=False
        )
        
        # Save individual results
        with open(results_dir / "data" / f"results_{algo}.json", 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_to_json_serializable(results), f, indent=2)
        
        results_dict[algo] = results
        print(f"  Accuracy: {results['network_stats']['accuracy']:.3f}")
        print(f"  Violations: {results['network_stats']['energy_violations']} energy, "
              f"{results['network_stats']['accuracy_violations']} accuracy")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_energy_dynamics(results_dict, results_dir)
    plot_accuracy_metrics(results_dict, results_dir)
    plot_performance_timeline(results_dict, results_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(results_dict, results_dir, config)
    
    # Save combined results
    with open(results_dir / "data" / "all_results.json", 'w') as f:
        json.dump({
            'test_config': {
                'duration': duration,
                'frequency': frequency,
                'config_path': config_path,
                'timestamp': datetime.now().isoformat()
            },
            'results': results_dict
        }, f, indent=2, default=str)
    
    print(f"\nTest completed successfully!")
    print(f"Results saved to: {results_dir}")
    print(f"  - Data files: {results_dir}/data/")
    print(f"  - Plots: {results_dir}/plots/")
    print(f"  - Report: {results_dir}/reports/test_summary.txt")
    
    return results_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive test suite for multi-camera classification system"
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=1000,
        help='Simulation duration (default: 1000)'
    )
    parser.add_argument(
        '--frequency',
        type=float,
        default=0.1,
        help='Classification frequency (default: 0.1)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    run_comprehensive_test(
        duration=args.duration,
        frequency=args.frequency,
        config_path=args.config
    )


if __name__ == "__main__":
    main()