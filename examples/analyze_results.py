#!/usr/bin/env python3
"""
Example: Analyze simulation results in detail.

This script loads simulation results and provides detailed analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(filename='results.json'):
    """Load simulation results from file."""
    with open(filename, 'r') as f:
        return json.load(f)


def analyze_classification_performance(results):
    """Analyze classification performance over time."""
    performance_history = results.get('performance_history', [])
    
    if not performance_history:
        print("No performance history available.")
        return
        
    # Extract metrics over time
    timestamps = []
    successes = []
    accuracies = []
    num_cameras = []
    
    # Calculate running accuracy
    total = 0
    correct = 0
    
    for event in performance_history:
        result = event['result']
        timestamps.append(event['timestamp'])
        
        total += 1
        if result['success']:
            correct += 1
            
        successes.append(result['success'])
        accuracies.append(correct / total)
        num_cameras.append(len(result['selected_cameras']))
        
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Classification success
    ax = axes[0]
    ax.scatter(timestamps, successes, alpha=0.6, s=20)
    ax.set_ylabel('Success')
    ax.set_title('Classification Success Over Time')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Running accuracy
    ax = axes[1]
    ax.plot(timestamps, accuracies, 'b-', linewidth=2)
    ax.axhline(y=results['network_stats']['accuracy'], color='r', 
               linestyle='--', label='Overall Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_title('Running Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Number of cameras used
    ax = axes[2]
    ax.plot(timestamps, num_cameras, 'g-', linewidth=1)
    ax.set_ylabel('Number of Cameras')
    ax.set_xlabel('Time')
    ax.set_title('Cameras per Classification')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_performance.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("\nClassification Performance Analysis")
    print("=" * 40)
    print(f"Total Classifications: {total}")
    print(f"Successful: {correct}")
    print(f"Failed: {total - correct}")
    print(f"Overall Accuracy: {correct/total:.3f}")
    print(f"Average Cameras per Classification: {np.mean(num_cameras):.2f}")
    print(f"Min Cameras Used: {min(num_cameras)}")
    print(f"Max Cameras Used: {max(num_cameras)}")


def analyze_energy_patterns(results):
    """Analyze energy consumption and harvesting patterns."""
    energy_history = results.get('energy_history', [])
    
    if not energy_history:
        print("No energy history available.")
        return
        
    # Extract data
    timestamps = [e['timestamp'] for e in energy_history]
    avg_energies = [e['avg_energy'] for e in energy_history]
    min_energies = [e['min_energy'] for e in energy_history]
    max_energies = [e['max_energy'] for e in energy_history]
    
    # Calculate energy spread
    energy_spread = [max_e - min_e for max_e, min_e in zip(max_energies, min_energies)]
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Energy levels
    ax = axes[0]
    ax.plot(timestamps, avg_energies, 'b-', label='Average', linewidth=2)
    ax.fill_between(timestamps, min_energies, max_energies, 
                    alpha=0.3, color='blue')
    ax.set_ylabel('Energy Level')
    ax.set_title('Camera Energy Levels Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=800, color='g', linestyle='--', alpha=0.5, label='High Threshold')
    ax.axhline(y=300, color='r', linestyle='--', alpha=0.5, label='Low Threshold')
    
    # Plot 2: Energy spread
    ax = axes[1]
    ax.plot(timestamps, energy_spread, 'r-', linewidth=2)
    ax.set_ylabel('Energy Spread')
    ax.set_xlabel('Time')
    ax.set_title('Energy Disparity Among Cameras')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_patterns.png', dpi=150)
    plt.show()
    
    # Calculate statistics
    print("\nEnergy Pattern Analysis")
    print("=" * 40)
    print(f"Initial Average Energy: {avg_energies[0]:.1f}")
    print(f"Final Average Energy: {avg_energies[-1]:.1f}")
    print(f"Energy Change: {avg_energies[-1] - avg_energies[0]:.1f}")
    print(f"Average Energy Spread: {np.mean(energy_spread):.1f}")
    print(f"Max Energy Spread: {max(energy_spread):.1f}")


def analyze_algorithm_behavior(results):
    """Analyze algorithm-specific behavior."""
    algorithm = results.get('algorithm_type', 'unknown')
    stats = results['network_stats']
    
    print(f"\nAlgorithm Behavior Analysis: {algorithm.upper()}")
    print("=" * 40)
    
    # General metrics
    print(f"Total Classifications: {stats['total_classifications']}")
    print(f"Accuracy: {stats['accuracy']:.3f}")
    print(f"Energy Violations: {stats['energy_violations']}")
    print(f"Accuracy Violations: {stats['accuracy_violations']}")
    
    # Algorithm-specific analysis
    if algorithm == 'fixed':
        print("\nFixed Frequency Algorithm:")
        print("- Uses round-robin scheduling")
        print("- Deterministic camera selection")
        print(f"- Average cameras per class: {stats['avg_cameras_per_classification']:.2f}")
        
    elif algorithm == 'variable':
        print("\nVariable Frequency Algorithm:")
        print("- Uses subclass rotation")
        print("- Maintains energy diversity")
        
    elif algorithm == 'unknown':
        print("\nUnknown Frequency Algorithm:")
        print("- Uses game-theoretic decisions")
        print("- Converges to Nash equilibrium")
        
        # Look for participation rate patterns
        if 'avg_participation_rate' in stats:
            print(f"- Average participation rate: {stats['avg_participation_rate']:.3f}")


def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"\nAlgorithm: {results['algorithm_type']}")
    print(f"Duration: {results['duration']} time units")
    print(f"Classification Frequency: {results['classification_frequency']} events/time")
    print(f"Total Events: {results['total_events']}")
    
    # Performance metrics
    stats = results['network_stats']
    print(f"\nPerformance Metrics:")
    print(f"  Overall Accuracy: {stats['accuracy']:.3f}")
    print(f"  Recent Accuracy: {stats['recent_accuracy']:.3f}")
    print(f"  Success Rate: {stats['successful_classifications']}/{stats['total_classifications']}")
    
    # Energy metrics
    print(f"\nEnergy Metrics:")
    print(f"  Average Energy: {stats['avg_energy']:.1f}")
    print(f"  Average Accuracy (energy-based): {stats['avg_accuracy']:.3f}")
    
    # Efficiency metrics
    print(f"\nEfficiency Metrics:")
    print(f"  Cameras per Classification: {stats['avg_cameras_per_classification']:.2f}")
    print(f"  Energy Violations: {stats['energy_violations']}")
    print(f"  Accuracy Violations: {stats['accuracy_violations']}")
    
    # Calculate energy efficiency
    if results['performance_history']:
        total_cameras_used = sum(len(e['result']['selected_cameras']) 
                               for e in results['performance_history'])
        energy_efficiency = stats['successful_classifications'] / total_cameras_used
        print(f"  Energy Efficiency: {energy_efficiency:.3f} successes/camera-use")


if __name__ == "__main__":
    # Check if results file is provided
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'results.json'
        
    print(f"Loading results from: {results_file}")
    
    try:
        results = load_results(results_file)
        
        # Generate summary report
        generate_summary_report(results)
        
        # Analyze classification performance
        analyze_classification_performance(results)
        
        # Analyze energy patterns
        analyze_energy_patterns(results)
        
        # Analyze algorithm behavior
        analyze_algorithm_behavior(results)
        
        print("\nAnalysis complete! Check generated plots.")
        
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        print("Run a simulation first using: python3 run_simulation.py")
    except Exception as e:
        print(f"Error analyzing results: {e}")