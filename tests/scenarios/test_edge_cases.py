#!/usr/bin/env python3
"""
Edge case testing for multi-camera classification system.

Tests extreme scenarios and boundary conditions:
1. Very low battery scenarios
2. High frequency classification
3. Single camera scenarios
4. All cameras depleted
5. Extreme accuracy thresholds
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import create_network_from_config, run_simulation, load_config
from src.core.network import CameraNetwork, NetworkConfig
from src.core.energy_model import EnergyParameters
from src.core.accuracy_model import AccuracyParameters
from src.utils.logger import setup_logging


def create_edge_case_configs():
    """Create configurations for various edge cases."""
    
    base_config = {
        'network': {
            'num_cameras': 10,
            'num_classes': 3,
            'num_objects': 5
        },
        'energy': {
            'battery_capacity': 1000,
            'recharge_rate': 10,
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
            'non_participation_penalty': 0.8,
            'discount_factor': 0.9
        }
    }
    
    edge_cases = {}
    
    # Edge Case 1: Very low initial battery
    edge_cases['low_battery'] = {
        **base_config,
        'test_params': {
            'initial_energy': 150,  # Just above min operational
            'duration': 500,
            'frequency': 0.1
        }
    }
    
    # Edge Case 2: High frequency classification
    edge_cases['high_frequency'] = {
        **base_config,
        'test_params': {
            'initial_energy': None,
            'duration': 500,
            'frequency': 0.5  # Very high frequency
        }
    }
    
    # Edge Case 3: Single camera scenario
    single_cam = base_config.copy()
    single_cam['network']['num_cameras'] = 1
    single_cam['network']['num_classes'] = 1
    edge_cases['single_camera'] = {
        **single_cam,
        'test_params': {
            'initial_energy': None,
            'duration': 500,
            'frequency': 0.1
        }
    }
    
    # Edge Case 4: High energy cost
    high_cost = base_config.copy()
    high_cost['energy']['classification_cost'] = 200  # Very high cost
    edge_cases['high_energy_cost'] = {
        **high_cost,
        'test_params': {
            'initial_energy': None,
            'duration': 500,
            'frequency': 0.1
        }
    }
    
    # Edge Case 5: Extreme accuracy threshold
    edge_cases['extreme_accuracy'] = {
        **base_config,
        'test_params': {
            'initial_energy': None,
            'duration': 500,
            'frequency': 0.1,
            'min_accuracy_threshold': 0.99  # Nearly impossible
        }
    }
    
    # Edge Case 6: No recharge
    no_recharge = base_config.copy()
    no_recharge['energy']['recharge_rate'] = 0  # No energy harvesting
    edge_cases['no_recharge'] = {
        **no_recharge,
        'test_params': {
            'initial_energy': None,
            'duration': 500,
            'frequency': 0.05
        }
    }
    
    # Edge Case 7: Many cameras, few classes
    many_cams = base_config.copy()
    many_cams['network']['num_cameras'] = 50
    many_cams['network']['num_classes'] = 2
    edge_cases['many_cameras'] = {
        **many_cams,
        'test_params': {
            'initial_energy': None,
            'duration': 300,
            'frequency': 0.2
        }
    }
    
    return edge_cases


def run_edge_case_test(case_name, config, test_params, output_dir):
    """Run a single edge case test."""
    print(f"\n{'='*60}")
    print(f"Running edge case: {case_name}")
    print(f"{'='*60}")
    
    # Extract test parameters
    initial_energy = test_params.get('initial_energy')
    duration = test_params['duration']
    frequency = test_params['frequency']
    min_accuracy = test_params.get('min_accuracy_threshold', 0.8)
    
    # Create network
    network = create_network_from_config(config)
    
    # Set initial energy if specified
    if initial_energy is not None:
        for camera in network.cameras:
            camera.state.energy = initial_energy
            camera.energy_history = [initial_energy]
    
    results = {}
    algorithms = ['fixed', 'variable', 'unknown']
    
    for algo in algorithms:
        print(f"\n  Testing {algo} algorithm...")
        
        # Reset network for each algorithm
        network.reset()
        if initial_energy is not None:
            for camera in network.cameras:
                camera.state.energy = initial_energy
                camera.energy_history = [initial_energy]
        
        try:
            # Run simulation
            result = run_simulation(
                network,
                algo,
                duration,
                time_step=1.0,
                classification_frequency=frequency,
                visualize=False
            )
            
            results[algo] = result
            
            # Print key metrics
            stats = result['network_stats']
            print(f"    Accuracy: {stats.get('accuracy', 0):.3f}")
            print(f"    Energy violations: {stats.get('energy_violations', 0)}")
            print(f"    Accuracy violations: {stats.get('accuracy_violations', 0)}")
            print(f"    Total classifications: {stats.get('total_classifications', 0)}")
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[algo] = {'error': str(e)}
    
    # Save results
    case_dir = output_dir / case_name
    case_dir.mkdir(exist_ok=True)
    
    with open(case_dir / 'results.json', 'w') as f:
        json.dump({
            'case_name': case_name,
            'config': config,
            'test_params': test_params,
            'results': results
        }, f, indent=2, default=str)
    
    # Create visualization
    create_edge_case_plot(case_name, results, case_dir)
    
    return results


def create_edge_case_plot(case_name, results, output_dir):
    """Create visualization for edge case results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    algorithms = []
    accuracies = []
    violations = []
    
    for algo, result in results.items():
        if 'error' not in result:
            algorithms.append(algo)
            stats = result.get('network_stats', {})
            accuracies.append(stats.get('accuracy', 0))
            total_violations = (stats.get('energy_violations', 0) + 
                              stats.get('accuracy_violations', 0))
            violations.append(total_violations)
    
    if algorithms:
        # Accuracy comparison
        x = np.arange(len(algorithms))
        bars1 = ax1.bar(x, accuracies, color=['blue', 'green', 'red'][:len(algorithms)])
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{case_name} - Classification Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.capitalize() for a in algorithms])
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Violations
        bars2 = ax2.bar(x, violations, color=['orange', 'purple', 'brown'][:len(algorithms)])
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Total Violations')
        ax2.set_title(f'{case_name} - Constraint Violations')
        ax2.set_xticks(x)
        ax2.set_xticklabels([a.capitalize() for a in algorithms])
        
        # Add value labels
        for bar, viol in zip(bars2, violations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(viol)}', ha='center', va='bottom')
    
    plt.suptitle(f'Edge Case Results: {case_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_edge_case_report(all_results, output_dir):
    """Generate summary report for all edge cases."""
    report_path = output_dir / 'edge_case_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EDGE CASE TESTING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for case_name, case_results in all_results.items():
            f.write(f"\n{'-'*60}\n")
            f.write(f"CASE: {case_name.replace('_', ' ').upper()}\n")
            f.write(f"{'-'*60}\n")
            
            # Write test parameters
            test_params = case_results.get('test_params', {})
            f.write("Test Parameters:\n")
            for param, value in test_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            # Write results for each algorithm
            results = case_results.get('results', {})
            for algo, result in results.items():
                f.write(f"\n{algo.upper()} Algorithm:\n")
                
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    stats = result.get('network_stats', {})
                    f.write(f"  Accuracy: {stats.get('accuracy', 0):.3f}\n")
                    f.write(f"  Energy Violations: {stats.get('energy_violations', 0)}\n")
                    f.write(f"  Accuracy Violations: {stats.get('accuracy_violations', 0)}\n")
                    f.write(f"  Total Classifications: {stats.get('total_classifications', 0)}\n")
                    f.write(
                        f"  Success Rate: "
                        f"{stats.get('successful_classifications', 0) / max(1, stats.get('total_classifications', 1)):.1%}\n"
                    )
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        
        # Analyze patterns
        f.write("\nAlgorithm Robustness:\n")
        for algo in ['fixed', 'variable', 'unknown']:
            failures = sum(1 for case_results in all_results.values() 
                          if 'error' in case_results.get('results', {}).get(algo, {}))
            f.write(f"  {algo.capitalize()}: {len(all_results) - failures}/{len(all_results)} cases successful\n")
        
        f.write("\nMost Challenging Cases:\n")
        for case_name, case_results in all_results.items():
            avg_accuracy = []
            for algo, result in case_results.get('results', {}).items():
                if 'error' not in result:
                    stats = result.get('network_stats', {})
                    avg_accuracy.append(stats.get('accuracy', 0))
            
            if avg_accuracy and np.mean(avg_accuracy) < 0.3:
                f.write(f"  - {case_name}: Average accuracy {np.mean(avg_accuracy):.3f}\n")
        
        f.write("\n" + "="*80 + "\n")


def main():
    """Run all edge case tests."""
    # Setup
    output_dir = Path(f"test_results_edge_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    setup_logging('INFO')
    
    print("="*80)
    print("MULTI-CAMERA CLASSIFICATION - EDGE CASE TESTING")
    print("="*80)
    
    # Get edge case configurations
    edge_cases = create_edge_case_configs()
    all_results = {}
    
    # Run each edge case
    for case_name, case_config in edge_cases.items():
        test_params = case_config.pop('test_params')
        results = run_edge_case_test(case_name, case_config, test_params, output_dir)
        
        all_results[case_name] = {
            'config': case_config,
            'test_params': test_params,
            'results': results
        }
    
    # Generate summary report
    generate_edge_case_report(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("EDGE CASE TESTING COMPLETED")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()