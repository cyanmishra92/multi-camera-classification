#!/usr/bin/env python3
"""
Comprehensive algorithm comparison for multi-camera classification system.

Compares all algorithms across multiple scenarios:
1. Different frequency regimes
2. Various energy conditions
3. Network sizes
4. Real-world scenarios
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
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import create_network_from_config, run_simulation
from src.utils.logger import setup_logging


def create_comparison_scenarios():
    """Create diverse scenarios for algorithm comparison."""
    
    base_config = {
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
    
    scenarios = {}
    
    # Scenario 1: Frequency regimes (as per paper)
    scenarios['high_frequency'] = {
        **base_config,
        'network': {
            'num_cameras': 15,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.2,  # High frequency (> 1/Δ)
            'description': 'High frequency classification (Algorithm 1 optimal)'
        }
    }
    
    scenarios['medium_frequency'] = {
        **base_config,
        'network': {
            'num_cameras': 15,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.05,  # Medium frequency
            'description': 'Medium frequency classification (Algorithm 2 optimal)'
        }
    }
    
    scenarios['low_frequency'] = {
        **base_config,
        'network': {
            'num_cameras': 15,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.02,  # Low/unknown frequency
            'description': 'Low/unknown frequency (Algorithm 3 optimal)'
        }
    }
    
    # Scenario 2: Energy scarcity
    energy_scarce = base_config.copy()
    energy_scarce['energy']['battery_capacity'] = 500
    energy_scarce['energy']['recharge_rate'] = 5
    scenarios['energy_scarce'] = {
        **energy_scarce,
        'network': {
            'num_cameras': 12,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.1,
            'description': 'Energy-scarce environment'
        }
    }
    
    # Scenario 3: Large network
    scenarios['large_network'] = {
        **base_config,
        'network': {
            'num_cameras': 50,
            'num_classes': 5,
            'num_objects': 10
        },
        'test_params': {
            'duration': 500,
            'frequency': 0.1,
            'description': 'Large camera network'
        }
    }
    
    # Scenario 4: High accuracy requirement
    high_accuracy = base_config.copy()
    scenarios['high_accuracy_req'] = {
        **high_accuracy,
        'network': {
            'num_cameras': 20,
            'num_classes': 4,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.1,
            'min_accuracy_threshold': 0.95,
            'description': 'High accuracy requirement (95%)'
        }
    }
    
    # Scenario 5: Variable frequency (simulated)
    scenarios['variable_frequency'] = {
        **base_config,
        'network': {
            'num_cameras': 15,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 1000,
            'frequency': 0.1,  # Will vary this during simulation
            'variable_frequency': True,
            'description': 'Variable classification frequency'
        }
    }
    
    # Scenario 6: Day/night cycle simulation
    day_night = base_config.copy()
    day_night['energy']['recharge_rate'] = 15  # Higher during "day"
    scenarios['day_night_cycle'] = {
        **day_night,
        'network': {
            'num_cameras': 15,
            'num_classes': 3,
            'num_objects': 5
        },
        'test_params': {
            'duration': 2000,
            'frequency': 0.1,
            'day_night_cycle': True,
            'description': 'Day/night energy harvesting cycle'
        }
    }
    
    return scenarios


def run_comparison_scenario(scenario_name, config, test_params, output_dir):
    """Run a comparison scenario for all algorithms."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Description: {test_params['description']}")
    print(f"{'='*60}")
    
    results = {}
    performance_data = []
    
    algorithms = ['fixed', 'variable', 'unknown']
    
    for algo in algorithms:
        print(f"\n  Running {algo} algorithm...")
        
        try:
            # Create network
            network = create_network_from_config(config)
            
            # Set algorithm with appropriate parameters
            min_accuracy = test_params.get('min_accuracy_threshold', 0.8)
            
            # Run simulation
            if test_params.get('variable_frequency'):
                # Simulate variable frequency
                result = run_variable_frequency_simulation(
                    network, algo, test_params['duration'], min_accuracy
                )
            elif test_params.get('day_night_cycle'):
                # Simulate day/night cycle
                result = run_day_night_simulation(
                    network, algo, test_params['duration'], 
                    test_params['frequency'], min_accuracy
                )
            else:
                # Standard simulation
                result = run_simulation(
                    network,
                    algo,
                    test_params['duration'],
                    time_step=1.0,
                    classification_frequency=test_params['frequency'],
                    visualize=False
                )
            
            results[algo] = result
            
            # Extract performance metrics
            stats = result['network_stats']
            performance_data.append({
                'algorithm': algo,
                'accuracy': stats.get('accuracy', 0),
                'energy_violations': stats.get('energy_violations', 0),
                'accuracy_violations': stats.get('accuracy_violations', 0),
                'total_classifications': stats.get('total_classifications', 0),
                'avg_cameras': stats.get('avg_cameras_per_classification', 0),
                'final_avg_energy': stats.get('avg_energy', 0)
            })
            
            print(f"    Accuracy: {stats.get('accuracy', 0):.3f}")
            print(f"    Violations: {stats.get('energy_violations', 0)} energy, "
                  f"{stats.get('accuracy_violations', 0)} accuracy")
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[algo] = {'error': str(e)}
            performance_data.append({
                'algorithm': algo,
                'error': str(e)
            })
    
    # Save results
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(scenario_dir / 'results.json', 'w') as f:
        json.dump({
            'scenario': scenario_name,
            'config': config,
            'test_params': test_params,
            'results': results
        }, f, indent=2, default=str)
    
    # Save performance summary
    pd.DataFrame(performance_data).to_csv(scenario_dir / 'performance_summary.csv', index=False)
    
    # Create visualizations
    create_scenario_visualizations(scenario_name, results, performance_data, scenario_dir)
    
    return results, performance_data


def run_variable_frequency_simulation(network, algo, duration, min_accuracy):
    """Simulate variable frequency classification."""
    # This is a simplified simulation - in reality would need custom implementation
    # For now, we'll use average of multiple frequencies
    frequencies = [0.05, 0.1, 0.2, 0.1, 0.05]  # Variable pattern
    
    # Run with average frequency
    avg_frequency = np.mean(frequencies)
    network.set_algorithm(algo, classification_frequency=avg_frequency, 
                         min_accuracy_threshold=min_accuracy)
    
    return run_simulation(
        network, algo, duration, time_step=1.0,
        classification_frequency=avg_frequency, visualize=False
    )


def run_day_night_simulation(network, algo, duration, frequency, min_accuracy):
    """Simulate day/night energy harvesting cycle."""
    # This is simplified - ideally would modify recharge rate during simulation
    network.set_algorithm(algo, classification_frequency=frequency,
                         min_accuracy_threshold=min_accuracy)
    
    return run_simulation(
        network, algo, duration, time_step=1.0,
        classification_frequency=frequency, visualize=False
    )


def create_scenario_visualizations(scenario_name, results, performance_data, output_dir):
    """Create visualizations for a scenario."""
    # Filter out error results
    valid_data = [d for d in performance_data if 'error' not in d]
    
    if not valid_data:
        return
    
    df = pd.DataFrame(valid_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy comparison
    algorithms = df['algorithm'].tolist()
    accuracies = df['accuracy'].tolist()
    
    bars1 = ax1.bar(algorithms, accuracies, color=['blue', 'green', 'red'])
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Algorithm Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Violations comparison
    energy_viols = df['energy_violations'].tolist()
    accuracy_viols = df['accuracy_violations'].tolist()
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, energy_viols, width, label='Energy', color='orange')
    bars3 = ax2.bar(x + width/2, accuracy_viols, width, label='Accuracy', color='purple')
    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Number of Violations')
    ax2.set_title('Constraint Violations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    
    # Efficiency metrics
    avg_cameras = df['avg_cameras'].tolist()
    bars4 = ax3.bar(algorithms, avg_cameras, color=['cyan', 'magenta', 'yellow'])
    ax3.set_ylabel('Average Cameras per Classification')
    ax3.set_title('Energy Efficiency')
    
    for bar, val in zip(bars4, avg_cameras):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # Final energy levels
    final_energies = df['final_avg_energy'].tolist()
    bars5 = ax4.bar(algorithms, final_energies, color=['darkblue', 'darkgreen', 'darkred'])
    ax4.set_ylabel('Average Energy Level')
    ax4.set_title('Final Energy State')
    
    for bar, val in zip(bars5, final_energies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom')
    
    plt.suptitle(f'Algorithm Comparison: {scenario_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_overall_comparison_matrix(all_results, output_dir):
    """Create a matrix visualization comparing all algorithms across scenarios."""
    # Prepare data
    scenarios = []
    algorithms = ['fixed', 'variable', 'unknown']
    accuracy_matrix = []
    violation_matrix = []
    
    for scenario_name, (results, perf_data) in all_results.items():
        scenarios.append(scenario_name)
        
        accuracy_row = []
        violation_row = []
        
        for algo in algorithms:
            # Find performance data for this algorithm
            algo_data = next((d for d in perf_data if d.get('algorithm') == algo), None)
            
            if algo_data and 'error' not in algo_data:
                accuracy_row.append(algo_data['accuracy'])
                total_violations = (algo_data['energy_violations'] + 
                                  algo_data['accuracy_violations'])
                violation_row.append(total_violations)
            else:
                accuracy_row.append(0)
                violation_row.append(-1)  # Indicate error
        
        accuracy_matrix.append(accuracy_row)
        violation_matrix.append(violation_row)
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Accuracy heatmap
    im1 = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(algorithms)))
    ax1.set_yticks(np.arange(len(scenarios)))
    ax1.set_xticklabels(algorithms)
    ax1.set_yticklabels(scenarios)
    ax1.set_title('Classification Accuracy')
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            text = ax1.text(j, i, f'{accuracy_matrix[i][j]:.2f}',
                           ha='center', va='center', color='black')
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Accuracy', rotation=270, labelpad=20)
    
    # Violations heatmap
    max_violations = max(max(row) for row in violation_matrix if max(row) >= 0)
    im2 = ax2.imshow(violation_matrix, cmap='YlOrRd', aspect='auto', 
                     vmin=0, vmax=max_violations)
    ax2.set_xticks(np.arange(len(algorithms)))
    ax2.set_yticks(np.arange(len(scenarios)))
    ax2.set_xticklabels(algorithms)
    ax2.set_yticklabels(scenarios)
    ax2.set_title('Total Violations')
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            val = violation_matrix[i][j]
            text = ax2.text(j, i, f'{int(val)}' if val >= 0 else 'ERR',
                           ha='center', va='center', color='black')
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Violations', rotation=270, labelpad=20)
    
    plt.suptitle('Algorithm Performance Matrix Across All Scenarios', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_report(all_results, output_dir):
    """Generate comprehensive comparison report."""
    report_path = output_dir / 'algorithm_comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ALGORITHM COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Scenario':<25} {'Algorithm':<10} {'Accuracy':<10} {'Violations':<12} {'Efficiency':<10}\n")
        f.write("-"*80 + "\n")
        
        for scenario_name, (results, perf_data) in all_results.items():
            for algo_data in perf_data:
                if 'error' not in algo_data:
                    algo = algo_data['algorithm']
                    acc = algo_data['accuracy']
                    viols = algo_data['energy_violations'] + algo_data['accuracy_violations']
                    eff = algo_data['avg_cameras']
                    
                    f.write(f"{scenario_name:<25} {algo:<10} {acc:<10.3f} {viols:<12} {eff:<10.2f}\n")
        
        # Best algorithm per scenario
        f.write("\n\nBEST ALGORITHM PER SCENARIO\n")
        f.write("-"*80 + "\n")
        
        for scenario_name, (results, perf_data) in all_results.items():
            valid_data = [d for d in perf_data if 'error' not in d]
            
            if valid_data:
                # Find best by accuracy
                best = max(valid_data, key=lambda x: x['accuracy'])
                f.write(f"{scenario_name}: {best['algorithm']} "
                       f"(accuracy={best['accuracy']:.3f})\n")
        
        # Algorithm strengths and weaknesses
        f.write("\n\nALGORITHM ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        f.write("\nFIXED FREQUENCY ALGORITHM:\n")
        f.write("Strengths:\n")
        f.write("- Predictable and deterministic behavior\n")
        f.write("- Low computational overhead\n")
        f.write("- Works well with high frequency classification\n")
        f.write("Weaknesses:\n")
        f.write("- May not adapt well to varying conditions\n")
        f.write("- Can be inefficient at low frequencies\n")
        
        f.write("\nVARIABLE FREQUENCY ALGORITHM:\n")
        f.write("Strengths:\n")
        f.write("- Energy diversity through subclass rotation\n")
        f.write("- Good for medium frequency scenarios\n")
        f.write("- Balances load across cameras\n")
        f.write("Weaknesses:\n")
        f.write("- More complex than fixed frequency\n")
        f.write("- Requires frequency estimation\n")
        
        f.write("\nUNKNOWN FREQUENCY ALGORITHM:\n")
        f.write("Strengths:\n")
        f.write("- Adapts to unknown/variable frequencies\n")
        f.write("- Game-theoretic approach optimizes decisions\n")
        f.write("- Can handle irregular patterns\n")
        f.write("Weaknesses:\n")
        f.write("- Higher computational complexity\n")
        f.write("- May have more violations initially\n")
        f.write("- Convergence time for Nash equilibrium\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        f.write("\n1. Use Fixed Frequency when:\n")
        f.write("   - Classification frequency is high and regular\n")
        f.write("   - Computational resources are limited\n")
        f.write("   - Predictability is important\n")
        
        f.write("\n2. Use Variable Frequency when:\n")
        f.write("   - Classification frequency is medium and known\n")
        f.write("   - Energy diversity is important\n")
        f.write("   - Load balancing is critical\n")
        
        f.write("\n3. Use Unknown Frequency when:\n")
        f.write("   - Classification pattern is irregular or unknown\n")
        f.write("   - System needs to adapt to changing conditions\n")
        f.write("   - Optimal long-term performance is priority\n")
        
        f.write("\n" + "="*80 + "\n")


def main():
    """Run comprehensive algorithm comparison."""
    # Setup
    output_dir = Path(f"test_results_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    setup_logging('INFO')
    
    print("="*80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*80)
    
    # Get comparison scenarios
    scenarios = create_comparison_scenarios()
    all_results = {}
    
    # Run each scenario
    for scenario_name, scenario_config in scenarios.items():
        test_params = scenario_config.pop('test_params')
        results, perf_data = run_comparison_scenario(
            scenario_name, scenario_config, test_params, output_dir
        )
        all_results[scenario_name] = (results, perf_data)
    
    # Create overall comparison
    print("\nGenerating overall comparison matrix...")
    create_overall_comparison_matrix(all_results, output_dir)
    
    # Generate report
    print("Generating comparison report...")
    generate_comparison_report(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON COMPLETED")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()