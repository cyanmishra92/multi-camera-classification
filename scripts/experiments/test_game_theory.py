#!/usr/bin/env python3
"""Test script for game-theoretic camera selection."""

import sys
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main_enhanced import create_enhanced_network_from_config, load_config, run_simulation


def run_game_theory_test():
    """Test game-theoretic selection vs standard approaches."""
    print("=" * 60)
    print("GAME-THEORETIC CAMERA SELECTION TEST")
    print("=" * 60)
    
    # Load config
    config = load_config('configs/default_config.yaml')
    
    # Test parameters
    duration = 3000  # Longer to see equilibrium behavior
    frequency = 0.1
    
    results = {}
    
    # Test configurations
    test_configs = [
        ("Standard", False, False),  # No enhancements
        ("Enhanced Only", True, False),  # Enhanced accuracy only
        ("Game Theory", True, True)  # Full game theory
    ]
    
    for test_name, use_enhanced, use_game_theory in test_configs:
        print(f"\n{test_name} Configuration:")
        print("-" * 40)
        
        # Create network
        network = create_enhanced_network_from_config(config, use_enhanced=use_enhanced)
        
        # Override game theory setting if needed
        if use_enhanced and use_game_theory:
            # Ensure utility params are set
            from src.game_theory.utility_functions import UtilityParameters
            network.config.utility_params = UtilityParameters(
                reward_scale=2.0,  # Increased reward for participation
                incorrect_penalty=0.3,  # Reduced penalty for mistakes
                non_participation_penalty=1.0,  # Increased penalty for not participating
                discount_factor=0.85  # Slightly lower discount
            )
        
        # Run simulation
        start_time = time.time()
        result = run_simulation(
            network,
            'fixed',  # Test with fixed algorithm
            duration,
            time_step=1.0,
            classification_frequency=frequency,
            visualize=False
        )
        elapsed = time.time() - start_time
        
        results[test_name] = result
        
        # Print summary
        stats = result['network_stats']
        print(f"  Overall Accuracy: {stats.get('accuracy', 0):.3f}")
        print(f"  Recent Accuracy: {stats.get('recent_accuracy', 0):.3f}")
        print(f"  Energy Violations: {stats.get('energy_violations', 0)}")
        print(f"  Accuracy Violations: {stats.get('accuracy_violations', 0)}")
        print(f"  Avg Cameras/Classification: {stats.get('avg_cameras_per_classification', 0):.2f}")
        print(f"  Runtime: {elapsed:.2f}s")
        
        # Game theory specific metrics
        if 'avg_participation_rate' in stats:
            print(f"  Avg Participation Rate: {stats['avg_participation_rate']:.3f}")
            print(f"  Avg Nash Accuracy: {stats['avg_nash_accuracy']:.3f}")
            print(f"  Avg Price of Anarchy: {stats['avg_price_of_anarchy']:.3f}")
    
    # Create comparison plots
    create_game_theory_plots(results)
    
    # Detailed analysis
    analyze_game_theory_performance(results)
    
    return results


def create_game_theory_plots(results):
    """Create plots comparing game theory approaches."""
    
    with PdfPages('game_theory_comparison.pdf') as pdf:
        
        # Page 1: Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Game-Theoretic Selection Performance Comparison', fontsize=16)
        
        test_names = list(results.keys())
        colors = ['blue', 'green', 'red']
        
        # Accuracy comparison
        ax = axes[0, 0]
        accuracies = [results[name]['network_stats'].get('accuracy', 0) for name in test_names]
        recent_accs = [results[name]['network_stats'].get('recent_accuracy', 0) for name in test_names]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Overall', alpha=0.7)
        bars2 = ax.bar(x + width/2, recent_accs, width, label='Recent', alpha=0.7)
        
        ax.axhline(y=0.8, color='red', linestyle='--', label='Target', alpha=0.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Energy efficiency
        ax = axes[0, 1]
        efficiencies = [results[name]['network_stats'].get('avg_cameras_per_classification', 0) 
                       for name in test_names]
        
        bars = ax.bar(test_names, efficiencies, color=colors, alpha=0.7)
        
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{eff:.2f}', ha='center', va='bottom')
        
        ax.set_ylabel('Cameras per Classification')
        ax.set_title('Energy Efficiency')
        ax.set_xticklabels(test_names, rotation=15)
        ax.grid(True, alpha=0.3)
        
        # Violations
        ax = axes[1, 0]
        energy_viols = [results[name]['network_stats'].get('energy_violations', 0) 
                       for name in test_names]
        acc_viols = [results[name]['network_stats'].get('accuracy_violations', 0) 
                    for name in test_names]
        
        x = np.arange(len(test_names))
        bars1 = ax.bar(x - width/2, energy_viols, width, label='Energy', color='orange', alpha=0.7)
        bars2 = ax.bar(x + width/2, acc_viols, width, label='Accuracy', color='red', alpha=0.7)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Number of Violations')
        ax.set_title('Constraint Violations')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Success rate over time (if available)
        ax = axes[1, 1]
        
        for name, color in zip(test_names, colors):
            if 'performance_history' in results[name]:
                # Calculate moving average success rate
                perf_history = results[name]['performance_history']
                window = 50
                
                if len(perf_history) > window:
                    success_rates = []
                    timestamps = []
                    
                    for i in range(window, len(perf_history), 10):
                        window_data = perf_history[i-window:i]
                        rate = sum(r['result']['success'] for r in window_data) / window
                        success_rates.append(rate)
                        timestamps.append(window_data[-1]['timestamp'])
                    
                    ax.plot(timestamps, success_rates, label=name, color=color, linewidth=2)
        
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Success Rate (50-sample window)')
        ax.set_title('Success Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Energy Dynamics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Energy Dynamics Comparison', fontsize=16)
        
        for idx, (name, color) in enumerate(zip(test_names, colors)):
            ax = axes[idx]
            
            if 'energy_history' in results[name]:
                energy_history = results[name]['energy_history'][:500]  # First 500 steps
                
                timestamps = [e['timestamp'] for e in energy_history]
                avg_energies = [e['avg_energy'] for e in energy_history]
                min_energies = [e['min_energy'] for e in energy_history]
                max_energies = [e['max_energy'] for e in energy_history]
                
                ax.plot(timestamps, avg_energies, color=color, label='Average', linewidth=2)
                ax.fill_between(timestamps, min_energies, max_energies,
                               alpha=0.3, color=color)
                
                # Mark classification events
                if 'performance_history' in results[name]:
                    event_times = [p['timestamp'] for p in results[name]['performance_history'][:50]]
                    event_energies = []
                    for t in event_times:
                        # Find closest energy reading
                        closest_idx = np.argmin(np.abs(np.array(timestamps) - t))
                        event_energies.append(avg_energies[closest_idx])
                    
                    ax.scatter(event_times, event_energies, color='black', 
                             s=20, alpha=0.5, marker='v', label='Classifications')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy Level')
                ax.set_title(f'{name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Game Theory Metrics (if available)
        if 'Game Theory' in results:
            game_result = results['Game Theory']
            if 'network_stats' in game_result and 'avg_participation_rate' in game_result['network_stats']:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('Game Theory Specific Metrics', fontsize=16)
                
                # Extract algorithm instance for detailed metrics
                # This would require storing more detailed data during simulation
                
                # Placeholder for game theory specific visualizations
                ax = axes[0, 0]
                ax.text(0.5, 0.5, 'Participation Rate Evolution', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Nash Equilibrium Participation')
                
                ax = axes[0, 1]
                ax.text(0.5, 0.5, 'Social Welfare Over Time', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Social Welfare')
                
                ax = axes[1, 0]
                ax.text(0.5, 0.5, 'Price of Anarchy', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Efficiency Loss')
                
                ax = axes[1, 1]
                stats = game_result['network_stats']
                metrics = ['avg_participation_rate', 'avg_nash_accuracy', 'avg_price_of_anarchy']
                values = [stats.get(m, 0) for m in metrics]
                
                bars = ax.bar(['Participation\nRate', 'Nash\nAccuracy', 'Price of\nAnarchy'], 
                             values, color=['blue', 'green', 'red'], alpha=0.7)
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.3f}', ha='center', va='bottom')
                
                ax.set_ylabel('Value')
                ax.set_title('Game Theory Summary Metrics')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
    
    print(f"\nPlots saved to: game_theory_comparison.pdf")


def analyze_game_theory_performance(results):
    """Detailed analysis of game theory performance."""
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Compare key metrics
    metrics_table = []
    headers = ['Metric', 'Standard', 'Enhanced Only', 'Game Theory']
    
    metrics = [
        ('Overall Accuracy', 'accuracy'),
        ('Recent Accuracy', 'recent_accuracy'),
        ('Energy Violations', 'energy_violations'),
        ('Accuracy Violations', 'accuracy_violations'),
        ('Avg Cameras/Classification', 'avg_cameras_per_classification')
    ]
    
    for metric_name, metric_key in metrics:
        row = [metric_name]
        for test_name in ['Standard', 'Enhanced Only', 'Game Theory']:
            if test_name in results:
                value = results[test_name]['network_stats'].get(metric_key, 0)
                if isinstance(value, float):
                    row.append(f"{value:.3f}")
                else:
                    row.append(str(value))
            else:
                row.append("N/A")
        metrics_table.append(row)
    
    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + metrics_table) 
                  for i in range(len(headers))]
    
    # Print header
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in metrics_table:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    
    # Analysis summary
    print("\nKey Findings:")
    print("-" * 40)
    
    # Check if game theory improves accuracy
    if 'Game Theory' in results and 'Standard' in results:
        gt_acc = results['Game Theory']['network_stats'].get('accuracy', 0)
        std_acc = results['Standard']['network_stats'].get('accuracy', 0)
        
        if gt_acc > std_acc:
            print(f"✓ Game theory improves accuracy by {(gt_acc - std_acc)*100:.1f}%")
        else:
            print(f"✗ Game theory reduces accuracy by {(std_acc - gt_acc)*100:.1f}%")
        
        # Check efficiency
        gt_eff = results['Game Theory']['network_stats'].get('avg_cameras_per_classification', 0)
        std_eff = results['Standard']['network_stats'].get('avg_cameras_per_classification', 0)
        
        if gt_eff < std_eff:
            print(f"✓ Game theory improves efficiency (uses {std_eff - gt_eff:.2f} fewer cameras)")
        else:
            print(f"✗ Game theory reduces efficiency (uses {gt_eff - std_eff:.2f} more cameras)")
        
        # Check violations
        gt_viols = (results['Game Theory']['network_stats'].get('energy_violations', 0) +
                   results['Game Theory']['network_stats'].get('accuracy_violations', 0))
        std_viols = (results['Standard']['network_stats'].get('energy_violations', 0) +
                    results['Standard']['network_stats'].get('accuracy_violations', 0))
        
        if gt_viols < std_viols:
            print(f"✓ Game theory reduces violations by {std_viols - gt_viols}")
        elif gt_viols > std_viols:
            print(f"✗ Game theory increases violations by {gt_viols - std_viols}")
        else:
            print("= Game theory maintains same violation count")
    
    # Save detailed results
    with open('game_theory_test_results.json', 'w') as f:
        summary = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': {}
        }
        
        for name in results:
            stats = results[name]['network_stats']
            summary['test_results'][name] = {
                'accuracy': stats.get('accuracy', 0),
                'recent_accuracy': stats.get('recent_accuracy', 0),
                'energy_violations': stats.get('energy_violations', 0),
                'accuracy_violations': stats.get('accuracy_violations', 0),
                'avg_cameras_per_classification': stats.get('avg_cameras_per_classification', 0),
                'game_theory_metrics': {
                    'avg_participation_rate': stats.get('avg_participation_rate', 0),
                    'avg_nash_accuracy': stats.get('avg_nash_accuracy', 0),
                    'avg_price_of_anarchy': stats.get('avg_price_of_anarchy', 0)
                } if 'avg_participation_rate' in stats else None
            }
        
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: game_theory_test_results.json")


if __name__ == "__main__":
    run_game_theory_test()
    print("\nGame theory test complete!")