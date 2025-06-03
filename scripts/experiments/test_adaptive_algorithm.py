#!/usr/bin/env python3
"""Test adaptive algorithm to achieve 80% accuracy target."""

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
from src.game_theory.utility_functions import UtilityParameters


def test_adaptive_algorithm():
    """Test the adaptive algorithm with best parameters."""
    print("=" * 60)
    print("ADAPTIVE ALGORITHM TEST")
    print("=" * 60)
    print("Testing accuracy-adaptive algorithm with tuned parameters\n")
    
    # Load config and best parameters
    config = load_config('configs/default_config.yaml')
    
    # Load best parameters if available
    try:
        with open('best_parameters.json', 'r') as f:
            best_params = json.load(f)['parameters']
        print("Loaded best parameters from tuning")
    except:
        best_params = {
            'min_accuracy_threshold': 0.8,
            'distance_decay': 0.01,
            'angle_penalty': 0.3,
            'overlap_bonus': 0.2,
            'reward_scale': 2.0,
            'incorrect_penalty': 0.3,
            'non_participation_penalty': 1.0,
            'discount_factor': 0.85
        }
        print("Using default parameters")
    
    # Update config with best parameters
    config['accuracy'].update({
        'distance_decay': best_params['distance_decay'],
        'angle_penalty': best_params['angle_penalty'],
        'overlap_bonus': best_params['overlap_bonus']
    })
    
    # Test configurations
    test_configs = [
        ("Standard Algorithm", False, False, {}),
        ("Enhanced Algorithm", True, False, {}),
        ("Game Theory Algorithm", True, True, {}),
        ("Adaptive Algorithm", True, True, {'use_adaptive': True})
    ]
    
    results = {}
    
    for test_name, use_enhanced, use_game_theory, extra_kwargs in test_configs:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        # Create network
        network = create_enhanced_network_from_config(config, use_enhanced=use_enhanced)
        
        # Set utility parameters for game theory algorithms
        if use_game_theory:
            network.config.utility_params = UtilityParameters(
                reward_scale=best_params['reward_scale'],
                incorrect_penalty=best_params['incorrect_penalty'],
                non_participation_penalty=best_params['non_participation_penalty'],
                discount_factor=best_params['discount_factor']
            )
        
        # Run simulation
        start_time = time.time()
        
        # Set algorithm with appropriate kwargs
        algo_kwargs = {
            'classification_frequency': 0.1,
            'min_accuracy_threshold': best_params['min_accuracy_threshold']
        }
        
        # Add position_weight only for enhanced algorithms
        if use_enhanced:
            algo_kwargs['position_weight'] = best_params.get('position_weight', 0.7)
            
        # Add extra kwargs (but remove use_adaptive as it's not a parameter)
        filtered_kwargs = {k: v for k, v in extra_kwargs.items() if k != 'use_adaptive'}
        algo_kwargs.update(filtered_kwargs)
        
        network.set_algorithm('fixed', **algo_kwargs)
        
        # Run longer simulation for better statistics
        duration = 5000
        
        # Custom simulation loop to track detailed metrics
        np.random.seed(42)
        classification_times = []
        current_time = 0
        
        while current_time < duration:
            interval = np.random.exponential(1.0 / 0.1)
            current_time += interval
            if current_time < duration:
                classification_times.append(current_time)
        
        # Track performance over time
        performance_over_time = []
        accuracy_checkpoints = []
        
        event_idx = 0
        for step in range(duration):
            current_time = float(step)
            
            # Process events
            while event_idx < len(classification_times) and classification_times[event_idx] <= current_time:
                object_position = np.random.uniform(-40, 40, size=3)
                object_position[2] = 0
                true_label = np.random.randint(0, 2)
                
                result = network.classify_object(object_position, true_label)
                event_idx += 1
            
            # Update network
            network.update_time(1.0)
            
            # Track performance every 100 steps
            if step % 100 == 0 and step > 0:
                stats = network.get_network_stats()
                performance_over_time.append({
                    'time': step,
                    'accuracy': stats.get('recent_accuracy', 0),
                    'avg_cameras': stats.get('avg_cameras_per_classification', 0)
                })
                
                # Checkpoint at specific times
                if step in [500, 1000, 2000, 5000]:
                    accuracy_checkpoints.append({
                        'time': step,
                        'overall': stats.get('accuracy', 0),
                        'recent': stats.get('recent_accuracy', 0)
                    })
        
        elapsed = time.time() - start_time
        
        # Get final stats
        final_stats = network.get_network_stats()
        
        # Store results
        results[test_name] = {
            'final_stats': final_stats,
            'performance_over_time': performance_over_time,
            'accuracy_checkpoints': accuracy_checkpoints,
            'runtime': elapsed
        }
        
        # Print summary
        print(f"  Overall Accuracy: {final_stats.get('accuracy', 0):.3f}")
        print(f"  Recent Accuracy: {final_stats.get('recent_accuracy', 0):.3f}")
        print(f"  Avg Cameras/Classification: {final_stats.get('avg_cameras_per_classification', 0):.2f}")
        print(f"  Energy Violations: {final_stats.get('energy_violations', 0)}")
        print(f"  Accuracy Violations: {final_stats.get('accuracy_violations', 0)}")
        
        # Print checkpoints
        print(f"\n  Accuracy Progression:")
        for checkpoint in accuracy_checkpoints:
            print(f"    t={checkpoint['time']:4d}: {checkpoint['overall']:.3f} (recent: {checkpoint['recent']:.3f})")
        
        print(f"\n  Runtime: {elapsed:.2f}s")
        
        # Check if target achieved
        if final_stats.get('accuracy', 0) >= 0.8:
            print("  ✅ TARGET ACHIEVED!")
    
    # Create comparison plots
    create_adaptive_comparison_plots(results)
    
    # Save results
    save_adaptive_results(results)
    
    return results


def create_adaptive_comparison_plots(results):
    """Create plots comparing adaptive algorithm performance."""
    
    with PdfPages('adaptive_algorithm_comparison.pdf') as pdf:
        
        # Page 1: Final Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Adaptive Algorithm Performance Comparison', fontsize=16)
        
        algorithms = list(results.keys())
        colors = ['blue', 'green', 'orange', 'red']
        
        # Overall accuracy
        ax = axes[0, 0]
        accuracies = [results[algo]['final_stats'].get('accuracy', 0) for algo in algorithms]
        bars = ax.bar(algorithms, accuracies, color=colors, alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', label='Target', linewidth=2)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{acc:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Overall Accuracy')
        ax.set_title('Final Overall Accuracy')
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Recent accuracy
        ax = axes[0, 1]
        recent_accs = [results[algo]['final_stats'].get('recent_accuracy', 0) for algo in algorithms]
        bars = ax.bar(algorithms, recent_accs, color=colors, alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2)
        
        for bar, acc in zip(bars, recent_accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{acc:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Recent Accuracy')
        ax.set_title('Final Recent Accuracy')
        ax.set_xticklabels(algorithms, rotation=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Efficiency
        ax = axes[1, 0]
        efficiencies = [results[algo]['final_stats'].get('avg_cameras_per_classification', 0) 
                       for algo in algorithms]
        bars = ax.bar(algorithms, efficiencies, color=colors, alpha=0.7)
        
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{eff:.2f}', ha='center', va='bottom')
        
        ax.set_ylabel('Cameras per Classification')
        ax.set_title('Energy Efficiency')
        ax.set_xticklabels(algorithms, rotation=15)
        ax.grid(True, alpha=0.3)
        
        # Violations
        ax = axes[1, 1]
        violations = [results[algo]['final_stats'].get('energy_violations', 0) + 
                     results[algo]['final_stats'].get('accuracy_violations', 0)
                     for algo in algorithms]
        bars = ax.bar(algorithms, violations, color=colors, alpha=0.7)
        
        for bar, viol in zip(bars, violations):
            if viol > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{int(viol)}', ha='center', va='bottom')
        
        ax.set_ylabel('Total Violations')
        ax.set_title('Constraint Violations')
        ax.set_xticklabels(algorithms, rotation=15)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Accuracy Evolution Over Time
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Accuracy Evolution During Simulation', fontsize=16)
        
        for idx, (algo, color) in enumerate(zip(algorithms, colors)):
            ax = axes[idx // 2, idx % 2]
            
            if 'performance_over_time' in results[algo]:
                perf_data = results[algo]['performance_over_time']
                times = [p['time'] for p in perf_data]
                accuracies = [p['accuracy'] for p in perf_data]
                
                ax.plot(times, accuracies, color=color, linewidth=2)
                ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
                
                # Mark checkpoints
                checkpoints = results[algo]['accuracy_checkpoints']
                cp_times = [cp['time'] for cp in checkpoints]
                cp_overall = [cp['overall'] for cp in checkpoints]
                
                ax.scatter(cp_times, cp_overall, color='black', s=50, 
                          marker='o', zorder=5, label='Overall Acc')
                
                # Annotate final accuracy
                final_acc = results[algo]['final_stats'].get('accuracy', 0)
                ax.text(0.95, 0.05, f'Final: {final_acc:.3f}',
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Accuracy')
            ax.set_title(algo)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Efficiency Evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Camera Usage Efficiency Over Time', fontsize=16)
        
        for algo, color in zip(algorithms, colors):
            if 'performance_over_time' in results[algo]:
                perf_data = results[algo]['performance_over_time']
                times = [p['time'] for p in perf_data]
                cameras = [p['avg_cameras'] for p in perf_data]
                
                ax.plot(times, cameras, color=color, linewidth=2, label=algo)
        
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Cameras per Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"\nPlots saved to: adaptive_algorithm_comparison.pdf")


def save_adaptive_results(results):
    """Save detailed results of adaptive algorithm test."""
    summary = {
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'algorithms': {}
    }
    
    for algo in results:
        summary['algorithms'][algo] = {
            'final_accuracy': results[algo]['final_stats'].get('accuracy', 0),
            'recent_accuracy': results[algo]['final_stats'].get('recent_accuracy', 0),
            'avg_cameras': results[algo]['final_stats'].get('avg_cameras_per_classification', 0),
            'violations': (results[algo]['final_stats'].get('energy_violations', 0) +
                          results[algo]['final_stats'].get('accuracy_violations', 0)),
            'runtime': results[algo]['runtime'],
            'target_achieved': results[algo]['final_stats'].get('accuracy', 0) >= 0.8,
            'checkpoints': results[algo]['accuracy_checkpoints']
        }
    
    with open('adaptive_algorithm_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: adaptive_algorithm_results.json")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Accuracy':>10} {'Recent':>10} {'Cameras':>10} {'Target Met':>12}")
    print("-" * 80)
    
    for algo in results:
        acc = results[algo]['final_stats'].get('accuracy', 0)
        recent = results[algo]['final_stats'].get('recent_accuracy', 0)
        cameras = results[algo]['final_stats'].get('avg_cameras_per_classification', 0)
        met = "✅ YES" if acc >= 0.8 else "❌ NO"
        
        print(f"{algo:<25} {acc:>10.3f} {recent:>10.3f} {cameras:>10.2f} {met:>12}")
    
    print("=" * 80)


if __name__ == "__main__":
    results = test_adaptive_algorithm()
    print("\nAdaptive algorithm test complete!")