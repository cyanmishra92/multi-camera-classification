#!/usr/bin/env python3
"""Test script to compare original vs enhanced accuracy model."""

import sys
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main_enhanced import create_enhanced_network_from_config, load_config, run_simulation


def run_comparison_test():
    """Run comparison between original and enhanced accuracy models."""
    print("=" * 60)
    print("ENHANCED ACCURACY MODEL COMPARISON TEST")
    print("=" * 60)
    
    # Load config
    config = load_config('configs/default_config.yaml')
    
    # Test parameters
    duration = 2000
    frequency = 0.1
    algorithms = ['fixed', 'variable', 'unknown']
    
    results = {}
    
    # Run with original accuracy model
    print("\n1. Testing with ORIGINAL accuracy model...")
    print("-" * 40)
    
    for algo in algorithms:
        print(f"\nRunning {algo} algorithm (original)...")
        network = create_enhanced_network_from_config(config, use_enhanced=False)
        
        start_time = time.time()
        result = run_simulation(
            network,
            algo,
            duration,
            time_step=1.0,
            classification_frequency=frequency,
            visualize=False
        )
        elapsed = time.time() - start_time
        
        results[f"{algo}_original"] = result
        
        stats = result['network_stats']
        print(f"  Accuracy: {stats.get('accuracy', 0):.3f}")
        print(f"  Recent Accuracy: {stats.get('recent_accuracy', 0):.3f}")
        print(f"  Energy Violations: {stats.get('energy_violations', 0)}")
        print(f"  Accuracy Violations: {stats.get('accuracy_violations', 0)}")
        print(f"  Avg Cameras/Classification: {stats.get('avg_cameras_per_classification', 0):.2f}")
        print(f"  Runtime: {elapsed:.2f}s")
    
    # Run with enhanced accuracy model
    print("\n\n2. Testing with ENHANCED accuracy model...")
    print("-" * 40)
    
    for algo in algorithms:
        print(f"\nRunning {algo} algorithm (enhanced)...")
        network = create_enhanced_network_from_config(config, use_enhanced=True)
        
        start_time = time.time()
        result = run_simulation(
            network,
            algo,
            duration,
            time_step=1.0,
            classification_frequency=frequency,
            visualize=False
        )
        elapsed = time.time() - start_time
        
        results[f"{algo}_enhanced"] = result
        
        stats = result['network_stats']
        print(f"  Accuracy: {stats.get('accuracy', 0):.3f}")
        print(f"  Recent Accuracy: {stats.get('recent_accuracy', 0):.3f}")
        print(f"  Energy Violations: {stats.get('energy_violations', 0)}")
        print(f"  Accuracy Violations: {stats.get('accuracy_violations', 0)}")
        print(f"  Avg Cameras/Classification: {stats.get('avg_cameras_per_classification', 0):.2f}")
        print(f"  Runtime: {elapsed:.2f}s")
        
        # Coverage analysis for enhanced model
        if result.get('coverage_stats'):
            coverage = result['coverage_stats']
            print(f"  Coverage Analysis:")
            print(f"    Avg Coverage: {coverage['avg_coverage']:.2f} cameras")
            print(f"    Blind Spots: {coverage['blind_spots']}")
            print(f"    Well Covered: {coverage['well_covered']}")
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Save results
    save_results_json(results)
    
    return results


def create_comparison_plots(results):
    """Create PDF plots comparing original vs enhanced models."""
    
    with PdfPages('enhanced_accuracy_comparison.pdf') as pdf:
        
        # Page 1: Accuracy Comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Accuracy Comparison: Original vs Enhanced Model', fontsize=16)
        
        algorithms = ['fixed', 'variable', 'unknown']
        
        for idx, algo in enumerate(algorithms):
            ax = axes[idx]
            
            orig_stats = results[f"{algo}_original"]['network_stats']
            enh_stats = results[f"{algo}_enhanced"]['network_stats']
            
            metrics = ['accuracy', 'recent_accuracy']
            orig_values = [orig_stats.get(m, 0) for m in metrics]
            enh_values = [enh_stats.get(m, 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, enh_values, width, label='Enhanced', color='green', alpha=0.7)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            
            # Add 80% threshold line
            ax.axhline(y=0.8, color='red', linestyle='--', label='Min Threshold')
            
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{algo.capitalize()} Algorithm')
            ax.set_xticks(x)
            ax.set_xticklabels(['Overall', 'Recent'])
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Energy Efficiency and Violations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Energy Efficiency and Constraint Violations', fontsize=16)
        
        # Row 1: Cameras per classification
        for idx, algo in enumerate(algorithms):
            ax = axes[0, idx]
            
            orig_stats = results[f"{algo}_original"]['network_stats']
            enh_stats = results[f"{algo}_enhanced"]['network_stats']
            
            values = [
                orig_stats.get('avg_cameras_per_classification', 0),
                enh_stats.get('avg_cameras_per_classification', 0)
            ]
            
            bars = ax.bar(['Original', 'Enhanced'], values, color=['blue', 'green'], alpha=0.7)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom')
            
            ax.set_ylabel('Cameras per Classification')
            ax.set_title(f'{algo.capitalize()} Algorithm')
            ax.grid(True, alpha=0.3)
        
        # Row 2: Violations
        for idx, algo in enumerate(algorithms):
            ax = axes[1, idx]
            
            orig_stats = results[f"{algo}_original"]['network_stats']
            enh_stats = results[f"{algo}_enhanced"]['network_stats']
            
            violation_types = ['energy_violations', 'accuracy_violations']
            orig_violations = [orig_stats.get(v, 0) for v in violation_types]
            enh_violations = [enh_stats.get(v, 0) for v in violation_types]
            
            x = np.arange(len(violation_types))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, orig_violations, width, label='Original', color='red', alpha=0.7)
            bars2 = ax.bar(x + width/2, enh_violations, width, label='Enhanced', color='orange', alpha=0.7)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
            
            ax.set_ylabel('Number of Violations')
            ax.set_title(f'{algo.capitalize()} Algorithm')
            ax.set_xticks(x)
            ax.set_xticklabels(['Energy', 'Accuracy'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Energy Dynamics Over Time
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Energy Dynamics: Original vs Enhanced Model', fontsize=16)
        
        for row, model_type in enumerate(['original', 'enhanced']):
            for col, algo in enumerate(algorithms):
                ax = axes[row, col]
                
                result = results[f"{algo}_{model_type}"]
                if 'energy_history' in result and len(result['energy_history']) > 0:
                    # Sample energy history
                    energy_history = result['energy_history'][:200]  # First 200 time steps
                    
                    timestamps = [e['timestamp'] for e in energy_history]
                    avg_energies = [e['avg_energy'] for e in energy_history]
                    min_energies = [e['min_energy'] for e in energy_history]
                    max_energies = [e['max_energy'] for e in energy_history]
                    
                    ax.plot(timestamps, avg_energies, 'b-', label='Average', linewidth=2)
                    ax.fill_between(timestamps, min_energies, max_energies,
                                   alpha=0.3, color='blue', label='Min-Max Range')
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Energy Level')
                    ax.set_title(f'{algo.capitalize()} - {model_type.capitalize()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 4: Accuracy Over Time Traces
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Accuracy Traces Over Time', fontsize=16)
        
        for row, model_type in enumerate(['original', 'enhanced']):
            for col, algo in enumerate(algorithms):
                ax = axes[row, col]
                
                result = results[f"{algo}_{model_type}"]
                if 'performance_history' in result:
                    # Calculate moving average accuracy
                    window_size = 20
                    perf_history = result['performance_history']
                    
                    if len(perf_history) > window_size:
                        accuracies = []
                        timestamps = []
                        
                        for i in range(window_size, len(perf_history)):
                            window = perf_history[i-window_size:i]
                            acc = sum(r['result']['success'] for r in window) / window_size
                            accuracies.append(acc)
                            timestamps.append(window[-1]['timestamp'])
                        
                        ax.plot(timestamps, accuracies, 'g-', linewidth=2)
                        ax.axhline(y=0.8, color='red', linestyle='--', 
                                  label='Min Threshold', alpha=0.7)
                        
                        # Calculate periods above/below threshold
                        above_threshold = sum(1 for a in accuracies if a >= 0.8)
                        below_threshold = len(accuracies) - above_threshold
                        
                        ax.text(0.02, 0.98, f'Above: {above_threshold/len(accuracies)*100:.1f}%',
                               transform=ax.transAxes, va='top', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Moving Average Accuracy')
                    ax.set_title(f'{algo.capitalize()} - {model_type.capitalize()}')
                    ax.set_ylim(0, 1.0)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"\nPlots saved to: enhanced_accuracy_comparison.pdf")


def save_results_json(results):
    """Save detailed results to JSON file."""
    # Convert to JSON-serializable format
    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(v) for v in obj]
        else:
            return obj
    
    # Create summary
    summary = {
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparison_results': {}
    }
    
    algorithms = ['fixed', 'variable', 'unknown']
    
    for algo in algorithms:
        orig_stats = results[f"{algo}_original"]['network_stats']
        enh_stats = results[f"{algo}_enhanced"]['network_stats']
        
        summary['comparison_results'][algo] = {
            'original': {
                'accuracy': orig_stats.get('accuracy', 0),
                'recent_accuracy': orig_stats.get('recent_accuracy', 0),
                'energy_violations': orig_stats.get('energy_violations', 0),
                'accuracy_violations': orig_stats.get('accuracy_violations', 0),
                'avg_cameras_per_classification': orig_stats.get('avg_cameras_per_classification', 0)
            },
            'enhanced': {
                'accuracy': enh_stats.get('accuracy', 0),
                'recent_accuracy': enh_stats.get('recent_accuracy', 0),
                'energy_violations': enh_stats.get('energy_violations', 0),
                'accuracy_violations': enh_stats.get('accuracy_violations', 0),
                'avg_cameras_per_classification': enh_stats.get('avg_cameras_per_classification', 0)
            },
            'improvement': {
                'accuracy_gain': enh_stats.get('accuracy', 0) - orig_stats.get('accuracy', 0),
                'efficiency_gain': orig_stats.get('avg_cameras_per_classification', 0) - 
                                 enh_stats.get('avg_cameras_per_classification', 0)
            }
        }
    
    with open('enhanced_accuracy_test_results.json', 'w') as f:
        json.dump(convert_to_json(summary), f, indent=2)
    
    print(f"Results saved to: enhanced_accuracy_test_results.json")


if __name__ == "__main__":
    run_comparison_test()
    print("\nEnhanced accuracy model test complete!")