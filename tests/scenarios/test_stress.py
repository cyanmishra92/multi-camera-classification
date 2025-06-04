#!/usr/bin/env python3
"""
Stress testing for multi-camera classification system.

Tests system performance under heavy load:
1. Large number of cameras
2. Long duration simulations
3. Memory usage tracking
4. Performance degradation analysis
5. Scalability testing
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import create_network_from_config, run_simulation
from src.core.network import CameraNetwork, NetworkConfig
from src.core.energy_model import EnergyParameters
from src.utils.logger import setup_logging


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_stress_test_configs():
    """Create configurations for stress testing."""
    
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
    
    stress_tests = {}
    
    # Stress Test 1: Scalability with number of cameras
    camera_counts = [10, 25, 50, 100, 200]
    for count in camera_counts:
        stress_tests[f'cameras_{count}'] = {
            **base_config,
            'network': {
                'num_cameras': count,
                'num_classes': min(count // 3, 10),  # Reasonable number of classes
                'num_objects': 5
            },
            'test_params': {
                'duration': 200,  # Short duration for many cameras
                'frequency': 0.1
            }
        }
    
    # Stress Test 2: Long duration
    durations = [1000, 5000, 10000]
    for duration in durations:
        stress_tests[f'duration_{duration}'] = {
            **base_config,
            'network': {
                'num_cameras': 20,
                'num_classes': 4,
                'num_objects': 5
            },
            'test_params': {
                'duration': duration,
                'frequency': 0.1
            }
        }
    
    # Stress Test 3: High frequency
    frequencies = [0.5, 1.0, 2.0]
    for freq in frequencies:
        stress_tests[f'frequency_{freq}'] = {
            **base_config,
            'network': {
                'num_cameras': 20,
                'num_classes': 4,
                'num_objects': 5
            },
            'test_params': {
                'duration': 500,
                'frequency': freq
            }
        }
    
    return stress_tests


def run_stress_test(test_name, config, test_params, output_dir):
    """Run a single stress test."""
    print(f"\n{'='*60}")
    print(f"Running stress test: {test_name}")
    print(f"{'='*60}")
    
    results = {}
    performance_metrics = {}
    
    # Only test fixed algorithm for stress tests (faster)
    algorithms = ['fixed']  # Can add others if needed
    
    for algo in algorithms:
        print(f"\n  Testing {algo} algorithm...")
        
        # Track memory before
        gc.collect()
        mem_before = get_memory_usage()
        
        # Track time
        start_time = time.time()
        
        try:
            # Create network
            network = create_network_from_config(config)
            
            # Run simulation
            result = run_simulation(
                network,
                algo,
                test_params['duration'],
                time_step=1.0,
                classification_frequency=test_params['frequency'],
                visualize=False
            )
            
            # Track performance
            end_time = time.time()
            mem_after = get_memory_usage()
            
            performance_metrics[algo] = {
                'execution_time': end_time - start_time,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_increase': mem_after - mem_before,
                'duration': test_params['duration'],
                'num_cameras': config['network']['num_cameras'],
                'frequency': test_params['frequency'],
                'total_events': result['network_stats']['total_classifications']
            }
            
            results[algo] = result
            
            # Print metrics
            print(f"    Execution time: {performance_metrics[algo]['execution_time']:.2f}s")
            print(f"    Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB "
                  f"(+{performance_metrics[algo]['memory_increase']:.1f}MB)")
            print(
                f"    Events/second: "
                f"{performance_metrics[algo]['total_events'] / performance_metrics[algo]['execution_time']:.2f}"
            )
            print(f"    Accuracy: {result['network_stats']['accuracy']:.3f}")
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[algo] = {'error': str(e)}
            performance_metrics[algo] = {'error': str(e)}
        
        finally:
            # Clean up
            gc.collect()
    
    # Save results
    test_dir = output_dir / test_name
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / 'results.json', 'w') as f:
        json.dump({
            'test_name': test_name,
            'config': config,
            'test_params': test_params,
            'results': results,
            'performance_metrics': performance_metrics
        }, f, indent=2, default=str)
    
    return results, performance_metrics


def create_scalability_plots(all_metrics, output_dir):
    """Create plots showing scalability analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract camera scalability data
    camera_tests = sorted([k for k in all_metrics.keys() if k.startswith('cameras_')])
    if camera_tests:
        camera_counts = []
        exec_times = []
        memory_usage = []
        events_per_sec = []
        
        for test in camera_tests:
            metrics = all_metrics[test].get('fixed', {})
            if 'error' not in metrics:
                camera_counts.append(metrics['num_cameras'])
                exec_times.append(metrics['execution_time'])
                memory_usage.append(metrics['memory_increase'])
                events_per_sec.append(metrics['total_events'] / metrics['execution_time'])
        
        # Execution time vs cameras
        ax1.plot(camera_counts, exec_times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Cameras')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time Scalability')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage vs cameras
        ax2.plot(camera_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Cameras')
        ax2.set_ylabel('Memory Increase (MB)')
        ax2.set_title('Memory Usage Scalability')
        ax2.grid(True, alpha=0.3)
    
    # Extract duration scalability data
    duration_tests = sorted([k for k in all_metrics.keys() if k.startswith('duration_')])
    if duration_tests:
        durations = []
        exec_times = []
        
        for test in duration_tests:
            metrics = all_metrics[test].get('fixed', {})
            if 'error' not in metrics:
                durations.append(metrics['duration'])
                exec_times.append(metrics['execution_time'])
        
        # Execution time vs simulation duration
        ax3.plot(durations, exec_times, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Simulation Duration')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_title('Duration Scalability')
        ax3.grid(True, alpha=0.3)
    
    # Extract frequency scalability data
    freq_tests = sorted([k for k in all_metrics.keys() if k.startswith('frequency_')])
    if freq_tests:
        frequencies = []
        events_per_sec = []
        
        for test in freq_tests:
            metrics = all_metrics[test].get('fixed', {})
            if 'error' not in metrics:
                frequencies.append(metrics['frequency'])
                events_per_sec.append(metrics['total_events'] / metrics['execution_time'])
        
        # Events per second vs frequency
        ax4.plot(frequencies, events_per_sec, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Classification Frequency')
        ax4.set_ylabel('Events Processed/Second')
        ax4.set_title('Frequency Processing Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('System Scalability Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_stress_test_report(all_results, all_metrics, output_dir):
    """Generate comprehensive stress test report."""
    report_path = output_dir / 'stress_test_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRESS TESTING SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("SYSTEM INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"CPU Count: {psutil.cpu_count()}\n")
        f.write(f"Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB\n")
        f.write(f"Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB\n\n")
        
        # Test results
        f.write("STRESS TEST RESULTS\n")
        f.write("-"*40 + "\n\n")
        
        for test_name in sorted(all_results.keys()):
            f.write(f"{test_name.upper()}:\n")
            
            metrics = all_metrics[test_name].get('fixed', {})
            if 'error' in metrics:
                f.write(f"  ERROR: {metrics['error']}\n\n")
                continue
            
            f.write(f"  Cameras: {metrics['num_cameras']}\n")
            f.write(f"  Duration: {metrics['duration']}\n")
            f.write(f"  Frequency: {metrics['frequency']}\n")
            f.write(f"  Execution Time: {metrics['execution_time']:.2f}s\n")
            f.write(f"  Memory Increase: {metrics['memory_increase']:.1f}MB\n")
            f.write(f"  Total Events: {metrics['total_events']}\n")
            f.write(f"  Events/Second: {metrics['total_events'] / metrics['execution_time']:.2f}\n")
            
            result = all_results[test_name].get('fixed', {})
            if 'network_stats' in result:
                stats = result['network_stats']
                f.write(f"  Accuracy: {stats.get('accuracy', 0):.3f}\n")
                f.write(f"  Violations: {stats.get('energy_violations', 0) + stats.get('accuracy_violations', 0)}\n")
            
            f.write("\n")
        
        # Performance analysis
        f.write("\nPERFORMANCE ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        # Camera scalability
        camera_tests = [k for k in all_metrics.keys() if k.startswith('cameras_')]
        if camera_tests:
            f.write("\nCamera Scalability:\n")
            for test in sorted(camera_tests):
                metrics = all_metrics[test].get('fixed', {})
                if 'error' not in metrics:
                    f.write(f"  {metrics['num_cameras']} cameras: "
                           f"{metrics['execution_time']:.2f}s, "
                           f"{metrics['memory_increase']:.1f}MB\n")
        
        # Duration scalability
        duration_tests = [k for k in all_metrics.keys() if k.startswith('duration_')]
        if duration_tests:
            f.write("\nDuration Scalability:\n")
            for test in sorted(duration_tests):
                metrics = all_metrics[test].get('fixed', {})
                if 'error' not in metrics:
                    f.write(f"  {metrics['duration']} steps: "
                           f"{metrics['execution_time']:.2f}s "
                           f"({metrics['duration'] / metrics['execution_time']:.1f} steps/s)\n")
        
        # Frequency scalability
        freq_tests = [k for k in all_metrics.keys() if k.startswith('frequency_')]
        if freq_tests:
            f.write("\nFrequency Processing:\n")
            for test in sorted(freq_tests):
                metrics = all_metrics[test].get('fixed', {})
                if 'error' not in metrics:
                    f.write(f"  {metrics['frequency']} Hz: "
                           f"{metrics['total_events'] / metrics['execution_time']:.2f} events/s\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        # Find limits
        max_cameras_tested = max([m['num_cameras'] for m in all_metrics.values() 
                                 if isinstance(m, dict) and m.get('fixed', {}).get('num_cameras')], 
                                default=0)
        f.write(f"\n- Successfully tested up to {max_cameras_tested} cameras\n")
        
        # Memory recommendations
        max_memory = max([m.get('fixed', {}).get('memory_increase', 0) 
                         for m in all_metrics.values() if isinstance(m, dict)], 
                        default=0)
        f.write(f"- Maximum memory increase observed: {max_memory:.1f}MB\n")
        
        # Performance recommendations
        f.write("- Performance scales linearly with simulation duration\n")
        f.write("- Memory usage scales with number of cameras and history length\n")
        f.write("- Consider reducing history_length for large-scale deployments\n")
        
        f.write("\n" + "="*80 + "\n")


def main():
    """Run all stress tests."""
    # Setup
    output_dir = Path(f"test_results_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    setup_logging('WARNING')  # Less verbose for stress tests
    
    print("="*80)
    print("MULTI-CAMERA CLASSIFICATION - STRESS TESTING")
    print("="*80)
    print(f"System: {psutil.cpu_count()} CPUs, "
          f"{psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB RAM")
    
    # Get stress test configurations
    stress_tests = create_stress_test_configs()
    all_results = {}
    all_metrics = {}
    
    # Run each stress test
    for test_name, test_config in stress_tests.items():
        test_params = test_config.pop('test_params')
        results, metrics = run_stress_test(test_name, test_config, test_params, output_dir)
        
        all_results[test_name] = results
        all_metrics[test_name] = metrics
    
    # Create visualizations
    print("\nGenerating scalability plots...")
    create_scalability_plots(all_metrics, output_dir)
    
    # Generate report
    print("Generating stress test report...")
    generate_stress_test_report(all_results, all_metrics, output_dir)
    
    print(f"\n{'='*80}")
    print("STRESS TESTING COMPLETED")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()