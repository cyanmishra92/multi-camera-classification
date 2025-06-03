#!/usr/bin/env python3
"""Parameter tuning script to achieve 80% accuracy target."""

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

from src.main_enhanced import create_enhanced_network_from_config, load_config
from src.algorithms.adaptive_parameter_tuner import AdaptiveParameterTuner, TuningParameters
from src.game_theory.utility_functions import UtilityParameters
from src.core.enhanced_accuracy_model import EnhancedAccuracyParameters


def run_parameter_tuning():
    """Run adaptive parameter tuning to achieve target accuracy."""
    print("=" * 60)
    print("ADAPTIVE PARAMETER TUNING")
    print("=" * 60)
    print("Target: 80% accuracy with <2.5 cameras per classification\n")
    
    # Load base config
    config = load_config('configs/default_config.yaml')
    
    # Initialize tuner
    tuner = AdaptiveParameterTuner(
        target_accuracy=0.8,
        target_efficiency=2.0,
        learning_rate=0.15,
        momentum=0.8
    )
    
    # Tuning iterations
    n_iterations = 10
    iteration_duration = 1000
    
    results = []
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        print("-" * 40)
        
        # Get current parameters
        params = tuner.params
        
        # Update network configuration with tuned parameters
        config['accuracy']['min_accuracy_threshold'] = params.min_accuracy_threshold
        config['accuracy']['distance_decay'] = params.distance_decay
        config['accuracy']['angle_penalty'] = params.angle_penalty
        config['accuracy']['overlap_bonus'] = params.overlap_bonus
        config['accuracy']['optimal_distance'] = 25.0  # Fixed for now
        
        # Create network with tuned parameters
        network = create_enhanced_network_from_config(config, use_enhanced=True)
        
        # Set game theory parameters
        network.config.utility_params = UtilityParameters(
            reward_scale=params.reward_scale,
            incorrect_penalty=params.incorrect_penalty,
            non_participation_penalty=params.non_participation_penalty,
            discount_factor=params.discount_factor
        )
        
        # Run short simulation
        start_time = time.time()
        
        # Set algorithm with tuned parameters
        network.set_algorithm(
            'fixed',
            classification_frequency=0.1,
            min_accuracy_threshold=params.min_accuracy_threshold,
            position_weight=params.position_weight,
            convergence_threshold=params.convergence_threshold
        )
        
        # Run simulation
        np.random.seed(42 + iteration)  # Different seed each iteration
        
        classification_times = []
        current_time = 0
        while current_time < iteration_duration:
            interval = np.random.exponential(1.0 / 0.1)
            current_time += interval
            if current_time < iteration_duration:
                classification_times.append(current_time)
        
        # Track detailed metrics
        accuracy_history = []
        efficiency_history = []
        violation_count = 0
        
        event_idx = 0
        for step in range(iteration_duration):
            current_time = float(step)
            
            # Process classification events
            while event_idx < len(classification_times) and classification_times[event_idx] <= current_time:
                # Generate object
                object_position = np.random.uniform(-40, 40, size=3)
                object_position[2] = 0
                true_label = np.random.randint(0, 2)
                
                # Classify with adaptive threshold
                if hasattr(network.algorithm, 'min_accuracy_threshold'):
                    # Adjust threshold dynamically
                    recent_acc = np.mean(accuracy_history[-20:]) if accuracy_history else 0.5
                    violation_rate = violation_count / max(1, len(accuracy_history))
                    
                    adaptive_threshold = tuner.get_adaptive_threshold(recent_acc, violation_rate)
                    network.algorithm.min_accuracy_threshold = adaptive_threshold
                
                result = network.classify_object(object_position, true_label)
                
                # Track metrics
                if result['success'] is not None:
                    accuracy_history.append(1 if result['success'] else 0)
                    efficiency_history.append(len(result.get('participating_cameras', [])))
                    
                    if result.get('collective_accuracy', 1.0) < params.min_accuracy_threshold:
                        violation_count += 1
                
                event_idx += 1
            
            # Update network time
            network.update_time(1.0)
        
        elapsed = time.time() - start_time
        
        # Calculate performance metrics
        overall_accuracy = np.mean(accuracy_history) if accuracy_history else 0
        recent_accuracy = np.mean(accuracy_history[-50:]) if len(accuracy_history) > 50 else overall_accuracy
        avg_efficiency = np.mean(efficiency_history) if efficiency_history else 0
        
        performance_metrics = {
            'accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'avg_cameras_per_classification': avg_efficiency,
            'energy_violations': network.algorithm.energy_violations,
            'accuracy_violations': network.algorithm.accuracy_violations,
            'total_violations': network.algorithm.energy_violations + network.algorithm.accuracy_violations,
            'runtime': elapsed
        }
        
        # Store results
        results.append({
            'iteration': iteration,
            'parameters': params.__dict__.copy(),
            'performance': performance_metrics
        })
        
        # Print current performance
        print(f"  Accuracy: {overall_accuracy:.3f} (recent: {recent_accuracy:.3f})")
        print(f"  Efficiency: {avg_efficiency:.2f} cameras/classification")
        print(f"  Violations: {performance_metrics['total_violations']}")
        print(f"  Min Acc Threshold: {params.min_accuracy_threshold:.3f}")
        print(f"  Position Weight: {params.position_weight:.3f}")
        print(f"  Reward Scale: {params.reward_scale:.3f}")
        
        # Update parameters for next iteration
        tuner.update_parameters(performance_metrics)
        
        # Early stopping if target achieved
        if overall_accuracy >= 0.78 and avg_efficiency <= 2.5 and performance_metrics['total_violations'] < 10:
            print(f"\n✓ Target achieved in {iteration + 1} iterations!")
            break
    
    # Save tuning history
    tuner.save_tuning_history('parameter_tuning_history.json')
    
    # Create visualization
    create_tuning_plots(results, tuner)
    
    # Test best parameters
    print("\n" + "=" * 60)
    print("TESTING BEST PARAMETERS")
    print("=" * 60)
    
    test_best_parameters(config, tuner.best_params)
    
    return results, tuner


def create_tuning_plots(results, tuner):
    """Create plots showing parameter tuning progress."""
    
    with PdfPages('parameter_tuning_results.pdf') as pdf:
        
        # Page 1: Performance Evolution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Parameter Tuning Progress', fontsize=16)
        
        iterations = [r['iteration'] for r in results]
        
        # Accuracy evolution
        ax = axes[0, 0]
        accuracies = [r['performance']['accuracy'] for r in results]
        recent_accs = [r['performance']['recent_accuracy'] for r in results]
        
        ax.plot(iterations, accuracies, 'b-', label='Overall', linewidth=2)
        ax.plot(iterations, recent_accs, 'g--', label='Recent', linewidth=2)
        ax.axhline(y=0.8, color='red', linestyle='--', label='Target', alpha=0.5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Efficiency evolution
        ax = axes[0, 1]
        efficiencies = [r['performance']['avg_cameras_per_classification'] for r in results]
        
        ax.plot(iterations, efficiencies, 'b-', linewidth=2)
        ax.axhline(y=2.0, color='red', linestyle='--', label='Target', alpha=0.5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cameras per Classification')
        ax.set_title('Efficiency Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Violations
        ax = axes[1, 0]
        violations = [r['performance']['total_violations'] for r in results]
        
        ax.plot(iterations, violations, 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Violations')
        ax.set_title('Constraint Violations')
        ax.grid(True, alpha=0.3)
        
        # Score evolution
        ax = axes[1, 1]
        scores = [tuner._calculate_score(r['performance']) for r in results]
        
        ax.plot(iterations, scores, 'g-', linewidth=2)
        ax.scatter([np.argmax(scores)], [max(scores)], color='red', s=100, 
                  marker='*', label='Best', zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance Score')
        ax.set_title('Overall Performance Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Parameter Evolution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Evolution During Tuning', fontsize=16)
        
        # Key parameters to plot
        param_names = [
            ('min_accuracy_threshold', 'Min Accuracy Threshold'),
            ('position_weight', 'Position Weight'),
            ('reward_scale', 'Reward Scale'),
            ('energy_weight', 'Energy Weight'),
            ('overlap_bonus', 'Overlap Bonus'),
            ('discount_factor', 'Discount Factor')
        ]
        
        for idx, (param, title) in enumerate(param_names):
            ax = axes[idx // 3, idx % 3]
            values = [r['parameters'][param] for r in results]
            
            ax.plot(iterations, values, 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Best Parameters Summary
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Best Parameters Found', fontsize=16)
        
        if tuner.best_params:
            params_dict = tuner.best_params.__dict__
            param_names = list(params_dict.keys())
            param_values = list(params_dict.values())
            
            y_pos = np.arange(len(param_names))
            
            bars = ax.barh(y_pos, param_values, alpha=0.7)
            
            # Color code by value
            for i, (bar, val) in enumerate(zip(bars, param_values)):
                if val < 0.3:
                    bar.set_color('red')
                elif val < 0.7:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
                
                # Add value label
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(param_names)
            ax.set_xlabel('Parameter Value')
            ax.set_title(f'Best Score: {tuner.best_score:.2f}')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"\nTuning plots saved to: parameter_tuning_results.pdf")


def test_best_parameters(config, best_params):
    """Test the best parameters on a longer simulation."""
    if not best_params:
        print("No best parameters found!")
        return
    
    print("\nTesting best parameters on extended simulation...")
    
    # Update config with best parameters
    config['accuracy']['min_accuracy_threshold'] = best_params.min_accuracy_threshold
    config['accuracy']['distance_decay'] = best_params.distance_decay
    config['accuracy']['angle_penalty'] = best_params.angle_penalty
    config['accuracy']['overlap_bonus'] = best_params.overlap_bonus
    
    # Create network
    network = create_enhanced_network_from_config(config, use_enhanced=True)
    
    # Set game theory parameters
    network.config.utility_params = UtilityParameters(
        reward_scale=best_params.reward_scale,
        incorrect_penalty=best_params.incorrect_penalty,
        non_participation_penalty=best_params.non_participation_penalty,
        discount_factor=best_params.discount_factor
    )
    
    # Run longer test
    from src.main_enhanced import run_simulation
    
    result = run_simulation(
        network,
        'fixed',
        duration=5000,
        time_step=1.0,
        classification_frequency=0.1,
        visualize=False
    )
    
    # Print results
    stats = result['network_stats']
    print(f"\nFinal Test Results:")
    print(f"  Overall Accuracy: {stats.get('accuracy', 0):.3f}")
    print(f"  Recent Accuracy: {stats.get('recent_accuracy', 0):.3f}")
    print(f"  Avg Cameras/Classification: {stats.get('avg_cameras_per_classification', 0):.2f}")
    print(f"  Energy Violations: {stats.get('energy_violations', 0)}")
    print(f"  Accuracy Violations: {stats.get('accuracy_violations', 0)}")
    
    # Save best configuration
    best_config = {
        'parameters': best_params.__dict__,
        'test_results': {
            'accuracy': stats.get('accuracy', 0),
            'recent_accuracy': stats.get('recent_accuracy', 0),
            'avg_cameras_per_classification': stats.get('avg_cameras_per_classification', 0),
            'violations': stats.get('energy_violations', 0) + stats.get('accuracy_violations', 0)
        }
    }
    
    with open('best_parameters.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest parameters saved to: best_parameters.json")
    
    # Check if we achieved the target
    if stats.get('accuracy', 0) >= 0.8:
        print("\n✅ SUCCESS: Achieved 80% accuracy target!")
    else:
        print(f"\n⚠️  Accuracy {stats.get('accuracy', 0):.3f} still below 80% target")


if __name__ == "__main__":
    results, tuner = run_parameter_tuning()
    print("\nParameter tuning complete!")