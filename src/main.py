"""Main entry point for multi-camera classification simulations."""

import argparse
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .core.network import CameraNetwork, NetworkConfig
from .core.energy_model import EnergyParameters
from .core.accuracy_model import AccuracyParameters
from .game_theory.utility_functions import UtilityParameters
from .utils.logger import setup_logging
from .visualization.live_dashboard import LiveDashboard

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_network_from_config(config: Dict[str, Any]) -> CameraNetwork:
    """Create camera network from configuration dictionary."""
    # Extract parameters
    network_config = config['network']
    energy_config = config['energy']
    accuracy_config = config['accuracy']
    
    # Create parameter objects
    energy_params = EnergyParameters(
        capacity=energy_config['battery_capacity'],
        recharge_rate=energy_config['recharge_rate'],
        classification_cost=energy_config['classification_cost'],
        min_operational=energy_config['min_operational']
    )
    
    accuracy_params = AccuracyParameters(
        max_accuracy=accuracy_config['max_accuracy'],
        min_accuracy_ratio=accuracy_config['min_accuracy_ratio'],
        correlation_factor=accuracy_config['correlation_factor']
    )
    
    utility_params = None
    if 'game_theory' in config:
        gt_config = config['game_theory']
        utility_params = UtilityParameters(
            reward_scale=gt_config['reward_scale'],
            incorrect_penalty=gt_config['incorrect_penalty'],
            non_participation_penalty=gt_config['non_participation_penalty'],
            discount_factor=gt_config['discount_factor']
        )
    
    # Create network configuration
    net_config = NetworkConfig(
        num_cameras=network_config['num_cameras'],
        num_classes=network_config['num_classes'],
        num_objects=network_config['num_objects'],
        energy_params=energy_params,
        accuracy_params=accuracy_params,
        utility_params=utility_params
    )
    
    return CameraNetwork(net_config)


def run_simulation(
    network: CameraNetwork,
    algorithm_type: str,
    duration: int,
    time_step: float,
    classification_frequency: float,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Run a simulation with the specified parameters.
    
    Args:
        network: Camera network
        algorithm_type: Type of algorithm to use
        duration: Simulation duration in time steps
        time_step: Time step size
        classification_frequency: Average classification frequency
        visualize: Whether to show live visualization
        
    Returns:
        Simulation results
    """
    # Set algorithm
    network.set_algorithm(
        algorithm_type,
        classification_frequency=classification_frequency,
        min_accuracy_threshold=0.8
    )
    
    # Initialize visualization if requested
    dashboard = None
    if visualize:
        dashboard = LiveDashboard(network)
        dashboard.start()
    
    # Simulation loop
    logger.info(f"Starting simulation with {algorithm_type} algorithm")
    
    # Generate classification events
    classification_times = []
    current_time = 0
    
    # Use Poisson process for arrivals
    while current_time < duration:
        interval = np.random.exponential(1.0 / classification_frequency)
        current_time += interval
        if current_time < duration:
            classification_times.append(current_time)
    
    logger.info(f"Generated {len(classification_times)} classification events")
    
    # Run simulation
    event_idx = 0
    
    for step in range(int(duration / time_step)):
        current_time = step * time_step
        
        # Check for classification events
        while event_idx < len(classification_times) and classification_times[event_idx] <= current_time:
            # Generate random object
            object_position = np.random.uniform(-50, 50, size=3)
            object_position[2] = 0  # Ground level
            
            true_label = np.random.randint(0, 2)  # Binary classification
            
            # Classify
            result = network.classify_object(object_position, true_label)
            
            logger.debug(
                f"Classification at t={current_time:.2f}: "
                f"success={result['success']}, "
                f"cameras={len(result['selected_cameras'])}"
            )
            
            event_idx += 1
        
        # Update network time
        network.update_time(time_step)
        
        # Update visualization
        if dashboard and step % 10 == 0:
            dashboard.update()
        
        # Log progress
        if step % 100 == 0:
            stats = network.get_network_stats()
            logger.info(
                f"Step {step}/{int(duration/time_step)}: "
                f"Accuracy={stats.get('recent_accuracy', 0):.3f}, "
                f"Avg Energy={stats.get('avg_energy', 0):.1f}"
            )
    
    # Cleanup visualization
    if dashboard:
        dashboard.stop()
    
    # Collect results
    results = {
        'network_stats': network.get_network_stats(),
        'algorithm_type': algorithm_type,
        'duration': duration,
        'classification_frequency': classification_frequency,
        'total_events': len(classification_times),
        'performance_history': network.performance_history,
        'energy_history': network.energy_history
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-camera classification simulation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['fixed', 'variable', 'unknown'],
        default='fixed',
        help='Algorithm type to use'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=1000,
        help='Simulation duration'
    )
    parser.add_argument(
        '--frequency',
        type=float,
        default=0.1,
        help='Classification frequency'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable live visualization'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create network
    network = create_network_from_config(config)
    
    # Run simulation
    results = run_simulation(
        network,
        args.algorithm,
        args.duration,
        time_step=1.0,
        classification_frequency=args.frequency,
        visualize=args.visualize
    )
    
    # Save results
    import json
    
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        else:
            return obj
    
    with open(args.output, 'w') as f:
        results_json = convert_to_json_serializable(results)
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    stats = results['network_stats']
    print("\nSimulation Summary:")
    print(f"Algorithm: {args.algorithm}")
    print(f"Duration: {args.duration}")
    print(f"Total Classifications: {stats['total_classifications']}")
    print(f"Overall Accuracy: {stats.get('accuracy', 0):.3f}")
    print(f"Recent Accuracy: {stats.get('recent_accuracy', 0):.3f}")
    print(f"Energy Violations: {stats.get('energy_violations', 0)}")
    print(f"Accuracy Violations: {stats.get('accuracy_violations', 0)}")


if __name__ == "__main__":
    main()