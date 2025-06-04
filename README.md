# Multi-Camera Energy-Aware Classification Framework

A game-theoretic framework for multi-camera classification systems with energy harvesting constraints. This project implements energy-aware camera coordination algorithms that optimize the trade-off between classification accuracy and energy sustainability.

## 🎯 Overview

This framework addresses the challenge of coordinating multiple energy-harvesting cameras for object classification tasks. Key features include:

- **Energy-Dependent Accuracy Modeling**: Camera accuracy degrades as battery depletes
- **Game-Theoretic Decision Making**: Cameras act as strategic agents optimizing long-term utility
- **Multiple Scheduling Algorithms**: Three algorithms for different frequency scenarios
- **Nash Equilibrium Analysis**: Provable convergence to stable participation patterns
- **Federated Learning Integration**: Adaptive model updates across the network

## 📊 Key Concepts

### Energy Model
- Cameras harvest energy at a constant rate
- Classification consumes fixed energy
- Accuracy depends on current energy level

### Game Theory
- Each camera is a strategic agent
- Utility = Reward - Cost
- Nash equilibrium ensures stable behavior

### Algorithms
1. **Fixed Frequency**: Round-robin scheduling for high-frequency classification
2. **Variable Frequency**: Subclass rotation for medium-frequency scenarios  
3. **Unknown Frequency**: Probabilistic participation with adaptive thresholds

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cyanmishra92/multi-camera-classification
cd multi_camera_classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .[dev,viz]
```

### Run Example Simulation

```bash
# Run with default settings
python src/main.py --algorithm fixed --duration 1000 --visualize

# Run with custom configuration
python src/main.py --config configs/my_config.yaml --algorithm unknown
```

### Basic Usage

```python
from src.core.network import CameraNetwork, NetworkConfig
from src.core.energy_model import EnergyParameters
from src.core.accuracy_model import AccuracyParameters

# Create network configuration
config = NetworkConfig(
    num_cameras=10,
    num_classes=3,
    num_objects=5,
    energy_params=EnergyParameters(
        capacity=1000,
        recharge_rate=10,
        classification_cost=50,
        min_operational=100
    ),
    accuracy_params=AccuracyParameters(
        max_accuracy=0.95,
        min_accuracy_ratio=0.3
    )
)

# Create network
network = CameraNetwork(config)

# Set algorithm
network.set_algorithm('fixed', min_accuracy_threshold=0.8)

# Run classification
result = network.classify_object(
    object_position=np.array([10, 20, 0]),
    true_label=1
)
```

## 📁 Project Structure

```
multi_camera_classification/
├── src/
│   ├── core/               # Core camera and network models
│   ├── algorithms/         # Classification algorithms
│   ├── game_theory/        # Game-theoretic components
│   ├── federated_learning/ # Federated learning modules
│   ├── utils/              # Utilities
│   └── visualization/      # Visualization tools
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files
├── experiments/            # Experiment scripts
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## 🔧 Configuration

The framework uses YAML configuration files. See `configs/default_config.yaml` for an example:

```yaml
network:
  num_cameras: 10
  num_classes: 3
  
energy:
  battery_capacity: 1000
  recharge_rate: 10
  classification_cost: 50
  
accuracy:
  max_accuracy: 0.95
  min_accuracy_ratio: 0.3
  
game_theory:
  reward_scale: 1.0
  incorrect_penalty: 0.5
  non_participation_penalty: 0.8
  discount_factor: 0.9
```

## 📚 Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Detailed setup and usage instructions
- [API Documentation](docs/api/) - Complete API reference
- [Theory Guide](docs/theory/) - Mathematical foundations
- [Examples](notebooks/) - Jupyter notebook examples

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## 📈 Performance

Expected performance characteristics:
- Energy Efficiency: 40-60% reduction vs always-on
- Accuracy Maintenance: Within 5% of full participation
- Nash Equilibrium: Convergence in 10-20 iterations
- Scalability: Tested up to 100 cameras

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{multi_camera_classification,
  title = {Multi-Camera Energy-Aware Classification Framework},
  author = {Cyan Subhra Mishra},
  year = {2025},
  url = {https://github.com/cyanmishra92/multi-camera-classification}
}
```

## 🙏 Acknowledgments

This work builds upon research in energy-harvesting sensor networks, game theory, and federated learning.

---

# GETTING_STARTED.md

# Getting Started with Multi-Camera Classification Framework

This guide will help you get started with the multi-camera classification framework, from installation to running your first simulation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Understanding the Framework](#understanding-the-framework)
4. [Running Your First Simulation](#running-your-first-simulation)
5. [Configuring Simulations](#configuring-simulations)
6. [Choosing an Algorithm](#choosing-an-algorithm)
7. [Analyzing Results](#analyzing-results)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- (Optional) CUDA-capable GPU for federated learning experiments

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/multi_camera_classification.git
cd multi_camera_classification
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install the Package

```bash
# Install with all dependencies
pip install -e .[dev,viz]

# Or install without visualization tools
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run tests to verify installation
pytest tests/test_camera.py -v

# Check if main script runs
python src/main.py --help
```

## Understanding the Framework

### Core Concepts

1. **Cameras**: Energy-harvesting sensors with position-dependent accuracy
2. **Energy Model**: Battery dynamics with constant harvesting rate
3. **Accuracy Model**: Energy-dependent accuracy degradation
4. **Algorithms**: Different scheduling strategies for camera coordination
5. **Game Theory**: Strategic decision-making for participation

### Key Parameters

- `num_cameras`: Total number of cameras in the network
- `num_classes`: Number of camera groups for round-robin scheduling
- `battery_capacity`: Maximum energy storage per camera
- `classification_cost`: Energy consumed per classification
- `min_accuracy_threshold`: Minimum required collective accuracy

## Running Your First Simulation

### Basic Simulation

```bash
# Run a basic simulation with the fixed frequency algorithm
python src/main.py \
    --algorithm fixed \
    --duration 1000 \
    --frequency 0.1 \
    --visualize
```

### Using Python API

```python
import numpy as np
from src.core.network import CameraNetwork, NetworkConfig
from src.core.energy_model import EnergyParameters

# Create network
config = NetworkConfig(
    num_cameras=10,
    num_classes=3,
    num_objects=5
)
network = CameraNetwork(config)

# Set algorithm
network.set_algorithm('fixed')

# Simulate classification events
for i in range(100):
    # Random object position
    position = np.random.uniform(-50, 50, size=3)
    position[2] = 0  # Ground level
    
    # Classify
    result = network.classify_object(position, true_label=0)
    
    # Update time
    network.update_time(1.0)

# Get statistics
stats = network.get_network_stats()
print(f"Accuracy: {stats['accuracy']:.3f}")
```

## Configuring Simulations

### Configuration File Structure

Create a custom configuration file `configs/my_config.yaml`:

```yaml
network:
  num_cameras: 20
  num_classes: 4
  num_objects: 10

energy:
  battery_capacity: 2000
  recharge_rate: 15
  classification_cost: 75
  min_operational: 150
  high_threshold: 0.8
  low_threshold: 0.3

accuracy:
  max_accuracy: 0.98
  min_accuracy_ratio: 0.2
  correlation_factor: 0.1

game_theory:
  reward_scale: 1.5
  incorrect_penalty: 0.6
  non_participation_penalty: 1.0
  discount_factor: 0.85

algorithms:
  min_accuracy_threshold: 0.85
  history_length: 15

simulation:
  duration: 5000
  time_step: 1.0
  random_seed: 42
```

### Using Custom Configuration

```bash
python src/main.py --config configs/my_config.yaml --algorithm variable
```

## Choosing an Algorithm

### Algorithm 1: Fixed Frequency
- Best for: High-frequency classification (≥ 1/Δ)
- Features: Round-robin scheduling, predictable behavior
- Usage: `--algorithm fixed`

### Algorithm 2: Variable Frequency
- Best for: Medium-frequency classification (< 1/Δ)
- Features: Subclass rotation, energy diversity
- Usage: `--algorithm variable --frequency 0.05`

### Algorithm 3: Unknown Frequency
- Best for: Unknown or variable classification frequency
- Features: Game-theoretic adaptation, probabilistic participation
- Usage: `--algorithm unknown`

## Analyzing Results

### Output Files

Simulations produce JSON output files with:
- Network statistics
- Performance metrics
- Energy history
- Classification results

### Visualization

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Plot energy history
energy_history = results['energy_history']
times = [e['timestamp'] for e in energy_history]
avg_energies = [e['avg_energy'] for e in energy_history]

plt.plot(times, avg_energies)
plt.xlabel('Time')
plt.ylabel('Average Energy')
plt.title('Network Energy Over Time')
plt.show()
```

### Performance Metrics

Key metrics to monitor:
- **Classification Accuracy**: Percentage of correct classifications
- **Energy Violations**: Times when cameras couldn't participate due to low energy
- **Accuracy Violations**: Times when collective accuracy was below threshold
- **Average Participation**: Number of cameras participating per classification

## Advanced Usage

### Custom Camera Positions

```python
# Define custom camera positions
positions = np.array([
    [0, 0, 30],    # Camera 1 at origin, 30m high
    [50, 0, 25],   # Camera 2 east, 25m high
    [-50, 0, 25],  # Camera 3 west, 25m high
    # ... more positions
])

config = NetworkConfig(
    num_cameras=len(positions),
    camera_positions=positions,
    # ... other parameters
)
```

### Implementing Custom Algorithms

```python
from src.algorithms.base_algorithm import BaseClassificationAlgorithm

class MyCustomAlgorithm(BaseClassificationAlgorithm):
    def select_cameras(self, instance_id, current_time):
        # Your custom selection logic
        selected = []
        for i, camera in enumerate(self.cameras):
            if camera.can_classify() and some_custom_condition:
                selected.append(i)
        return selected
```

### Running Experiments

```python
# Run multiple simulations with different parameters
import itertools

frequencies = [0.05, 0.1, 0.2]
algorithms = ['fixed', 'variable', 'unknown']

for freq, alg in itertools.product(frequencies, algorithms):
    network = create_network_from_config(config)
    results = run_simulation(
        network, 
        algorithm_type=alg,
        classification_frequency=freq,
        duration=1000
    )
    
    # Save results
    with open(f'results_{alg}_{freq}.json', 'w') as f:
        json.dump(results, f)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project directory
   - Check virtual environment is activated
   - Reinstall with `pip install -e .`

2. **Low Accuracy**
   - Increase `num_cameras` or decrease `num_classes`
   - Reduce `classification_frequency`
   - Adjust energy parameters

3. **Energy Violations**
   - Increase `battery_capacity`
   - Reduce `classification_cost`
   - Increase `recharge_rate`

### Debug Mode

Enable detailed logging:

```bash
python src/main.py --log-level DEBUG
```

### Getting Help

- Check the [API Documentation](docs/api/)
- Review example notebooks in `notebooks/`
- Open an issue on GitHub

## Next Steps

1. Explore different algorithm configurations
2. Implement custom utility functions
3. Try federated learning experiments
4. Contribute improvements back to the project

Happy simulating! 🎥🔋
