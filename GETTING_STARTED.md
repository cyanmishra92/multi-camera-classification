# Multi-Camera Classification System - Getting Started Guide

## Overview

The Multi-Camera Classification System is a game-theoretic framework for coordinating energy-harvesting cameras to perform object classification tasks. The system balances classification accuracy with energy sustainability through strategic participation decisions.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [Running Simulations](#running-simulations)
5. [Configuration](#configuration)
6. [Algorithms](#algorithms)
7. [Understanding Results](#understanding-results)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- PyYAML
- Pandas (optional, for data analysis)
- Matplotlib (optional, for visualization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MULTICAMv1/multi_camera_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python3 run_simulation.py --help
```

## Quick Start

Run your first simulation with default settings:

```bash
python3 run_simulation.py --algorithm fixed --duration 1000 --frequency 0.1
```

This will:
- Use the fixed frequency algorithm
- Run for 1000 time units
- Generate classification events at 0.1 events per time unit
- Save results to `results.json`

## System Architecture

### Core Components

1. **Camera Model** (`src/core/camera.py`)
   - Energy harvesting and consumption
   - Position-dependent accuracy
   - Classification capabilities

2. **Energy Model** (`src/core/energy_model.py`)
   - Battery dynamics
   - Recharge rates
   - Energy constraints

3. **Accuracy Model** (`src/core/accuracy_model.py`)
   - Energy-dependent accuracy
   - Collective accuracy calculations
   - Correlation factors

4. **Network Coordinator** (`src/core/network.py`)
   - Manages camera fleet
   - Coordinates algorithms
   - Tracks performance

### Algorithms

1. **Fixed Frequency** (Algorithm 1)
   - For high-frequency classification (≥ 1/Δ)
   - Round-robin scheduling
   - Deterministic participation

2. **Variable Frequency** (Algorithm 2)
   - For medium-frequency classification (< 1/Δ)
   - Subclass rotation
   - Energy diversity maintenance

3. **Unknown Frequency** (Algorithm 3)
   - For unknown classification rates
   - Game-theoretic decisions
   - Nash equilibrium convergence

## Running Simulations

### Basic Command Structure

```bash
python3 run_simulation.py [OPTIONS]
```

### Available Options

- `--config`: Path to configuration file (default: `configs/default_config.yaml`)
- `--algorithm`: Algorithm type: `fixed`, `variable`, or `unknown`
- `--duration`: Simulation duration in time units
- `--frequency`: Average classification frequency
- `--visualize`: Enable live visualization (if implemented)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--output`: Output file for results (default: `results.json`)

### Examples

1. **High-frequency scenario with fixed algorithm:**
```bash
python3 run_simulation.py --algorithm fixed --duration 5000 --frequency 0.2
```

2. **Low-frequency scenario with variable algorithm:**
```bash
python3 run_simulation.py --algorithm variable --duration 10000 --frequency 0.05
```

3. **Unknown frequency with game theory:**
```bash
python3 run_simulation.py --algorithm unknown --duration 10000 --frequency 0.1 --log-level DEBUG
```

## Configuration

### Configuration File Structure

The system uses YAML configuration files located in `configs/`:

```yaml
# Network configuration
network:
  num_cameras: 10        # Number of cameras
  num_classes: 3         # Number of camera classes
  num_objects: 5         # Number of object types

# Energy parameters
energy:
  battery_capacity: 1000      # Maximum battery capacity (cap)
  recharge_rate: 10          # Energy harvest rate (r)
  classification_cost: 50    # Energy per classification (δ)
  min_operational: 100       # Minimum operational energy (δ_min)

# Accuracy parameters
accuracy:
  max_accuracy: 0.95         # Maximum accuracy (α_max)
  min_accuracy_ratio: 0.3    # Minimum accuracy ratio (β)
  correlation_factor: 0.2    # Error correlation factor (ρ)

# Game theory parameters
game_theory:
  reward_scale: 1.0               # Reward scaling factor (γ)
  incorrect_penalty: 0.5          # Penalty for incorrect classification
  non_participation_penalty: 0.8  # Penalty for not participating
  discount_factor: 0.9           # Future value discount (β)

# Algorithm parameters
algorithms:
  min_accuracy_threshold: 0.8    # Minimum collective accuracy (α_min)
  history_length: 10             # Classification history length (k)
```

### Creating Custom Configurations

1. Copy the default configuration:
```bash
cp configs/default_config.yaml configs/my_config.yaml
```

2. Edit parameters as needed

3. Run with custom configuration:
```bash
python3 run_simulation.py --config configs/my_config.yaml
```

## Understanding Results

### Output File Structure

The simulation saves results in JSON format with the following structure:

```json
{
  "network_stats": {
    "total_cameras": 10,
    "avg_energy": 850.5,
    "avg_accuracy": 0.92,
    "total_classifications": 100,
    "successful_classifications": 85,
    "accuracy": 0.85,
    "energy_violations": 2,
    "accuracy_violations": 3
  },
  "algorithm_type": "fixed",
  "duration": 1000,
  "classification_frequency": 0.1,
  "performance_history": [...],
  "energy_history": [...]
}
```

### Key Metrics

1. **Accuracy**: Overall classification success rate
2. **Energy Violations**: Times when no cameras had sufficient energy
3. **Accuracy Violations**: Times when collective accuracy was below threshold
4. **Average Energy**: Mean energy level across all cameras
5. **Participation Rate**: Average number of cameras per classification

### Analyzing Results

Use the provided analysis scripts or load results in Python:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Extract metrics
stats = results['network_stats']
print(f"Overall Accuracy: {stats['accuracy']:.3f}")
print(f"Energy Efficiency: {stats['avg_cameras_per_classification']:.2f} cameras/classification")

# Plot energy history
energy_history = results['energy_history']
timestamps = [e['timestamp'] for e in energy_history]
avg_energies = [e['avg_energy'] for e in energy_history]

plt.plot(timestamps, avg_energies)
plt.xlabel('Time')
plt.ylabel('Average Energy')
plt.title('Camera Energy Over Time')
plt.show()
```

## Advanced Usage

### Running Batch Experiments

Create a script to run multiple simulations:

```python
import subprocess
import json

algorithms = ['fixed', 'variable', 'unknown']
frequencies = [0.05, 0.1, 0.2]

results = {}

for algo in algorithms:
    for freq in frequencies:
        output_file = f"results_{algo}_{freq}.json"
        
        # Run simulation
        subprocess.run([
            'python3', 'run_simulation.py',
            '--algorithm', algo,
            '--frequency', str(freq),
            '--duration', '5000',
            '--output', output_file
        ])
        
        # Load and store results
        with open(output_file, 'r') as f:
            results[f"{algo}_{freq}"] = json.load(f)

# Analyze results...
```

### Extending the System

1. **Adding New Algorithms**: 
   - Inherit from `BaseClassificationAlgorithm`
   - Implement `select_cameras()` method
   - Add to `network.py` algorithm selection

2. **Custom Energy Models**:
   - Modify `EnergyModel` class
   - Add solar patterns, weather effects, etc.

3. **Enhanced Accuracy Models**:
   - Extend `AccuracyModel` class
   - Add object-specific accuracy, environmental factors

### Running Tests

Run the test suite to verify system functionality:

```bash
# Run all tests
python3 -m unittest discover tests

# Run specific test modules
python3 tests/test_energy_model.py
python3 tests/test_accuracy_model.py
python3 tests/test_camera.py
python3 tests/test_algorithms_integration.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the project root directory
   - Check that Python path includes the src directory

2. **JSON Serialization Errors**
   - Usually caused by NumPy types
   - Already handled in the code, but check for custom modifications

3. **Low Accuracy Results**
   - Check energy parameters - cameras may be depleting too quickly
   - Verify accuracy thresholds are reasonable
   - Ensure sufficient cameras for collective accuracy

4. **No Classification Events**
   - Classification frequency may be too low
   - Increase duration or frequency

### Performance Tips

1. **Memory Usage**: For long simulations, consider:
   - Reducing history length in algorithms
   - Sampling performance metrics instead of storing all
   - Using smaller time steps

2. **Speed Optimization**:
   - Use vectorized operations where possible
   - Consider parallel camera updates
   - Profile code to identify bottlenecks

### Getting Help

1. Check the documentation in `docs/`
2. Review the example configurations
3. Examine test cases for usage examples
4. Check logs with `--log-level DEBUG`

## Next Steps

1. **Experiment with Parameters**: Try different energy and accuracy settings
2. **Compare Algorithms**: Run the same scenario with different algorithms
3. **Visualize Results**: Create plots to understand system behavior
4. **Extend the System**: Add features like heterogeneous cameras or dynamic networks
5. **Optimize Performance**: Fine-tune parameters for your specific use case

Happy simulating!