# How to Run - Multi-Camera Classification System

This guide will help you get started with the multi-camera classification system. Follow these steps to set up and run the simulation.

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cyanmishra92/multi-camera-classification.git
cd multi-camera-classification
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Basic Simulation

```bash
# Run the main simulation with default settings
python run_simulation.py

# For more options:
python run_simulation.py --help
```

## Detailed Usage

### Running Different Algorithms

The system includes three main algorithms:

1. **Fixed Frequency Algorithm** (default)
   ```bash
   python run_simulation.py --algorithm fixed
   ```

2. **Variable Frequency Algorithm**
   ```bash
   python run_simulation.py --algorithm variable
   ```

3. **Unknown Frequency Algorithm**
   ```bash
   python run_simulation.py --algorithm unknown
   ```

### Advanced Features

#### Enhanced Algorithms with Game Theory

Run the enhanced algorithms that include game theory and position awareness:

```bash
# Run enhanced fixed frequency algorithm
python src/main_enhanced.py --algorithm enhanced_fixed

# Run game-theoretic algorithm
python src/main_enhanced.py --algorithm game_theoretic

# Run adaptive parameter algorithm
python src/main_enhanced.py --algorithm adaptive
```

#### Full Demonstration

To see all features in action:

```bash
python run_full_demo.py
```

This will:
- Run all algorithm variants
- Compare performance metrics
- Generate comprehensive PDF reports
- Save results to JSON files

### Configuration

#### Basic Configuration

Edit `configs/default_config.yaml` to modify:
- Number of cameras
- Simulation duration
- Energy parameters
- Accuracy thresholds

#### Advanced Configuration

For detailed simulations, use `configs/simulation_config.yaml`:

```yaml
simulation:
  duration: 1000
  num_cameras: 5
  
cameras:
  energy_capacity: 1000
  solar_generation_rate: 2.0
  
algorithms:
  energy_threshold: 100
  min_cameras_active: 2
```

### Analyzing Results

#### View Results

After running simulations, check:
- `results.json` - Raw simulation data
- `DEMO_SUMMARY.json` - Summary of all algorithms
- `*.png` files - Visualization plots
- `comprehensive_analysis_*.pdf` - Full analysis reports

#### Compare Algorithms

```bash
python examples/compare_algorithms.py
```

#### Analyze Specific Results

```bash
python examples/analyze_results.py --file results.json
```

### Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run specific test categories:
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/test_algorithms_integration.py
```

### Visualization

The system generates several types of visualizations:

1. **Energy Dynamics** - Shows energy levels over time
2. **Accuracy Plots** - Classification accuracy trends
3. **Algorithm Comparison** - Bar charts comparing metrics
4. **Comprehensive Analysis** - Multi-page PDF reports

All visualizations are saved as PNG or PDF files in the working directory.

## Common Issues

### ImportError

If you encounter import errors:
```bash
# Install in development mode
pip install -e .
```

### Memory Issues

For large simulations, increase available memory or reduce:
- Number of cameras
- Simulation duration
- Visualization frequency

### No Module Named 'src'

Ensure you're running from the project root directory:
```bash
cd multi-camera-classification
python run_simulation.py
```

## Next Steps

1. **Explore Examples**
   - Check the `examples/` directory for more usage patterns
   - Review `notebooks/example_simulation.ipynb` for interactive exploration

2. **Customize Algorithms**
   - Extend `src/algorithms/base_algorithm.py`
   - Implement your own selection strategies

3. **Add New Features**
   - Integrate real camera hardware
   - Add new ML models for classification
   - Implement distributed deployment

## Getting Help

- Check existing documentation in `docs/`
- Review test files for usage examples
- Open an issue on GitHub for bugs or questions

## Performance Tips

- Start with small simulations (100-500 timesteps)
- Use fixed frequency algorithm for baseline
- Enable visualization only when needed
- Monitor memory usage for long simulations

Happy simulating!