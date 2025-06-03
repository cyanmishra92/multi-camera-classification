# Project Structure

This document describes the organization of the Multi-Camera Classification System codebase.

## Directory Structure

```
multi_camera_classification/
├── src/                          # Source code
│   ├── algorithms/               # Classification algorithms
│   │   ├── base_algorithm.py     # Abstract base class
│   │   ├── fixed_frequency.py    # Algorithm 1: Fixed frequency
│   │   ├── variable_frequency.py # Algorithm 2: Variable frequency
│   │   ├── unknown_frequency.py  # Algorithm 3: Unknown frequency
│   │   └── enhanced_*.py         # Enhanced algorithm variants
│   ├── core/                     # Core system components
│   │   ├── camera.py            # Camera model
│   │   ├── energy_model.py      # Energy harvesting/consumption
│   │   ├── accuracy_model.py    # Accuracy calculations
│   │   ├── network.py           # Network coordinator
│   │   └── enhanced_*.py        # Enhanced model variants
│   ├── game_theory/             # Game-theoretic components
│   │   ├── strategic_agent.py   # Strategic decision making
│   │   ├── utility_functions.py # Utility calculations
│   │   └── nash_equilibrium.py  # Nash equilibrium solver
│   ├── federated_learning/      # Federated learning (placeholder)
│   │   └── federated_trainer.py
│   ├── utils/                   # Utilities
│   │   ├── config_parser.py    # Configuration parsing
│   │   ├── logger.py           # Logging setup
│   │   └── metrics.py          # Performance metrics
│   ├── visualization/           # Visualization tools
│   │   ├── accuracy_plots.py   
│   │   ├── energy_plots.py
│   │   └── live_dashboard.py
│   └── main.py                  # Main entry point
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   │   ├── test_camera.py
│   │   ├── test_energy_model.py
│   │   ├── test_accuracy_model.py
│   │   └── test_algorithms.py
│   ├── integration/             # Integration tests
│   │   └── test_algorithms_integration.py
│   ├── scenarios/               # Comprehensive test scenarios
│   │   ├── run_comprehensive_test.py  # Basic system test
│   │   ├── test_edge_cases.py        # Edge case testing
│   │   ├── test_stress.py            # Stress & scalability
│   │   ├── test_parameter_sensitivity.py # Parameter analysis
│   │   ├── test_algorithm_comparison.py  # Algorithm comparison
│   │   ├── run_all_tests.py          # Master test runner
│   │   └── README.md                  # Test documentation
│   └── README.md                # Test overview
│
├── scripts/                     # Utility scripts
│   ├── demos/                   # Demonstration scripts
│   │   ├── demonstrate_improvements.py
│   │   └── run_full_demo.py
│   ├── experiments/             # Experimental scripts
│   │   ├── test_adaptive_algorithm.py
│   │   ├── test_enhanced_accuracy.py
│   │   ├── test_game_theory.py
│   │   └── tune_parameters.py
│   └── tools/                   # Utility tools
│
├── configs/                     # Configuration files
│   ├── default_config.yaml      # Default parameters
│   └── simulation_config.yaml   # Simulation settings
│
├── docs/                        # Documentation
│   ├── api/                     # API documentation
│   ├── concept/                 # Conceptual papers
│   │   └── *.pdf               # Research papers
│   ├── theory/                  # Theoretical background
│   └── PROJECT_STRUCTURE.md     # This file
│
├── notebooks/                   # Jupyter notebooks
│   └── example_simulation.ipynb
│
├── data/                        # Data directories
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
│
├── examples/                    # Example scripts
│   ├── analyze_results.py
│   └── compare_algorithms.py
│
├── run_simulation.py            # Main simulation runner
├── run_tests.sh                 # Test launcher script
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── README.md                    # Project overview
├── GETTING_STARTED.md          # Getting started guide
├── HOW_TO_RUN.md              # Running instructions
└── .gitignore                  # Git ignore file
```

## Key Components

### Core System (`src/core/`)
- **camera.py**: Camera model with energy-dependent accuracy
- **energy_model.py**: Battery dynamics and energy harvesting
- **accuracy_model.py**: Accuracy calculations based on energy
- **network.py**: Coordinates multiple cameras

### Algorithms (`src/algorithms/`)
- **base_algorithm.py**: Abstract base class for all algorithms
- **fixed_frequency.py**: Round-robin scheduling for high frequency
- **variable_frequency.py**: Subclass rotation for medium frequency
- **unknown_frequency.py**: Game-theoretic approach for unknown frequency

### Game Theory (`src/game_theory/`)
- **strategic_agent.py**: Camera as strategic decision maker
- **utility_functions.py**: Reward and cost calculations
- **nash_equilibrium.py**: Finds stable participation patterns

### Testing (`tests/`)
- **unit/**: Unit tests for individual components
- **integration/**: Integration tests for system behavior
- **scenarios/**: Comprehensive test scenarios including:
  - Edge case testing
  - Stress testing
  - Parameter sensitivity analysis
  - Algorithm comparison

## Usage Patterns

### Running Simulations
```bash
# Basic simulation
python run_simulation.py --algorithm fixed --duration 1000

# Using specific config
python run_simulation.py --config configs/my_config.yaml
```

### Running Tests
```bash
# All scenario tests
python tests/scenarios/run_all_tests.py

# Specific test suite
./run_tests.sh --duration 500

# Unit tests
python -m pytest tests/unit/
```

### Analyzing Results
```bash
# Compare algorithms
python examples/compare_algorithms.py

# Analyze specific results
python examples/analyze_results.py --file results.json
```

## Development Guidelines

### Adding New Features
1. Implement in appropriate module under `src/`
2. Add unit tests in `tests/unit/`
3. Add integration tests if needed
4. Update documentation

### Adding New Algorithms
1. Inherit from `BaseClassificationAlgorithm`
2. Implement `select_cameras()` method
3. Add to algorithm selection in `network.py`
4. Create tests in `tests/unit/test_algorithms.py`

### Running Experiments
1. Use scripts in `scripts/experiments/`
2. Save results with timestamps
3. Use visualization tools for analysis

## Configuration

### Config Files
- YAML format for easy editing
- Hierarchical structure
- Parameter validation in `config_parser.py`

### Key Parameters
- Network: num_cameras, num_classes
- Energy: battery_capacity, recharge_rate, classification_cost
- Accuracy: max_accuracy, min_accuracy_ratio
- Game Theory: rewards, penalties, discount_factor