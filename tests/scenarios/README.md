# Test Scenarios for Multi-Camera Classification System

This directory contains comprehensive test scenarios for evaluating the multi-camera classification system under various conditions.

## Test Suite Overview

### 1. Comprehensive Test (`run_comprehensive_test.py`)
Basic functionality test that runs all three algorithms and generates standard visualizations.
- **Purpose**: Baseline performance evaluation
- **Output**: Energy dynamics, accuracy metrics, performance timeline plots
- **Duration**: ~1-2 minutes

### 2. Edge Case Testing (`test_edge_cases.py`)
Tests system behavior under extreme conditions:
- Very low battery scenarios
- High frequency classification
- Single camera operation
- No energy harvesting
- Extreme accuracy requirements
- **Purpose**: Identify system limits and failure modes
- **Output**: Comparison plots for each edge case

### 3. Stress Testing (`test_stress.py`)
Evaluates system scalability and performance:
- Large numbers of cameras (up to 200)
- Long duration simulations
- High frequency operations
- Memory usage tracking
- **Purpose**: Determine scalability limits
- **Output**: Scalability analysis plots and performance metrics

### 4. Parameter Sensitivity Analysis (`test_parameter_sensitivity.py`)
Analyzes how sensitive the system is to parameter changes:
- Energy parameters (capacity, recharge rate, cost)
- Accuracy parameters (max accuracy, min ratio)
- Game theory parameters (rewards, penalties)
- **Purpose**: Identify critical parameters for tuning
- **Output**: Sensitivity plots and heatmap analysis

### 5. Algorithm Comparison (`test_algorithm_comparison.py`)
Comprehensive comparison across diverse scenarios:
- Different frequency regimes
- Energy-scarce environments
- Large networks
- Variable frequency patterns
- Day/night cycles
- **Purpose**: Determine optimal algorithm for each use case
- **Output**: Performance matrix and comparison plots

## Running Tests

### Run All Tests
```bash
python tests/scenarios/run_all_tests.py
```

### Run Specific Tests
```bash
# Run only edge case and stress tests
python tests/scenarios/run_all_tests.py --only edge stress

# Skip sensitivity analysis
python tests/scenarios/run_all_tests.py --skip sensitivity
```

### Run Individual Tests
```bash
# Basic comprehensive test
python tests/scenarios/run_comprehensive_test.py

# Edge cases
python tests/scenarios/test_edge_cases.py

# Stress testing
python tests/scenarios/test_stress.py

# Parameter sensitivity
python tests/scenarios/test_parameter_sensitivity.py

# Algorithm comparison
python tests/scenarios/test_algorithm_comparison.py
```

## Output Structure

Each test creates a timestamped directory with results:
```
test_results_TESTNAME_YYYYMMDD_HHMMSS/
├── data/          # Raw JSON results
├── plots/         # Visualization PNGs
└── reports/       # Text summary reports
```

## Interpreting Results

### Key Metrics
- **Accuracy**: Classification success rate (0-1)
- **Energy Violations**: Times when no cameras could participate
- **Accuracy Violations**: Times when collective accuracy was below threshold
- **Avg Cameras/Classification**: Energy efficiency metric

### Algorithm Selection Guide
Based on test results:
- **Fixed Frequency**: Best for high, regular classification rates
- **Variable Frequency**: Best for medium, known frequencies
- **Unknown Frequency**: Best for irregular or unknown patterns

### Performance Expectations
- **Small networks (≤20 cameras)**: All algorithms perform well
- **Large networks (>50 cameras)**: Fixed algorithm most efficient
- **Energy-scarce**: Unknown algorithm adapts best
- **High accuracy requirements**: May need multiple cameras per event

## Customizing Tests

### Adding New Scenarios
1. Create new test file following the pattern
2. Define scenarios with configurations
3. Implement visualization and reporting
4. Add to `run_all_tests.py`

### Modifying Parameters
Edit configuration dictionaries in each test file:
```python
base_config = {
    'network': {...},
    'energy': {...},
    'accuracy': {...},
    'game_theory': {...}
}
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce simulation duration or camera count
2. **Display errors**: Matplotlib uses 'Agg' backend for headless operation
3. **Import errors**: Ensure running from project root

### Debug Mode
Enable verbose logging:
```python
setup_logging('DEBUG')  # Instead of 'INFO' or 'WARNING'
```

## Performance Notes
- Stress tests may take 5-10 minutes
- Sensitivity analysis runs many simulations (10-15 minutes)
- Use `--skip stress sensitivity` for quick tests