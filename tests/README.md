# Multi-Camera Classification System - Tests

This directory contains unit tests and comprehensive test scripts for the multi-camera classification system.

## Test Structure

### Unit Tests
- `test_energy_model.py` - Tests for energy harvesting and consumption
- `test_accuracy_model.py` - Tests for accuracy calculations
- `test_camera.py` - Tests for camera functionality
- `test_algorithms.py` - Tests for classification algorithms
- `test_algorithms_integration.py` - Integration tests

### Comprehensive Test Suite
- `run_comprehensive_test.py` - Full system test with visualizations

## Running Tests

### Unit Tests
Run all unit tests:
```bash
python -m pytest tests/
```

Run specific test:
```bash
python tests/test_camera.py
```

### Comprehensive Test Suite
The comprehensive test suite runs all three algorithms, generates comparative visualizations, and creates a detailed report.

#### Quick Start
```bash
# From project root
./run_tests.sh
```

#### Custom Parameters
```bash
# Long simulation with low frequency
./run_tests.sh --duration 5000 --frequency 0.05

# Short test run
./run_tests.sh --duration 200 --frequency 0.2

# Custom configuration
./run_tests.sh --config configs/my_config.yaml
```

#### Direct Python Execution
```bash
python tests/run_comprehensive_test.py --duration 1000 --frequency 0.1
```

## Test Outputs

The comprehensive test creates a timestamped directory with:

```
test_results_YYYYMMDD_HHMMSS/
├── data/
│   ├── test_config.json      # Configuration used
│   ├── results_fixed.json    # Fixed algorithm results
│   ├── results_variable.json # Variable algorithm results
│   ├── results_unknown.json  # Unknown algorithm results
│   └── all_results.json      # Combined results
├── plots/
│   ├── energy_dynamics.png      # Energy levels over time
│   ├── accuracy_metrics.png     # Accuracy comparison
│   └── performance_timeline.png # Performance over time
└── reports/
    └── test_summary.txt         # Comprehensive summary report
```

## Understanding Results

### Energy Dynamics Plot
Shows how battery levels change over time for each algorithm:
- Average energy (solid line)
- Min-max range (shaded area)
- Comparative view of all algorithms

### Accuracy Metrics Plot
Compares algorithm performance:
- Overall classification accuracy
- Number of constraint violations
- Energy vs accuracy violations

### Performance Timeline Plot
Shows temporal behavior:
- Rolling accuracy (10-event window)
- Camera participation patterns
- Real-time performance trends

### Summary Report
Text report containing:
- Configuration summary
- Per-algorithm performance metrics
- Comparative analysis
- Best performer identification

## Interpreting Results

### Key Metrics
1. **Accuracy**: Percentage of correct classifications
2. **Energy Violations**: Times when no cameras had sufficient energy
3. **Accuracy Violations**: Times when collective accuracy was below threshold
4. **Avg Cameras/Classification**: Energy efficiency metric

### Expected Behavior
- **Fixed Algorithm**: Consistent performance, deterministic scheduling
- **Variable Algorithm**: Better energy distribution, adaptive scheduling  
- **Unknown Algorithm**: Game-theoretic adaptation, may have more violations initially

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from project root
2. **Display Issues**: Script uses Agg backend for headless operation
3. **Memory Issues**: Reduce duration for large simulations

### Debug Mode
Enable detailed logging:
```bash
python tests/run_comprehensive_test.py --duration 100 --frequency 0.1 2>&1 | tee debug.log
```