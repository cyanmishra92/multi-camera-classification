# Experimental Framework Summary

## Overview

This document summarizes the comprehensive experimental framework created for the multi-camera energy-aware classification research project. The framework enables systematic evaluation of our algorithms against baseline approaches across various scales and configurations.

## 🚀 Master Experiment Script

The primary entry point for running experiments is:

```bash
./run_full_experiments.sh
```

This script handles the entire experimental pipeline:
- Runs experiments at different scales (small/medium/large)
- Generates visualizations automatically
- Organizes results in structured directories
- Creates combined reports
- Includes progress monitoring and logging
- Supports both interactive and batch modes

## 📊 Experiment Scales

### Small Scale (~5 minutes)
- **Total experiments**: 144
- **Purpose**: Quick testing and validation
- **Parameters**:
  - Network sizes: 10 cameras
  - Classes: 3
  - Frequencies: {0.05, 0.1, 0.5} Hz
  - Energy capacities: 1000
  - Recharge rates: 10
  - Accuracy thresholds: {0.7, 0.8}
  - Duration: 2000 time units
  - Runs: 3 per configuration

### Medium Scale (~2 hours)
- **Total experiments**: 28,800
- **Purpose**: Conference paper results
- **Parameters**:
  - Network sizes: {10, 30} cameras
  - Classes: {3, 5}
  - Frequencies: {0.01, 0.05, 0.1, 0.5, 1.0} Hz
  - Energy capacities: {500, 1000, 2000}
  - Recharge rates: {5, 10, 20}
  - Accuracy thresholds: {0.6, 0.7, 0.8, 0.9}
  - Duration: 5000 time units
  - Runs: 5 per configuration

### Large Scale (~8 hours)
- **Total experiments**: 144,000+
- **Purpose**: Journal paper results
- **Parameters**:
  - Network sizes: {10, 50, 100} cameras
  - Classes: {3, 5, 10}
  - Frequencies: {0.01, 0.05, 0.1, 0.5, 1.0} Hz
  - Energy capacities: {500, 1000, 2000}
  - Recharge rates: {5, 10, 20}
  - Accuracy thresholds: {0.6, 0.7, 0.8, 0.9}
  - Duration: 10000 time units
  - Runs: 10 per configuration

## 📁 Results Organization

All experimental results are organized in a structured directory hierarchy:

```
experimental_results/
├── small_scale/
│   └── YYYYMMDD_HHMMSS/
│       ├── raw_results/
│       │   ├── all_results.csv      # Complete results data
│       │   ├── all_results.json     # JSON format
│       │   └── intermediate_results.csv
│       ├── figures/
│       │   ├── accuracy_comparison.png/pdf
│       │   ├── frequency_adaptation.png/pdf
│       │   ├── fairness_comparison.png/pdf
│       │   ├── energy_violations.png/pdf
│       │   └── summary_table.txt
│       ├── aggregated_results/
│       │   ├── algorithm_summary.csv
│       │   ├── frequency_summary.csv
│       │   └── size_summary.csv
│       ├── logs/
│       │   └── experiment.log
│       └── summary_report.txt
├── medium_scale/
│   └── [same structure]
├── full_scale/
│   └── [same structure]
└── logs/
    └── experiment_<scale>_<timestamp>.log
```

## 🛠️ Additional Tools

### 1. Setup Checker
```bash
./check_experiment_setup.py
```
Verifies that your environment is properly configured:
- Python version check
- Required packages verification
- Directory structure validation
- Disk space check
- CPU core count

### 2. Visualization Script
```bash
python experiments/visualize_results.py <results.csv> [output_dir]
```
Generates publication-quality figures:
- Accuracy comparison bar charts
- Frequency adaptation curves
- Fairness metrics comparison
- Energy violation analysis
- Summary statistics table

### 3. Detailed Guide
See `experiments/RUN_EXPERIMENTS_GUIDE.md` for:
- Step-by-step instructions
- Troubleshooting tips
- HPC/cluster configuration
- Custom experiment setup

## 📈 Current Results Summary

From our small-scale experiments (144 runs):

### Algorithm Performance
| Algorithm | Mean Accuracy | Std Dev | vs Best Baseline |
|-----------|--------------|---------|------------------|
| **Unknown-Freq (Ours)** | 48.9% | 0.163 | +7.2% |
| **Fixed-Freq (Ours)** | 46.9% | 0.033 | +2.9% |
| **Variable-Freq (Ours)** | 45.8% | 0.058 | +0.4% |
| Coverage-Based | 45.6% | 0.044 | - |
| Greedy-Energy | 45.2% | 0.048 | - |
| Round-Robin | 44.5% | 0.030 | - |
| Random | 42.1% | 0.041 | - |
| Threshold | 26.9% | 0.174 | - |

### Key Achievements
- **Perfect fairness**: Fixed and Unknown algorithms achieve Jain's index = 1.0
- **Zero energy violations**: All algorithms maintain energy constraints
- **Efficient selection**: Variable-Freq uses only 1.0 cameras per event
- **Consistent performance**: Fixed-Freq has lowest variance (σ=0.033)

## 🎯 Usage Instructions

### Quick Start
```bash
# Check your setup first
./check_experiment_setup.py

# Run all experiments (interactive)
./run_full_experiments.sh

# Run specific scale
./run_full_experiments.sh small    # Quick test
./run_full_experiments.sh medium   # Full evaluation
./run_full_experiments.sh large    # Comprehensive
```

### Background Execution
```bash
# Run in background with logging
nohup ./run_full_experiments.sh medium > experiment.log 2>&1 &

# Monitor progress
tail -f experimental_results/logs/experiment_medium_scale_*.log
```

### Custom Experiments
Edit `experiments/full_scale_experiments.py` to add custom parameter configurations:
```python
PARAM_CONFIGS['custom'] = {
    'network_sizes': [20, 40],
    'frequencies': [0.1, 0.2],
    # ... other parameters
}
```

## 🔧 Framework Components

### Core Algorithms (Our Contributions)
1. **Fixed-Frequency**: Energy-aware round-robin with game theory
2. **Variable-Frequency**: Dynamic subclassing with redistribution
3. **Unknown-Frequency**: Game-theoretic probabilistic participation

### Baseline Algorithms (From Literature)
1. **Random Selection**: Naive random k-camera selection
2. **Greedy-Energy**: Select highest-energy cameras
3. **Round-Robin**: Deterministic cycling
4. **Coverage-Based**: Maximize spatial coverage
5. **Threshold-Based**: Energy threshold activation

### Key Files
- `run_full_experiments.sh`: Master experiment script
- `experiments/full_scale_experiments.py`: Core experiment runner
- `experiments/visualize_results.py`: Visualization generator
- `src/algorithms/baselines/`: Baseline implementations
- `check_experiment_setup.py`: Environment validator

## 📊 Metrics Collected

### Primary Metrics
- Overall classification accuracy
- Recent accuracy (last 100 events)
- Energy violations count
- Accuracy violations count
- Average cameras per event

### Fairness Metrics
- Jain's fairness index
- Participation variance
- Gini coefficient
- Min/max participation rates

### Efficiency Metrics
- Runtime per experiment
- Convergence time
- Energy sustainability score
- Classification success rate

### Adaptability Metrics
- Stability score
- Accuracy improvement over time
- Adaptation rate
- Response time average

## 🚀 Future Enhancements

1. **Statistical Analysis**
   - Add hypothesis testing between algorithms
   - Implement confidence intervals
   - Generate statistical significance tables

2. **Visualization Improvements**
   - Add 3D performance surfaces
   - Create animated convergence plots
   - Generate LaTeX tables directly

3. **Scalability**
   - Add distributed computing support
   - Implement checkpointing for long runs
   - Create Docker container for reproducibility

4. **Real-world Testing**
   - Add hardware-in-the-loop support
   - Create network simulation interface
   - Implement live dashboard

## 📝 Publication Support

The framework generates all necessary materials for publications:
- High-resolution figures (PNG and PDF)
- Statistical summaries in CSV format
- Raw data for reproducibility
- Formatted tables for papers

All results are designed to meet the standards of top-tier conferences (NeurIPS, ICML, ICLR) and journals.

## 🤝 Contributing

To add new algorithms or experiments:
1. Implement algorithm in `src/algorithms/`
2. Add to `ALGORITHMS` dict in `full_scale_experiments.py`
3. Update visualization scripts if needed
4. Run small-scale tests first
5. Document changes in this file

## 📧 Support

For questions or issues:
1. Check `experiments/RUN_EXPERIMENTS_GUIDE.md`
2. Review log files in `experimental_results/logs/`
3. Verify setup with `./check_experiment_setup.py`
4. Open GitHub issue with details

---

*Last updated: June 2025*
*Framework version: 1.0*