# Guide to Running Full Experimental Evaluation

## Quick Start

The easiest way to run all experiments is using the provided bash script:

```bash
# Run all experiments (will prompt before each scale)
./run_full_experiments.sh

# Run specific scale
./run_full_experiments.sh small    # 144 experiments, ~5 minutes
./run_full_experiments.sh medium   # 28,800 experiments, ~2 hours
./run_full_experiments.sh large    # 144,000 experiments, ~8 hours
```

## Experiment Scales

### Small Scale (Quick Test)
- **Experiments**: 144
- **Duration**: ~5 minutes
- **Parameters**:
  - Network sizes: 10 cameras
  - Classes: 3
  - Frequencies: {0.05, 0.1, 0.5} Hz
  - Energy capacities: 1000
  - Recharge rates: 10
  - Accuracy thresholds: {0.7, 0.8}
  - Simulation duration: 2000 time units
  - Runs: 3 per configuration

### Medium Scale (Conference Paper)
- **Experiments**: 28,800
- **Duration**: ~2 hours
- **Parameters**:
  - Network sizes: {10, 30} cameras
  - Classes: {3, 5}
  - Frequencies: {0.01, 0.05, 0.1, 0.5, 1.0} Hz
  - Energy capacities: {500, 1000, 2000}
  - Recharge rates: {5, 10, 20}
  - Accuracy thresholds: {0.6, 0.7, 0.8, 0.9}
  - Simulation duration: 5000 time units
  - Runs: 5 per configuration

### Full Scale (Journal Paper)
- **Experiments**: 144,000+
- **Duration**: ~8 hours
- **Parameters**:
  - Network sizes: {10, 50, 100} cameras
  - Classes: {3, 5, 10}
  - Frequencies: {0.01, 0.05, 0.1, 0.5, 1.0} Hz
  - Energy capacities: {500, 1000, 2000}
  - Recharge rates: {5, 10, 20}
  - Accuracy thresholds: {0.6, 0.7, 0.8, 0.9}
  - Simulation duration: 10000 time units
  - Runs: 10 per configuration

## Manual Execution

### 1. Run Experiments

```bash
# Small scale with 4 workers
python experiments/full_scale_experiments.py \
    --scale small_scale \
    --parallel \
    --workers 4

# Medium scale with 8 workers
python experiments/full_scale_experiments.py \
    --scale medium_scale \
    --parallel \
    --workers 8

# Full scale with 16 workers (recommended for cluster)
python experiments/full_scale_experiments.py \
    --scale full_scale \
    --parallel \
    --workers 16
```

### 2. Generate Visualizations

```bash
# After experiments complete
python experiments/visualize_results.py \
    research_results_<scale>_<timestamp>/raw_results/all_results.csv \
    research_results_<scale>_<timestamp>/figures/
```

### 3. Generate Research Plots (Advanced)

```bash
# For publication-quality figures
python src/visualization/research_plots.py \
    research_results_<scale>_<timestamp>/raw_results/all_results.csv \
    paper_figures/
```

## Output Directory Structure

```
experimental_results/
├── small_scale/
│   └── YYYYMMDD_HHMMSS/
│       ├── raw_results/
│       │   ├── all_results.csv
│       │   └── all_results.json
│       ├── aggregated_results/
│       │   ├── algorithm_summary.csv
│       │   ├── frequency_summary.csv
│       │   └── size_summary.csv
│       ├── figures/
│       │   ├── accuracy_comparison.png/pdf
│       │   ├── frequency_adaptation.png/pdf
│       │   ├── fairness_comparison.png/pdf
│       │   └── energy_violations.png/pdf
│       ├── logs/
│       │   └── experiment.log
│       └── summary_report.txt
├── medium_scale/
│   └── ...
├── full_scale/
│   └── ...
└── logs/
    ├── experiment_small_scale_TIMESTAMP.log
    ├── experiment_medium_scale_TIMESTAMP.log
    └── experiment_full_scale_TIMESTAMP.log
```

## Monitoring Progress

### During Execution
```bash
# Watch experiment progress
tail -f experimental_results/logs/experiment_<scale>_<timestamp>.log

# Check intermediate results
ls -la research_results_<scale>_<timestamp>/raw_results/
```

### Resource Usage
```bash
# Monitor CPU usage
htop

# Monitor memory
free -h

# Check disk space
df -h
```

## Customizing Experiments

### Modify Parameters
Edit `experiments/full_scale_experiments.py`:

```python
PARAM_CONFIGS = {
    'custom_scale': {
        'network_sizes': [20, 40],
        'num_classes': [4],
        'frequencies': [0.1, 0.2, 0.3],
        'energy_capacities': [1500],
        'recharge_rates': [15],
        'accuracy_thresholds': [0.75],
        'durations': [3000],
        'runs': 5
    }
}
```

### Add New Algorithms
1. Implement in `src/algorithms/`
2. Add to `ALGORITHMS` dict in `full_scale_experiments.py`
3. Add to algorithm creation logic

## Troubleshooting

### Out of Memory
- Reduce number of parallel workers
- Run smaller scale first
- Use sequential mode: `--no-parallel`

### Experiments Timeout
- Increase timeout in bash script
- Check for infinite loops in algorithms
- Reduce simulation duration

### Missing Results
- Check logs for errors
- Verify all dependencies installed
- Ensure sufficient disk space

## Performance Tips

1. **CPU-bound**: Use more workers (up to CPU cores - 1)
2. **Memory-bound**: Reduce workers or batch size
3. **I/O-bound**: Use SSD for results directory
4. **Network experiments**: Run on cluster with job scheduler

## For HPC/Cluster

Create a SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=multicam_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G

module load python/3.8
source venv/bin/activate

./run_full_experiments.sh large
```

## Results Analysis

After experiments complete:

1. **Check summary reports** in each results directory
2. **View visualizations** in figures/ subdirectory
3. **Analyze raw data** using pandas:

```python
import pandas as pd

# Load results
df = pd.read_csv('path/to/all_results.csv')

# Custom analysis
print(df.groupby('algorithm')['overall_accuracy'].describe())
```

## Publishing Results

1. Use figures from `figures/` directory
2. Summary statistics from `aggregated_results/`
3. Raw data available for reproducibility
4. All configurations saved in `experiment_list.json`

## Contact

For issues or questions about running experiments:
- Check logs first
- Verify environment setup
- Ensure all dependencies installed