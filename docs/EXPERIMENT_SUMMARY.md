# Multi-Camera Energy-Harvesting Network: Experimental Summary

## Overview

We have developed a comprehensive research framework for evaluating energy-aware camera selection algorithms in multi-camera networks. Our approach introduces three novel algorithms optimized for different operational scenarios and compares them against five established baselines from the literature.

## Our Algorithms

### 1. **Fixed Frequency Algorithm** (High-frequency scenarios)
- **Key Innovation**: Energy-aware round-robin with optional game theory
- **Best for**: f ≥ 1/Δ (high classification rates)
- **Complexity**: O(log m) per selection

### 2. **Variable Frequency Algorithm** (Low-frequency scenarios)  
- **Key Innovation**: Dynamic subclassing with energy redistribution
- **Best for**: f < 1/Δ (low classification rates)
- **Complexity**: O(m) per selection

### 3. **Unknown Frequency Algorithm** (Adaptive scenarios)
- **Key Innovation**: Game-theoretic probabilistic participation
- **Best for**: Unknown or variable classification rates
- **Complexity**: O(m) per selection

## Baseline Algorithms

1. **Random Selection** (RAND) - Naive random k-camera selection
2. **Greedy Energy** (GREEDY) - Select highest-energy cameras
3. **Round Robin** (RR) - Deterministic cycling
4. **Coverage-Based** (COV) - Maximize spatial coverage
5. **Threshold-Based** (THRESH) - Energy threshold activation

## Key Metrics

### Primary Metrics
- **Classification Accuracy**: Overall and recent (last 100 events)
- **Energy Efficiency**: Violations, lifetime, sustainability
- **Fairness**: Jain's index, participation variance

### Secondary Metrics
- **Response Quality**: Collective accuracy, cameras per event
- **Adaptability**: Convergence time, frequency robustness
- **Computational**: Runtime, memory usage

## Experimental Design

### Network Configurations
- **Small**: 10 cameras, 3 classes
- **Medium**: 50 cameras, 5 classes
- **Large**: 100 cameras, 10 classes

### Parameter Space
- **Frequencies**: {0.01, 0.05, 0.1, 0.5, 1.0} Hz
- **Energy Capacity**: {500, 1000, 2000} units
- **Recharge Rate**: {5, 10, 20} units/time
- **Accuracy Threshold**: {0.6, 0.7, 0.8, 0.9}

### Total Experiments
- 8 algorithms × 3 network sizes × 5 frequencies × 3 capacities × 3 recharge rates × 4 thresholds × 5 runs = **14,400 experiments**

## Expected Results

### Performance Improvements Over Baselines
1. **Accuracy**: 15-30% improvement
2. **Energy Efficiency**: 20-40% better lifetime
3. **Fairness**: More balanced camera utilization
4. **Adaptability**: Superior handling of variable workloads

### Algorithm Selection Guide
| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| Predictable high frequency | Fixed Frequency | Optimal scheduling with minimal overhead |
| Predictable low frequency | Variable Frequency | Ensures energy availability through subclassing |
| Unpredictable workload | Unknown Frequency | Adaptive game-theoretic approach |
| Energy-critical | Variable/Unknown | Better energy management |
| Accuracy-critical | Unknown with high threshold | Strategic selection for quality |

## Visualization Suite

### Main Figures (8 publication-ready plots)
1. **Accuracy Comparison** - Bar charts across network sizes
2. **Energy Efficiency** - Multi-panel analysis
3. **Scalability Analysis** - Performance vs network size
4. **Frequency Adaptation** - Algorithm performance across frequencies
5. **Fairness Metrics** - Jain's index and variance
6. **Parameter Sensitivity** - Impact of key parameters
7. **Convergence Analysis** - Adaptation behavior
8. **Ablation Study** - Component contributions

## Implementation

### Code Structure
```
src/algorithms/
├── fixed_frequency.py         # Our Algorithm 1
├── variable_frequency.py      # Our Algorithm 2  
├── unknown_frequency.py       # Our Algorithm 3
└── baselines/                 # 5 baseline implementations
    ├── random_selection.py
    ├── greedy_energy.py
    ├── round_robin.py
    ├── coverage_based.py
    └── threshold_based.py

experiments/
├── research_experiments.py    # Full experimental framework
└── demo_research_experiments.py # Quick demo

src/visualization/
└── research_plots.py         # Publication-quality figures
```

### Running Experiments

1. **Quick Demo** (5 minutes):
```bash
python experiments/demo_research_experiments.py
```

2. **Full Experiments** (several hours):
```bash
python experiments/research_experiments.py --parallel
```

3. **Generate Figures**:
```bash
python src/visualization/research_plots.py results.csv paper_figures/
```

## Theoretical Contributions

1. **Game-Theoretic Framework**: First application of Nash equilibrium to camera selection
2. **Adaptive Subclassing**: Novel approach to handle low-frequency scenarios
3. **Energy-Accuracy Co-optimization**: Joint optimization rather than sequential
4. **Regret Bounds**: Theoretical guarantees on learning performance

## Practical Impact

### Applications
- Smart city surveillance
- Wildlife monitoring  
- Industrial IoT
- Disaster response

### Advantages
- **No training required**: Algorithms work out-of-the-box
- **Distributed implementation**: Cameras can make local decisions
- **Robust**: Handles failures and dynamic environments
- **Scalable**: Linear complexity in camera count

## Next Steps

1. Run full experiments on cluster
2. Statistical significance testing
3. Additional ablation studies
4. Real-world deployment testing
5. Extension to multi-hop networks

## Reproducibility

All code, configurations, and scripts are available for reproduction:
- Automated experiment runner
- Docker container for environment
- Configuration files for all scenarios
- Analysis and plotting scripts