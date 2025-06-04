#!/usr/bin/env python3
"""Simple visualization of experimental results."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Algorithm mapping
ALGO_MAP = {
    'fixed': 'Fixed-Freq (Ours)',
    'variable': 'Variable-Freq (Ours)', 
    'unknown': 'Unknown-Freq (Ours)',
    'random': 'Random',
    'greedy': 'Greedy-Energy',
    'round_robin': 'Round-Robin',
    'coverage': 'Coverage-Based',
    'threshold': 'Threshold'
}

# Colors for our algorithms vs baselines
ALGO_COLORS = {
    'fixed': '#1f77b4',      # Blue
    'variable': '#ff7f0e',   # Orange  
    'unknown': '#2ca02c',    # Green
    'random': '#d62728',     # Red
    'greedy': '#9467bd',     # Purple
    'round_robin': '#8c564b', # Brown
    'coverage': '#e377c2',   # Pink
    'threshold': '#7f7f7f'   # Gray
}

def plot_accuracy_comparison(df, output_dir):
    """Plot accuracy comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by algorithm
    algo_stats = df.groupby('algorithm').agg({
        'overall_accuracy': ['mean', 'std']
    }).round(3)
    
    algo_stats.columns = ['mean', 'std']
    algo_stats = algo_stats.sort_values('mean', ascending=False)
    
    # Create bar plot
    algorithms = algo_stats.index
    x_pos = np.arange(len(algorithms))
    means = algo_stats['mean'].values
    stds = algo_stats['std'].values
    
    # Color bars differently for our algorithms
    colors = [ALGO_COLORS[algo] for algo in algorithms]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_MAP[algo] for algo in algorithms], rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Highlight our algorithms
    for i, algo in enumerate(algorithms):
        if algo in ['fixed', 'variable', 'unknown']:
            bars[i].set_linewidth(2)
            bars[i].set_edgecolor('darkblue')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_frequency_adaptation(df, output_dir):
    """Plot performance across different frequencies."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get unique frequencies
    frequencies = sorted(df['frequency'].unique())
    
    # Plot for each algorithm
    for algo in ['fixed', 'variable', 'unknown', 'random', 'greedy']:
        if algo not in df['algorithm'].unique():
            continue
            
        algo_data = df[df['algorithm'] == algo]
        freq_stats = algo_data.groupby('frequency')['overall_accuracy'].agg(['mean', 'std'])
        
        ax.plot(frequencies, freq_stats['mean'], 'o-', 
                label=ALGO_MAP[algo], color=ALGO_COLORS[algo], 
                linewidth=2, markersize=8)
        
        # Add error bars
        ax.fill_between(frequencies, 
                       freq_stats['mean'] - freq_stats['std'],
                       freq_stats['mean'] + freq_stats['std'],
                       alpha=0.2, color=ALGO_COLORS[algo])
    
    ax.set_xlabel('Classification Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Performance vs Classification Frequency', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_adaptation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'frequency_adaptation.pdf', bbox_inches='tight')
    plt.close()

def plot_fairness_comparison(df, output_dir):
    """Plot fairness metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Jain's fairness index
    fairness_stats = df.groupby('algorithm')['jains_fairness'].agg(['mean', 'std'])
    fairness_stats = fairness_stats.sort_values('mean', ascending=False)
    
    algorithms = fairness_stats.index
    x_pos = np.arange(len(algorithms))
    means = fairness_stats['mean'].values
    stds = fairness_stats['std'].values
    colors = [ALGO_COLORS[algo] for algo in algorithms]
    
    ax1.bar(x_pos, means, yerr=stds, capsize=5,
            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax1.set_title('Camera Utilization Fairness', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([ALGO_MAP[algo] for algo in algorithms], rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Fairness')
    
    # Participation variance
    var_stats = df.groupby('algorithm')['participation_variance'].agg(['mean', 'std'])
    var_stats = var_stats.sort_values('mean')
    
    algorithms = var_stats.index
    x_pos = np.arange(len(algorithms))
    means = var_stats['mean'].values
    stds = var_stats['std'].values
    colors = [ALGO_COLORS[algo] for algo in algorithms]
    
    ax2.bar(x_pos, means, yerr=stds, capsize=5,
            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Participation Variance', fontsize=12)
    ax2.set_title('Camera Participation Variance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([ALGO_MAP[algo] for algo in algorithms], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fairness_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fairness_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_energy_violations(df, output_dir):
    """Plot energy violation statistics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by algorithm
    violation_stats = df.groupby('algorithm')['energy_violations'].sum()
    violation_stats = violation_stats.sort_values()
    
    algorithms = violation_stats.index
    x_pos = np.arange(len(algorithms))
    violations = violation_stats.values
    colors = [ALGO_COLORS[algo] for algo in algorithms]
    
    bars = ax.bar(x_pos, violations, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Total Energy Violations', fontsize=12)
    ax.set_title('Energy Constraint Violations', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_MAP[algo] for algo in algorithms], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(violations):
        ax.text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_violations.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'energy_violations.pdf', bbox_inches='tight')
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate a summary table of results."""
    summary = df.groupby('algorithm').agg({
        'overall_accuracy': ['mean', 'std'],
        'energy_violations': 'sum',
        'jains_fairness': 'mean',
        'avg_cameras_per_event': 'mean',
        'runtime': 'mean'
    }).round(3)
    
    summary.columns = ['Accuracy (mean)', 'Accuracy (std)', 'Energy Violations', 
                      'Fairness', 'Avg Cameras', 'Runtime (s)']
    summary = summary.sort_values('Accuracy (mean)', ascending=False)
    
    # Map algorithm names
    summary.index = [ALGO_MAP[algo] for algo in summary.index]
    
    # Save as CSV and text
    summary.to_csv(output_dir / 'summary_table.csv')
    
    with open(output_dir / 'summary_table.txt', 'w') as f:
        f.write("EXPERIMENTAL RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Unique configurations: {len(df.groupby(['num_cameras', 'frequency', 'accuracy_threshold']))}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results_csv> [output_dir]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} experimental results")
    
    # Generate plots
    print("Generating visualizations...")
    plot_accuracy_comparison(df, output_dir)
    print("  - Accuracy comparison done")
    
    plot_frequency_adaptation(df, output_dir)
    print("  - Frequency adaptation done")
    
    plot_fairness_comparison(df, output_dir)
    print("  - Fairness comparison done")
    
    plot_energy_violations(df, output_dir)
    print("  - Energy violations done")
    
    generate_summary_table(df, output_dir)
    print("  - Summary table done")
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()