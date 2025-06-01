"""Comprehensive visualization tools with PDF output."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ComprehensiveVisualizer:
    """
    Generate comprehensive plots for multi-camera classification results.
    
    All plots are saved to PDF files for terminal-only environments.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with style settings."""
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        
    def create_full_analysis_report(self, results_file: str, output_pdf: str = 'full_analysis_report.pdf'):
        """
        Create comprehensive analysis report from results file.
        
        Args:
            results_file: Path to JSON results file
            output_pdf: Output PDF filename
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        with PdfPages(output_pdf) as pdf:
            # Page 1: Overview Dashboard
            self._create_overview_dashboard(results, pdf)
            
            # Page 2: Energy Dynamics
            self._create_energy_dynamics_plots(results, pdf)
            
            # Page 3: Accuracy Traces
            self._create_accuracy_traces(results, pdf)
            
            # Page 4: Camera Utilization
            self._create_camera_utilization_plots(results, pdf)
            
            # Page 5: Violation Analysis
            self._create_violation_analysis(results, pdf)
            
            # Page 6: Performance Heatmaps
            self._create_performance_heatmaps(results, pdf)
            
            # Page 7: Algorithm Comparison
            self._create_algorithm_comparison(results, pdf)
            
            # Page 8: Network Topology
            self._create_network_topology_view(results, pdf)
            
        logger.info(f"Comprehensive analysis report saved to {output_pdf}")
    
    def _create_overview_dashboard(self, results: Dict, pdf: PdfPages):
        """Create overview dashboard page."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Multi-Camera Classification System - Performance Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Key metrics
        stats = results.get('network_stats', {})
        
        # Accuracy gauge
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._create_gauge_plot(ax1, stats.get('accuracy', 0), 'Overall Accuracy', 
                               target=0.8, color='green')
        
        # Recent accuracy gauge
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._create_gauge_plot(ax2, stats.get('recent_accuracy', 0), 'Recent Accuracy',
                               target=0.8, color='blue')
        
        # Energy efficiency
        ax3 = fig.add_subplot(gs[1, 0:2])
        efficiency = stats.get('avg_cameras_per_classification', 0)
        self._create_bar_metric(ax3, efficiency, 'Cameras per Classification',
                               target=2.0, lower_better=True)
        
        # Violations
        ax4 = fig.add_subplot(gs[1, 2:4])
        violations = stats.get('energy_violations', 0) + stats.get('accuracy_violations', 0)
        self._create_bar_metric(ax4, violations, 'Total Violations',
                               target=0, lower_better=True, color='red')
        
        # Performance timeline
        ax5 = fig.add_subplot(gs[2, :])
        if 'performance_history' in results:
            self._create_performance_timeline(ax5, results['performance_history'])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_gauge_plot(self, ax, value: float, label: str, target: float = 0.8, color='green'):
        """Create a gauge plot for metrics."""
        # Create semi-circular gauge
        theta = np.linspace(0, np.pi, 100)
        r_inner = 0.7
        r_outer = 1.0
        
        # Background
        ax.fill_between(theta, r_inner, r_outer, color='lightgray', alpha=0.3)
        
        # Value arc
        value_theta = value * np.pi
        theta_value = np.linspace(0, value_theta, 50)
        ax.fill_between(theta_value, r_inner, r_outer, color=color, alpha=0.7)
        
        # Target line
        target_theta = target * np.pi
        ax.plot([target_theta, target_theta], [r_inner-0.1, r_outer+0.1], 
               'r--', linewidth=3, label=f'Target: {target:.1%}')
        
        # Convert to Cartesian
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        
        # Labels
        ax.text(0, -0.1, f'{value:.1%}', ha='center', va='center', 
               fontsize=24, fontweight='bold')
        ax.text(0, 1.3, label, ha='center', va='center', fontsize=14)
        
        ax.axis('off')
        ax.legend(loc='upper right')
    
    def _create_bar_metric(self, ax, value: float, label: str, target: float = None,
                          lower_better: bool = False, color='blue'):
        """Create bar metric visualization."""
        ax.bar([label], [value], color=color, alpha=0.7)
        
        if target is not None:
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2,
                      label=f'Target: {target}')
        
        ax.set_ylabel('Value')
        ax.text(0, value + 0.1, f'{value:.2f}', ha='center', va='bottom',
               fontsize=14, fontweight='bold')
        
        if lower_better and value > target:
            ax.text(0, value/2, '⚠️', ha='center', va='center', fontsize=20)
        elif not lower_better and value < target:
            ax.text(0, value/2, '⚠️', ha='center', va='center', fontsize=20)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_performance_timeline(self, ax, performance_history: List[Dict]):
        """Create performance timeline."""
        if len(performance_history) < 2:
            return
            
        timestamps = [p['timestamp'] for p in performance_history]
        success_rate = []
        
        window = 20
        for i in range(window, len(performance_history)):
            window_data = performance_history[i-window:i]
            rate = sum(r['result']['success'] for r in window_data) / window
            success_rate.append(rate)
        
        ax.plot(timestamps[window:], success_rate, 'b-', linewidth=2)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(timestamps[window:], 0, success_rate, alpha=0.3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Success Rate (20-sample window)')
        ax.set_title('Classification Success Rate Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    def _create_energy_dynamics_plots(self, results: Dict, pdf: PdfPages):
        """Create detailed energy dynamics plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Energy Dynamics Analysis', fontsize=16)
        
        if 'energy_history' not in results:
            pdf.savefig(fig)
            plt.close()
            return
        
        energy_history = results['energy_history'][:1000]  # Limit to first 1000
        
        # Average energy over time
        ax = axes[0, 0]
        timestamps = [e['timestamp'] for e in energy_history]
        avg_energies = [e['avg_energy'] for e in energy_history]
        min_energies = [e['min_energy'] for e in energy_history]
        max_energies = [e['max_energy'] for e in energy_history]
        
        ax.plot(timestamps, avg_energies, 'b-', label='Average', linewidth=2)
        ax.fill_between(timestamps, min_energies, max_energies, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy Level')
        ax.set_title('Network Energy Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy distribution
        ax = axes[0, 1]
        final_energies = energy_history[-1]['camera_energies'] if energy_history else []
        if final_energies:
            ax.hist(final_energies, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(final_energies), color='red', linestyle='--',
                      label=f'Mean: {np.mean(final_energies):.1f}')
            ax.set_xlabel('Energy Level')
            ax.set_ylabel('Number of Cameras')
            ax.set_title('Final Energy Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Energy consumption patterns
        ax = axes[1, 0]
        if len(energy_history) > 1:
            consumption_rates = []
            for i in range(1, len(energy_history)):
                dt = energy_history[i]['timestamp'] - energy_history[i-1]['timestamp']
                if dt > 0:
                    de = energy_history[i-1]['avg_energy'] - energy_history[i]['avg_energy']
                    consumption_rates.append(de / dt)
            
            ax.plot(timestamps[1:], consumption_rates, 'r-', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy Consumption Rate')
            ax.set_title('Energy Consumption Pattern')
            ax.grid(True, alpha=0.3)
        
        # Energy efficiency metrics
        ax = axes[1, 1]
        if 'performance_history' in results:
            perf_history = results['performance_history']
            
            # Calculate energy per successful classification
            energy_per_success = []
            for i in range(100, len(perf_history), 10):
                window = perf_history[i-100:i]
                successes = sum(r['result']['success'] for r in window)
                if successes > 0:
                    # Estimate energy used
                    energy_used = len([r for r in window if r['result']['selected_cameras']]) * 50
                    energy_per_success.append(energy_used / successes)
            
            if energy_per_success:
                ax.plot(energy_per_success, 'g-', linewidth=2)
                ax.set_xlabel('Time Window')
                ax.set_ylabel('Energy per Success')
                ax.set_title('Energy Efficiency Trend')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_accuracy_traces(self, results: Dict, pdf: PdfPages):
        """Create detailed accuracy traces."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Accuracy Analysis', fontsize=16)
        
        if 'performance_history' not in results:
            pdf.savefig(fig)
            plt.close()
            return
        
        perf_history = results['performance_history']
        
        # Moving average accuracy
        ax = axes[0, 0]
        window_sizes = [10, 50, 100]
        colors = ['blue', 'green', 'red']
        
        for window, color in zip(window_sizes, colors):
            if len(perf_history) > window:
                moving_avg = []
                timestamps = []
                
                for i in range(window, len(perf_history)):
                    window_data = perf_history[i-window:i]
                    avg = sum(r['result']['success'] for r in window_data) / window
                    moving_avg.append(avg)
                    timestamps.append(window_data[-1]['timestamp'])
                
                ax.plot(timestamps, moving_avg, color=color, 
                       label=f'{window}-sample', linewidth=2)
        
        ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        ax.set_title('Moving Average Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Accuracy by number of cameras
        ax = axes[0, 1]
        acc_by_cameras = {}
        
        for record in perf_history:
            n_cams = len(record['result'].get('participating_cameras', []))
            if n_cams > 0:
                if n_cams not in acc_by_cameras:
                    acc_by_cameras[n_cams] = []
                acc_by_cameras[n_cams].append(record['result']['success'])
        
        if acc_by_cameras:
            cam_counts = sorted(acc_by_cameras.keys())
            accuracies = [np.mean(acc_by_cameras[n]) for n in cam_counts]
            
            ax.bar(cam_counts, accuracies, color='purple', alpha=0.7)
            ax.set_xlabel('Number of Cameras')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Accuracy vs Camera Count')
            ax.grid(True, alpha=0.3)
        
        # Accuracy distribution
        ax = axes[1, 0]
        recent_results = [r['result']['success'] for r in perf_history[-1000:]]
        if recent_results:
            success_rate = sum(recent_results) / len(recent_results)
            
            # Binomial confidence interval
            n = len(recent_results)
            se = np.sqrt(success_rate * (1 - success_rate) / n)
            ci_low = success_rate - 1.96 * se
            ci_high = success_rate + 1.96 * se
            
            ax.bar(['Accuracy'], [success_rate], color='green', alpha=0.7)
            ax.errorbar(['Accuracy'], [success_rate], 
                       yerr=[[success_rate - ci_low], [ci_high - success_rate]],
                       fmt='none', color='black', capsize=10)
            
            ax.axhline(y=0.8, color='red', linestyle='--')
            ax.text(0, success_rate + 0.02, f'{success_rate:.3f}', 
                   ha='center', va='bottom', fontsize=14)
            ax.set_ylabel('Success Rate')
            ax.set_title('Recent Accuracy with 95% CI')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        
        # Accuracy evolution phases
        ax = axes[1, 1]
        if len(perf_history) > 100:
            phases = {
                'Initial': perf_history[:len(perf_history)//4],
                'Learning': perf_history[len(perf_history)//4:len(perf_history)//2],
                'Stable': perf_history[len(perf_history)//2:3*len(perf_history)//4],
                'Final': perf_history[3*len(perf_history)//4:]
            }
            
            phase_accuracies = {}
            for phase_name, phase_data in phases.items():
                acc = sum(r['result']['success'] for r in phase_data) / len(phase_data)
                phase_accuracies[phase_name] = acc
            
            ax.bar(phase_accuracies.keys(), phase_accuracies.values(), 
                  color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
            ax.axhline(y=0.8, color='black', linestyle='--')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Simulation Phase')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            
            for i, (phase, acc) in enumerate(phase_accuracies.items()):
                ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_camera_utilization_plots(self, results: Dict, pdf: PdfPages):
        """Create camera utilization analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Camera Utilization Analysis', fontsize=16)
        
        if 'performance_history' not in results:
            pdf.savefig(fig)
            plt.close()
            return
        
        perf_history = results['performance_history']
        
        # Camera participation frequency
        ax = axes[0, 0]
        camera_participation = {}
        
        for record in perf_history:
            for cam_id in record['result'].get('participating_cameras', []):
                camera_participation[cam_id] = camera_participation.get(cam_id, 0) + 1
        
        if camera_participation:
            cam_ids = sorted(camera_participation.keys())
            participations = [camera_participation[i] for i in cam_ids]
            
            ax.bar(cam_ids, participations, color='blue', alpha=0.7)
            ax.set_xlabel('Camera ID')
            ax.set_ylabel('Participation Count')
            ax.set_title('Camera Participation Frequency')
            ax.grid(True, alpha=0.3)
        
        # Average cameras per classification over time
        ax = axes[0, 1]
        window = 50
        avg_cameras_time = []
        timestamps = []
        
        for i in range(window, len(perf_history)):
            window_data = perf_history[i-window:i]
            avg_cams = np.mean([len(r['result'].get('participating_cameras', [])) 
                               for r in window_data])
            avg_cameras_time.append(avg_cams)
            timestamps.append(window_data[-1]['timestamp'])
        
        if avg_cameras_time:
            ax.plot(timestamps, avg_cameras_time, 'g-', linewidth=2)
            ax.axhline(y=2.0, color='red', linestyle='--', label='Target')
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Cameras')
            ax.set_title('Camera Usage Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Camera efficiency (success per participation)
        ax = axes[1, 0]
        camera_success = {}
        
        for record in perf_history:
            if record['result']['success']:
                for cam_id in record['result'].get('participating_cameras', []):
                    if cam_id not in camera_success:
                        camera_success[cam_id] = {'success': 0, 'total': 0}
                    camera_success[cam_id]['success'] += 1
                    camera_success[cam_id]['total'] += 1
            else:
                for cam_id in record['result'].get('participating_cameras', []):
                    if cam_id not in camera_success:
                        camera_success[cam_id] = {'success': 0, 'total': 0}
                    camera_success[cam_id]['total'] += 1
        
        if camera_success:
            cam_ids = sorted(camera_success.keys())
            efficiencies = [camera_success[i]['success'] / camera_success[i]['total'] 
                           if camera_success[i]['total'] > 0 else 0 for i in cam_ids]
            
            ax.bar(cam_ids, efficiencies, color='orange', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--')
            ax.set_xlabel('Camera ID')
            ax.set_ylabel('Success Rate')
            ax.set_title('Camera Classification Efficiency')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        
        # Camera workload distribution
        ax = axes[1, 1]
        if camera_participation:
            workloads = list(camera_participation.values())
            ax.hist(workloads, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(workloads), color='red', linestyle='--',
                      label=f'Mean: {np.mean(workloads):.1f}')
            ax.set_xlabel('Participation Count')
            ax.set_ylabel('Number of Cameras')
            ax.set_title('Workload Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_violation_analysis(self, results: Dict, pdf: PdfPages):
        """Create violation analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Constraint Violation Analysis', fontsize=16)
        
        stats = results.get('network_stats', {})
        
        # Violation summary
        ax = axes[0, 0]
        violations = {
            'Energy': stats.get('energy_violations', 0),
            'Accuracy': stats.get('accuracy_violations', 0)
        }
        
        ax.bar(violations.keys(), violations.values(), color=['orange', 'red'], alpha=0.7)
        ax.set_ylabel('Number of Violations')
        ax.set_title('Violation Summary')
        ax.grid(True, alpha=0.3)
        
        for i, (vtype, count) in enumerate(violations.items()):
            ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=14)
        
        # Violation timeline (if detailed history available)
        ax = axes[0, 1]
        ax.text(0.5, 0.5, 'Violation Timeline\n(Requires detailed history)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Violations Over Time')
        
        # Violation causes
        ax = axes[1, 0]
        ax.text(0.5, 0.5, 'Violation Root Causes\n(Analysis pending)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Violation Analysis')
        
        # Mitigation effectiveness
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Mitigation Strategies\n(To be implemented)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Mitigation Effectiveness')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_performance_heatmaps(self, results: Dict, pdf: PdfPages):
        """Create performance heatmaps."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Heatmaps', fontsize=16)
        
        # Placeholder for various heatmaps
        titles = [
            'Camera Performance Matrix',
            'Time-based Performance',
            'Spatial Coverage Quality',
            'Energy-Accuracy Trade-off'
        ]
        
        for ax, title in zip(axes.flat, titles):
            # Create dummy heatmap for demonstration
            data = np.random.rand(10, 10)
            sns.heatmap(data, ax=ax, cmap='YlOrRd', cbar=True)
            ax.set_title(title)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_algorithm_comparison(self, results: Dict, pdf: PdfPages):
        """Create algorithm comparison plots."""
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        # If this is comparison data
        if 'algorithms' in results:
            algorithms = results['algorithms']
            metrics = ['accuracy', 'efficiency', 'violations']
            
            # Create grouped bar chart
            x = np.arange(len(algorithms))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [algorithms[algo].get(metric, 0) for algo in algorithms]
                ax.bar(x + i*width, values, width, label=metric)
            
            ax.set_xlabel('Algorithm')
            ax.set_xticks(x + width)
            ax.set_xticklabels(algorithms.keys())
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Single Algorithm Results\n(No comparison data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        pdf.savefig(fig)
        plt.close()
    
    def _create_network_topology_view(self, results: Dict, pdf: PdfPages):
        """Create network topology visualization."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Camera Network Topology', fontsize=16)
        
        # Generate example camera positions
        n_cameras = results.get('network_stats', {}).get('total_cameras', 10)
        
        # Create hemisphere positions
        positions = []
        for i in range(n_cameras):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi/2)
            r = np.random.uniform(20, 50)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            positions.append([x, y, z])
        
        positions = np.array(positions)
        
        # Plot cameras
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=100, alpha=0.6, edgecolors='black')
        
        # Add ground plane
        xx, yy = np.meshgrid(np.linspace(-60, 60, 10), np.linspace(-60, 60, 10))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='green')
        
        # Add coverage cones (simplified)
        for pos in positions[:5]:  # Show a few coverage cones
            # Create cone pointing downward
            height = pos[2]
            radius = height * 0.5
            
            theta = np.linspace(0, 2*np.pi, 20)
            x_cone = pos[0] + radius * np.cos(theta)
            y_cone = pos[1] + radius * np.sin(theta)
            z_cone = np.zeros_like(theta)
            
            for i in range(len(theta)):
                ax.plot([pos[0], x_cone[i]], [pos[1], y_cone[i]], 
                       [pos[2], z_cone[i]], 'b-', alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Positions and Coverage')
        
        pdf.savefig(fig)
        plt.close()


def create_comparison_report(results_files: List[str], output_pdf: str = 'comparison_report.pdf'):
    """
    Create comparison report from multiple result files.
    
    Args:
        results_files: List of result file paths
        output_pdf: Output PDF filename
    """
    visualizer = ComprehensiveVisualizer()
    
    # Load all results
    all_results = {}
    for filepath in results_files:
        name = filepath.split('/')[-1].replace('.json', '')
        with open(filepath, 'r') as f:
            all_results[name] = json.load(f)
    
    with PdfPages(output_pdf) as pdf:
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Algorithm Comparison', fontsize=16)
        
        # Accuracy comparison
        ax = axes[0, 0]
        names = list(all_results.keys())
        accuracies = [r.get('network_stats', {}).get('accuracy', 0) for r in all_results.values()]
        
        ax.bar(names, accuracies, color=visualizer.colors[:len(names)], alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--')
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Accuracy Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
        
        # Add other comparison plots...
        
        pdf.savefig(fig)
        plt.close()
    
    logger.info(f"Comparison report saved to {output_pdf}")


# Example usage
if __name__ == "__main__":
    visualizer = ComprehensiveVisualizer()
    
    # Create sample results for testing
    sample_results = {
        'network_stats': {
            'accuracy': 0.75,
            'recent_accuracy': 0.85,
            'avg_cameras_per_classification': 2.5,
            'energy_violations': 10,
            'accuracy_violations': 5,
            'total_cameras': 10
        },
        'performance_history': [
            {'timestamp': i, 'result': {'success': np.random.random() > 0.3,
                                       'participating_cameras': list(np.random.choice(10, 3))}}
            for i in range(1000)
        ],
        'energy_history': [
            {'timestamp': i, 'avg_energy': 900 - i*0.1 + np.random.randn()*10,
             'min_energy': 800 - i*0.1 + np.random.randn()*10,
             'max_energy': 1000 - i*0.1 + np.random.randn()*10,
             'camera_energies': [900 - i*0.1 + np.random.randn()*20 for _ in range(10)]}
            for i in range(1000)
        ]
    }
    
    # Save sample results
    with open('sample_results.json', 'w') as f:
        json.dump(sample_results, f)
    
    # Create report
    visualizer.create_full_analysis_report('sample_results.json', 'sample_report.pdf')
    print("Sample report created: sample_report.pdf")