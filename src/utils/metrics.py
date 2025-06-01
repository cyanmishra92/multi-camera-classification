"""Performance metrics and evaluation."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    accuracy: float
    energy_efficiency: float
    participation_rate: float
    accuracy_violations: int
    energy_violations: int
    avg_cameras_per_classification: float
    total_energy_consumed: float
    total_energy_harvested: float
    

class MetricsCalculator:
    """Calculate and analyze performance metrics."""
    
    @staticmethod
    def calculate_accuracy(results: List[Dict]) -> float:
        """Calculate overall classification accuracy."""
        if not results:
            return 0.0
            
        successful = sum(1 for r in results if r.get('success', False))
        return successful / len(results)
    
    @staticmethod
    def calculate_energy_efficiency(
        energy_history: List[Dict],
        classification_history: List[Dict]
    ) -> float:
        """Calculate energy efficiency (classifications per unit energy)."""
        if not energy_history or not classification_history:
            return 0.0
            
        # Total energy consumed
        initial_energy = sum(energy_history[0]['camera_energies'])
        final_energy = sum(energy_history[-1]['camera_energies'])
        
        # Account for harvesting
        time_elapsed = energy_history[-1]['timestamp'] - energy_history[0]['timestamp']
        harvest_rate = 10  # Default, should be from config
        num_cameras = len(energy_history[0]['camera_energies'])
        total_harvested = harvest_rate * time_elapsed * num_cameras
        
        total_consumed = initial_energy - final_energy + total_harvested
        
        # Classifications per unit energy
        num_classifications = len([c for c in classification_history if c.get('success')])
        
        return num_classifications / total_consumed if total_consumed > 0 else 0.0
    
    @staticmethod
    def calculate_participation_metrics(
        classification_history: List[Dict]
    ) -> Dict[str, float]:
        """Calculate participation-related metrics."""
        if not classification_history:
            return {
                'avg_participation_rate': 0.0,
                'std_participation_rate': 0.0,
                'min_participation_rate': 0.0,
                'max_participation_rate': 0.0
            }
            
        participation_rates = []
        
        for record in classification_history:
            num_selected = len(record.get('selected_cameras', []))
            total_cameras = record.get('total_cameras', 10)  # Default
            rate = num_selected / total_cameras
            participation_rates.append(rate)
            
        return {
            'avg_participation_rate': np.mean(participation_rates),
            'std_participation_rate': np.std(participation_rates),
            'min_participation_rate': np.min(participation_rates),
            'max_participation_rate': np.max(participation_rates)
        }
    
    @staticmethod
    def analyze_violations(
        classification_history: List[Dict],
        min_accuracy_threshold: float = 0.8
    ) -> Dict[str, int]:
        """Analyze accuracy and energy violations."""
        accuracy_violations = 0
        energy_violations = 0
        
        for record in classification_history:
            # Check accuracy violation
            if record.get('collective_accuracy', 1.0) < min_accuracy_threshold:
                accuracy_violations += 1
                
            # Check energy violation
            if record.get('reason') == 'no_cameras_available':
                energy_violations += 1
                
        return {
            'accuracy_violations': accuracy_violations,
            'energy_violations': energy_violations,
            'total_violations': accuracy_violations + energy_violations
        }
    
    @staticmethod
    def create_summary_report(
        network_stats: Dict,
        performance_history: List[Dict],
        energy_history: List[Dict]
    ) -> str:
        """Create a summary report of simulation results."""
        report = []
        report.append("=" * 50)
        report.append("SIMULATION SUMMARY REPORT")
        report.append("=" * 50)
        
        # Basic stats
        report.append(f"\nNetwork Configuration:")
        report.append(f"  Total Cameras: {network_stats.get('total_cameras', 'N/A')}")
        report.append(f"  Active Cameras: {network_stats.get('active_cameras', 'N/A')}")
        report.append(f"  Simulation Time: {network_stats.get('simulation_time', 0):.1f}")
        
        # Performance metrics
        report.append(f"\nPerformance Metrics:")
        report.append(f"  Total Classifications: {network_stats.get('total_classifications', 0)}")
        report.append(f"  Overall Accuracy: {network_stats.get('accuracy', 0):.3f}")
        report.append(f"  Recent Accuracy: {network_stats.get('recent_accuracy', 0):.3f}")
        
        # Energy metrics
        report.append(f"\nEnergy Metrics:")
        report.append(f"  Average Energy: {network_stats.get('avg_energy', 0):.1f}")
        report.append(f"  Average Accuracy: {network_stats.get('avg_accuracy', 0):.3f}")
        
        # Violations
        report.append(f"\nViolations:")
        report.append(f"  Energy Violations: {network_stats.get('energy_violations', 0)}")
        report.append(f"  Accuracy Violations: {network_stats.get('accuracy_violations', 0)}")
        
        # Participation
        report.append(f"\nParticipation:")
        report.append(f"  Avg Cameras per Classification: {network_stats.get('avg_cameras_per_classification', 0):.2f}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)