"""Adaptive parameter tuning for algorithm optimization."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class TuningParameters:
    """Parameters that can be tuned."""
    min_accuracy_threshold: float = 0.8
    position_weight: float = 0.7
    participation_bonus: float = 0.2
    energy_weight: float = 0.3
    overlap_bonus: float = 0.2
    distance_decay: float = 0.01
    angle_penalty: float = 0.3
    
    # Game theory parameters
    reward_scale: float = 2.0
    incorrect_penalty: float = 0.3
    non_participation_penalty: float = 1.0
    discount_factor: float = 0.85
    
    # Algorithm-specific
    convergence_threshold: float = 0.01
    future_value_weight: float = 0.3
    

class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning using online learning.
    
    Adjusts parameters based on performance feedback to achieve target metrics.
    """
    
    def __init__(
        self,
        target_accuracy: float = 0.8,
        target_efficiency: float = 2.0,
        learning_rate: float = 0.1,
        momentum: float = 0.9
    ):
        """
        Initialize parameter tuner.
        
        Args:
            target_accuracy: Target classification accuracy
            target_efficiency: Target cameras per classification
            learning_rate: Learning rate for parameter updates
            momentum: Momentum for gradient updates
        """
        self.target_accuracy = target_accuracy
        self.target_efficiency = target_efficiency
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Current parameters
        self.params = TuningParameters()
        
        # Performance history
        self.performance_history = []
        self.parameter_history = []
        
        # Gradient estimates
        self.gradients = {field: 0.0 for field in self.params.__dataclass_fields__}
        self.velocity = {field: 0.0 for field in self.params.__dataclass_fields__}
        
        # Best parameters found
        self.best_params = None
        self.best_score = -float('inf')
        
    def update_parameters(self, performance_metrics: Dict) -> TuningParameters:
        """
        Update parameters based on performance feedback.
        
        Args:
            performance_metrics: Dictionary with accuracy, efficiency, violations
            
        Returns:
            Updated parameters
        """
        # Record performance
        self.performance_history.append(performance_metrics)
        
        # Calculate performance score
        score = self._calculate_score(performance_metrics)
        
        # Update best if improved
        if score > self.best_score:
            self.best_score = score
            self.best_params = TuningParameters(**self.params.__dict__)
            logger.info(f"New best score: {score:.3f}")
        
        # Estimate gradients using finite differences
        if len(self.performance_history) > 1:
            self._estimate_gradients()
        
        # Update parameters using gradient descent with momentum
        self._update_with_momentum()
        
        # Apply constraints
        self._apply_constraints()
        
        # Record parameter state
        self.parameter_history.append(self.params.__dict__.copy())
        
        return self.params
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall performance score."""
        accuracy = metrics.get('accuracy', 0)
        efficiency = metrics.get('avg_cameras_per_classification', float('inf'))
        violations = metrics.get('total_violations', 0)
        
        # Accuracy component (heavily weighted)
        accuracy_score = 100 * min(accuracy / self.target_accuracy, 1.0)
        
        # Efficiency component
        efficiency_score = 20 * min(self.target_efficiency / efficiency, 1.0)
        
        # Violation penalty
        violation_penalty = 0.1 * violations
        
        # Recent accuracy bonus
        recent_acc = metrics.get('recent_accuracy', accuracy)
        recent_bonus = 10 * max(0, recent_acc - accuracy)
        
        return accuracy_score + efficiency_score - violation_penalty + recent_bonus
    
    def _estimate_gradients(self):
        """Estimate gradients using performance differences."""
        if len(self.performance_history) < 2:
            return
            
        # Get recent performance change
        current_perf = self.performance_history[-1]
        previous_perf = self.performance_history[-2]
        
        # Performance improvement
        current_score = self._calculate_score(current_perf)
        previous_score = self._calculate_score(previous_perf)
        score_change = current_score - previous_score
        
        # Accuracy gradient - most important
        acc_change = current_perf.get('accuracy', 0) - previous_perf.get('accuracy', 0)
        
        # Update gradients based on performance changes
        if acc_change != 0:
            # Accuracy-related parameters
            self.gradients['min_accuracy_threshold'] = -0.5 * acc_change
            self.gradients['position_weight'] = 2.0 * acc_change
            self.gradients['overlap_bonus'] = 1.5 * acc_change
            self.gradients['participation_bonus'] = 1.0 * acc_change
            
        # Efficiency gradient
        eff_change = (current_perf.get('avg_cameras_per_classification', 0) - 
                     previous_perf.get('avg_cameras_per_classification', 0))
        
        if eff_change != 0:
            self.gradients['energy_weight'] = -0.3 * eff_change
            self.gradients['reward_scale'] = -0.5 * eff_change
            
        # Violation gradient
        viol_change = (current_perf.get('total_violations', 0) - 
                      previous_perf.get('total_violations', 0))
        
        if viol_change != 0:
            self.gradients['min_accuracy_threshold'] = 0.2 * viol_change
            self.gradients['convergence_threshold'] = -0.1 * viol_change
    
    def _update_with_momentum(self):
        """Update parameters using momentum-based gradient descent."""
        for param_name in self.gradients:
            # Update velocity
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] + 
                self.learning_rate * self.gradients[param_name]
            )
            
            # Update parameter
            current_value = getattr(self.params, param_name)
            new_value = current_value + self.velocity[param_name]
            setattr(self.params, param_name, new_value)
    
    def _apply_constraints(self):
        """Apply constraints to keep parameters in valid ranges."""
        # Accuracy threshold constraints
        self.params.min_accuracy_threshold = np.clip(
            self.params.min_accuracy_threshold, 0.7, 0.9
        )
        
        # Weight constraints [0, 1]
        for param in ['position_weight', 'energy_weight', 'participation_bonus']:
            setattr(self.params, param, np.clip(getattr(self.params, param), 0.0, 1.0))
        
        # Positive value constraints
        for param in ['overlap_bonus', 'distance_decay', 'angle_penalty',
                     'reward_scale', 'incorrect_penalty', 'non_participation_penalty']:
            setattr(self.params, param, max(0.01, getattr(self.params, param)))
        
        # Discount factor constraint
        self.params.discount_factor = np.clip(self.params.discount_factor, 0.5, 0.99)
        
        # Convergence threshold
        self.params.convergence_threshold = np.clip(
            self.params.convergence_threshold, 0.001, 0.1
        )
    
    def get_adaptive_threshold(self, current_accuracy: float, 
                             violation_rate: float) -> float:
        """
        Get adaptive accuracy threshold based on current performance.
        
        Args:
            current_accuracy: Current accuracy level
            violation_rate: Rate of accuracy violations
            
        Returns:
            Adjusted accuracy threshold
        """
        base_threshold = self.params.min_accuracy_threshold
        
        # If accuracy is consistently high, can be more strict
        if current_accuracy > 0.85 and violation_rate < 0.05:
            return min(base_threshold + 0.05, 0.9)
        
        # If too many violations, be more lenient
        if violation_rate > 0.2:
            return max(base_threshold - 0.1, 0.7)
        
        # Gradual adjustment
        adjustment = 0.02 * (current_accuracy - self.target_accuracy)
        return np.clip(base_threshold + adjustment, 0.7, 0.9)
    
    def suggest_camera_count(self, object_position: Optional[np.ndarray] = None,
                           current_performance: Optional[Dict] = None) -> int:
        """
        Suggest optimal camera count based on current performance.
        
        Args:
            object_position: Object position for context
            current_performance: Current performance metrics
            
        Returns:
            Suggested number of cameras
        """
        if not current_performance:
            return 3  # Default
        
        current_acc = current_performance.get('accuracy', 0)
        current_eff = current_performance.get('avg_cameras_per_classification', 3)
        
        # If accuracy too low, add more cameras
        if current_acc < self.target_accuracy - 0.1:
            return int(current_eff + 1)
        
        # If accuracy high and using too many cameras, reduce
        if current_acc > self.target_accuracy + 0.05 and current_eff > self.target_efficiency:
            return max(1, int(current_eff - 0.5))
        
        return int(current_eff)
    
    def save_tuning_history(self, filepath: str):
        """Save tuning history for analysis."""
        history = {
            'performance_history': self.performance_history,
            'parameter_history': self.parameter_history,
            'best_params': self.best_params.__dict__ if self.best_params else None,
            'best_score': self.best_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_best_params(self, filepath: str) -> TuningParameters:
        """Load best parameters from file."""
        with open(filepath, 'r') as f:
            history = json.load(f)
            
        if history.get('best_params'):
            self.params = TuningParameters(**history['best_params'])
            self.best_params = TuningParameters(**history['best_params'])
            self.best_score = history.get('best_score', -float('inf'))
            
        return self.params