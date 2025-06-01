"""Accuracy-adaptive algorithm that adjusts based on performance."""

import numpy as np
from typing import List, Dict, Optional
import logging

from .game_theoretic_fixed_frequency import GameTheoreticFixedFrequencyAlgorithm
from ..core.camera import Camera
from ..game_theory.utility_functions import UtilityParameters

logger = logging.getLogger(__name__)


class AccuracyAdaptiveAlgorithm(GameTheoreticFixedFrequencyAlgorithm):
    """
    Adaptive algorithm that dynamically adjusts to maintain target accuracy.
    
    Features:
    - Dynamic accuracy threshold adjustment
    - Warm-up period with relaxed constraints
    - Performance-based camera count adjustment
    - Adaptive participation strategies
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        utility_params: Optional[UtilityParameters] = None,
        position_weight: float = 0.7,
        use_nash_equilibrium: bool = True,
        convergence_threshold: float = 0.01,
        warm_up_period: int = 50,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize accuracy-adaptive algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes
            min_accuracy_threshold: Target accuracy threshold
            history_length: Length of classification history
            utility_params: Utility function parameters
            position_weight: Weight for position-based selection
            use_nash_equilibrium: Whether to use Nash equilibrium
            convergence_threshold: Nash equilibrium convergence threshold
            warm_up_period: Initial warm-up period with relaxed constraints
            adaptation_rate: Rate of threshold adaptation
        """
        super().__init__(
            cameras=cameras,
            num_classes=num_classes,
            min_accuracy_threshold=min_accuracy_threshold,
            history_length=history_length,
            utility_params=utility_params,
            position_weight=position_weight,
            use_nash_equilibrium=use_nash_equilibrium,
            convergence_threshold=convergence_threshold
        )
        
        self.warm_up_period = warm_up_period
        self.adaptation_rate = adaptation_rate
        self.target_accuracy = min_accuracy_threshold
        
        # Adaptive thresholds
        self.current_threshold = 0.7  # Start lower
        self.min_cameras = 2
        self.max_cameras = 5
        
        # Performance tracking
        self.accuracy_window = []
        self.window_size = 20
        
    def select_cameras(self, instance_id: int, current_time: float,
                      object_position: Optional[np.ndarray] = None) -> List[int]:
        """
        Adaptive camera selection based on current performance.
        
        Args:
            instance_id: Classification instance ID
            current_time: Current simulation time
            object_position: Object position
            
        Returns:
            Selected camera IDs
        """
        # Update adaptive threshold
        self._update_adaptive_threshold()
        
        # During warm-up, use more cameras
        if self.total_classifications < self.warm_up_period:
            self.min_cameras = 3
            self.max_cameras = 6
        else:
            # Adjust camera count based on performance
            self._adjust_camera_count()
        
        # Get base selection from parent
        selected = super().select_cameras(instance_id, current_time, object_position)
        
        # Ensure minimum cameras during warm-up
        if len(selected) < self.min_cameras:
            selected = self._augment_to_minimum(selected, instance_id, object_position)
        
        # Cap at maximum
        if len(selected) > self.max_cameras:
            selected = self._reduce_to_maximum(selected, object_position)
        
        return selected
    
    def _update_adaptive_threshold(self):
        """Update accuracy threshold based on recent performance."""
        if len(self.classification_history) < 10:
            return
            
        # Calculate recent accuracy
        recent_results = self.classification_history[-self.window_size:]
        recent_accuracy = sum(r['success'] for r in recent_results) / len(recent_results)
        
        # Update window
        self.accuracy_window.append(recent_accuracy)
        if len(self.accuracy_window) > 5:
            self.accuracy_window.pop(0)
        
        # Smooth accuracy estimate
        smooth_accuracy = np.mean(self.accuracy_window)
        
        # Adapt threshold
        if smooth_accuracy < self.target_accuracy - 0.05:
            # Below target, relax threshold
            self.current_threshold = max(
                0.65,
                self.current_threshold - self.adaptation_rate * 0.02
            )
        elif smooth_accuracy > self.target_accuracy + 0.05:
            # Above target, can be stricter
            self.current_threshold = min(
                self.target_accuracy,
                self.current_threshold + self.adaptation_rate * 0.01
            )
        
        # Update algorithm threshold
        self.min_accuracy_threshold = self.current_threshold
    
    def _adjust_camera_count(self):
        """Adjust min/max camera count based on performance."""
        if not self.accuracy_window:
            return
            
        avg_accuracy = np.mean(self.accuracy_window)
        
        # Get efficiency metrics
        recent_cameras = [
            len(r['participating_cameras']) 
            for r in self.classification_history[-20:]
            if 'participating_cameras' in r
        ]
        avg_cameras = np.mean(recent_cameras) if recent_cameras else 3
        
        # Adjust limits
        if avg_accuracy < self.target_accuracy - 0.1:
            # Need more cameras
            self.min_cameras = min(4, self.min_cameras + 1)
            self.max_cameras = min(7, self.max_cameras + 1)
        elif avg_accuracy > self.target_accuracy and avg_cameras > 2.5:
            # Can use fewer cameras
            self.min_cameras = max(2, self.min_cameras - 1)
            self.max_cameras = max(4, self.max_cameras - 1)
    
    def _augment_to_minimum(self, selected: List[int], instance_id: int,
                           object_position: Optional[np.ndarray]) -> List[int]:
        """Add cameras to meet minimum requirement."""
        # Determine active class
        active_class = instance_id % self.num_classes
        class_cameras = self.camera_classes[active_class]
        
        # Get available cameras not yet selected
        available = [c for c in class_cameras if c not in selected and self.cameras[c].can_classify()]
        
        # Sort by energy
        available.sort(key=lambda i: self.cameras[i].current_energy, reverse=True)
        
        # Add cameras
        augmented = selected.copy()
        for cam_id in available:
            if len(augmented) >= self.min_cameras:
                break
            augmented.append(cam_id)
        
        return augmented
    
    def _reduce_to_maximum(self, selected: List[int], 
                          object_position: Optional[np.ndarray]) -> List[int]:
        """Reduce cameras to maximum limit while maintaining quality."""
        if len(selected) <= self.max_cameras:
            return selected
            
        # Score cameras by contribution
        scores = []
        for cam_id in selected:
            camera = self.cameras[cam_id]
            
            # Energy score
            energy_score = camera.current_energy / camera.energy_model.capacity
            
            # Position score if available
            if object_position is not None and self.using_enhanced:
                pos_score = camera.accuracy_model.get_position_based_accuracy(
                    camera, object_position
                )
            else:
                pos_score = camera.current_accuracy
            
            # Combined score
            score = 0.6 * pos_score + 0.4 * energy_score
            scores.append((cam_id, score))
        
        # Sort by score and keep top cameras
        scores.sort(key=lambda x: x[1], reverse=True)
        return [cam_id for cam_id, _ in scores[:self.max_cameras]]
    
    def get_adaptive_metrics(self) -> Dict:
        """Get adaptive algorithm specific metrics."""
        base_metrics = self.get_game_theory_metrics()
        
        adaptive_metrics = {
            **base_metrics,
            'current_threshold': self.current_threshold,
            'min_cameras': self.min_cameras,
            'max_cameras': self.max_cameras,
            'smooth_accuracy': np.mean(self.accuracy_window) if self.accuracy_window else 0,
            'in_warm_up': self.total_classifications < self.warm_up_period
        }
        
        return adaptive_metrics