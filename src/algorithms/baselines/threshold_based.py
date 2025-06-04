"""Threshold-based Selection Baseline Algorithm.

Implements a threshold-based approach where cameras participate if their
energy exceeds a threshold. Common in event-driven sensor networks.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..base_algorithm import BaseClassificationAlgorithm
from ...core.camera import Camera

logger = logging.getLogger(__name__)


class ThresholdBasedAlgorithm(BaseClassificationAlgorithm):
    """
    Threshold-based camera selection.
    
    Cameras participate if energy > threshold. Based on threshold-based
    activation in WSN (e.g., Kar & Banerjee, 2003; Tian & Georganas, 2002).
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        energy_threshold_ratio: float = 0.5,
        adaptive_threshold: bool = True
    ):
        """
        Initialize threshold-based algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            energy_threshold_ratio: Energy threshold as ratio of capacity
            adaptive_threshold: Whether to adapt threshold based on performance
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.energy_threshold_ratio = energy_threshold_ratio
        self.adaptive_threshold = adaptive_threshold
        self.threshold_history = [energy_threshold_ratio]
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras based on energy threshold.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        # Get current threshold
        current_threshold = self.threshold_history[-1]
        
        # Select cameras above threshold
        selected = []
        for i, cam in enumerate(self.cameras):
            energy_ratio = cam.current_energy / cam.energy_model.capacity
            if cam.can_classify() and energy_ratio >= current_threshold:
                selected.append(i)
        
        # If adaptive, adjust threshold based on selection outcome
        if self.adaptive_threshold:
            self._adapt_threshold(selected)
        
        return selected
    
    def _adapt_threshold(self, selected: List[int]):
        """Adapt threshold based on selection outcome."""
        current_threshold = self.threshold_history[-1]
        
        if not selected:
            # No cameras selected, lower threshold
            new_threshold = max(0.1, current_threshold * 0.9)
        else:
            # Check if we're selecting too many or too few
            selected_cameras = [self.cameras[i] for i in selected]
            collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
            
            if len(selected) > 5:  # Too many cameras
                new_threshold = min(0.9, current_threshold * 1.1)
            elif collective_accuracy < self.min_accuracy_threshold:
                # Not enough accuracy, lower threshold
                new_threshold = max(0.1, current_threshold * 0.95)
            else:
                # Good selection, maintain threshold
                new_threshold = current_threshold
        
        self.threshold_history.append(new_threshold)