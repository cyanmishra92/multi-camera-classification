"""Greedy Energy-based Selection Baseline Algorithm.

Implements a greedy algorithm that always selects cameras with highest energy.
This is a common baseline in energy-aware sensor networks.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..base_algorithm import BaseClassificationAlgorithm
from ...core.camera import Camera

logger = logging.getLogger(__name__)


class GreedyEnergyAlgorithm(BaseClassificationAlgorithm):
    """
    Greedy energy-based camera selection.
    
    Always selects cameras with highest available energy until accuracy
    threshold is met. Common baseline in energy-harvesting WSN literature
    (e.g., Noh & Kang, 2011; Vigorito et al., 2007).
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        energy_threshold: float = 0.2
    ):
        """
        Initialize greedy energy algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            energy_threshold: Minimum energy ratio to consider camera
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.energy_threshold = energy_threshold
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras greedily based on energy levels.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        # Get cameras with sufficient energy
        available_cameras = []
        for i, cam in enumerate(self.cameras):
            energy_ratio = cam.current_energy / cam.energy_model.capacity
            if cam.can_classify() and energy_ratio >= self.energy_threshold:
                available_cameras.append((i, cam.current_energy))
        
        if not available_cameras:
            return []
        
        # Sort by energy (descending)
        available_cameras.sort(key=lambda x: x[1], reverse=True)
        
        # Greedily select cameras until accuracy threshold is met
        selected = []
        for cam_id, _ in available_cameras:
            selected.append(cam_id)
            
            # Check if we meet accuracy threshold
            selected_cameras = [self.cameras[i] for i in selected]
            collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
            
            if collective_accuracy >= self.min_accuracy_threshold:
                break
        
        return selected