"""Round Robin Baseline Algorithm.

Implements a simple round-robin scheduling without energy awareness.
This represents traditional scheduling approaches in distributed systems.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..base_algorithm import BaseClassificationAlgorithm
from ...core.camera import Camera

logger = logging.getLogger(__name__)


class RoundRobinAlgorithm(BaseClassificationAlgorithm):
    """
    Simple round-robin camera selection.
    
    Cycles through cameras in order without considering energy levels.
    Common baseline in distributed systems (e.g., Rhee et al., 2009).
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        cameras_per_round: int = 3
    ):
        """
        Initialize round-robin algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            cameras_per_round: Number of cameras to select each round
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.cameras_per_round = cameras_per_round
        self.current_index = 0
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras in round-robin fashion.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        selected = []
        attempts = 0
        
        # Try to select cameras_per_round cameras
        while len(selected) < self.cameras_per_round and attempts < len(self.cameras):
            cam_id = self.current_index % len(self.cameras)
            
            if self.cameras[cam_id].can_classify():
                selected.append(cam_id)
            
            self.current_index = (self.current_index + 1) % len(self.cameras)
            attempts += 1
        
        return selected