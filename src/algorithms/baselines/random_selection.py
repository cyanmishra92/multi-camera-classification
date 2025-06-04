"""Random Selection Baseline Algorithm.

This implements a naive random selection approach commonly used as a baseline
in sensor network literature. It randomly selects cameras without considering
energy or accuracy constraints.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..base_algorithm import BaseClassificationAlgorithm
from ...core.camera import Camera

logger = logging.getLogger(__name__)


class RandomSelectionAlgorithm(BaseClassificationAlgorithm):
    """
    Random camera selection baseline.
    
    This algorithm randomly selects k cameras from available cameras
    without considering energy levels or accuracy requirements.
    Common baseline in WSN literature (e.g., Alaei & Barcelo-Ordinas, 2010).
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        selection_probability: float = 0.3,
        min_cameras: int = 1,
        max_cameras: int = 5
    ):
        """
        Initialize random selection algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            selection_probability: Probability of selecting each camera
            min_cameras: Minimum number of cameras to select
            max_cameras: Maximum number of cameras to select
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.selection_probability = selection_probability
        self.min_cameras = min_cameras
        self.max_cameras = max_cameras
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Randomly select cameras.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of randomly selected camera IDs
        """
        # Get cameras with sufficient energy
        available_cameras = [
            i for i, cam in enumerate(self.cameras)
            if cam.can_classify()
        ]
        
        if not available_cameras:
            return []
        
        # Randomly decide how many cameras to select
        num_to_select = np.random.randint(
            self.min_cameras, 
            min(self.max_cameras + 1, len(available_cameras) + 1)
        )
        
        # Random selection without replacement
        selected = np.random.choice(
            available_cameras, 
            size=min(num_to_select, len(available_cameras)),
            replace=False
        ).tolist()
        
        return selected