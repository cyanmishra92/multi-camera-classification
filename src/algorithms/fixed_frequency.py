"""Algorithm 1: Fixed frequency classification."""

import numpy as np
from typing import List, Dict, Optional
import logging

from .base_algorithm import BaseClassificationAlgorithm
from ..core.camera import Camera

logger = logging.getLogger(__name__)


class FixedFrequencyAlgorithm(BaseClassificationAlgorithm):
    """
    Algorithm 1: Fixed frequency classification with round-robin scheduling.
    
    Suitable when classification frequency ≥ Δ^(-1) where Δ is recharge time.
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        use_game_theory: bool = False
    ):
        """
        Initialize fixed frequency algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes (n)
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            use_game_theory: Whether to use game-theoretic selection
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.num_classes = num_classes
        self.use_game_theory = use_game_theory
        
        # Assign cameras to classes
        self._assign_camera_classes()
        
        # Initialize energy tracking
        self.energy_tracking = {
            i: camera.current_energy 
            for i, camera in enumerate(cameras)
        }
        
    def _assign_camera_classes(self) -> None:
        """Assign cameras to classes using round-robin."""
        cameras_per_class = len(self.cameras) // self.num_classes
        
        for i, camera in enumerate(self.cameras):
            camera.class_assignment = i % self.num_classes
            
        # Create class mapping
        self.camera_classes = {}
        for class_id in range(self.num_classes):
            self.camera_classes[class_id] = [
                i for i, cam in enumerate(self.cameras)
                if cam.class_assignment == class_id
            ]
            
        logger.info(f"Assigned {len(self.cameras)} cameras to {self.num_classes} classes")
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras using round-robin scheduling.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        # Determine active class
        active_class = instance_id % self.num_classes
        class_cameras = self.camera_classes[active_class]
        
        if self.use_game_theory:
            # Use strategic selection within class
            return self._select_strategic(class_cameras)
        else:
            # Select subset to meet accuracy threshold
            return self._select_greedy(class_cameras)
            
    def _select_greedy(self, candidate_cameras: List[int]) -> List[int]:
        """
        Greedy selection to meet accuracy threshold.
        
        Args:
            candidate_cameras: List of candidate camera IDs
            
        Returns:
            Selected camera IDs
        """
        # Sort by energy level (descending)
        sorted_cameras = sorted(
            candidate_cameras,
            key=lambda i: self.cameras[i].current_energy,
            reverse=True
        )
        
        selected = []
        
        for cam_id in sorted_cameras:
            camera = self.cameras[cam_id]
            
            if not camera.can_classify():
                continue
                
            selected.append(cam_id)
            
            # Check if accuracy threshold met
            selected_cameras = [self.cameras[i] for i in selected]
            collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
            
            if collective_accuracy >= self.min_accuracy_threshold:
                break
                
        return selected
    
    def _select_strategic(self, candidate_cameras: List[int]) -> List[int]:
        """
        Strategic selection using game theory.
        
        Args:
            candidate_cameras: List of candidate camera IDs
            
        Returns:
            Selected camera IDs
        """
        # This would integrate with the game theory module
        # For now, fall back to greedy
        logger.debug("Strategic selection not fully implemented, using greedy")
        return self._select_greedy(candidate_cameras)
    
    def update_energy_tracking(self) -> None:
        """Update energy tracking for all cameras."""
        for i, camera in enumerate(self.cameras):
            self.energy_tracking[i] = camera.current_energy