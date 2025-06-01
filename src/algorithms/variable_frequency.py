"""Algorithm 2: Variable frequency classification."""

import numpy as np
from typing import List, Dict, Optional
import logging

from .base_algorithm import BaseClassificationAlgorithm
from ..core.camera import Camera

logger = logging.getLogger(__name__)


class VariableFrequencyAlgorithm(BaseClassificationAlgorithm):
    """
    Algorithm 2: Variable frequency classification with subclasses.
    
    Suitable when classification frequency < Δ^(-1).
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        classification_frequency: float,
        recharge_time: float,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10
    ):
        """
        Initialize variable frequency algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes (n)
            classification_frequency: Expected classification frequency
            recharge_time: Time to recharge to minimum operational (Δ)
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.num_classes = num_classes
        self.classification_frequency = classification_frequency
        self.recharge_time = recharge_time
        
        # Calculate number of subclasses
        self.subclasses_per_class = max(1, int(np.ceil(
            classification_frequency * recharge_time
        )))
        
        # Assign cameras to classes and subclasses
        self._assign_camera_subclasses()
        
        # Track subclass turns
        self.class_turns = {i: 0 for i in range(num_classes)}
        
    def _assign_camera_subclasses(self) -> None:
        """Assign cameras to classes and subclasses with energy diversity."""
        # First assign to classes
        for i, camera in enumerate(self.cameras):
            camera.class_assignment = i % self.num_classes
            
        # Create subclass structure
        self.camera_subclasses = {}
        
        for class_id in range(self.num_classes):
            class_cameras = [
                i for i, cam in enumerate(self.cameras)
                if cam.class_assignment == class_id
            ]
            
            # Distribute across subclasses maintaining energy diversity
            subclasses = [[] for _ in range(self.subclasses_per_class)]
            
            # Sort by energy to ensure diversity
            class_cameras.sort(key=lambda i: self.cameras[i].current_energy)
            
            # Distribute in round-robin to maintain diversity
            for idx, cam_id in enumerate(class_cameras):
                subclass_idx = idx % self.subclasses_per_class
                subclasses[subclass_idx].append(cam_id)
                
            self.camera_subclasses[class_id] = subclasses
            
        logger.info(
            f"Created {self.subclasses_per_class} subclasses per class "
            f"for frequency {self.classification_frequency:.3f}"
        )
        
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras using subclass rotation.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        # Determine active class
        active_class = instance_id % self.num_classes
        
        # Get current subclass turn
        subclass_idx = self.class_turns[active_class]
        subclass_cameras = self.camera_subclasses[active_class][subclass_idx]
        
        # Update turn for next time
        self.class_turns[active_class] = (
            (subclass_idx + 1) % self.subclasses_per_class
        )
        
        # Select from subclass
        selected = self._select_from_subclass(subclass_cameras)
        
        # If insufficient, borrow from next subclass
        if not self._check_accuracy_threshold(selected):
            next_subclass_idx = (subclass_idx + 1) % self.subclasses_per_class
            next_subclass = self.camera_subclasses[active_class][next_subclass_idx]
            
            selected = self._borrow_cameras(selected, next_subclass)
            
        return selected
    
    def _select_from_subclass(self, subclass_cameras: List[int]) -> List[int]:
        """Select all capable cameras from subclass."""
        selected = []
        
        for cam_id in subclass_cameras:
            if self.cameras[cam_id].can_classify():
                selected.append(cam_id)
                
        return selected
    
    def _check_accuracy_threshold(self, camera_ids: List[int]) -> bool:
        """Check if selected cameras meet accuracy threshold."""
        if not camera_ids:
            return False
            
        selected_cameras = [self.cameras[i] for i in camera_ids]
        collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
        
        return collective_accuracy >= self.min_accuracy_threshold
    
    def _borrow_cameras(
        self,
        current_selection: List[int],
        next_subclass: List[int]
    ) -> List[int]:
        """
        Borrow cameras from next subclass if needed.
        
        Args:
            current_selection: Currently selected cameras
            next_subclass: Cameras in next subclass
            
        Returns:
            Updated selection
        """
        # Sort next subclass by energy
        available_next = [
            cam_id for cam_id in next_subclass
            if self.cameras[cam_id].can_classify()
        ]
        
        available_next.sort(
            key=lambda i: self.cameras[i].current_energy,
            reverse=True
        )
        
        # Add cameras until threshold met
        selection = current_selection.copy()
        
        for cam_id in available_next:
            selection.append(cam_id)
            
            if self._check_accuracy_threshold(selection):
                break
                
        return selection
    
    def rebalance_subclasses(self) -> None:
        """Rebalance subclasses based on current energy distribution."""
        for class_id in range(self.num_classes):
            class_cameras = []
            
            # Collect all cameras in class
            for subclass in self.camera_subclasses[class_id]:
                class_cameras.extend(subclass)
                
            # Redistribute based on energy
            class_cameras.sort(key=lambda i: self.cameras[i].current_energy)
            
            # Clear and redistribute
            subclasses = [[] for _ in range(self.subclasses_per_class)]
            
            for idx, cam_id in enumerate(class_cameras):
                subclass_idx = idx % self.subclasses_per_class
                subclasses[subclass_idx].append(cam_id)
                
            self.camera_subclasses[class_id] = subclasses