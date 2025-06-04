# src/core/camera.py
"""Camera model with energy-dependent accuracy."""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import heapq
from dataclasses import dataclass, field
import logging

from .energy_model import EnergyModel
from .accuracy_model import AccuracyModel

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    """State of a camera at a given time."""
    energy: float
    position: np.ndarray
    is_active: bool = True
    last_classification_time: Optional[float] = None
    classification_count: int = 0
    correct_classifications: int = 0
    

class Camera:
    """
    Camera with energy harvesting and energy-dependent accuracy.
    
    This class models a single camera in the multi-camera network,
    including its energy dynamics, accuracy model, and decision-making.
    """
    
    def __init__(
        self,
        camera_id: int,
        position: np.ndarray,
        energy_model: EnergyModel,
        accuracy_model: AccuracyModel,
        initial_energy: Optional[float] = None,
        class_assignment: Optional[int] = None,
        num_classes: int = 2
    ):
        """
        Initialize a camera.
        
        Args:
            camera_id: Unique identifier for the camera
            position: 3D position of the camera
            energy_model: Energy harvesting and consumption model
            accuracy_model: Energy-dependent accuracy model
            initial_energy: Initial battery level (defaults to full capacity)
            class_assignment: Class assignment for round-robin scheduling
            num_classes: Number of object classes for classification
        """
        self.camera_id = camera_id
        self.position = position
        self.energy_model = energy_model
        self.accuracy_model = accuracy_model
        self.class_assignment = class_assignment
        self.num_classes = num_classes
        
        # Initialize state
        initial_energy = initial_energy or energy_model.capacity
        self.state = CameraState(
            energy=initial_energy,
            position=position.copy()
        )

        # History tracking
        self.energy_history = [initial_energy]
        self.accuracy_history = []
        self.participation_history = []

        # Reference to algorithm energy heap for fast selection
        self.energy_heap: Optional[List[Tuple[float, int]]] = None
        
    @property
    def current_energy(self) -> float:
        """Get current energy level."""
        return self.state.energy
    
    @property
    def current_accuracy(self) -> float:
        """Get current accuracy based on energy level."""
        return self.accuracy_model.get_accuracy(self.state.energy)
    
    def can_classify(self) -> bool:
        """Check if camera has enough energy to perform classification."""
        return self.state.energy >= self.energy_model.classification_cost
    
    def update_energy(self, time_delta: float, is_classifying: bool = False) -> None:
        """
        Update energy level based on harvesting and consumption.
        
        NOTE: This method is kept for compatibility but should not be used
        when the camera is part of a CameraNetwork, as the network manages
        energy updates centrally for performance.
        
        Args:
            time_delta: Time elapsed since last update
            is_classifying: Whether camera is performing classification
        """
        # Harvest energy
        harvested = self.energy_model.harvest(time_delta)
        self.state.energy = min(
            self.state.energy + harvested,
            self.energy_model.capacity
        )
        
        # Consume energy if classifying
        if is_classifying:
            # Only consume if we have enough energy
            if self.state.energy >= self.energy_model.classification_cost:
                self.state.energy -= self.energy_model.classification_cost
            else:
                # This should not happen if can_classify() is checked properly
                logger.warning(f"Camera {self.camera_id} insufficient energy for classification")
                self.state.energy = 0

        self.energy_history.append(self.state.energy)

        # Update algorithm's energy heap if available
        if self.energy_heap is not None:
            heapq.heappush(self.energy_heap, (-self.state.energy, self.camera_id))
        
    def classify(self, object_position: np.ndarray, true_label: int) -> Tuple[int, bool]:
        """
        Perform classification with energy-dependent accuracy.
        
        Args:
            object_position: Position of object to classify
            true_label: True label of the object
            
        Returns:
            Tuple of (predicted_label, is_correct)
        """
        if not self.can_classify():
            raise ValueError(f"Camera {self.camera_id} has insufficient energy")
            
        # Get accuracy based on current energy and position
        base_accuracy = self.current_accuracy
        position_factor = self._compute_position_factor(object_position)
        effective_accuracy = base_accuracy * position_factor
        
        # Simulate classification
        is_correct = np.random.random() < effective_accuracy
        if is_correct:
            predicted_label = true_label
        else:
            # For incorrect prediction, choose a random different label
            other_labels = [i for i in range(self.num_classes) if i != true_label]
            predicted_label = np.random.choice(other_labels) if other_labels else true_label
        
        # Update state
        self.state.classification_count += 1
        if is_correct:
            self.state.correct_classifications += 1
            
        self.accuracy_history.append(effective_accuracy)
        
        return predicted_label, is_correct
    
    def _compute_position_factor(self, object_position: np.ndarray) -> float:
        """
        Compute accuracy factor based on camera-object geometry.
        
        Args:
            object_position: Position of object
            
        Returns:
            Position-based accuracy factor in [0, 1]
        """
        distance = np.linalg.norm(self.position - object_position)
        max_distance = 100.0  # Maximum effective distance
        
        # Simple inverse distance model
        if distance >= max_distance:
            return 0.1
        return 1.0 - (distance / max_distance) * 0.9
    
    def reset(self) -> None:
        """Reset camera state to initial conditions."""
        self.state.energy = self.energy_model.capacity
        self.state.classification_count = 0
        self.state.correct_classifications = 0
        self.state.last_classification_time = None
        
        self.energy_history = [self.state.energy]
        self.accuracy_history = []
        self.participation_history = []
        
    def get_stats(self) -> Dict[str, Any]:
        """Get camera statistics."""
        avg_accuracy = (
            self.state.correct_classifications / self.state.classification_count
            if self.state.classification_count > 0 else 0
        )
        
        return {
            "camera_id": self.camera_id,
            "current_energy": self.current_energy,
            "current_accuracy": self.current_accuracy,
            "classification_count": self.state.classification_count,
            "average_accuracy": avg_accuracy,
            "energy_history": self.energy_history[-100:],  # Last 100 samples
            "position": self.position.tolist()
        }
    
    def get_recent_success_rate(self, window_size: int = 10) -> float:
        """Get recent classification success rate."""
        if self.state.classification_count == 0:
            return 0.0
        
        # Use overall success rate for now
        return self.state.correct_classifications / self.state.classification_count


