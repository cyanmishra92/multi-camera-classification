"""Enhanced base class for classification algorithms with position awareness."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .base_algorithm import BaseClassificationAlgorithm
from ..core.camera import Camera
from ..core.enhanced_accuracy_model import EnhancedAccuracyModel

logger = logging.getLogger(__name__)


class EnhancedBaseClassificationAlgorithm(BaseClassificationAlgorithm):
    """
    Enhanced base class with position-aware accuracy calculations.
    
    Extends base functionality to support:
    - Position-based accuracy calculations
    - Optimal camera selection strategies
    - Object position awareness
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        energy_budget_factor: float = 2.0
    ):
        """
        Initialize enhanced base algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history to maintain
            energy_budget_factor: Factor to determine energy budget (x min required)
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        self.energy_budget_factor = energy_budget_factor
        
        # Check if using enhanced accuracy model
        self.using_enhanced = isinstance(
            cameras[0].accuracy_model if cameras else None,
            EnhancedAccuracyModel
        )
        
        if self.using_enhanced:
            logger.info("Using enhanced position-aware accuracy calculations")
    
    @abstractmethod
    def select_cameras(self, instance_id: int, current_time: float, 
                      object_position: Optional[np.ndarray] = None) -> List[int]:
        """
        Select cameras for classification with optional position awareness.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            object_position: Optional position of object for position-aware selection
            
        Returns:
            List of camera IDs to participate
        """
        pass
    
    def classify(
        self,
        instance_id: int,
        object_position: np.ndarray,
        true_label: int,
        current_time: float
    ) -> Dict:
        """
        Enhanced classification with position-aware accuracy.
        
        Args:
            instance_id: ID of classification instance
            object_position: Position of object to classify
            true_label: True label of object
            current_time: Current simulation time
            
        Returns:
            Classification result dictionary
        """
        # Select cameras (with position if using enhanced model)
        if self.using_enhanced:
            selected_camera_ids = self.select_cameras(
                instance_id, current_time, object_position
            )
        else:
            selected_camera_ids = self.select_cameras(instance_id, current_time)
        
        if not selected_camera_ids:
            self.energy_violations += 1
            logger.warning(f"No cameras selected for instance {instance_id}")
            return {
                'instance_id': instance_id,
                'success': False,
                'reason': 'no_cameras_available',
                'selected_cameras': [],
                'participating_cameras': [],
                'collective_accuracy': 0.0,
                'object_position': object_position.tolist()
            }
        
        # Get selected cameras
        selected_cameras = [self.cameras[i] for i in selected_camera_ids]
        
        # Calculate collective accuracy
        if self.using_enhanced:
            collective_accuracy = self._calculate_enhanced_collective_accuracy(
                selected_cameras, object_position
            )
        else:
            collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
        
        if collective_accuracy < self.min_accuracy_threshold:
            self.accuracy_violations += 1
            logger.warning(
                f"Collective accuracy {collective_accuracy:.3f} below threshold "
                f"{self.min_accuracy_threshold:.3f} for instance {instance_id}"
            )
        
        # Perform classification with position-aware individual accuracies
        predictions = []
        correct_count = 0
        participating_cameras = []
        
        for i, cam in enumerate(selected_cameras):
            if cam.can_classify():
                # Get position-based accuracy if using enhanced model
                if self.using_enhanced:
                    cam_accuracy = cam.accuracy_model.get_position_based_accuracy(
                        cam, object_position
                    )
                else:
                    cam_accuracy = cam.current_accuracy
                
                # Simulate classification with position-based accuracy
                is_correct = np.random.random() < cam_accuracy
                pred = true_label if is_correct else (1 - true_label)
                
                predictions.append(pred)
                if is_correct:
                    correct_count += 1
                
                participating_cameras.append(selected_camera_ids[i])
                
                
                # Update camera's classification history
                # TODO: Add classification record tracking to Camera class
                # cam.add_classification_record({
                #     'timestamp': current_time,
                #     'object_position': object_position,
                #     'prediction': pred,
                #     'true_label': true_label,
                #     'is_correct': is_correct,
                #     'accuracy': cam_accuracy
                # })
            else:
                self.energy_violations += 1
        
        # Majority voting
        if predictions:
            final_prediction = max(set(predictions), key=predictions.count)
            is_successful = final_prediction == true_label
            self.total_classifications += 1
            if is_successful:
                self.successful_classifications += 1
        else:
            final_prediction = -1
            is_successful = False
        
        # Calculate confidence based on agreement
        confidence = (predictions.count(final_prediction) / len(predictions) 
                     if predictions else 0.0)
        
        # Update history
        result = {
            'instance_id': instance_id,
            'success': is_successful,
            'prediction': final_prediction,
            'true_label': true_label,
            'selected_cameras': selected_camera_ids,
            'participating_cameras': participating_cameras,
            'collective_accuracy': collective_accuracy,
            'individual_correct': correct_count,
            'total_participating': len(participating_cameras),
            'timestamp': current_time,
            'object_position': object_position.tolist(),
            'confidence': confidence,
            'using_enhanced': self.using_enhanced
        }
        
        self._update_history(result)
        
        return result
    
    def _calculate_enhanced_collective_accuracy(
        self,
        cameras: List[Camera],
        object_position: np.ndarray,
        distances: Optional[np.ndarray] = None
    ) -> float:
        """Calculate collective accuracy with position awareness."""
        if not cameras or not self.using_enhanced:
            return self._calculate_collective_accuracy(cameras)
        
        # Use enhanced accuracy model
        accuracy_model = cameras[0].accuracy_model
        return accuracy_model.get_collective_accuracy_with_positions(
            cameras, object_position, distances=distances
        )
    
    def get_optimal_camera_count(
        self, object_position: Optional[np.ndarray] = None
    ) -> int:
        """
        Determine optimal number of cameras for classification.
        
        Args:
            object_position: Optional object position for enhanced calculation
            
        Returns:
            Optimal number of cameras
        """
        if not self.using_enhanced or object_position is None:
            # Fallback to energy-based calculation
            available_cameras = [cam for cam in self.cameras if cam.can_classify()]
            energies = np.array([cam.current_energy for cam in available_cameras])
            energies.sort()[::-1]  # Sort descending
            
            for n in range(1, len(available_cameras) + 1):
                acc = self.cameras[0].accuracy_model.get_collective_accuracy(energies[:n])
                if acc >= self.min_accuracy_threshold:
                    return n
            return len(available_cameras)
        
        # Enhanced calculation with position
        accuracy_model = self.cameras[0].accuracy_model
        available_cameras = [cam for cam in self.cameras if cam.can_classify()]
        
        # Calculate energy budget
        min_cost = min(cam.energy_model.classification_cost for cam in available_cameras)
        energy_budget = min_cost * len(available_cameras) * self.energy_budget_factor
        
        # Get optimal selection
        selected_indices = accuracy_model.optimal_camera_selection(
            available_cameras, object_position, energy_budget, self.min_accuracy_threshold
        )
        
        return len(selected_indices)
    
    def get_camera_effectiveness(
        self, camera_id: int, object_position: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate effectiveness score for a camera.
        
        Args:
            camera_id: Camera ID
            object_position: Optional object position
            
        Returns:
            Effectiveness score (0-1)
        """
        camera = self.cameras[camera_id]
        
        if not camera.can_classify():
            return 0.0
        
        if self.using_enhanced and object_position is not None:
            # Position-based accuracy
            accuracy = camera.accuracy_model.get_position_based_accuracy(
                camera, object_position
            )
        else:
            # Energy-based accuracy
            accuracy = camera.current_accuracy
        
        # Factor in energy efficiency
        energy_factor = camera.current_energy / camera.energy_model.capacity
        
        # Factor in recent performance
        recent_success_rate = camera.get_recent_success_rate()
        
        # Combined effectiveness
        effectiveness = (
            0.5 * accuracy +
            0.3 * energy_factor +
            0.2 * recent_success_rate
        )
        
        return effectiveness