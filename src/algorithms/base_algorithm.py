"""Base class for classification algorithms."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from ..core.camera import Camera

logger = logging.getLogger(__name__)


class BaseClassificationAlgorithm(ABC):
    """
    Abstract base class for multi-camera classification algorithms.
    
    Provides common functionality and interface for different scheduling strategies.
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10
    ):
        """
        Initialize base algorithm.
        
        Args:
            cameras: List of camera objects
            min_accuracy_threshold: Minimum required collective accuracy (α_min)
            history_length: Length of classification history to maintain (k)
        """
        self.cameras = cameras
        self.min_accuracy_threshold = min_accuracy_threshold
        self.history_length = history_length
        
        # Classification history
        self.classification_history = []
        
        # Performance metrics
        self.total_classifications = 0
        self.successful_classifications = 0
        self.energy_violations = 0
        self.accuracy_violations = 0
        
    @abstractmethod
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras for classification at given instance.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
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
        Perform classification using selected cameras.
        
        Args:
            instance_id: ID of classification instance
            object_position: Position of object to classify
            true_label: True label of object
            current_time: Current simulation time
            
        Returns:
            Classification result dictionary
        """
        # Select cameras
        selected_camera_ids = self.select_cameras(instance_id, current_time)
        
        if not selected_camera_ids:
            self.energy_violations += 1
            logger.warning(f"No cameras selected for instance {instance_id}")
            return {
                'instance_id': instance_id,
                'success': False,
                'reason': 'no_cameras_available',
                'selected_cameras': [],
                'collective_accuracy': 0.0
            }
        
        # Check collective accuracy
        selected_cameras = [self.cameras[i] for i in selected_camera_ids]
        energies = np.array([cam.current_energy for cam in selected_cameras])
        
        # Calculate collective accuracy
        collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
        
        if collective_accuracy < self.min_accuracy_threshold:
            self.accuracy_violations += 1
            logger.warning(
                f"Collective accuracy {collective_accuracy:.3f} below threshold "
                f"{self.min_accuracy_threshold:.3f} for instance {instance_id}"
            )
        
        # Perform classification
        predictions = []
        correct_count = 0
        
        for cam in selected_cameras:
            if cam.can_classify():
                pred, is_correct = cam.classify(object_position, true_label)
                predictions.append(pred)
                if is_correct:
                    correct_count += 1
                    
                # Update camera energy
                cam.update_energy(0, is_classifying=True)
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
            
        # Update history
        result = {
            'instance_id': instance_id,
            'success': is_successful,
            'prediction': final_prediction,
            'true_label': true_label,
            'selected_cameras': selected_camera_ids,
            'collective_accuracy': collective_accuracy,
            'individual_correct': correct_count,
            'total_participating': len(selected_cameras),
            'timestamp': current_time
        }
        
        self._update_history(result)
        
        return result
    
    def _calculate_collective_accuracy(self, cameras: List[Camera]) -> float:
        """Calculate collective accuracy of camera group."""
        if not cameras:
            return 0.0
            
        # Assume cameras share the same accuracy model
        energies = np.array([cam.current_energy for cam in cameras])
        return cameras[0].accuracy_model.get_collective_accuracy(energies)
    
    def _update_history(self, result: Dict) -> None:
        """Update classification history with size limit."""
        self.classification_history.append(result)
        
        # Maintain history length
        if len(self.classification_history) > self.history_length:
            self.classification_history.pop(0)
            
    def get_recent_participants(self) -> List[int]:
        """Get list of cameras that participated recently."""
        recent_participants = set()
        
        for record in self.classification_history:
            recent_participants.update(record.get('selected_cameras', []))
            
        return list(recent_participants)
    
    def get_performance_metrics(self) -> Dict:
        """Get algorithm performance metrics."""
        accuracy = (
            self.successful_classifications / self.total_classifications
            if self.total_classifications > 0 else 0
        )
        
        return {
            'total_classifications': self.total_classifications,
            'successful_classifications': self.successful_classifications,
            'accuracy': accuracy,
            'energy_violations': self.energy_violations,
            'accuracy_violations': self.accuracy_violations,
            'avg_cameras_per_classification': np.mean([
                len(r['selected_cameras']) 
                for r in self.classification_history
            ]) if self.classification_history else 0
        }