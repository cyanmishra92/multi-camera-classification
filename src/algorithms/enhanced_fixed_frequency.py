"""Enhanced Algorithm 1: Fixed frequency with position-aware selection."""

import numpy as np
from typing import List, Dict, Optional
import logging

from .enhanced_base_algorithm import EnhancedBaseClassificationAlgorithm
from ..core.camera import Camera
from ..core.enhanced_accuracy_model import EnhancedAccuracyModel

logger = logging.getLogger(__name__)


class EnhancedFixedFrequencyAlgorithm(EnhancedBaseClassificationAlgorithm):
    """
    Enhanced fixed frequency algorithm with position-aware camera selection.
    
    Improvements:
    - Position-based camera selection
    - Optimal camera count determination
    - Strategic selection with spatial awareness
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        use_game_theory: bool = True,
        position_weight: float = 0.7
    ):
        """
        Initialize enhanced fixed frequency algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes (n)
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history
            use_game_theory: Whether to use game-theoretic selection
            position_weight: Weight for position-based selection (0-1)
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.num_classes = num_classes
        self.use_game_theory = use_game_theory
        self.position_weight = position_weight
        
        # Assign cameras to classes with spatial diversity
        self._assign_camera_classes_spatial()
        
        # Initialize performance tracking per class
        self.class_performance = {i: [] for i in range(num_classes)}
        
    def _assign_camera_classes_spatial(self) -> None:
        """Assign cameras to classes ensuring spatial diversity."""
        # If not using enhanced model, fall back to round-robin
        if not self.using_enhanced:
            self._assign_camera_classes_simple()
            return
            
        # Cluster cameras spatially first
        positions = np.array([cam.position for cam in self.cameras])
        
        # Simple k-means style assignment for spatial diversity
        # Initialize class centers
        class_centers = []
        indices = np.random.choice(len(self.cameras), self.num_classes, replace=False)
        for idx in indices:
            class_centers.append(positions[idx])
        
        # Assign cameras to nearest class center
        for i, camera in enumerate(self.cameras):
            distances = [np.linalg.norm(camera.position - center) 
                        for center in class_centers]
            camera.class_assignment = np.argmin(distances)
        
        # Balance classes by reassigning if needed
        class_counts = {i: 0 for i in range(self.num_classes)}
        for cam in self.cameras:
            class_counts[cam.class_assignment] += 1
            
        target_size = len(self.cameras) // self.num_classes
        
        # Rebalance overloaded classes
        for class_id in range(self.num_classes):
            cameras_in_class = [i for i, cam in enumerate(self.cameras) 
                               if cam.class_assignment == class_id]
            
            if len(cameras_in_class) > target_size + 1:
                # Move excess cameras to underloaded classes
                excess = len(cameras_in_class) - target_size
                for i in range(excess):
                    # Find underloaded class
                    for other_class in range(self.num_classes):
                        other_count = sum(1 for cam in self.cameras 
                                        if cam.class_assignment == other_class)
                        if other_count < target_size:
                            self.cameras[cameras_in_class[-(i+1)]].class_assignment = other_class
                            break
        
        # Create class mapping
        self.camera_classes = {}
        for class_id in range(self.num_classes):
            self.camera_classes[class_id] = [
                i for i, cam in enumerate(self.cameras)
                if cam.class_assignment == class_id
            ]
            
        logger.info(f"Assigned {len(self.cameras)} cameras to {self.num_classes} "
                   f"classes with spatial diversity")
        
    def _assign_camera_classes_simple(self) -> None:
        """Simple round-robin assignment."""
        for i, camera in enumerate(self.cameras):
            camera.class_assignment = i % self.num_classes
            
        self.camera_classes = {}
        for class_id in range(self.num_classes):
            self.camera_classes[class_id] = [
                i for i, cam in enumerate(self.cameras)
                if cam.class_assignment == class_id
            ]
    
    def select_cameras(self, instance_id: int, current_time: float,
                      object_position: Optional[np.ndarray] = None) -> List[int]:
        """
        Enhanced camera selection with position awareness.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            object_position: Position of object to classify
            
        Returns:
            List of selected camera IDs
        """
        # Determine active class
        active_class = instance_id % self.num_classes
        class_cameras = self.camera_classes[active_class]
        
        if self.use_game_theory and object_position is not None:
            # Use strategic selection with position awareness
            return self._select_strategic_enhanced(class_cameras, object_position)
        elif object_position is not None and self.using_enhanced:
            # Position-aware greedy selection
            return self._select_position_aware(class_cameras, object_position)
        else:
            # Standard greedy selection
            return self._select_greedy(class_cameras)
    
    def _select_position_aware(self, candidate_cameras: List[int],
                               object_position: np.ndarray) -> List[int]:
        """
        Position-aware camera selection.
        
        Args:
            candidate_cameras: List of candidate camera IDs
            object_position: Object position
            
        Returns:
            Selected camera IDs
        """
        if not self.using_enhanced:
            return self._select_greedy(candidate_cameras)
            
        # Precompute distances to all candidate cameras
        positions = np.array([self.cameras[i].position for i in candidate_cameras])
        distances = np.linalg.norm(positions - object_position, axis=1)
        distance_map = {cid: d for cid, d in zip(candidate_cameras, distances)}

        # Score cameras by position and energy
        camera_scores = []

        for cam_id in candidate_cameras:
            camera = self.cameras[cam_id]

            if not camera.can_classify():
                continue

            dist = distance_map[cam_id]

            # Calculate position-based accuracy
            pos_accuracy = camera.accuracy_model.get_position_based_accuracy(
                camera, object_position, precomputed_distance=dist
            )

            # Energy factor
            energy_factor = camera.current_energy / camera.energy_model.capacity

            # Combined score
            score = (
                self.position_weight * pos_accuracy
                + (1 - self.position_weight) * energy_factor
            )

            camera_scores.append((cam_id, score, dist))
        
        # Sort by score
        camera_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select cameras to meet accuracy threshold
        selected = []
        
        for cam_id, score, dist in camera_scores:
            selected.append(cam_id)

            # Check if accuracy threshold met
            selected_cameras = [self.cameras[i] for i in selected]
            selected_distances = np.array([distance_map[i] for i in selected])
            collective_accuracy = self._calculate_enhanced_collective_accuracy(
                selected_cameras, object_position, selected_distances
            )
            
            if collective_accuracy >= self.min_accuracy_threshold:
                break
                
        return selected
    
    def _select_strategic_enhanced(self, candidate_cameras: List[int],
                                  object_position: np.ndarray) -> List[int]:
        """
        Strategic selection with game theory and position awareness.
        
        Args:
            candidate_cameras: List of candidate camera IDs
            object_position: Object position
            
        Returns:
            Selected camera IDs
        """
        if not self.using_enhanced:
            return self._select_greedy(candidate_cameras)
            
        # Precompute distances
        positions = np.array([self.cameras[i].position for i in candidate_cameras])
        distances = np.linalg.norm(positions - object_position, axis=1)
        distance_map = {cid: d for cid, d in zip(candidate_cameras, distances)}

        # Calculate utilities for each camera
        camera_utilities = []

        for cam_id in candidate_cameras:
            camera = self.cameras[cam_id]

            if not camera.can_classify():
                continue

            dist = distance_map[cam_id]

            # Position-based accuracy
            accuracy = camera.accuracy_model.get_position_based_accuracy(
                camera, object_position, precomputed_distance=dist
            )
            
            # Expected reward
            reward = accuracy * 1.0  # Normalized reward
            
            # Energy cost consideration
            energy_ratio = camera.current_energy / camera.energy_model.capacity
            future_value = energy_ratio * 0.9  # Discount factor
            
            # Participation history bonus
            recent_participation = cam_id in self.get_recent_participants()
            diversity_bonus = 0.1 if not recent_participation else 0
            
            # Total utility
            utility = reward + future_value + diversity_bonus
            
            camera_utilities.append((cam_id, utility, accuracy))
        
        # Sort by utility
        camera_utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Select strategically
        selected = []
        remaining_budget = len(candidate_cameras) * 50  # Energy budget
        
        for cam_id, utility, _ in camera_utilities:
            camera = self.cameras[cam_id]
            
            if camera.energy_model.classification_cost <= remaining_budget:
                selected.append(cam_id)
                remaining_budget -= camera.energy_model.classification_cost
                
                # Check accuracy with current selection
                selected_cameras = [self.cameras[i] for i in selected]
                selected_distances = np.array([distance_map[i] for i in selected])
                collective_accuracy = self._calculate_enhanced_collective_accuracy(
                    selected_cameras, object_position, selected_distances
                )
                
                if collective_accuracy >= self.min_accuracy_threshold:
                    break
                    
        # If we couldn't meet threshold, add more cameras greedily
        if len(selected) < len(camera_utilities):
            for cam_id, _, _ in camera_utilities:
                if cam_id not in selected:
                    selected.append(cam_id)

                    selected_cameras = [self.cameras[i] for i in selected]
                    selected_distances = np.array([distance_map[i] for i in selected])
                    collective_accuracy = self._calculate_enhanced_collective_accuracy(
                        selected_cameras, object_position, selected_distances
                    )
                    
                    if collective_accuracy >= self.min_accuracy_threshold:
                        break
                        
        return selected
    
    def _select_greedy(self, candidate_cameras: List[int]) -> List[int]:
        """Standard greedy selection by energy."""
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
    
    def get_class_performance(self, class_id: int) -> Dict:
        """Get performance metrics for a specific class."""
        class_results = [r for r in self.classification_history
                        if r['instance_id'] % self.num_classes == class_id]
        
        if not class_results:
            return {
                'class_id': class_id,
                'total_classifications': 0,
                'accuracy': 0.0,
                'avg_cameras_used': 0.0,
                'energy_efficiency': 0.0
            }
        
        accuracy = sum(r['success'] for r in class_results) / len(class_results)
        avg_cameras = np.mean([len(r['participating_cameras']) for r in class_results])
        
        # Energy efficiency: accuracy per camera
        energy_efficiency = accuracy / avg_cameras if avg_cameras > 0 else 0
        
        return {
            'class_id': class_id,
            'total_classifications': len(class_results),
            'accuracy': accuracy,
            'avg_cameras_used': avg_cameras,
            'energy_efficiency': energy_efficiency
        }