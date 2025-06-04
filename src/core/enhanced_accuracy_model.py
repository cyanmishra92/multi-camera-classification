# src/core/enhanced_accuracy_model.py
"""Enhanced accuracy modeling with spatial awareness and view overlap."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .accuracy_model import AccuracyModel, AccuracyParameters


@dataclass
class EnhancedAccuracyParameters(AccuracyParameters):
    """Extended parameters for enhanced accuracy model."""
    distance_decay: float = 0.01  # Distance impact factor
    angle_penalty: float = 0.3    # Viewing angle penalty
    overlap_bonus: float = 0.2    # Bonus for overlapping views
    optimal_distance: float = 20.0  # Optimal viewing distance


class EnhancedAccuracyModel(AccuracyModel):
    """
    Enhanced accuracy model with spatial awareness.
    
    Improvements:
    1. Distance-based accuracy degradation
    2. Viewing angle considerations
    3. Overlapping field-of-view bonuses
    4. Object-position aware calculations
    """
    
    def __init__(self, params: EnhancedAccuracyParameters, energy_model):
        """Initialize enhanced accuracy model."""
        super().__init__(params, energy_model)
        self.distance_decay = params.distance_decay
        self.angle_penalty = params.angle_penalty
        self.overlap_bonus = params.overlap_bonus
        self.optimal_distance = params.optimal_distance
        
    def get_position_based_accuracy(self, camera, object_position: np.ndarray,
                                   energy: Optional[float] = None,
                                   precomputed_distance: Optional[float] = None
                                   ) -> float:
        """
        Calculate accuracy considering camera position and object location.
        
        Args:
            camera: Camera object with position information
            object_position: 3D position of object
            energy: Camera energy (if None, uses camera's current energy)
            
        Returns:
            Position-adjusted accuracy
        """
        if energy is None:
            energy = camera.current_energy
            
        # Base accuracy from energy
        base_accuracy = self.get_accuracy(energy)
        
        # Distance factor
        if precomputed_distance is None:
            distance = np.linalg.norm(camera.position - object_position)
        else:
            distance = precomputed_distance
        distance_factor = self._calculate_distance_factor(distance)
        
        # Viewing angle factor
        angle_factor = self._calculate_angle_factor(camera, object_position)
        
        # Combined accuracy
        return base_accuracy * distance_factor * angle_factor
    
    def _calculate_distance_factor(self, distance: float) -> float:
        """Calculate accuracy degradation based on distance."""
        if distance <= self.optimal_distance:
            return 1.0
        else:
            # Exponential decay beyond optimal distance
            excess = distance - self.optimal_distance
            return np.exp(-self.distance_decay * excess)
    
    def _calculate_angle_factor(self, camera, object_position: np.ndarray) -> float:
        """Calculate accuracy based on viewing angle."""
        # Vector from camera to object
        to_object = object_position - camera.position
        to_object_norm = to_object / (np.linalg.norm(to_object) + 1e-6)
        
        # Assume cameras face origin by default (can be enhanced)
        camera_facing = -camera.position / (np.linalg.norm(camera.position) + 1e-6)
        
        # Dot product gives cosine of angle
        cos_angle = np.dot(camera_facing, to_object_norm)
        
        # Convert to factor (1.0 when facing directly, decreases with angle)
        angle_factor = max(0, cos_angle)
        
        # Apply penalty for extreme angles
        if angle_factor < 0.5:
            angle_factor *= (1 - self.angle_penalty)
            
        return angle_factor
    
    def get_collective_accuracy_with_positions(self, cameras: List,
                                              object_position: np.ndarray,
                                              energies: Optional[np.ndarray] = None,
                                              distances: Optional[np.ndarray] = None) -> float:
        """
        Calculate collective accuracy considering camera positions and overlap.
        
        Args:
            cameras: List of camera objects with positions
            object_position: 3D position of object to classify
            energies: Optional energy levels (uses camera energies if None)
            
        Returns:
            Enhanced collective accuracy
        """
        if len(cameras) == 0:
            return 0.0
            
        # Get individual position-based accuracies
        if energies is None:
            energies = [cam.current_energy for cam in cameras]

        if distances is None:
            distances = [np.linalg.norm(cam.position - object_position) for cam in cameras]

        accuracies = [
            self.get_position_based_accuracy(cam, object_position, energy, dist)
            for cam, energy, dist in zip(cameras, energies, distances)
        ]
        
        # Calculate overlap bonuses
        overlap_factor = self._calculate_overlap_factor(cameras, object_position)
        
        # Base collective accuracy
        base_collective = self._calculate_collective_with_correlation(accuracies)
        
        # Apply overlap bonus
        enhanced_collective = base_collective * (1 + overlap_factor * self.overlap_bonus)
        
        return min(enhanced_collective, self.max_accuracy)
    
    def _calculate_overlap_factor(self, cameras: List, object_position: np.ndarray) -> float:
        """
        Calculate overlap factor for multiple cameras viewing same object.
        
        Returns value in [0, 1] indicating degree of view overlap.
        """
        if len(cameras) <= 1:
            return 0.0
            
        # Calculate viewing angles for each camera pair
        overlap_scores = []
        
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                # Vectors from object to cameras
                v1 = cameras[i].position - object_position
                v2 = cameras[j].position - object_position
                
                # Normalize
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
                
                # Angle between cameras (from object's perspective)
                cos_angle = np.dot(v1_norm, v2_norm)
                angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
                angle_deg = np.degrees(angle_rad)
                
                # Good overlap if cameras are 30-150 degrees apart
                if 30 <= angle_deg <= 150:
                    overlap_scores.append(1.0)
                elif angle_deg < 30:
                    # Too close together
                    overlap_scores.append(0.3)
                else:
                    # Opposite sides
                    overlap_scores.append(0.7)
                    
        return np.mean(overlap_scores) if overlap_scores else 0.0
    
    def _calculate_collective_with_correlation(self, accuracies: List[float]) -> float:
        """Enhanced collective accuracy calculation with correlation."""
        if not accuracies:
            return 0.0
            
        accuracies = np.array(accuracies)
        
        if self.correlation_factor == 0:
            # Independent errors
            return 1 - np.prod(1 - accuracies)
        else:
            # Correlated errors with variance consideration
            avg_accuracy = np.mean(accuracies)
            var_accuracy = np.var(accuracies)
            n = len(accuracies)
            
            # Adjust correlation based on variance (less correlation if diverse accuracies)
            effective_correlation = self.correlation_factor * (1 - var_accuracy)
            exponent = n * (1 - effective_correlation)
            
            return 1 - (1 - avg_accuracy) ** exponent
    
    def optimal_camera_selection(self, cameras: List, object_position: np.ndarray,
                               energy_constraint: float, min_accuracy: float) -> List[int]:
        """
        Select optimal set of cameras for classification.
        
        Args:
            cameras: List of all available cameras
            object_position: Object to classify
            energy_constraint: Maximum total energy to use
            min_accuracy: Minimum required accuracy
            
        Returns:
            Indices of selected cameras
        """
        # Score each camera by accuracy per unit energy
        scores = []
        for i, cam in enumerate(cameras):
            if cam.current_energy >= self.min_operational:
                accuracy = self.get_position_based_accuracy(cam, object_position)
                efficiency = accuracy / cam.energy_model.classification_cost
                scores.append((i, efficiency, accuracy))
        
        # Sort by efficiency
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection
        selected = []
        total_energy = 0
        
        for idx, efficiency, accuracy in scores:
            cam = cameras[idx]
            if total_energy + cam.energy_model.classification_cost <= energy_constraint:
                selected.append(idx)
                total_energy += cam.energy_model.classification_cost
                
                # Check if we meet accuracy requirement
                selected_cams = [cameras[i] for i in selected]
                collective_acc = self.get_collective_accuracy_with_positions(
                    selected_cams, object_position)
                
                if collective_acc >= min_accuracy:
                    break
                    
        return selected