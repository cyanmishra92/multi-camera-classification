"""Enhanced multi-camera network with improved accuracy modeling."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Type
import logging
from dataclasses import dataclass

from .camera import Camera
from .network import CameraNetwork, NetworkConfig
from .energy_model import EnergyModel, EnergyParameters
from .enhanced_accuracy_model import EnhancedAccuracyModel, EnhancedAccuracyParameters
from ..algorithms.base_algorithm import BaseClassificationAlgorithm
from ..algorithms.fixed_frequency import FixedFrequencyAlgorithm
from ..algorithms.variable_frequency import VariableFrequencyAlgorithm
from ..algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from ..algorithms.enhanced_fixed_frequency import EnhancedFixedFrequencyAlgorithm
from ..algorithms.game_theoretic_fixed_frequency import GameTheoreticFixedFrequencyAlgorithm
from ..algorithms.accuracy_adaptive_algorithm import AccuracyAdaptiveAlgorithm
from ..game_theory.utility_functions import UtilityParameters
from ..game_theory.nash_equilibrium import NashEquilibriumSolver

logger = logging.getLogger(__name__)


@dataclass
class EnhancedNetworkConfig(NetworkConfig):
    """Extended configuration for enhanced network."""
    enhanced_accuracy_params: Optional[EnhancedAccuracyParameters] = None
    use_enhanced_accuracy: bool = True


class EnhancedCameraNetwork(CameraNetwork):
    """
    Enhanced camera network with improved accuracy modeling.
    
    Features:
    - Position-aware accuracy calculations
    - Overlapping field-of-view bonuses
    - Optimal camera selection strategies
    """
    
    def __init__(self, config: EnhancedNetworkConfig):
        """Initialize enhanced camera network."""
        self.use_enhanced_accuracy = config.use_enhanced_accuracy
        self.enhanced_accuracy_params = config.enhanced_accuracy_params
        super().__init__(config)
        
    def _initialize_cameras(self) -> List[Camera]:
        """Initialize camera objects with enhanced models if enabled."""
        cameras = []
        
        # Generate camera positions if not provided
        if self.config.camera_positions is None:
            positions = self._generate_camera_positions()
        else:
            positions = self.config.camera_positions
            
        # Create energy model
        energy_model = EnergyModel(
            self.config.energy_params or EnergyParameters(
                capacity=1000,
                recharge_rate=10,
                classification_cost=50,
                min_operational=100
            )
        )
        
        # Create accuracy model (enhanced or standard)
        if self.use_enhanced_accuracy:
            accuracy_model = EnhancedAccuracyModel(
                self.enhanced_accuracy_params or EnhancedAccuracyParameters(
                    max_accuracy=0.95,
                    min_accuracy_ratio=0.3,
                    correlation_factor=0.2,
                    distance_decay=0.01,
                    angle_penalty=0.3,
                    overlap_bonus=0.2,
                    optimal_distance=20.0
                ),
                energy_model
            )
            logger.info("Using enhanced accuracy model with spatial awareness")
        else:
            from .accuracy_model import AccuracyModel, AccuracyParameters
            accuracy_model = AccuracyModel(
                self.config.accuracy_params or AccuracyParameters(
                    max_accuracy=0.95,
                    min_accuracy_ratio=0.3,
                    correlation_factor=0.2
                ),
                energy_model
            )
            logger.info("Using standard accuracy model")
        
        # Store accuracy model for network use
        self.accuracy_model = accuracy_model
        
        # Create cameras
        for i in range(self.config.num_cameras):
            camera = Camera(
                camera_id=i,
                position=positions[i],
                energy_model=energy_model,
                accuracy_model=accuracy_model
            )
            cameras.append(camera)
            
        logger.info(f"Initialized {len(cameras)} cameras")
        return cameras
    
    def _generate_camera_positions(self) -> np.ndarray:
        """Generate strategic camera positions for good coverage."""
        positions = []
        
        # Use a more structured placement for better coverage
        n_rings = int(np.sqrt(self.config.num_cameras))
        cameras_per_ring = self.config.num_cameras // n_rings
        extra = self.config.num_cameras % n_rings
        
        cam_idx = 0
        for ring in range(n_rings):
            # Determine number of cameras in this ring
            n_in_ring = cameras_per_ring + (1 if ring < extra else 0)
            
            # Ring radius and height
            r = 20 + ring * 15  # Radial distance
            z = 10 + ring * 5   # Height
            
            for i in range(n_in_ring):
                if cam_idx >= self.config.num_cameras:
                    break
                    
                # Evenly space cameras around ring
                theta = 2 * np.pi * i / n_in_ring
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                positions.append([x, y, z])
                cam_idx += 1
                
        return np.array(positions)
    
    def set_algorithm(
        self,
        algorithm_type: str,
        classification_frequency: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Set the classification algorithm with enhanced support.
        
        Args:
            algorithm_type: Type of algorithm ('fixed', 'variable', 'unknown')
            classification_frequency: Expected classification frequency
            **kwargs: Additional algorithm parameters
        """
        if algorithm_type == 'fixed':
            if self.use_enhanced_accuracy:
                # Use adaptive algorithm if specified
                if kwargs.get('use_adaptive', False) and self.config.utility_params is not None:
                    self.algorithm = AccuracyAdaptiveAlgorithm(
                        cameras=self.cameras,
                        num_classes=self.config.num_classes,
                        utility_params=self.config.utility_params,
                        **kwargs
                    )
                # Use game-theoretic version if utility params available
                elif self.config.utility_params is not None:
                    self.algorithm = GameTheoreticFixedFrequencyAlgorithm(
                        cameras=self.cameras,
                        num_classes=self.config.num_classes,
                        utility_params=self.config.utility_params,
                        **kwargs
                    )
                else:
                    self.algorithm = EnhancedFixedFrequencyAlgorithm(
                        cameras=self.cameras,
                        num_classes=self.config.num_classes,
                        **kwargs
                    )
            else:
                self.algorithm = FixedFrequencyAlgorithm(
                    cameras=self.cameras,
                    num_classes=self.config.num_classes,
                    **kwargs
                )
        elif algorithm_type == 'variable':
            if classification_frequency is None:
                raise ValueError("Variable frequency algorithm requires classification_frequency")
                
            recharge_time = self.cameras[0].energy_model.min_recharge_time
            self.algorithm = VariableFrequencyAlgorithm(
                cameras=self.cameras,
                num_classes=self.config.num_classes,
                classification_frequency=classification_frequency,
                recharge_time=recharge_time,
                **kwargs
            )
        elif algorithm_type == 'unknown':
            utility_params = self.config.utility_params or UtilityParameters()
            self.algorithm = UnknownFrequencyAlgorithm(
                cameras=self.cameras,
                num_classes=self.config.num_classes,
                utility_params=utility_params,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
            
        logger.info(f"Set algorithm to {algorithm_type} (enhanced={self.use_enhanced_accuracy})")
    
    def classify_object(
        self,
        object_position: np.ndarray,
        true_label: int
    ) -> Dict:
        """
        Enhanced classification with position-aware accuracy.
        
        Args:
            object_position: Position of object to classify
            true_label: True label of object
            
        Returns:
            Classification result with enhanced metrics
        """
        if self.algorithm is None:
            raise ValueError("No algorithm set")
            
        # Store object position for algorithms to use
        self._current_object_position = object_position
            
        # Perform classification
        result = self.algorithm.classify(
            instance_id=self.classification_count,
            object_position=object_position,
            true_label=true_label,
            current_time=self.current_time
        )
        
        # If using enhanced accuracy, recalculate actual accuracy
        if self.use_enhanced_accuracy and isinstance(self.accuracy_model, EnhancedAccuracyModel):
            participating_cameras = [
                self.cameras[idx] for idx in result.get('participating_cameras', [])
            ]
            
            if participating_cameras:
                # Recalculate with position awareness
                enhanced_accuracy = self.accuracy_model.get_collective_accuracy_with_positions(
                    participating_cameras, object_position
                )
                
                # Update result with enhanced accuracy
                result['enhanced_accuracy'] = enhanced_accuracy
                result['original_accuracy'] = result.get('collective_accuracy', 0)
                result['collective_accuracy'] = enhanced_accuracy
                
                # Recalculate success based on enhanced accuracy
                if 'confidence' in result:
                    result['success'] = (np.random.random() < enhanced_accuracy)
        
        self.classification_count += 1
        
        # Track performance
        self._update_performance_tracking(result)
        
        return result
    
    def get_optimal_cameras(self, object_position: np.ndarray, 
                          energy_budget: float, min_accuracy: float) -> List[int]:
        """
        Get optimal camera selection for given constraints.
        
        Args:
            object_position: Object to classify
            energy_budget: Maximum energy to use
            min_accuracy: Minimum required accuracy
            
        Returns:
            List of camera indices
        """
        if isinstance(self.accuracy_model, EnhancedAccuracyModel):
            return self.accuracy_model.optimal_camera_selection(
                self.cameras, object_position, energy_budget, min_accuracy
            )
        else:
            # Fallback to energy-based selection
            available = [(i, cam) for i, cam in enumerate(self.cameras)
                        if cam.can_classify()]
            available.sort(key=lambda x: x[1].current_energy, reverse=True)
            
            selected = []
            total_cost = 0
            
            for idx, cam in available:
                if total_cost + cam.classification_cost <= energy_budget:
                    selected.append(idx)
                    total_cost += cam.classification_cost
                    
                    # Check accuracy
                    energies = [self.cameras[i].current_energy for i in selected]
                    acc = self.accuracy_model.get_collective_accuracy(np.array(energies))
                    if acc >= min_accuracy:
                        break
                        
            return selected
    
    def analyze_coverage(self, grid_size: int = 20) -> Dict:
        """
        Analyze network coverage quality.
        
        Args:
            grid_size: Grid resolution for analysis
            
        Returns:
            Coverage statistics
        """
        # Create grid of test points
        x_range = np.linspace(-50, 50, grid_size)
        y_range = np.linspace(-50, 50, grid_size)
        z = 0  # Ground level
        
        coverage_map = np.zeros((grid_size, grid_size))
        accuracy_map = np.zeros((grid_size, grid_size))
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                pos = np.array([x, y, z])
                
                # Count cameras that can see this position well
                good_cameras = 0
                max_accuracy = 0
                
                for cam in self.cameras:
                    if isinstance(self.accuracy_model, EnhancedAccuracyModel):
                        acc = self.accuracy_model.get_position_based_accuracy(cam, pos)
                    else:
                        acc = cam.current_accuracy
                        
                    if acc > 0.5:  # Threshold for "good" coverage
                        good_cameras += 1
                    max_accuracy = max(max_accuracy, acc)
                
                coverage_map[i, j] = good_cameras
                accuracy_map[i, j] = max_accuracy
                
        return {
            'coverage_map': coverage_map,
            'accuracy_map': accuracy_map,
            'avg_coverage': np.mean(coverage_map),
            'min_coverage': np.min(coverage_map),
            'max_coverage': np.max(coverage_map),
            'blind_spots': np.sum(coverage_map == 0),
            'well_covered': np.sum(coverage_map >= 3)
        }