"""Multi-camera network coordination."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Type
import logging
from dataclasses import dataclass
import time

from .camera import Camera
from .energy_model import EnergyModel, EnergyParameters
from .accuracy_model import AccuracyModel, AccuracyParameters
from ..algorithms.base_algorithm import BaseClassificationAlgorithm
from ..algorithms.fixed_frequency import FixedFrequencyAlgorithm
from ..algorithms.variable_frequency import VariableFrequencyAlgorithm
from ..algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from ..game_theory.utility_functions import UtilityParameters
from ..game_theory.nash_equilibrium import NashEquilibriumSolver

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for multi-camera network."""
    num_cameras: int
    num_classes: int
    num_objects: int
    camera_positions: Optional[np.ndarray] = None
    energy_params: Optional[EnergyParameters] = None
    accuracy_params: Optional[AccuracyParameters] = None
    utility_params: Optional[UtilityParameters] = None
    

class CameraNetwork:
    """
    Multi-camera network with coordination and energy management.
    
    Manages the network of cameras and coordinates classification tasks.
    """
    
    def __init__(self, config: NetworkConfig):
        """
        Initialize camera network.
        
        Args:
            config: Network configuration
        """
        self.config = config
        
        # Initialize cameras
        self.cameras = self._initialize_cameras()
        
        # Classification algorithm
        self.algorithm = None
        
        # Network state
        self.current_time = 0.0
        self.classification_count = 0
        
        # Performance tracking
        self.performance_history = []
        self.energy_history = []
        
        # Nash equilibrium solver
        self.nash_solver = NashEquilibriumSolver()
        
    def _initialize_cameras(self) -> List[Camera]:
        """Initialize camera objects with positions and models."""
        cameras = []
        
        # Generate camera positions if not provided
        if self.config.camera_positions is None:
            positions = self._generate_camera_positions()
        else:
            positions = self.config.camera_positions
            
        # Create energy and accuracy models
        energy_model = EnergyModel(
            self.config.energy_params or EnergyParameters(
                capacity=1000,
                recharge_rate=10,
                classification_cost=50,
                min_operational=100
            )
        )
        
        accuracy_model = AccuracyModel(
            self.config.accuracy_params or AccuracyParameters(
                max_accuracy=0.95,
                min_accuracy_ratio=0.3,
                correlation_factor=0.2
            ),
            energy_model
        )
        
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
        """Generate random camera positions in 3D space."""
        # Place cameras in a hemisphere above ground
        positions = []
        
        for i in range(self.config.num_cameras):
            # Random position in hemisphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi / 2)
            r = np.random.uniform(20, 50)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            positions.append([x, y, z])
            
        return np.array(positions)
    
    def set_algorithm(
        self,
        algorithm_type: str,
        classification_frequency: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Set the classification algorithm.
        
        Args:
            algorithm_type: Type of algorithm ('fixed', 'variable', 'unknown')
            classification_frequency: Expected classification frequency
            **kwargs: Additional algorithm parameters
        """
        if algorithm_type == 'fixed':
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
            
        logger.info(f"Set algorithm to {algorithm_type}")
        
    def classify_object(
        self,
        object_position: np.ndarray,
        true_label: int
    ) -> Dict:
        """
        Classify an object using the current algorithm.
        
        Args:
            object_position: Position of object to classify
            true_label: True label of object
            
        Returns:
            Classification result
        """
        if self.algorithm is None:
            raise ValueError("No algorithm set")
            
        # Perform classification
        result = self.algorithm.classify(
            instance_id=self.classification_count,
            object_position=object_position,
            true_label=true_label,
            current_time=self.current_time
        )
        
        self.classification_count += 1
        
        # Track performance
        self._update_performance_tracking(result)
        
        return result
    
    def update_time(self, time_delta: float) -> None:
        """
        Update network time and camera states.
        
        Args:
            time_delta: Time elapsed since last update
        """
        self.current_time += time_delta
        
        # Update camera energy
        for camera in self.cameras:
            camera.update_energy(time_delta, is_classifying=False)
            
        # Track energy state
        self._track_energy_state()
        
    def _update_performance_tracking(self, result: Dict) -> None:
        """Update performance tracking with classification result."""
        self.performance_history.append({
            'timestamp': self.current_time,
            'classification_count': self.classification_count,
            'result': result
        })
        
    def _track_energy_state(self) -> None:
        """Track current energy state of all cameras."""
        energy_state = {
            'timestamp': self.current_time,
            'camera_energies': [cam.current_energy for cam in self.cameras],
            'avg_energy': np.mean([cam.current_energy for cam in self.cameras]),
            'min_energy': np.min([cam.current_energy for cam in self.cameras]),
            'max_energy': np.max([cam.current_energy for cam in self.cameras])
        }
        
        self.energy_history.append(energy_state)
        
    def get_network_stats(self) -> Dict:
        """Get current network statistics."""
        # Camera stats
        camera_stats = {
            'total_cameras': len(self.cameras),
            'active_cameras': sum(1 for cam in self.cameras if cam.state.is_active),
            'avg_energy': np.mean([cam.current_energy for cam in self.cameras]),
            'avg_accuracy': np.mean([cam.current_accuracy for cam in self.cameras])
        }
        
        # Algorithm stats
        if self.algorithm:
            algorithm_stats = self.algorithm.get_performance_metrics()
        else:
            algorithm_stats = {}
            
        # Classification stats
        if self.performance_history:
            recent_accuracy = np.mean([
                r['result']['success'] 
                for r in self.performance_history[-100:]
            ])
        else:
            recent_accuracy = 0.0
            
        return {
            **camera_stats,
            **algorithm_stats,
            'total_classifications': self.classification_count,
            'recent_accuracy': recent_accuracy,
            'simulation_time': self.current_time
        }
    
    def reset(self) -> None:
        """Reset network to initial state."""
        for camera in self.cameras:
            camera.reset()
            
        self.current_time = 0.0
        self.classification_count = 0
        self.performance_history = []
        self.energy_history = []
        
        if self.algorithm:
            self.algorithm.classification_history = []
            self.algorithm.total_classifications = 0
            self.algorithm.successful_classifications = 0