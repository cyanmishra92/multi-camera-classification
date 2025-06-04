"""Algorithm 3: Unknown frequency classification."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .base_algorithm import BaseClassificationAlgorithm
from ..core.camera import Camera
from ..game_theory.strategic_agent import StrategicAgent
from ..game_theory.utility_functions import UtilityParameters

logger = logging.getLogger(__name__)


class UnknownFrequencyAlgorithm(BaseClassificationAlgorithm):
    """
    Algorithm 3: Unknown frequency classification with probabilistic participation.
    
    Uses game theory and adaptive thresholds for unknown classification frequency.
    """
    
    def __init__(
        self,
        cameras: List[Camera],
        num_classes: int,
        utility_params: UtilityParameters,
        min_accuracy_threshold: float = 0.8,
        history_length: int = 10,
        initial_threshold: float = 0.5
    ):
        """
        Initialize unknown frequency algorithm.
        
        Args:
            cameras: List of camera objects
            num_classes: Number of camera classes (n)
            utility_params: Parameters for utility calculations
            min_accuracy_threshold: Minimum required collective accuracy
            history_length: Length of classification history (k)
            initial_threshold: Initial participation threshold
        """
        super().__init__(cameras, min_accuracy_threshold, history_length)
        
        self.num_classes = num_classes
        self.utility_params = utility_params
        
        # Create strategic agents
        self.agents = [
            StrategicAgent(camera, utility_params, initial_threshold)
            for camera in cameras
        ]
        
        # Assign cameras to classes
        self._assign_camera_classes()
        
        # Initialize participation thresholds
        self._initialize_thresholds()
        
        # Track participation rates
        self.participation_rates = []
        
    def _assign_camera_classes(self) -> None:
        """Assign cameras to classes and build mapping."""
        # Assign classes if none are set
        if all(cam.class_assignment is None for cam in self.cameras):
            for i, camera in enumerate(self.cameras):
                camera.class_assignment = i % self.num_classes

        # Create class mapping
        self.camera_classes = {}
        for class_id in range(self.num_classes):
            self.camera_classes[class_id] = [
                i for i, cam in enumerate(self.cameras)
                if cam.class_assignment == class_id
            ]
            
    def _initialize_thresholds(self) -> None:
        """Initialize participation thresholds based on energy-weighted intervals."""
        for class_id in range(self.num_classes):
            class_agents = [self.agents[i] for i in self.camera_classes[class_id]]
            
            # Calculate energy-weighted thresholds
            total_weighted_accuracy = sum(
                agent.camera.current_accuracy * agent.camera.current_energy
                for agent in class_agents
            )
            
            for agent in class_agents:
                weight = (
                    agent.camera.current_accuracy * agent.camera.current_energy /
                    total_weighted_accuracy
                )
                agent.participation_threshold = weight
                
    def select_cameras(self, instance_id: int, current_time: float) -> List[int]:
        """
        Select cameras using probabilistic game-theoretic approach.
        
        Args:
            instance_id: ID of classification instance
            current_time: Current simulation time
            
        Returns:
            List of selected camera IDs
        """
        # Determine active class
        active_class = instance_id % self.num_classes
        class_camera_ids = self.camera_classes[active_class]
        
        # Generate random number for this instance
        random_value = np.random.random()
        
        # Get recent participants to avoid
        recent_participants = set(self.get_recent_participants())
        
        selected = []
        participating_accuracies = []
        
        # Each camera makes strategic decision
        for cam_id in class_camera_ids:
            agent = self.agents[cam_id]
            
            # Skip if recently participated
            if cam_id in recent_participants:
                continue
                
            # Check if camera can participate with the classification cost
            # This prevents selecting cameras that will fail later
            if agent.camera.state.energy < agent.camera.energy_model.classification_cost:
                continue
                
            # Update future value estimate
            agent.update_future_value()
            
            # Make participation decision
            network_state = {
                'participating_accuracies': participating_accuracies,
                'random_value': random_value
            }
            
            should_participate, expected_utility = agent.decide_participation(
                current_state={'energy': agent.camera.current_energy},
                network_state=network_state
            )
            
            # Also check probabilistic threshold
            if should_participate and random_value <= agent.participation_threshold:
                selected.append(cam_id)
                participating_accuracies.append(agent.camera.current_accuracy)
                
        # If insufficient accuracy, redistribute thresholds
        if not self._check_accuracy_threshold(selected):
            selected = self._redistribute_and_select(
                class_camera_ids, 
                recent_participants,
                random_value
            )
            
        # Track participation rate
        participation_rate = len(selected) / len(class_camera_ids) if class_camera_ids else 0
        self.participation_rates.append(participation_rate)
        
        return selected
    
    def _check_accuracy_threshold(self, camera_ids: List[int]) -> bool:
        """Check if selected cameras meet accuracy threshold."""
        if not camera_ids:
            return False
            
        selected_cameras = [self.cameras[i] for i in camera_ids]
        collective_accuracy = self._calculate_collective_accuracy(selected_cameras)
        
        return collective_accuracy >= self.min_accuracy_threshold
    
    def _redistribute_and_select(
        self,
        class_camera_ids: List[int],
        recent_participants: set,
        random_value: float
    ) -> List[int]:
        """
        Redistribute thresholds and reselect if accuracy threshold not met.
        
        Args:
            class_camera_ids: Camera IDs in active class
            recent_participants: Set of recent participant IDs
            random_value: Random value for this instance
            
        Returns:
            Updated selection
        """
        # Get available cameras with sufficient energy
        available_cameras = [
            cam_id for cam_id in class_camera_ids
            if cam_id not in recent_participants and 
            self.cameras[cam_id].state.energy >= self.cameras[cam_id].energy_model.classification_cost
        ]
        
        if not available_cameras:
            return []
            
        # Sort by energy-accuracy product
        available_cameras.sort(
            key=lambda i: (
                self.cameras[i].current_energy * 
                self.cameras[i].current_accuracy
            ),
            reverse=True
        )
        
        # Select greedily until threshold met
        selected = []
        for cam_id in available_cameras:
            selected.append(cam_id)
            
            if self._check_accuracy_threshold(selected):
                break
                
        # Update thresholds for failed cameras
        for cam_id in class_camera_ids:
            if cam_id in selected:
                # Increase threshold for selected cameras
                self.agents[cam_id].participation_threshold *= 1.1
            else:
                # Decrease threshold for non-selected cameras
                self.agents[cam_id].participation_threshold *= 0.9
                
            # Keep thresholds in valid range
            self.agents[cam_id].participation_threshold = np.clip(
                self.agents[cam_id].participation_threshold,
                0.1, 0.9
            )
            
        return selected
    
    def update_thresholds_adaptive(self) -> None:
        """Update participation thresholds based on recent performance."""
        if len(self.participation_rates) < 10:
            return
            
        # Calculate recent average participation rate
        recent_rate = np.mean(self.participation_rates[-10:])
        
        # Target participation rate based on energy availability
        avg_energy_ratio = np.mean([
            cam.current_energy / cam.energy_model.capacity
            for cam in self.cameras
        ])
        
        target_rate = 0.3 + 0.4 * avg_energy_ratio  # Adaptive target
        
        # Update all agent thresholds
        for agent in self.agents:
            agent.update_threshold(recent_rate, target_rate)
            
    def adapt_to_frequency(self, observed_frequency: float) -> None:
        """
        Adapt algorithm parameters based on observed frequency.
        
        Args:
            observed_frequency: Observed classification frequency
        """
        # Calculate expected interval between classifications
        expected_interval = 1.0 / observed_frequency if observed_frequency > 0 else float('inf')
        
        # Adjust history length
        recharge_time = self.cameras[0].energy_model.min_recharge_time
        self.history_length = max(
            10,
            int(recharge_time / expected_interval)
        )
        
        # Adjust utility parameters based on frequency
        if observed_frequency > 1.0 / recharge_time:
            # High frequency - be more conservative
            self.utility_params.non_participation_penalty *= 0.9
        else:
            # Low frequency - encourage participation
            self.utility_params.non_participation_penalty *= 1.1
            
        # Update agents with new parameters
        for agent in self.agents:
            agent.utility_function.params = self.utility_params
            
    def get_algorithm_stats(self) -> Dict:
        """Get algorithm-specific statistics."""
        base_stats = self.get_performance_metrics()
        
        # Add unknown frequency specific stats
        threshold_stats = {
            f'avg_threshold_class_{i}': np.mean([
                self.agents[j].participation_threshold
                for j in self.camera_classes[i]
            ])
            for i in range(self.num_classes)
        }
        
        participation_stats = {
            'avg_participation_rate': np.mean(self.participation_rates) if self.participation_rates else 0,
            'std_participation_rate': np.std(self.participation_rates) if self.participation_rates else 0,
        }
        
        return {
            **base_stats,
            **threshold_stats,
            **participation_stats
        }