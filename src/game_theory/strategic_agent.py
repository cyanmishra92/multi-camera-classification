"""Strategic camera agent implementation."""

import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

from ..core.camera import Camera
from .utility_functions import UtilityFunction, UtilityParameters

logger = logging.getLogger(__name__)


class StrategicAgent:
    """
    Strategic agent wrapper for camera with game-theoretic decision making.
    
    Implements best-response dynamics and Nash equilibrium strategies.
    """
    
    def __init__(
        self,
        camera: Camera,
        utility_params: UtilityParameters,
        initial_threshold: float = 0.5
    ):
        """
        Initialize strategic agent.
        
        Args:
            camera: Underlying camera object
            utility_params: Parameters for utility calculations
            initial_threshold: Initial participation threshold
        """
        self.camera = camera
        self.utility_function = UtilityFunction(utility_params)
        self.participation_threshold = initial_threshold
        
        # State tracking
        self.utility_history = []
        self.decision_history = []
        self.future_value_estimate = 0.0
        
    def decide_participation(
        self,
        current_state: Dict,
        network_state: Dict
    ) -> Tuple[bool, float]:
        """
        Decide whether to participate in classification.
        
        Args:
            current_state: Current camera state
            network_state: State of other cameras in network
            
        Returns:
            Tuple of (should_participate, expected_utility)
        """
        # Check if camera can physically participate
        if not self.camera.can_classify():
            return False, -float('inf')
            
        # Calculate marginal accuracy improvement
        marginal_accuracy = self._calculate_marginal_accuracy(network_state)
        
        # Calculate utilities for both actions
        utility_participate = self.utility_function.expected_utility_participate(
            accuracy=self.camera.current_accuracy,
            marginal_accuracy=marginal_accuracy,
            energy_cost=self.camera.energy_model.classification_cost,
            future_value=self.future_value_estimate
        )
        
        utility_not_participate = self.utility_function.expected_utility_not_participate(
            future_value=self.future_value_estimate
        )
        
        # Make decision
        should_participate = utility_participate >= utility_not_participate
        expected_utility = max(utility_participate, utility_not_participate)
        
        # Update history
        self.utility_history.append(expected_utility)
        self.decision_history.append(should_participate)
        
        return should_participate, expected_utility
    
    def _calculate_marginal_accuracy(self, network_state: Dict) -> float:
        """
        Calculate marginal accuracy improvement from participation.
        
        Args:
            network_state: State of other cameras
            
        Returns:
            Marginal accuracy improvement ΔA_i
        """
        # Get other participating cameras' accuracies
        other_accuracies = network_state.get('participating_accuracies', [])
        
        if not other_accuracies:
            # If no others participating, full contribution
            return self.camera.current_accuracy
            
        # Calculate collective accuracy without this camera
        collective_without = 1 - np.prod([1 - a for a in other_accuracies])
        
        # Calculate collective accuracy with this camera
        all_accuracies = other_accuracies + [self.camera.current_accuracy]
        collective_with = 1 - np.prod([1 - a for a in all_accuracies])
        
        # Marginal improvement
        return collective_with - collective_without
    
    def update_future_value(self, time_horizon: int = 100) -> None:
        """
        Update estimate of future utility value.
        
        Args:
            time_horizon: Time steps to consider for future value
        """
        if len(self.utility_history) < 10:
            # Not enough history, use heuristic
            self.future_value_estimate = 0.5
            return
            
        # Use exponential moving average of past utilities
        recent_utilities = self.utility_history[-20:]
        weights = np.exp(-0.1 * np.arange(len(recent_utilities)))
        weights = weights / weights.sum()
        
        avg_utility = np.average(recent_utilities, weights=weights)
        
        # Discount over time horizon
        discount = self.utility_function.params.discount_factor
        self.future_value_estimate = avg_utility * (1 - discount**time_horizon) / (1 - discount)
    
    def update_threshold(self, participation_rate: float, target_rate: float) -> None:
        """
        Update participation threshold based on network participation rate.
        
        Args:
            participation_rate: Current network participation rate
            target_rate: Target participation rate
        """
        # Simple proportional control
        error = target_rate - participation_rate
        adjustment = 0.1 * error
        
        self.participation_threshold = np.clip(
            self.participation_threshold + adjustment,
            0.1, 0.9
        )