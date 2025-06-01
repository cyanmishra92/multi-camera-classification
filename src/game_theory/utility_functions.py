"""Utility function definitions and calculations."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UtilityParameters:
    """Parameters for utility calculations."""
    reward_scale: float = 1.0  # γ
    incorrect_penalty: float = 0.5  # δ
    non_participation_penalty: float = 0.8  # η
    discount_factor: float = 0.9  # β
    

class UtilityFunction:
    """
    Utility function for camera decision making.
    
    Implements the reward and cost structure for strategic participation.
    """
    
    def __init__(self, params: UtilityParameters):
        """Initialize utility function with parameters."""
        self.params = params
        
    def calculate_reward(
        self, 
        participated: bool,
        was_correct: bool,
        marginal_accuracy: float
    ) -> float:
        """
        Calculate reward for a camera's action.
        
        Args:
            participated: Whether camera participated
            was_correct: Whether classification was correct (if participated)
            marginal_accuracy: Marginal accuracy improvement from participation
            
        Returns:
            Reward value
        """
        if not participated:
            return -self.params.non_participation_penalty
            
        if was_correct:
            return self.params.reward_scale * marginal_accuracy
        else:
            return -self.params.incorrect_penalty
    
    def calculate_cost(
        self,
        energy_cost: float,
        future_value: float,
        max_energy: float = 1000.0
    ) -> float:
        """
        Calculate cost including future opportunity cost.
        
        Args:
            energy_cost: Energy consumed by action
            future_value: Expected future utility
            max_energy: Maximum energy capacity for normalization
            
        Returns:
            Total cost
        """
        # Normalize energy cost to [0, 1] range
        normalized_cost = energy_cost / max_energy
        return normalized_cost + self.params.discount_factor * future_value
    
    def calculate_utility(
        self,
        reward: float,
        cost: float
    ) -> float:
        """
        Calculate net utility.
        
        Args:
            reward: Reward from action
            cost: Cost of action
            
        Returns:
            Net utility
        """
        return reward - cost
    
    def expected_utility_participate(
        self,
        accuracy: float,
        marginal_accuracy: float,
        energy_cost: float,
        future_value: float
    ) -> float:
        """
        Calculate expected utility from participation.
        
        Args:
            accuracy: Camera's current accuracy
            marginal_accuracy: Marginal accuracy improvement
            energy_cost: Energy cost of classification
            future_value: Expected future utility
            
        Returns:
            Expected utility from participating
        """
        # Expected reward = P(correct) * reward - P(incorrect) * penalty
        expected_reward = (
            accuracy * self.params.reward_scale * marginal_accuracy -
            (1 - accuracy) * self.params.incorrect_penalty
        )
        
        # Normalize energy cost by typical battery capacity
        cost = self.calculate_cost(energy_cost, future_value, max_energy=1000.0)
        return expected_reward - cost
    
    def expected_utility_not_participate(
        self,
        future_value: float
    ) -> float:
        """
        Calculate expected utility from not participating.
        
        Args:
            future_value: Expected future utility
            
        Returns:
            Expected utility from not participating
        """
        reward = -self.params.non_participation_penalty
        cost = self.params.discount_factor * future_value
        return reward - cost