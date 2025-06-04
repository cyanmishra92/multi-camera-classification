# src/core/energy_model.py
"""Energy harvesting and consumption models."""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnergyParameters:
    """Parameters for energy model."""
    capacity: float  # Battery capacity (cap)
    recharge_rate: float  # Energy harvest rate (r)
    classification_cost: float  # Energy per classification (δ)
    min_operational: float  # Minimum operational energy (δ_min)
    high_threshold: float = 0.8  # E_high as fraction of capacity
    low_threshold: float = 0.3  # E_low as fraction of capacity


class EnergyModel:
    """
    Energy harvesting and consumption model for cameras.

    Models battery dynamics including:
    - Constant rate energy harvesting
    - Fixed cost per classification
    - Energy constraints
    """

    def __init__(self, params: EnergyParameters):
        """
        Initialize energy model.

        Args:
            params: Energy model parameters
        """
        self.params = params
        self.capacity = params.capacity
        self.recharge_rate = params.recharge_rate
        self.classification_cost = params.classification_cost
        self.min_operational = params.min_operational

        # Compute derived parameters
        if self.recharge_rate > 0:
            self.full_recharge_time = self.capacity / self.recharge_rate
            self.min_recharge_time = self.min_operational / self.recharge_rate
        else:
            logger.warning("Recharge rate is non-positive; recharge times set to infinity")
            self.full_recharge_time = float('inf')
            self.min_recharge_time = float('inf')

        # Energy thresholds
        self.e_high = params.high_threshold * self.capacity
        self.e_low = params.low_threshold * self.capacity

    def harvest(self, time_delta: float) -> float:
        """
        Calculate energy harvested over time period.

        Args:
            time_delta: Time period for harvesting

        Returns:
            Amount of energy harvested
        """
        return self.recharge_rate * time_delta

    def time_to_recharge(self, current_energy: float, target_energy: Optional[float] = None) -> float:
        """
        Calculate time needed to recharge to target level.

        Args:
            current_energy: Current energy level
            target_energy: Target energy level (defaults to full capacity)

        Returns:
            Time needed to reach target energy
        """
        if target_energy is None:
            target_energy = self.capacity

        if current_energy >= target_energy:
            return 0.0

        if self.recharge_rate <= 0:
            logger.warning("Recharge rate is non-positive; time to recharge is infinite")
            return float('inf')

        return (target_energy - current_energy) / self.recharge_rate

    def get_snr(self, energy: float) -> float:
        """
        Get Signal-to-Noise Ratio based on energy level.

        Args:
            energy: Current energy level

        Returns:
            SNR value
        """
        if energy <= 0:
            return 0.0
        return np.sqrt(energy / self.e_high)

    def get_processing_cost_factor(self, energy: float) -> float:
        """
        Get computational cost factor based on energy.

        Low energy may require using lower-fidelity models.

        Args:
            energy: Current energy level

        Returns:
            Cost factor in [0.3, 1.0]
        """
        if energy >= self.e_high:
            return 1.0
        elif energy >= self.e_low:
            return 0.7 + 0.3 * (energy - self.e_low) / (self.e_high - self.e_low)
        else:
            return 0.3

