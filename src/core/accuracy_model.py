# src/core/accuracy_model.py
"""Energy-dependent accuracy modeling."""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AccuracyParameters:
    """Parameters for accuracy model."""
    max_accuracy: float  # α_max
    min_accuracy_ratio: float  # β
    correlation_factor: float = 0.0  # ρ for correlated errors


class AccuracyModel:
    """
    Energy-dependent accuracy model for cameras.

    Implements the piecewise linear accuracy function:
    - Full accuracy above E_high
    - Linear degradation between E_low and E_high
    - Minimum accuracy below E_low
    """

    def __init__(self, params: AccuracyParameters, energy_model):
        """
        Initialize accuracy model.

        Args:
            params: Accuracy model parameters
            energy_model: Associated energy model for thresholds
        """
        self.params = params
        self.max_accuracy = params.max_accuracy
        self.min_accuracy_ratio = params.min_accuracy_ratio
        self.correlation_factor = params.correlation_factor

        # Get energy thresholds from energy model
        self.e_high = energy_model.e_high
        self.e_low = energy_model.e_low
        self.min_operational = energy_model.min_operational

    def get_accuracy(self, energy: float) -> float:
        """
        Calculate accuracy based on energy level.

        Args:
            energy: Current energy level

        Returns:
            Accuracy value α(E)
        """
        if energy < self.min_operational:
            return 0.0

        f_e = self._energy_accuracy_function(energy)
        return self.max_accuracy * f_e

    def _energy_accuracy_function(self, energy: float) -> float:
        """
        Calculate energy-accuracy function f(E).

        Args:
            energy: Current energy level

        Returns:
            Energy accuracy factor in [0, 1]
        """
        if energy >= self.e_high:
            return 1.0
        elif energy > self.e_low:
            # Linear interpolation
            return (energy - self.e_low) / (self.e_high - self.e_low)
        else:
            return self.min_accuracy_ratio

    def get_collective_accuracy(self, energies: np.ndarray) -> float:
        """
        Calculate effective accuracy for a group of cameras.

        Args:
            energies: Array of energy levels for participating cameras

        Returns:
            Collective accuracy α_eff
        """
        if len(energies) == 0:
            return 0.0

        # Get individual accuracies
        accuracies = np.array([self.get_accuracy(e) for e in energies])

        if self.correlation_factor == 0:
            # Independent errors
            return 1 - np.prod(1 - accuracies)
        else:
            # Correlated errors
            avg_accuracy = np.mean(accuracies)
            n = len(accuracies)
            exponent = n * (1 - self.correlation_factor)
            return 1 - (1 - avg_accuracy) ** exponent

    def min_cameras_for_threshold(self, energies: np.ndarray, threshold: float) -> int:
        """
        Find minimum number of cameras needed to achieve accuracy threshold.

        Args:
            energies: Array of available camera energies (sorted descending)
            threshold: Minimum required collective accuracy

        Returns:
            Minimum number of cameras needed
        """
        for n in range(1, len(energies) + 1):
            if self.get_collective_accuracy(energies[:n]) >= threshold:
                return n
        return len(energies)  # Need all cameras
