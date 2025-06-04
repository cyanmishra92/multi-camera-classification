"""Unit tests for classification algorithms."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm


class TestFixedFrequencyAlgorithm(unittest.TestCase):
    """Tests for the FixedFrequencyAlgorithm."""

    def setUp(self):
        """Create a simple camera network for testing."""
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100,
        )
        self.energy_model = EnergyModel(energy_params)

        accuracy_params = AccuracyParameters(
            max_accuracy=0.95,
            min_accuracy_ratio=0.3,
            correlation_factor=0.0,
        )
        self.accuracy_model = AccuracyModel(accuracy_params, self.energy_model)

        # Create a small set of cameras with ample energy
        self.cameras = []
        for i in range(3):
            cam = Camera(
                camera_id=i,
                position=np.array([0, 0, 0]),
                energy_model=self.energy_model,
                accuracy_model=self.accuracy_model,
                initial_energy=600,
            )
            cam.class_assignment = i % 2  # Two classes
            self.cameras.append(cam)

    def test_select_cameras_with_sufficient_energy(self):
        """Algorithm should select at least one camera when energy is sufficient."""
        algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=2,
            min_accuracy_threshold=0.5,
        )
        selected = algorithm.select_cameras(0, 0.0)
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)

    def test_select_cameras_insufficient_energy(self):
        """No cameras should be selected if all lack energy."""
        for cam in self.cameras:
            cam.state.energy = 10  # Below classification cost

        algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=2,
            min_accuracy_threshold=0.5,
        )
        selected = algorithm.select_cameras(0, 0.0)
        self.assertEqual(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
