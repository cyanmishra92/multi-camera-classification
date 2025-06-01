"""Unit tests for accuracy model."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters


class TestAccuracyModel(unittest.TestCase):
    """Test cases for AccuracyModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create energy model first
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100,
            high_threshold=0.8,
            low_threshold=0.3
        )
        self.energy_model = EnergyModel(energy_params)
        
        # Create accuracy model
        self.accuracy_params = AccuracyParameters(
            max_accuracy=0.95,
            min_accuracy_ratio=0.3,
            correlation_factor=0.2
        )
        self.model = AccuracyModel(self.accuracy_params, self.energy_model)
        
    def test_initialization(self):
        """Test proper initialization of accuracy model."""
        self.assertEqual(self.model.max_accuracy, 0.95)
        self.assertEqual(self.model.min_accuracy_ratio, 0.3)
        self.assertEqual(self.model.correlation_factor, 0.2)
        self.assertEqual(self.model.e_high, 800)
        self.assertEqual(self.model.e_low, 300)
        
    def test_get_accuracy(self):
        """Test accuracy calculation based on energy."""
        # Test at high energy
        accuracy = self.model.get_accuracy(900)
        self.assertAlmostEqual(accuracy, 0.95, places=3)
        
        # Test at exact high threshold
        accuracy = self.model.get_accuracy(800)
        self.assertAlmostEqual(accuracy, 0.95, places=3)
        
        # Test in linear region
        accuracy = self.model.get_accuracy(550)  # Midpoint
        # Linear interpolation: f(E) = (550-300)/(800-300) = 0.5
        expected = 0.95 * 0.5
        self.assertAlmostEqual(accuracy, expected, places=3)
        
        # Test at low energy
        accuracy = self.model.get_accuracy(200)
        expected = 0.95 * 0.3  # min_accuracy_ratio
        self.assertAlmostEqual(accuracy, expected, places=3)
        
        # Test below operational threshold
        accuracy = self.model.get_accuracy(50)
        self.assertEqual(accuracy, 0.0)
        
    def test_energy_accuracy_function(self):
        """Test the internal energy-accuracy function."""
        # Test at various energy levels
        self.assertEqual(self.model._energy_accuracy_function(900), 1.0)
        self.assertEqual(self.model._energy_accuracy_function(300), 0.3)
        self.assertAlmostEqual(
            self.model._energy_accuracy_function(550),
            0.5,
            places=2
        )
        
    def test_collective_accuracy_independent(self):
        """Test collective accuracy with independent errors."""
        # Set correlation factor to 0 for independent errors
        self.model.correlation_factor = 0
        
        # Test with single camera
        energies = np.array([800])
        accuracy = self.model.get_collective_accuracy(energies)
        self.assertAlmostEqual(accuracy, 0.95, places=3)
        
        # Test with two cameras at full energy
        energies = np.array([800, 800])
        accuracy = self.model.get_collective_accuracy(energies)
        expected = 1 - (1 - 0.95) ** 2  # 0.9975
        self.assertAlmostEqual(accuracy, expected, places=3)
        
        # Test with empty array
        energies = np.array([])
        accuracy = self.model.get_collective_accuracy(energies)
        self.assertEqual(accuracy, 0.0)
        
    def test_collective_accuracy_correlated(self):
        """Test collective accuracy with correlated errors."""
        # Use default correlation factor of 0.2
        
        # Test with two cameras
        energies = np.array([800, 800])
        accuracy = self.model.get_collective_accuracy(energies)
        # With correlation, accuracy should be lower than independent case
        independent_accuracy = 1 - (1 - 0.95) ** 2
        self.assertLess(accuracy, independent_accuracy)
        
    def test_min_cameras_for_threshold(self):
        """Test minimum cameras calculation."""
        # All cameras at high energy
        energies = np.array([900, 900, 900, 900, 900])
        
        # Should need only 1 camera for low threshold
        min_cams = self.model.min_cameras_for_threshold(energies, 0.9)
        self.assertEqual(min_cams, 1)
        
        # Should need more cameras for very high threshold
        min_cams = self.model.min_cameras_for_threshold(energies, 0.999)
        self.assertGreater(min_cams, 1)
        
        # Test with mixed energy levels
        energies = np.array([900, 600, 400, 200])
        min_cams = self.model.min_cameras_for_threshold(energies, 0.8)
        self.assertGreaterEqual(min_cams, 1)
        self.assertLessEqual(min_cams, 4)


if __name__ == "__main__":
    unittest.main()