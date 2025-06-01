"""Unit tests for camera model."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera, CameraState
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters


class TestCamera(unittest.TestCase):
    """Test cases for Camera class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create energy model
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100
        )
        self.energy_model = EnergyModel(energy_params)
        
        # Create accuracy model
        accuracy_params = AccuracyParameters(
            max_accuracy=0.95,
            min_accuracy_ratio=0.3,
            correlation_factor=0.0
        )
        self.accuracy_model = AccuracyModel(accuracy_params, self.energy_model)
        
        # Create camera
        self.camera = Camera(
            camera_id=1,
            position=np.array([10, 20, 30]),
            energy_model=self.energy_model,
            accuracy_model=self.accuracy_model,
            initial_energy=800
        )
        
    def test_initialization(self):
        """Test proper initialization of camera."""
        self.assertEqual(self.camera.camera_id, 1)
        np.testing.assert_array_equal(self.camera.position, [10, 20, 30])
        self.assertEqual(self.camera.current_energy, 800)
        self.assertTrue(self.camera.state.is_active)
        self.assertEqual(len(self.camera.energy_history), 1)
        
    def test_current_properties(self):
        """Test current energy and accuracy properties."""
        self.assertEqual(self.camera.current_energy, 800)
        # At energy 800 (e_high), accuracy should be max
        self.assertAlmostEqual(self.camera.current_accuracy, 0.95, places=3)
        
    def test_can_classify(self):
        """Test classification capability check."""
        # With 800 energy, should be able to classify
        self.assertTrue(self.camera.can_classify())
        
        # Set energy below classification cost
        self.camera.state.energy = 40
        self.assertFalse(self.camera.can_classify())
        
    def test_update_energy(self):
        """Test energy update mechanism."""
        # Test harvesting only
        initial_energy = self.camera.current_energy
        self.camera.update_energy(5.0, is_classifying=False)
        expected_energy = min(initial_energy + 50, 1000)  # 5 * 10 recharge rate
        self.assertEqual(self.camera.current_energy, expected_energy)
        
        # Test classification energy consumption
        self.camera.state.energy = 500
        self.camera.update_energy(0, is_classifying=True)
        self.assertEqual(self.camera.current_energy, 450)  # 500 - 50
        
        # Test energy doesn't go negative
        self.camera.state.energy = 30
        self.camera.update_energy(0, is_classifying=True)
        self.assertEqual(self.camera.current_energy, 0)
        
    def test_position_factor(self):
        """Test position-based accuracy factor."""
        # Object at same position
        factor = self.camera._compute_position_factor(np.array([10, 20, 30]))
        self.assertEqual(factor, 1.0)
        
        # Object far away
        factor = self.camera._compute_position_factor(np.array([200, 200, 200]))
        self.assertLess(factor, 0.2)
        self.assertGreater(factor, 0)
        
    def test_classify(self):
        """Test classification process."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        object_position = np.array([15, 25, 35])  # Close to camera
        true_label = 1
        
        # Perform classification
        predicted, is_correct = self.camera.classify(object_position, true_label)
        
        # Check that classification was performed
        self.assertEqual(self.camera.state.classification_count, 1)
        self.assertIn(predicted, [0, 1])  # Binary classification
        
        # Check energy was updated (should be called separately in real use)
        self.assertEqual(len(self.camera.accuracy_history), 1)
        
    def test_reset(self):
        """Test camera reset functionality."""
        # Perform some operations
        self.camera.update_energy(10, is_classifying=True)
        self.camera.state.classification_count = 5
        self.camera.state.correct_classifications = 3
        
        # Reset
        self.camera.reset()
        
        # Check reset state
        self.assertEqual(self.camera.current_energy, 1000)  # Full capacity
        self.assertEqual(self.camera.state.classification_count, 0)
        self.assertEqual(self.camera.state.correct_classifications, 0)
        self.assertEqual(len(self.camera.energy_history), 1)
        self.assertEqual(len(self.camera.accuracy_history), 0)
        
    def test_get_stats(self):
        """Test statistics gathering."""
        # Perform some classifications
        self.camera.state.classification_count = 10
        self.camera.state.correct_classifications = 8
        
        stats = self.camera.get_stats()
        
        self.assertEqual(stats['camera_id'], 1)
        self.assertEqual(stats['current_energy'], 800)
        self.assertAlmostEqual(stats['current_accuracy'], 0.95, places=3)
        self.assertEqual(stats['classification_count'], 10)
        self.assertEqual(stats['average_accuracy'], 0.8)
        self.assertIn('position', stats)
        self.assertIn('energy_history', stats)
        
    def test_class_assignment(self):
        """Test class assignment for round-robin scheduling."""
        self.camera.class_assignment = 2
        self.assertEqual(self.camera.class_assignment, 2)


if __name__ == "__main__":
    unittest.main()