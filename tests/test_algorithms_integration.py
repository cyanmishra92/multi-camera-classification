"""Integration tests for classification algorithms."""

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
from src.algorithms.variable_frequency import VariableFrequencyAlgorithm
from src.algorithms.unknown_frequency import UnknownFrequencyAlgorithm
from src.game_theory.utility_functions import UtilityParameters


class TestAlgorithmsIntegration(unittest.TestCase):
    """Integration tests for classification algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create energy and accuracy models
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100
        )
        self.energy_model = EnergyModel(energy_params)
        
        accuracy_params = AccuracyParameters(
            max_accuracy=0.95,
            min_accuracy_ratio=0.3,
            correlation_factor=0.0
        )
        self.accuracy_model = AccuracyModel(accuracy_params, self.energy_model)
        
        # Create cameras
        self.num_cameras = 9
        self.cameras = []
        for i in range(self.num_cameras):
            position = np.array([i * 10, i * 10, 30])
            camera = Camera(
                camera_id=i,
                position=position,
                energy_model=self.energy_model,
                accuracy_model=self.accuracy_model,
                initial_energy=800 + i * 20  # Vary initial energy
            )
            self.cameras.append(camera)
            
    def test_fixed_frequency_algorithm(self):
        """Test fixed frequency algorithm."""
        algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=3,
            min_accuracy_threshold=0.8
        )
        
        # Test camera selection
        selected = algorithm.select_cameras(0, 0.0)
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # Test classification
        result = algorithm.classify(
            instance_id=0,
            object_position=np.array([50, 50, 0]),
            true_label=1,
            current_time=0.0
        )
        
        self.assertIn('success', result)
        self.assertIn('selected_cameras', result)
        self.assertIn('collective_accuracy', result)
        
        # Test round-robin behavior
        selected_class_0 = algorithm.select_cameras(0, 0.0)
        selected_class_1 = algorithm.select_cameras(1, 0.0)
        selected_class_2 = algorithm.select_cameras(2, 0.0)
        
        # Different classes should select different cameras
        self.assertNotEqual(set(selected_class_0), set(selected_class_1))
        
    def test_variable_frequency_algorithm(self):
        """Test variable frequency algorithm."""
        algorithm = VariableFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=3,
            classification_frequency=0.2,  # Higher frequency to create multiple subclasses
            recharge_time=10,
            min_accuracy_threshold=0.8
        )
        
        # Test subclass creation
        self.assertGreater(algorithm.subclasses_per_class, 0)
        
        # Test camera selection
        selected = algorithm.select_cameras(0, 0.0)
        self.assertIsInstance(selected, list)
        
        # Test classification
        result = algorithm.classify(
            instance_id=0,
            object_position=np.array([50, 50, 0]),
            true_label=0,
            current_time=0.0
        )
        
        self.assertIn('success', result)
        
        # Test subclass rotation
        initial_turn = algorithm.class_turns[0]
        algorithm.select_cameras(3, 0.0)  # Same class, next turn
        self.assertNotEqual(algorithm.class_turns[0], initial_turn)
        
    def test_unknown_frequency_algorithm(self):
        """Test unknown frequency algorithm."""
        utility_params = UtilityParameters(
            reward_scale=1.0,
            incorrect_penalty=0.5,
            non_participation_penalty=0.8,
            discount_factor=0.9
        )
        
        algorithm = UnknownFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=3,
            utility_params=utility_params,
            min_accuracy_threshold=0.8
        )
        
        # Test agent creation
        self.assertEqual(len(algorithm.agents), self.num_cameras)
        
        # Test camera selection with game theory
        selected = algorithm.select_cameras(0, 0.0)
        self.assertIsInstance(selected, list)
        
        # Test classification
        result = algorithm.classify(
            instance_id=0,
            object_position=np.array([50, 50, 0]),
            true_label=1,
            current_time=0.0
        )
        
        self.assertIn('success', result)
        
        # Test adaptive threshold updates
        initial_thresholds = [agent.participation_threshold for agent in algorithm.agents]
        algorithm.update_thresholds_adaptive()
        # Thresholds might not change if not enough history
        
    def test_algorithm_performance_metrics(self):
        """Test performance metric tracking."""
        algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=3,
            min_accuracy_threshold=0.8
        )
        
        # Perform multiple classifications
        for i in range(10):
            algorithm.classify(
                instance_id=i,
                object_position=np.array([50, 50, 0]),
                true_label=i % 2,
                current_time=i * 10.0
            )
            
        # Get metrics
        metrics = algorithm.get_performance_metrics()
        
        self.assertIn('total_classifications', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('energy_violations', metrics)
        self.assertIn('accuracy_violations', metrics)
        
        self.assertEqual(metrics['total_classifications'], 10)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        
    def test_energy_depletion_handling(self):
        """Test handling of energy depletion."""
        # Deplete camera energy
        for camera in self.cameras[:3]:
            camera.state.energy = 30  # Below classification cost
            
        algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=3,
            min_accuracy_threshold=0.8
        )
        
        # Try to select from class with depleted cameras
        selected = algorithm.select_cameras(0, 0.0)
        
        # Should only select cameras with sufficient energy
        for cam_id in selected:
            self.assertGreaterEqual(self.cameras[cam_id].current_energy, 50)
            
    def test_accuracy_threshold_enforcement(self):
        """Test that algorithms enforce accuracy thresholds."""
        # Create cameras with varying accuracy
        low_accuracy_cameras = []
        for i in range(6):
            camera = Camera(
                camera_id=i,
                position=np.array([i * 10, i * 10, 30]),
                energy_model=self.energy_model,
                accuracy_model=self.accuracy_model,
                initial_energy=200  # Low energy = low accuracy
            )
            low_accuracy_cameras.append(camera)
            
        algorithm = FixedFrequencyAlgorithm(
            cameras=low_accuracy_cameras,
            num_classes=2,
            min_accuracy_threshold=0.9  # High threshold
        )
        
        # Algorithm should select multiple cameras to meet threshold
        selected = algorithm.select_cameras(0, 0.0)
        self.assertGreater(len(selected), 1)  # Need multiple low-accuracy cameras


if __name__ == "__main__":
    unittest.main()