import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.enhanced_accuracy_model import EnhancedAccuracyModel, EnhancedAccuracyParameters
from src.algorithms.enhanced_fixed_frequency import EnhancedFixedFrequencyAlgorithm


class TestEnhancedFixedFrequencyPrecompute(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100,
        )
        energy_model = EnergyModel(energy_params)
        acc_params = EnhancedAccuracyParameters(
            max_accuracy=0.9,
            min_accuracy_ratio=0.3,
            correlation_factor=0.0,
        )
        accuracy_model = EnhancedAccuracyModel(acc_params, energy_model)

        self.cameras = []
        positions = [
            np.array([0, 0, 0]),
            np.array([10, 0, 0]),
            np.array([0, 10, 0]),
            np.array([10, 10, 0]),
        ]
        for i, pos in enumerate(positions):
            cam = Camera(
                camera_id=i,
                position=pos,
                energy_model=energy_model,
                accuracy_model=accuracy_model,
                initial_energy=800,
            )
            cam.class_assignment = i % 2
            self.cameras.append(cam)

        self.algorithm = EnhancedFixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=2,
            min_accuracy_threshold=0.5,
            use_game_theory=False,
            position_weight=0.7,
        )

    def baseline_select(self, candidate_cameras, object_position):
        camera_scores = []
        for cid in candidate_cameras:
            cam = self.cameras[cid]
            if not cam.can_classify():
                continue
            pos_acc = cam.accuracy_model.get_position_based_accuracy(cam, object_position)
            energy_factor = cam.current_energy / cam.energy_model.capacity
            score = 0.7 * pos_acc + 0.3 * energy_factor
            camera_scores.append((cid, score))
        camera_scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for cid, _ in camera_scores:
            selected.append(cid)
            cams = [self.cameras[i] for i in selected]
            acc = self.algorithm._calculate_enhanced_collective_accuracy(cams, object_position)
            if acc >= self.algorithm.min_accuracy_threshold:
                break
        return selected

    def test_precomputed_distance_consistency(self):
        obj_pos = np.array([5, 5, 0])
        class_id = 0
        candidates = self.algorithm.camera_classes[class_id]

        expected = self.baseline_select(candidates, obj_pos)
        # Use instance_id that maps to class_id 0
        instance_id = class_id  # Since instance_id % num_classes gives class_id
        result = self.algorithm.select_cameras(instance_id, 0.0, obj_pos)
        self.assertEqual(expected, result)

        selected_cams = [self.cameras[i] for i in result]
        distances = np.linalg.norm([cam.position for cam in selected_cams] - obj_pos, axis=1)
        acc1 = self.algorithm._calculate_enhanced_collective_accuracy(selected_cams, obj_pos, distances)
        acc2 = self.algorithm._calculate_enhanced_collective_accuracy(selected_cams, obj_pos)
        self.assertAlmostEqual(acc1, acc2, places=6)


if __name__ == "__main__":
    unittest.main()
