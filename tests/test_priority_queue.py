import unittest
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters
from src.algorithms.fixed_frequency import FixedFrequencyAlgorithm


class TestPriorityQueuePerformance(unittest.TestCase):
    def setUp(self):
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
            correlation_factor=0.0,
        )
        self.accuracy_model = AccuracyModel(accuracy_params, self.energy_model)

        self.cameras = []
        for i in range(50):
            cam = Camera(
                camera_id=i,
                position=np.array([0, 0, 0]),
                energy_model=self.energy_model,
                accuracy_model=self.accuracy_model,
                initial_energy=500 + i * 5,
            )
            cam.class_assignment = i % 5
            self.cameras.append(cam)

        self.algorithm = FixedFrequencyAlgorithm(
            cameras=self.cameras,
            num_classes=5,
            min_accuracy_threshold=0.5,
        )

    def baseline_select(self, candidate_cameras):
        sorted_cams = sorted(
            candidate_cameras,
            key=lambda i: self.algorithm.cameras[i].current_energy,
            reverse=True,
        )
        selected = []
        for cam_id in sorted_cams:
            cam = self.algorithm.cameras[cam_id]
            if not cam.can_classify():
                continue
            selected.append(cam_id)
            selected_cameras = [self.algorithm.cameras[i] for i in selected]
            if (
                self.algorithm._calculate_collective_accuracy(selected_cameras)
                >= self.algorithm.min_accuracy_threshold
            ):
                break
        return selected

    def test_heap_selection_faster(self):
        class_id = 0
        candidate = self.algorithm.camera_classes[class_id]
        iterations = 200

        start = time.perf_counter()
        for _ in range(iterations):
            self.baseline_select(candidate)
        baseline_time = time.perf_counter() - start

        heap_time = 0.0
        for _ in range(iterations):
            heap_copy = list(self.algorithm.energy_heaps[class_id])
            self.algorithm.energy_heaps[class_id] = heap_copy
            start = time.perf_counter()
            self.algorithm._select_greedy(class_id)
            heap_time += time.perf_counter() - start
        self.algorithm.energy_heaps[class_id] = heap_copy

        self.assertLess(heap_time, baseline_time)


if __name__ == "__main__":
    unittest.main()
