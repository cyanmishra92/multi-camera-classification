import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import Camera
from src.core.energy_model import EnergyModel, EnergyParameters
from src.core.accuracy_model import AccuracyModel, AccuracyParameters
from src.game_theory.nash_equilibrium import NashEquilibriumSolver
from src.game_theory.strategic_agent import StrategicAgent
from src.game_theory.utility_functions import UtilityParameters


class TestNashEquilibriumOptimalActions(unittest.TestCase):
    def setUp(self):
        energy_params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100,
        )
        self.energy_model = EnergyModel(energy_params)

        accuracy_params = AccuracyParameters(
            max_accuracy=0.9,
            min_accuracy_ratio=0.3,
            correlation_factor=0.0,
        )
        self.accuracy_model = AccuracyModel(accuracy_params, self.energy_model)

        util_params = UtilityParameters(
            reward_scale=1.0,
            incorrect_penalty=0.5,
            non_participation_penalty=0.8,
            discount_factor=0.9,
        )

        self.agents = []
        for i in range(4):
            cam = Camera(
                camera_id=i,
                position=np.array([0, 0, 30]),
                energy_model=self.energy_model,
                accuracy_model=self.accuracy_model,
                initial_energy=800,
            )
            self.agents.append(StrategicAgent(cam, util_params))

        self.solver = NashEquilibriumSolver()

    def _brute_force_optimal(self):
        n = len(self.agents)
        best_welfare = -float("inf")
        best_actions = [False] * n
        for mask in range(1 << n):
            actions = [(mask >> j) & 1 == 1 for j in range(n)]
            valid = all(
                not actions[j] or self.agents[j].camera.can_classify() for j in range(n)
            )
            if not valid:
                continue
            welfare = self.solver.compute_social_welfare(self.agents, actions)
            if welfare > best_welfare:
                best_welfare = welfare
                best_actions = actions
        return best_actions

    def test_optimal_actions_match_brute_force(self):
        dp_actions = self.solver._find_optimal_actions(self.agents)
        brute_actions = self._brute_force_optimal()
        self.assertEqual(dp_actions, brute_actions)

    def test_with_energy_constraints(self):
        # Make first agent unable to participate
        self.agents[0].camera.state.energy = 20
        dp_actions = self.solver._find_optimal_actions(self.agents)
        brute_actions = self._brute_force_optimal()
        self.assertEqual(dp_actions, brute_actions)


if __name__ == "__main__":
    unittest.main()
