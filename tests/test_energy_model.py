"""Unit tests for energy model."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.energy_model import EnergyModel, EnergyParameters


class TestEnergyModel(unittest.TestCase):
    """Test cases for EnergyModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = EnergyParameters(
            capacity=1000,
            recharge_rate=10,
            classification_cost=50,
            min_operational=100,
            high_threshold=0.8,
            low_threshold=0.3
        )
        self.model = EnergyModel(self.params)
        
    def test_initialization(self):
        """Test proper initialization of energy model."""
        self.assertEqual(self.model.capacity, 1000)
        self.assertEqual(self.model.recharge_rate, 10)
        self.assertEqual(self.model.classification_cost, 50)
        self.assertEqual(self.model.min_operational, 100)
        self.assertEqual(self.model.e_high, 800)  # 0.8 * 1000
        self.assertEqual(self.model.e_low, 300)   # 0.3 * 1000
        
    def test_harvest(self):
        """Test energy harvesting calculation."""
        # Test harvesting for 10 time units
        harvested = self.model.harvest(10)
        self.assertEqual(harvested, 100)  # 10 * 10
        
        # Test harvesting for 0 time units
        harvested = self.model.harvest(0)
        self.assertEqual(harvested, 0)
        
    def test_time_to_recharge(self):
        """Test recharge time calculation."""
        # Test recharge from empty to full
        time = self.model.time_to_recharge(0, 1000)
        self.assertEqual(time, 100)  # 1000 / 10
        
        # Test recharge from half to full
        time = self.model.time_to_recharge(500, 1000)
        self.assertEqual(time, 50)  # 500 / 10
        
        # Test when already at target
        time = self.model.time_to_recharge(1000, 1000)
        self.assertEqual(time, 0)
        
        # Test default target (full capacity)
        time = self.model.time_to_recharge(700)
        self.assertEqual(time, 30)  # (1000 - 700) / 10
        
    def test_get_snr(self):
        """Test SNR calculation."""
        # Test at high energy
        snr = self.model.get_snr(800)
        self.assertEqual(snr, 1.0)  # sqrt(800/800)
        
        # Test at zero energy
        snr = self.model.get_snr(0)
        self.assertEqual(snr, 0.0)
        
        # Test at arbitrary energy
        snr = self.model.get_snr(200)
        self.assertAlmostEqual(snr, 0.5, places=2)  # sqrt(200/800)
        
    def test_get_processing_cost_factor(self):
        """Test processing cost factor calculation."""
        # Test at high energy
        factor = self.model.get_processing_cost_factor(900)
        self.assertEqual(factor, 1.0)
        
        # Test at medium energy
        factor = self.model.get_processing_cost_factor(500)
        expected_factor = 0.7 + 0.3 * (500 - 300) / (800 - 300)  # 0.82
        self.assertAlmostEqual(factor, expected_factor, places=2)
        
        # Test at low energy
        factor = self.model.get_processing_cost_factor(200)
        self.assertEqual(factor, 0.3)
        
    def test_recharge_time_calculations(self):
        """Test derived recharge time calculations."""
        self.assertEqual(self.model.full_recharge_time, 100)  # 1000 / 10
        self.assertEqual(self.model.min_recharge_time, 10)   # 100 / 10

    def test_zero_recharge_rate(self):
        """Time to recharge should be infinite when recharge rate is zero."""
        params = EnergyParameters(
            capacity=100,
            recharge_rate=0,
            classification_cost=10,
            min_operational=10
        )
        model = EnergyModel(params)

        time = model.time_to_recharge(0, 50)
        self.assertEqual(time, float('inf'))


if __name__ == "__main__":
    unittest.main()