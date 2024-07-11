import unittest
import numpy as np
from src.cow_movement_simulator import CowMovementSimulator

class TestCowMovementSimulator(unittest.TestCase):

    def setUp(self):
        self.cow_id = 1
        self.simulator = CowMovementSimulator(self.cow_id)

    def test_initialization(self):
        self.assertEqual(self.simulator.cow_id, self.cow_id)
        self.assertEqual(self.simulator.FIELD_WIDTH, 600)
        self.assertEqual(self.simulator.FIELD_HEIGHT, 200)
        self.assertEqual(self.simulator.TIME_STEP, 0.25)
        self.assertEqual(self.simulator.TOTAL_TIME, 3000)
        self.assertEqual(self.simulator.NUM_POINTS, int(self.simulator.TOTAL_TIME / self.simulator.TIME_STEP))

    def test_add_gps_noise(self):
        x, y = 100, 100
        noise_level = 1.0
        noisy_x, noisy_y = self.simulator.add_gps_noise(x, y, noise_level)
        self.assertNotEqual((x, y), (noisy_x, noisy_y))
        self.assertTrue(np.abs(noisy_x - x) <= 3 * noise_level)
        self.assertTrue(np.abs(noisy_y - y) <= 3 * noise_level)

    def test_xy_to_wgs84(self):
        x, y = 100, 100
        lat, lon = self.simulator.xy_to_wgs84(x, y)
        expected_lat = self.simulator.reference_lat + (y / self.simulator.meters_per_degree_lat)
        expected_lon = self.simulator.reference_lon + (x / self.simulator.meters_per_degree_lon)
        self.assertAlmostEqual(lat, expected_lat, places=6)
        self.assertAlmostEqual(lon, expected_lon, places=6)

    def test_is_stationary(self):
        stationary_trajectory = [(0, 0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        non_stationary_trajectory = [(0, 0), (10, 10), (20, 20), (30, 30)]
        self.assertTrue(self.simulator.is_stationary(stationary_trajectory, 4, 3))
        self.assertFalse(self.simulator.is_stationary(non_stationary_trajectory, 4, 3))

    def test_next_step(self):
        initial_position = (self.simulator.x, self.simulator.y)
        for _ in range(10):  # Check movement over 10 steps
            cow_id, lat, lon = self.simulator.next_step()
        new_position = (self.simulator.x, self.simulator.y)
        self.assertNotAlmostEqual(initial_position[0], new_position[0], places=5)
        self.assertNotAlmostEqual(initial_position[1], new_position[1], places=5)
        self.assertTrue(0 <= self.simulator.x <= self.simulator.FIELD_WIDTH)
        self.assertTrue(0 <= self.simulator.y <= self.simulator.FIELD_HEIGHT)

    def test_trajectory_simulation(self):
        trajectory = []
        for _ in range(self.simulator.NUM_POINTS):
            cow_id, lat, lon = self.simulator.next_step()
            trajectory.append((cow_id, lat, lon))
        self.assertEqual(len(trajectory), self.simulator.NUM_POINTS)
        self.assertTrue(all(0 <= lat <= 90 and 0 <= lon <= 180 for _, lat, lon in trajectory))

if __name__ == "__main__":
    unittest.main()
