import unittest
from unittest.mock import Mock, patch
import asyncio
import websockets
import json
import matplotlib.pyplot as plt
from asynctest import TestCase, MagicMock
from src.tool1 import CowMovementSimulator  # Adjust import based on your project structure
from src.tool2 import MovementGenerator


class TestMovementGenerator(TestCase):

    def setUp(self):
        self.num_animals = 3
        self.generator = MovementGenerator(self.num_animals)

    @patch('src.tool1.CowMovementSimulator')
    @patch('websockets.serve')
    def test_send_movements(self, mock_serve, MockCowMovementSimulator):
        # Mock CowMovementSimulator's next_step method
        mock_simulators = [MockCowMovementSimulator(cow_id) for cow_id in range(self.num_animals)]
        for simulator in mock_simulators:
            simulator.next_step.return_value = (simulator.cow_id, 1.0, 2.0)

        # Mock asyncio event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._test_send_movements(loop, mock_simulators))

    async def _test_send_movements(self, loop, mock_simulators):
        mock_websocket = MagicMock()
        mock_path = '/test_path'
        await self.generator.send_movements(mock_websocket, mock_path)

        # Check if send method of websocket is called with correct data
        expected_data = {
            "cow_id": 0,  # Assuming cow_id starts from 0
            "latitude": 1.0,
            "longitude": 2.0
        }
        mock_websocket.send.assert_called_with(json.dumps(expected_data))

        # Check if trajectories are updated
        for simulator in mock_simulators:
            self.assertIn((1.0, 2.0), self.generator.trajectories[simulator.cow_id])

    def test_plot_trajectories(self):
        # Populate trajectories for testing
        self.generator.trajectories = {
            0: [(1.0, 2.0), (2.0, 3.0)],
            1: [(0.5, 1.5), (1.5, 2.5)],
            2: [(2.0, 3.0), (3.0, 4.0)]
        }

        # Mock matplotlib's plot method
        with patch('matplotlib.pyplot.show'):
            self.generator.plot_trajectories()

            # You can add more assertions here to check the plot, e.g., plot title, labels, etc.

if __name__ == '__main__':
    unittest.main()

