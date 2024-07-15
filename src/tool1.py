import asyncio
import websockets
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from cow_movement_simulator import CowMovementSimulator  # Ensure this import is correct

class MovementGenerator:
    def __init__(self, num_animals):
        self.simulators = [CowMovementSimulator(cow_id) for cow_id in range(num_animals)]
        self.trajectories = {simulator.cow_id: [] for simulator in self.simulators}
        self.connections = set()

    async def register(self, websocket):
        self.connections.add(websocket)

    async def unregister(self, websocket):
        self.connections.remove(websocket)

    async def send_movements(self):
        while True:
            data_batch = []
            for simulator in self.simulators:
                cow_id, lat, lon = simulator.next_step()
                self.trajectories[cow_id].append((lat, lon))
                data = {
                    "cow_id": cow_id,
                    "latitude": lat,
                    "longitude": lon
                }
                data_batch.append(json.dumps(data))

            if self.connections:
                # Send data to all connected clients
                await asyncio.wait([ws.send(data) for data in data_batch for ws in self.connections])

            await asyncio.sleep(simulator.TIME_STEP)  # Sleep for the time step duration

    async def handler(self, websocket, path):
        # Register new connection
        await self.register(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self.unregister(websocket)

    def plot_trajectories(self):
        plt.figure(figsize=(14, 8))
        for cow_id, trajectory in self.trajectories.items():
            if trajectory:
                lats, lons = zip(*trajectory)
                x, y = self.convert_to_meters(lats, lons)
                plt.plot(x, y, label=f'Cow {cow_id}')
        plt.title('Animal Movements Over Time')
        plt.xlabel('Distance (meters)')
        plt.ylabel('Distance (meters)')
        plt.xlim(0, 600)
        plt.ylim(0, 200)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def convert_to_meters(lats, lons):
        R = 6371000  # Radius of Earth in meters
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        lat0 = lats_rad[0]
        lon0 = lons_rad[0]
        x = R * (lons_rad - lon0) * np.cos(lat0)
        y = R * (lats_rad - lat0)
        return x, y

def main():
    num_animals = int(os.getenv("NUM_ANIMALS", 5))  # Default to 5 if the environment variable is not set
    generator = MovementGenerator(num_animals)
    start_server = websockets.serve(generator.handler, "0.0.0.0", 5678)

    # Run the movement data generator in the background
    asyncio.get_event_loop().create_task(generator.send_movements())

    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        # When interrupted, plot the trajectories
        generator.plot_trajectories()

if __name__ == "__main__":
    main()
