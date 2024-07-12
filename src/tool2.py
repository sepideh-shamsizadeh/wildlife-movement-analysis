import asyncio
import json
import os
import websockets
from statistics import mean

animal_positions = {}

async def calculate_metrics():
    while True:
        await asyncio.sleep(5)  # Metric calculation interval
        for animal_id, positions in animal_positions.items():
            if len(positions) > 1:
                distances = [
                    haversine(positions[i], positions[i + 1])
                    for i in range(len(positions) - 1)
                ]
                average_distance = mean(distances)
                print(f"Animal {animal_id} average distance: {average_distance:.2f} meters")
            else:
                print(f"Animal {animal_id} has insufficient data")

def haversine(coord1, coord2):
    import math
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

async def consume_movement_data():
    server_url = os.getenv('SERVER_URL', 'ws://localhost:5678')
    async with websockets.connect(server_url) as websocket:
        async for message in websocket:
            data = json.loads(message)
            animal_id = data['cow_id']
            position = (data['latitude'], data['longitude'])
            if animal_id not in animal_positions:
                animal_positions[animal_id] = []
            animal_positions[animal_id].append(position)

async def main():
    await asyncio.gather(
        consume_movement_data(),
        calculate_metrics()
    )

if __name__ == "__main__":
    asyncio.run(main())
