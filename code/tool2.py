import asyncio
import websockets
import numpy as np
import matplotlib.pyplot as plt

animal_positions = {}
log_file = open("animal_positions.log", "w")

async def process_data():
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    data = await websocket.recv()
                    animal_id, lat, lon = map(float, data.split(','))
                    log_file.write(f"{animal_id},{lat},{lon}\n")
                    if animal_id not in animal_positions:
                        animal_positions[animal_id] = []
                    animal_positions[animal_id].append((lat, lon))

                    # Calculate metrics (example: average distance traveled)
                    total_distance = 0
                    count = 0
                    for positions in animal_positions.values():
                        if len(positions) > 1:
                            for i in range(1, len(positions)):
                                prev_lat, prev_lon = positions[i - 1]
                                curr_lat, curr_lon = positions[i]
                                distance = np.sqrt((curr_lat - prev_lat) ** 2 + (curr_lon - prev_lon) ** 2)
                                total_distance += distance
                                count += 1
                    if count > 0:
                        average_distance = total_distance / count
                        print(f"Average Distance Traveled: {average_distance:.6f}")

                    # Plot positions
                    if len(animal_positions[animal_id]) % 40 == 0:  # Plot every 40 updates
                        plt.figure(figsize=(10, 6))
                        for animal_id, positions in animal_positions.items():
                            lats, lons = zip(*positions)
                            plt.plot(lons, lats, marker='o', label=f"Animal {animal_id}")
                        plt.title('Animal Movements')
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        plt.legend()
                        plt.show()
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode) as e:
            print(f"Connection error: {e}. Retrying...")
            await asyncio.sleep(1)  # Wait a bit before retrying

asyncio.run(process_data())
log_file.close()
