import asyncio
import json
import csv
from datetime import datetime
import os
import websockets

# Function to consume movement data and save to CSV
async def consume_and_save_data(server_url, output_file):
    animal_positions = {}

    async def consume_movement_data():
        async with websockets.connect(server_url) as websocket:
            async for message in websocket:
                data = json.loads(message)
                animal_id = data['cow_id']
                position = (data['latitude'], data['longitude'])
                if animal_id not in animal_positions:
                    animal_positions[animal_id] = []
                animal_positions[animal_id].append(position)

    async def save_positions_periodically():
        while True:
            await asyncio.sleep(60)  # Save data every 60 seconds
            save_positions(output_file, animal_positions)

    await asyncio.gather(
        consume_movement_data(),
        save_positions_periodically()
    )

# Function to save positions to CSV
def save_positions(filename, animal_positions):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'cow_id', 'latitude', 'longitude'])
        timestamp = datetime.now().isoformat()
        for animal_id, positions in animal_positions.items():
            for lat, lon in positions:
                writer.writerow([timestamp, animal_id, lat, lon])

# Main function to start the process
def main():
    server_url = os.getenv('SERVER_URL', 'ws://localhost:5678')
    output_file = 'historical_data.csv'
    asyncio.run(consume_and_save_data(server_url, output_file))

if __name__ == "__main__":
    main()
