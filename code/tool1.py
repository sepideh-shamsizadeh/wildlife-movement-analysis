import asyncio
import random
import numpy as np
import websockets

# Constants
FIELD_WIDTH = 600
FIELD_HEIGHT = 200
TIME_STEP = 0.25
TOTAL_TIME = 3000
NUM_POINTS = int(TOTAL_TIME / TIME_STEP)
NUM_ANIMALS = 10
GPS_NOISE_LEVEL = 1.0
REFERENCE_LAT = 47.0
REFERENCE_LON = 8.0
METERS_PER_DEGREE_LAT = 111320
METERS_PER_DEGREE_LON = 111320 * np.cos(np.radians(REFERENCE_LAT))

# Speed parameters
speed_grazing = 0.1  # m/s

# Function to convert x/y coordinates to WGS84
def xy_to_wgs84(x, y):
    lat = REFERENCE_LAT + (y / METERS_PER_DEGREE_LAT)
    lon = REFERENCE_LON + (x / METERS_PER_DEGREE_LON)
    return lat, lon

# Function to add GPS noise
def add_gps_noise(x, y, noise_level):
    return x + np.random.normal(0, noise_level), y + np.random.normal(0, noise_level)

async def simulate_animal_movement(animal_id, websocket):
    x, y = random.uniform(0, FIELD_WIDTH), random.uniform(0, FIELD_HEIGHT)
    for _ in range(NUM_POINTS):
        angle = random.uniform(0, 2 * np.pi)
        x += speed_grazing * np.cos(angle) * TIME_STEP
        y += speed_grazing * np.sin(angle) * TIME_STEP
        x, y = add_gps_noise(x, y, GPS_NOISE_LEVEL)
        x = np.clip(x, 0, FIELD_WIDTH)
        y = np.clip(y, 0, FIELD_HEIGHT)
        lat, lon = xy_to_wgs84(x, y)
        await websocket.send(f"{animal_id},{lat},{lon}")
        await asyncio.sleep(TIME_STEP)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

async def handler(websocket, path):
    tasks = []
    for animal_id in range(1, NUM_ANIMALS + 1):
        tasks.append(simulate_animal_movement(animal_id, websocket))
    await asyncio.gather(*tasks)

asyncio.run(main())
