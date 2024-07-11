import asyncio
import random
import numpy as np
import argparse
import websockets
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description='Simulate animal movements.')
parser.add_argument('--num_animals', type=int, default=10, help='Number of animals to simulate')
parser.add_argument('--update_interval', type=float, default=0.25, help='Time interval between updates (seconds)')
parser.add_argument('--duration', type=int, default=3000, help='Total duration of the simulation (seconds)')
args = parser.parse_args()

# Constants
FIELD_WIDTH = 600  # meters
FIELD_HEIGHT = 200  # meters
TIME_STEP = args.update_interval
TOTAL_TIME = args.duration
NUM_POINTS = int(TOTAL_TIME / TIME_STEP)
NUM_ANIMALS = args.num_animals
GPS_NOISE_LEVEL = 1.0  # meters
REFERENCE_LAT = 47.0  # degrees
REFERENCE_LON = 8.0  # degrees
METERS_PER_DEGREE_LAT = 111320  # Approx value, varies with latitude
METERS_PER_DEGREE_LON = 111320 * np.cos(np.radians(REFERENCE_LAT))  # Adjusted for latitude

# Milking times
MORNING_MILKING_TIME = 600  # seconds (10 minutes)
EVENING_MILKING_TIME = 1200  # seconds (20 minutes)

# Speed parameters
speed_grazing = 0.1  # m/s
speed_milking = 1.0  # m/s

# Define milking shed location
milking_shed_location = (500, 150)
milking_shed_radius = 5  # meters

# Function to add GPS noise
def add_gps_noise(x, y, noise_level):
    return x + np.random.normal(0, noise_level), y + np.random.normal(0, noise_level)

# Function to convert x/y coordinates to WGS84
def xy_to_wgs84(x, y):
    lat = REFERENCE_LAT + (y / METERS_PER_DEGREE_LAT)
    lon = REFERENCE_LON + (x / METERS_PER_DEGREE_LON)
    return lat, lon

async def simulate_animal_movement(animal_id, websocket):
    x, y = random.uniform(0, FIELD_WIDTH), random.uniform(0, FIELD_HEIGHT)
    for t in range(NUM_POINTS):
        current_time = t * TIME_STEP
        
        if current_time < MORNING_MILKING_TIME or (current_time > MORNING_MILKING_TIME and current_time < EVENING_MILKING_TIME):
            # Grazing or resting
            behavior = random.choices(['resting', 'grazing'], weights=(1, 5), k=1)[0]
        else:
            # Moving to the milking shed
            behavior = 'milking'
        
        if behavior == 'resting':
            x, y = add_gps_noise(x, y, GPS_NOISE_LEVEL)
            
        elif behavior == 'grazing':
            angle = random.uniform(0, 2 * np.pi)
            x += speed_grazing * np.cos(angle) * TIME_STEP
            y += speed_grazing * np.sin(angle) * TIME_STEP
            x, y = add_gps_noise(x, y, GPS_NOISE_LEVEL)
            
        elif behavior == 'milking':
            distance_to_shed = np.sqrt((milking_shed_location[0] - x)**2 + (milking_shed_location[1] - y)**2)
            if distance_to_shed > milking_shed_radius:
                direction = np.arctan2(milking_shed_location[1] - y, milking_shed_location[0] - x)
                x += speed_milking * np.cos(direction) * TIME_STEP
                y += speed_milking * np.sin(direction) * TIME_STEP
                # Add GPS noise during the directed movement
                x, y = add_gps_noise(x, y, GPS_NOISE_LEVEL)
            else:
                x, y = add_gps_noise(*milking_shed_location, GPS_NOISE_LEVEL)  # Stay at the milking shed with GPS noise
        
        # Ensure the cow stays within the field boundaries
        x = np.clip(x, 0, FIELD_WIDTH)
        y = np.clip(y, 0, FIELD_HEIGHT)
        
        # Convert to WGS84
        lat, lon = xy_to_wgs84(x, y)
        
        data = {
            'animal_id': animal_id,
            'timestamp': current_time,
            'lat': lat,
            'lon': lon,
            'behavior': behavior
        }

        # Send data to WebSocket
        await websocket.send(json.dumps(data))
        
        await asyncio.sleep(TIME_STEP)

async def handler(websocket, path):
    tasks = [simulate_animal_movement(animal_id, websocket) for animal_id in range(1, NUM_ANIMALS + 1)]
    await asyncio.gather(*tasks)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):  # Bind to all network interfaces
        await asyncio.Future()  # run forever

asyncio.run(main())
