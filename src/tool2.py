import asyncio
import websockets
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataStreamAnalyzer:
    def __init__(self):
        self.animal_data = {}

    def update_metrics(self, cow_id, lat, lon):
        if cow_id not in self.animal_data:
            self.animal_data[cow_id] = {
                "total_distance": 0,
                "num_positions": 0,
                "last_position": None
            }
        data = self.animal_data[cow_id]
        if data["last_position"] is not None:
            last_lat, last_lon = data["last_position"]
            distance = self.haversine_distance(last_lat, last_lon, lat, lon)
            data["total_distance"] += distance
            data["num_positions"] += 1
        data["last_position"] = (lat, lon)

    def average_distance(self, cow_id):
        data = self.animal_data.get(cow_id)
        if data is None or data["num_positions"] == 0:
            return 0
        return data["total_distance"] / data["num_positions"]

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

async def receive_movements():
    uri = "ws://localhost:5678"
    analyzer = DataStreamAnalyzer()

    try:
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                cow_id = data.get("cow_id")
                lat = data.get("latitude")
                lon = data.get("longitude")
                
                if cow_id is not None and lat is not None and lon is not None:
                    analyzer.update_metrics(cow_id, lat, lon)
                    logger.info(f"Received data for Cow ID: {cow_id}, Latitude: {lat}, Longitude: {lon}")
                    logger.info(f"Avg Distance for Cow ID {cow_id}: {analyzer.average_distance(cow_id):.4f} m")
                else:
                    logger.warning("Received incomplete data from the server.")

    except websockets.ConnectionClosed:
        logger.error("Connection to WebSocket server closed.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    asyncio.get_event_loop().run_until_complete(receive_movements())

if __name__ == "__main__":
    main()
