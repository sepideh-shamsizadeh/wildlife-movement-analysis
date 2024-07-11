import asyncio
import websockets
import json
import numpy as np
import joblib
import argparse
import logging

# Constants
REFERENCE_LAT = 47.0  # degrees
REFERENCE_LON = 8.0  # degrees
METERS_PER_DEGREE_LAT = 111320  # Approx value, varies with latitude
METERS_PER_DEGREE_LON = 111320 * np.cos(np.radians(REFERENCE_LAT))  # Adjusted for latitude

# Setup logging
logging.basicConfig(filename='anomaly_detection.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth's radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

async def read_data_stream():
    uri = "ws://simulator-service:80"  # Use the service name for animal-movement-simulator
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                data = await websocket.recv()
                yield json.loads(data)
            except websockets.ConnectionClosed:
                break

async def calculate_metrics_and_detect_anomalies(model):
    distances = {}
    counts = {}
    anomaly_points = []

    async for row in read_data_stream():
        animal_id = int(row['animal_id'])
        lat = float(row['lat'])
        lon = float(row['lon'])

        # Calculate distance traveled
        if animal_id in distances:
            prev_lat, prev_lon = distances[animal_id]
            distance = haversine_distance(prev_lat, prev_lon, lat, lon)
            distances[animal_id] = (lat, lon)
            counts[animal_id].append(distance)
        else:
            distances[animal_id] = (lat, lon)
            counts[animal_id] = []

        avg_distances = {aid: sum(dists) / len(dists) if dists else 0 for aid, dists in counts.items()}
        logging.info(f"Average distances: {avg_distances}")

        # Detect anomalies using the GMM model
        anomaly_score = model.score_samples([[lat, lon]])[0]
        is_anomalous = anomaly_score < -10  # Example threshold, adjust based on historic data
        if is_anomalous:
            anomaly_points.append((lat, lon))
            logging.warning(f"Animal ID: {animal_id}, Anomalous: {is_anomalous}, Anomaly Score: {anomaly_score}")

    # Plot real-time data and anomalies
    latitudes = [coords[0] for coords in distances.values()]
    longitudes = [coords[1] for coords in distances.values()]
    
    plt.scatter(longitudes, latitudes, s=1, label='Animal Positions')
    
    if anomaly_points:
        anomaly_lats = [point[0] for point in anomaly_points]
        anomaly_lons = [point[1] for point in anomaly_points]
        plt.scatter(anomaly_lons, anomaly_lats, color='red', s=5, label='Anomalies')

    plt.title('Animal Positions and Anomalies')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process animal movement data.')
    parser.add_argument('--model_file', type=str, default='gmm_model.pkl', help='File to load the GMM model')
    args = parser.parse_args()

    model = joblib.load(args.model_file)
    asyncio.run(calculate_metrics_and_detect_anomalies(model))
