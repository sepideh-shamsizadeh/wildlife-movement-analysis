import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
FIELD_WIDTH = 600  # meters
FIELD_HEIGHT = 200  # meters
TIME_STEP = 0.25  # seconds
TOTAL_TIME = 3000  # seconds (30 minutes)
NUM_POINTS = int(TOTAL_TIME / TIME_STEP)

# Speed parameters
mean_velocity = 1.5  # m/s
std_dev_velocity = 0.1  # m/s
gps_noise_level = 1.0  # meters

# Turning angle mean for 10s interval (in degrees)
mean_turning_angle_10s = 83.272

# Constants for geographic conversions
reference_lat = 47.0  # degrees
reference_lon = 8.0   # degrees
meters_per_degree_lat = 111320  # Approx value, varies with latitude
meters_per_degree_lon = 111320 * np.cos(np.radians(reference_lat))  # Adjusted for latitude

# Function to add GPS noise
def add_gps_noise(x, y, noise_level):
    return x + np.random.normal(0, noise_level), y + np.random.normal(0, noise_level)

# Function to convert x/y coordinates to WGS84
def xy_to_wgs84(x, y):
    lat = reference_lat + (y / meters_per_degree_lat)
    lon = reference_lon + (x / meters_per_degree_lon)
    return lat, lon

# Function to check if the cow is stationary
def is_stationary(trajectory, window_size, threshold_distance):
    if len(trajectory) < window_size:
        return False
    distances = [np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) for i in range(1, window_size)]
    average_distance = np.mean(distances)
    return average_distance < threshold_distance

# Initialize position
x, y = 600, 0  # Starting at the top-right corner of the field

# Lists to store trajectory data
trajectory = [(x, y)]
trajectory_lat = []
trajectory_lon = []

# Main loop to simulate movement
for t in range(NUM_POINTS):
    # Grazing behavior
    if not is_stationary(trajectory, window_size=300, threshold_distance=3):
        speed_grazing = np.random.normal(mean_velocity, std_dev_velocity)  # Speed from normal distribution
        angle = np.radians(np.random.normal(mean_turning_angle_10s, 30))  # Use mean turning angle with more variability
        x += speed_grazing * np.cos(angle) * TIME_STEP
        y += speed_grazing * np.sin(angle) * TIME_STEP
        x, y = add_gps_noise(x, y, gps_noise_level)
    
    # Ensure the animal stays within the field boundaries
    x = np.clip(x, 0, FIELD_WIDTH)
    y = np.clip(y, 0, FIELD_HEIGHT)
    
    # Convert to WGS84
    lat, lon = xy_to_wgs84(x, y)
    
    # Append to trajectory
    trajectory.append((x, y))
    trajectory_lat.append(lat)
    trajectory_lon.append(lon)

# Visualization
trajectory_x, trajectory_y = zip(*trajectory)
plt.figure(figsize=(14, 8))
plt.plot(trajectory_x, trajectory_y, color='green', label='grazing')

plt.title('Animal Grazing Movement Over 30 Minutes')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.legend()
plt.grid(True)
plt.show()

# Print some sample WGS84 coordinates
for i in range(0, len(trajectory_lat), NUM_POINTS // 10):  # Print 10 sample points
    print(f"Time: {i * TIME_STEP:.2f}s, Lat: {trajectory_lat[i]:.6f}, Lon: {trajectory_lon[i]:.6f}")
