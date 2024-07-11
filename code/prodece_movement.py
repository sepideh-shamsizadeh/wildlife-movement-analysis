import numpy as np
import matplotlib.pyplot as plt

class CowMovementSimulator:
    def __init__(self, field_width=600, field_height=200, time_step=0.25, total_time=3000, 
                 mean_velocity=1.5, std_dev_velocity=0.1, gps_noise_level=1.0, 
                 mean_turning_angle_10s=83.272, reference_lat=47.0, reference_lon=8.0):
        self.FIELD_WIDTH = field_width
        self.FIELD_HEIGHT = field_height
        self.TIME_STEP = time_step
        self.TOTAL_TIME = total_time
        self.NUM_POINTS = int(self.TOTAL_TIME / self.TIME_STEP)
        self.mean_velocity = mean_velocity
        self.std_dev_velocity = std_dev_velocity
        self.gps_noise_level = gps_noise_level
        self.mean_turning_angle_10s = mean_turning_angle_10s
        
        # Geographic conversion constants
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.meters_per_degree_lat = 111320
        self.meters_per_degree_lon = 111320 * np.cos(np.radians(reference_lat))
        
        # Initialize position
        self.x, self.y = 600, 0  # Starting at the top-right corner of the field
        
        # Lists to store trajectory data
        self.trajectory = [(self.x, self.y)]
        self.trajectory_lat = []
        self.trajectory_lon = []
        self.speeds = []
        
    def add_gps_noise(self, x, y, noise_level):
        return x + np.random.normal(0, noise_level), y + np.random.normal(0, noise_level)
    
    def xy_to_wgs84(self, x, y):
        lat = self.reference_lat + (y / self.meters_per_degree_lat)
        lon = self.reference_lon + (x / self.meters_per_degree_lon)
        return lat, lon
    
    def is_stationary(self, trajectory, window_size, threshold_distance):
        if len(trajectory) < window_size:
            return False
        distances = [np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) 
                     for i in range(1, window_size)]
        average_distance = np.mean(distances)
        return average_distance < threshold_distance
    
    def calculate_median_speed(self, speeds, window_size):
        if len(speeds) < window_size:
            return np.median(speeds)
        return np.median(speeds[-window_size:])
    
    def simulate_movement(self):
        stationary_check_interval = 300  # seconds
        speed_measure_interval = 60  # seconds
        window_size_stationary = int(stationary_check_interval / self.TIME_STEP)
        window_size_speed = int(speed_measure_interval / self.TIME_STEP)
        
        for t in range(self.NUM_POINTS):
            if not self.is_stationary(self.trajectory, window_size=window_size_stationary, threshold_distance=3):
                speed_grazing = np.random.normal(self.mean_velocity, self.std_dev_velocity)  # Speed from normal distribution
                angle = np.radians(np.random.normal(self.mean_turning_angle_10s, 30))  # Use mean turning angle with more variability
                self.x += speed_grazing * np.cos(angle) * self.TIME_STEP
                self.y += speed_grazing * np.sin(angle) * self.TIME_STEP
                self.x, self.y = self.add_gps_noise(self.x, self.y, self.gps_noise_level)
                
                self.speeds.append(speed_grazing)
            
            # Ensure the animal stays within the field boundaries
            self.x = np.clip(self.x, 0, self.FIELD_WIDTH)
            self.y = np.clip(self.y, 0, self.FIELD_HEIGHT)
            
            # Convert to WGS84
            lat, lon = self.xy_to_wgs84(self.x, self.y)
            
            # Append to trajectory
            self.trajectory.append((self.x, self.y))
            self.trajectory_lat.append(lat)
            self.trajectory_lon.append(lon)
    
    def plot_trajectory(self):
        trajectory_x, trajectory_y = zip(*self.trajectory)
        plt.figure(figsize=(14, 8))
        plt.plot(trajectory_x, trajectory_y, color='green', label='movement')

        plt.title('Animal Movement Over 30 Minutes')
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def print_sample_coordinates(self):
        window_size = int(60 / self.TIME_STEP)
        for i in range(0, len(self.trajectory_lat), self.NUM_POINTS // 10):  # Print 10 sample points
            median_speed = self.calculate_median_speed(self.speeds, window_size)
            print(f"Time: {i * self.TIME_STEP:.2f}s, Lat: {self.trajectory_lat[i]:.6f}, Lon: {self.trajectory_lon[i]:.6f}, Median Speed: {median_speed:.3f} m/s")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cow Movement Simulator")
    parser.add_argument("--field_width", type=int, default=600, help="Field width in meters")
    parser.add_argument("--field_height", type=int, default=200, help="Field height in meters")
    parser.add_argument("--time_step", type=float, default=0.25, help="Time step in seconds")
    parser.add_argument("--total_time", type=int, default=3000, help="Total simulation time in seconds")
    parser.add_argument("--mean_velocity", type=float, default=1.5, help="Mean velocity in m/s")
    parser.add_argument("--std_dev_velocity", type=float, default=0.1, help="Standard deviation of velocity in m/s")
    parser.add_argument("--gps_noise_level", type=float, default=1.0, help="GPS noise level in meters")
    parser.add_argument("--mean_turning_angle_10s", type=float, default=83.272, help="Mean turning angle in degrees")

    args = parser.parse_args()

    simulator = CowMovementSimulator(
        field_width=args.field_width,
        field_height=args.field_height,
        time_step=args.time_step,
        total_time=args.total_time,
        mean_velocity=args.mean_velocity,
        std_dev_velocity=args.std_dev_velocity,
        gps_noise_level=args.gps_noise_level,
        mean_turning_angle_10s=args.mean_turning_angle_10s
    )
    simulator.simulate_movement()
    simulator.plot_trajectory()
    simulator.print_sample_coordinates()
