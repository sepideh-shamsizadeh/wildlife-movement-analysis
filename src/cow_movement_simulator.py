import numpy as np
import matplotlib.pyplot as plt

class CowMovementSimulator:
    def __init__(self, cow_id, field_width=600, field_height=200, time_step=0.25, total_time=3000, 
                 mean_velocity=1.5, std_dev_velocity=0.1, gps_noise_level=1.0, 
                 mean_turning_angle_10s=83.272, reference_lat=47.0, reference_lon=8.0):
        self.cow_id = cow_id
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
        self.trajectory = [(self.x, self.y)]
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
        distances = [np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i-1][1] - trajectory[i][1])**2) 
                     for i in range(1, window_size)]
        average_distance = np.mean(distances)
        return average_distance < threshold_distance
    
    def calculate_median_speed(self, speeds, window_size):
        if len(speeds) < window_size:
            return np.median(speeds)
        return np.median(speeds[-window_size:])
    
    def next_step(self):
        stationary_check_interval = 300  # seconds
        window_size_stationary = int(stationary_check_interval / self.TIME_STEP)
        
        if not self.is_stationary(self.trajectory, window_size=window_size_stationary, threshold_distance=3):
            behavior = np.random.choice(['grazing', 'walking', 'resting'], p=[0.7, 0.2, 0.1])
            
            if behavior == 'grazing':
                speed = np.random.normal(self.mean_velocity, self.std_dev_velocity)
            elif behavior == 'walking':
                speed = np.random.normal(2.0, 0.2)
            else:
                speed = 0
            
            angle = np.radians(np.random.normal(self.mean_turning_angle_10s, 30))
            self.x += speed * np.cos(angle) * self.TIME_STEP
            self.y += speed * np.sin(angle) * self.TIME_STEP
            self.x, self.y = self.add_gps_noise(self.x, self.y, self.gps_noise_level)
            self.speeds.append(speed)
        
        self.x = np.clip(self.x, 0, self.FIELD_WIDTH)
        self.y = np.clip(self.y, 0, self.FIELD_HEIGHT)
        
        lat, lon = self.xy_to_wgs84(self.x, self.y)
        self.trajectory.append((self.x, self.y))
        
        return self.cow_id, lat, lon

    def plot_trajectory(self):
        trajectory_x, trajectory_y = zip(*self.trajectory)
        plt.figure(figsize=(14, 8))
        plt.plot(trajectory_x, trajectory_y, color='green', label='movement')

        plt.title('Animal Movement Over Time')
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.legend()
        plt.grid(True)
        plt.show()




def main():
    # Create an instance of the CowMovementSimulator
    cow_id = 1
    simulator = CowMovementSimulator(cow_id)
    
    # Simulate the movement
    trajectory = []
    for _ in range(simulator.NUM_POINTS):
        cow_id, lat, lon = simulator.next_step()
        trajectory.append((cow_id, lat, lon))
    
    # Print a few sample outputs
    print("Sample trajectory points (cow_id, lat, lon):")
    for i in range(0, len(trajectory), len(trajectory)//10):
        print(trajectory[i])
    
    # Plot the trajectory
    simulator.plot_trajectory()

if __name__ == "__main__":
    main()
