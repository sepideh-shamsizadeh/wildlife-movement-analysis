import numpy as np
import matplotlib.pyplot as plt

class CowMovementSimulator:
    def __init__(self, cow_id, field_width=600, field_height=200, time_step=1, total_time=18000, 
                 gps_noise_level=1.0, reference_lat=47.0, reference_lon=8.0):
        self.cow_id = cow_id
        self.FIELD_WIDTH = field_width
        self.FIELD_HEIGHT = field_height
        self.TIME_STEP = time_step
        self.TOTAL_TIME = total_time
        self.NUM_POINTS = int(self.TOTAL_TIME / self.TIME_STEP)
        
        self.gps_noise_level = gps_noise_level
        
        # Temporal scales based on the table
        self.temporal_scales = {
            '5s': {'speed': 0.336, 'turning_angle': 69.005},
            '10s': {'speed': 0.281, 'turning_angle': 83.272},
            '1min': {'speed': 0.164, 'turning_angle': 47.334},
            '15min': {'speed': 0.107, 'turning_angle': 62.018},
            '10min': {'speed': 0.115, 'turning_angle': 42.497},
            '30min': {'speed': 0.125, 'turning_angle': 71.831},
        }
        
        # Geographic conversion constants
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.meters_per_degree_lat = 111320
        self.meters_per_degree_lon = 111320 * np.cos(np.radians(reference_lat))
        
        # Initialize position to (0, 0)
        self.x = 0
        self.y = 0
        self.trajectory = [(self.x, self.y)]
        self.speeds = []
        
        self.movement_duration = 0  # Track duration of continuous movement
        self.angle = np.random.uniform(0, 2 * np.pi)  # Initialize direction angle randomly

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
    
    def get_temporal_scale(self, duration):
        if duration < 10:
            return '5s'
        elif duration < 60:
            return '10s'
        elif duration < 900:
            return '1min'
        elif duration < 600:
            return '15min'
        elif duration < 1800:
            return '10min'
        else:
            return '30min'
    
    def next_step(self):
        self.movement_duration += self.TIME_STEP
        if self.movement_duration >= 2700:  # 45 minutes
            self.movement_duration = 0
        
        scale = self.get_temporal_scale(self.movement_duration)
        speed_mean = self.temporal_scales[scale]['speed']
        turning_angle_mean = self.temporal_scales[scale]['turning_angle']
        
        behavior = np.random.choice(['grazing', 'walking'], p=[0.5, 0.5])
        
        if behavior == 'grazing':
            speed = np.random.normal(speed_mean, speed_mean * 0.1)
        else:
            speed = np.random.normal(speed_mean * 2, speed_mean * 0.2)
        
        # Update position with boundary checking
        new_x = self.x + speed * np.cos(self.angle) * self.TIME_STEP
        new_y = self.y + speed * np.sin(self.angle) * self.TIME_STEP
        
        if new_x < 0 or new_x > self.FIELD_WIDTH or new_y < 0 or new_y > self.FIELD_HEIGHT:
            turn_angle = np.radians(np.random.uniform(90, 180))  # Random turn angle between 90 and 180 degrees
            self.angle += turn_angle
            new_x = self.x + speed * np.cos(self.angle) * self.TIME_STEP
            new_y = self.y + speed * np.sin(self.angle) * self.TIME_STEP
        
        # Apply GPS noise
        self.x, self.y = self.add_gps_noise(new_x, new_y, self.gps_noise_level)
        self.speeds.append(speed)
        
        # Ensure the cow remains within bounds
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
