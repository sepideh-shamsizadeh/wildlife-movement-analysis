import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
FIELD_WIDTH = 600
FIELD_HEIGHT = 200
TIME_STEP = 0.25  # seconds
TOTAL_TIME = 3000  # seconds (30 minutes)
NUM_POINTS = int(TOTAL_TIME / TIME_STEP)
MORNING_MILKING_TIME = 600  # seconds (10 minutes)
EVENING_MILKING_TIME = 1200  # seconds (20 minutes)

# Speed parameters
speed_grazing = 0.1  # m/s
speed_milking = 1.0  # m/s
gps_noise_level = 1.0  # meters

# Define milking shed location
milking_shed_location = (500, 150)
milking_shed_radius = 5  # meters

# Function to add GPS noise
def add_gps_noise(x, y, noise_level):
    return x + np.random.normal(0, noise_level), y + np.random.normal(0, noise_level)

# Initialize position
x, y = 300, 100  # Starting at the center of the field

# Lists to store trajectory data
trajectory_x = [x]
trajectory_y = [y]
behavior_list = []

# Main loop
for t in range(NUM_POINTS):
    current_time = t * TIME_STEP
    
    if current_time < MORNING_MILKING_TIME or (current_time > MORNING_MILKING_TIME and current_time < EVENING_MILKING_TIME):
        # Grazing or resting
        behavior = random.choices(['resting', 'grazing'], weights=(1, 5), k=1)[0]
    else:
        # Moving to the milking shed
        behavior = 'milking'
    
    if behavior == 'resting':
        x, y = add_gps_noise(x, y, gps_noise_level)
        
    elif behavior == 'grazing':
        angle = random.uniform(0, 2 * np.pi)
        x += speed_grazing * np.cos(angle) * TIME_STEP
        y += speed_grazing * np.sin(angle) * TIME_STEP
        x, y = add_gps_noise(x, y, gps_noise_level)
        
    elif behavior == 'milking':
        distance_to_shed = np.sqrt((milking_shed_location[0] - x)**2 + (milking_shed_location[1] - y)**2)
        if distance_to_shed > milking_shed_radius:
            direction = np.arctan2(milking_shed_location[1] - y, milking_shed_location[0] - x)
            x += speed_milking * np.cos(direction) * TIME_STEP
            y += speed_milking * np.sin(direction) * TIME_STEP
            # Add GPS noise during the directed movement
            x, y = add_gps_noise(x, y, gps_noise_level)
        else:
            x, y = add_gps_noise(*milking_shed_location, gps_noise_level)  # Stay at the milking shed with GPS noise
    
    # Ensure the cow stays within the field boundaries
    x = np.clip(x, 0, FIELD_WIDTH)
    y = np.clip(y, 0, FIELD_HEIGHT)
    
    # Append to trajectory
    trajectory_x.append(x)
    trajectory_y.append(y)
    behavior_list.append(behavior)

# Visualization
colors = {'resting': 'blue', 'grazing': 'green', 'milking': 'orange'}
plt.figure(figsize=(10, 6))

for i in range(len(trajectory_x) - 1):
    plt.plot(trajectory_x[i:i+2], trajectory_y[i:i+2], color=colors[behavior_list[i]])

plt.title('Animal Movement Over 3000 Seconds')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.legend(handles=[plt.Line2D([0], [0], color='blue', lw=2, label='resting'),
                    plt.Line2D([0], [0], color='green', lw=2, label='grazing'),
                    plt.Line2D([0], [0], color='orange', lw=2, label='milking')])
plt.grid(True)
plt.show()
