import asyncio
import json
import os
import websockets
from statistics import mean
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

animal_positions = {}
model_path = 'lstm_autoencoder.pth'
seq_length = 10
hidden_dim = 50
latent_dim = 10

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.view(-1, self.hidden_dim)
        out = self.decoder(h)
        return out

def load_model(model_path):
    model = LSTMAutoencoder(seq_length, 2, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model(model_path)
scaler = MinMaxScaler()

def preprocess_positions(positions):
    scaled_positions = scaler.fit_transform(positions)
    sequences = [scaled_positions[i:i + seq_length] for i in range(len(scaled_positions) - seq_length)]
    return torch.tensor(sequences, dtype=torch.float32), scaled_positions

def haversine(coord1, coord2):
    import math
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

async def calculate_metrics():
    while True:
        await asyncio.sleep(5)  # Metric calculation interval
        for animal_id, positions in animal_positions.items():
            if len(positions) > seq_length:
                distances = [
                    haversine(positions[i], positions[i + 1])
                    for i in range(len(positions) - 1)
                ]
                average_distance = mean(distances)
                print(f"Animal {animal_id} average distance: {average_distance:.2f} meters")
                
                # Anomaly detection and variation calculation
                positions_tensor, scaled_positions = preprocess_positions(positions)
                with torch.no_grad():
                    output = model(positions_tensor)
                variations = np.linalg.norm(output.numpy() - positions_tensor[:, -1, :].numpy(), axis=1)
                mean_variation = np.mean(variations)
                print(f"Animal {animal_id} Variations:")
                print(f"  Mean Variation: {mean_variation:.4f}")

async def retrain_model():
    while True:
        await asyncio.sleep(3600)  # Retraining interval (e.g., every hour)
        all_positions = []
        for positions in animal_positions.values():
            all_positions.extend(positions)
        if len(all_positions) > seq_length:
            positions_tensor, _ = preprocess_positions(all_positions)
            print("Retraining model with new data...")
            train_model(model, positions_tensor)
            print("Model retrained with new data")

def train_model(model, positions_tensor, num_epochs=5, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(positions_tensor) - 1):
            x = positions_tensor[i].unsqueeze(0)
            y = positions_tensor[i + 1].unsqueeze(0)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Retraining Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(positions_tensor):.4f}")

async def consume_movement_data():
    server_url = os.getenv('SERVER_URL', 'ws://localhost:5678')
    async with websockets.connect(server_url) as websocket:
        async for message in websocket:
            data = json.loads(message)
            animal_id = data['cow_id']
            position = (data['latitude'], data['longitude'])
            if animal_id not in animal_positions:
                animal_positions[animal_id] = []
            animal_positions[animal_id].append(position)

async def main():
    await asyncio.gather(
        consume_movement_data(),
        calculate_metrics(),
        retrain_model()
    )

if __name__ == "__main__":
    asyncio.run(main())
