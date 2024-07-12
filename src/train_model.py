import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

# Load and visualize historical data
def load_historical_data(filename):
    data = pd.read_csv(filename)
    return data

def visualize_data(data):
    plt.figure(figsize=(14, 8))
    for cow_id in data['cow_id'].unique():
        cow_data = data[data['cow_id'] == cow_id]
        plt.plot(cow_data['longitude'], cow_data['latitude'], label=f'Cow {cow_id}')
    plt.title('Animal Movements Over Time')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Dataset class for LSTM
class MovementDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(data[['latitude', 'longitude']])

    def __len__(self):
        return len(self.scaled_data) - self.seq_length

    def __getitem__(self, idx):
        x = self.scaled_data[idx:idx + self.seq_length]
        y = self.scaled_data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LSTM Autoencoder model
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

# Train the autoencoder
def train_model(model, train_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# Detect anomalies
def detect_anomalies(model, data_loader, scaler, threshold=0.01):
    model.eval()
    anomalies = []
    reconstruction_errors = []
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            loss = torch.mean((output - y[:, -1, :])**2, dim=1)
            reconstruction_errors.extend(loss.numpy())
            anomalies.extend(loss.numpy() > threshold)
    return anomalies, reconstruction_errors

# Determine the threshold using the 99th percentile of reconstruction errors
def determine_threshold(errors):
    return np.percentile(errors, 99)

# Plot results
def plot_results(data, anomalies, scaler):
    scaled_data = scaler.transform(data[['latitude', 'longitude']])
    anomaly_indices = np.where(anomalies)[0] + 10  # offset by sequence length
    plt.figure(figsize=(14, 8))
    plt.plot(scaled_data[:, 1], scaled_data[:, 0], label='Normal', color='blue')
    plt.scatter(scaled_data[anomaly_indices, 1], scaled_data[anomaly_indices, 0], color='red', label='Anomalies')
    plt.title('Detected Anomalies in Animal Movement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

# Save the trained model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Main function
def main():
    filename = 'historical_data.csv'
    data = load_historical_data(filename)
    visualize_data(data)

    seq_length = 10
    input_dim = 2
    hidden_dim = 50
    latent_dim = 10
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20

    dataset = MovementDataset(data, seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(seq_length, input_dim, hidden_dim, latent_dim)
    train_model(model, data_loader, num_epochs, learning_rate)

    # Detect anomalies
    anomalies, reconstruction_errors = detect_anomalies(model, data_loader, dataset.scaler)
    threshold = determine_threshold(reconstruction_errors)
    anomalies, _ = detect_anomalies(model, data_loader, dataset.scaler, threshold)
    
    print(f"Anomalies detected: {sum(anomalies)}")

    # Plot results
    plot_results(data, anomalies, dataset.scaler)

    # Save the model
    model_filename = 'lstm_autoencoder.pth'
    save_model(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
