import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np

def train_gmm_model(data_file, model_file):
    df = pd.read_csv(data_file)
    data = df[['lat', 'lon']].values

    # Train the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
    gmm.fit(data)

    # Save the trained model
    joblib.dump(gmm, model_file)
    print(f"Model trained and saved to {model_file}")

    # Print evaluation metrics
    print(f"BIC: {gmm.bic(data)}")
    print(f"AIC: {gmm.aic(data)}")

    # Visualize the data and the Gaussian components
    plt.scatter(data[:, 1], data[:, 0], s=1, label='Data')
    
    # Create grid for plotting
    x = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
    y = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)

    plt.contour(X, Y, Z, levels=10, cmap='viridis')
    plt.title('GMM components')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GMM model using animal_movement_data.csv')
    parser.add_argument('--data_file', type=str, default='animal_movement_data.csv', help='CSV file containing animal movement data')
    parser.add_argument('--model_file', type=str, default='gmm_model.pkl', help='File to save the trained GMM model')
    args = parser.parse_args()

    train_gmm_model(args.data_file, args.model_file)
