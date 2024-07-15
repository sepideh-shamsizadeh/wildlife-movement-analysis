# Wildlife Movement Analysis

Animal Behavior Tracker monitors wildlife and livestock using distributed IoT data acquisition. It generates simulated animal movements, streams data, calculates metrics, and deploys on Kubernetes. Features include real-time location updates, anomaly detection, and scalable performance analysis.

## Getting Started

### Clone the Repository

```sh
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/sepideh-shamsizadeh/wildlife-movement-analysis.git
cd wildlife-movement-analysis
```

### Running the Simulation Locally

To test `tool1` and `tool2` locally:

#### Terminal 1 (for `tool1`)

1. **Set the environment variable:**

    ```sh
    export NUM_ANIMALS=10
    ```

2. **Run the script:**

    ```sh
    python src/tool1.py
    ```

#### Terminal 2 (for `tool2`)

1. **Set the environment variable:**

    ```sh
    export SERVER_URL="ws://localhost:5678"
    ```

2. **Run the script:**

    ```sh
    python src/tool2.py
    ```

### Running the Simulation Script for Trajectory Data

Run the simulation script to generate the trajectory data and plot it for one animal:

```sh
python src/cow_movement_simulator.py
```

## Docker Images

The Docker images for `tool1` and `tool2` are available on Docker Hub:

- `sepideh92sh/tool1:latest`
- `sepideh92sh/tool2:latest`

You can pull these images directly or build them yourself.

### Pulling Docker Images

To pull the Docker images from Docker Hub:

```sh
minikube ssh
docker pull sepideh92sh/tool1:latest
docker pull sepideh92sh/tool2:latest
```

### Building Docker Images

If you need to build the Docker images yourself, the Dockerfiles are located in the root of the repository:

- `Dockerfile.tool1`
- `Dockerfile.tool2`

To build the Docker images:

1. **Build `tool1` Docker image:**

    ```sh
    docker build -t your-docker-repo/tool1:latest -f Dockerfile.tool1 .
    ```

2. **Build `tool2` Docker image:**

    ```sh
    docker build -t your-docker-repo/tool2:latest -f Dockerfile.tool2 .
    ```

3. **Push the Docker images to your Docker repository:**

    ```sh
    docker push your-docker-repo/tool1:latest
    docker push your-docker-repo/tool2:latest
    ```

Replace `your-docker-repo` with your actual Docker repository name if you choose to build and push your images.

## Using Kubernetes

To deploy `tool1` and `tool2` on Kubernetes, you will need to use the provided Kubernetes configuration files. Here are the steps to do so:

### Prerequisites

Ensure you have `kubectl` installed and configured to communicate with your Kubernetes cluster.

### Starting Minikube

1. **Start Minikube:**

    ```sh
    minikube start
    ```

2. **Set Docker environment to use Minikube's Docker daemon:**

    ```sh
    eval $(minikube docker-env)
    ```

### Deploying `tool1` and `tool2` on Kubernetes

1. **Navigate to the Kubernetes directory:**

    ```sh
    cd wildlife-movement-analysis/k8s-Demo
    ```

2. **Apply the ConfigMap for `tool1`:**

    ```sh
    kubectl apply -f tool1-configmap.yaml
    ```

3. **Deploy `tool1` using the Deployment configuration:**

    ```sh
    kubectl apply -f tool1-deployment.yaml
    ```

4. **Set up Horizontal Pod Autoscaler (HPA) for `tool1`:**

    ```sh
    kubectl apply -f tool1-hpa.yaml
    ```

5. **Set up Vertical Pod Autoscaler (VPA) for `tool1`:**

    ```sh
    kubectl apply -f tool1-vpa.yaml
    ```

6. **Apply the ConfigMap for `tool2`:**

    ```sh
    kubectl apply -f tool2-configmap.yaml
    ```

7. **Deploy `tool2` using the Deployment configuration:**

    ```sh
    kubectl apply -f tool2-deployment.yaml
    ```

### Running the Load Generator Job

To simulate load on `tool1`, run the load generator job:

1. **Apply the load generator job:**

    ```sh
    kubectl apply -f load-generator-job.yaml
    ```

This job will create multiple WebSocket connections to `tool1` to simulate load and trigger the autoscaling mechanisms.

### Verifying the Deployment

You can check the status of your deployments using:

```sh
kubectl get all
```

This command will show the status of pods, services, deployments, and other Kubernetes objects.

## Monitoring and Logging

To deploy monitoring tools such as Prometheus and Grafana, follow these steps:

### Install Helm

1. **Install Helm:**

    ```sh
    sudo snap install helm --classic
    ```

2. **Add Helm Repositories:**

    ```sh
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    ```

3. **Install Prometheus:**

    ```sh
    helm install prometheus prometheus-community/prometheus
    ```

4. **Install Grafana:**

    ```sh
    helm install grafana grafana/grafana
    ```

### Verifying the Installation

1. **Check Helm Releases:**

    ```sh
    helm list
    ```

2. **Check Kubernetes Pods:**

    ```sh
    kubectl get pods
    ```

### Accessing Grafana

1. **Get Grafana Admin Password:**

    ```sh
    kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
    ```

2. **Port Forward to Access Grafana UI:**

    ```sh
    kubectl port-forward --namespace default svc/grafana 3000:80
    ```

    Access Grafana at `http://localhost:3000` in your web browser.

This section provides a step-by-step guide to installing Helm, Prometheus, and Grafana, ensuring your Kubernetes cluster is properly monitored.


## Additional Resources

- `historical_data.csv`: Contains historical data used for training models.
- `lstm_autoencoder.pth`: The pre-trained model file.
- `requirements.txt`: Lists the Python dependencies needed for the project.
- `tests/`: Contains test scripts for the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Kubernetes Deployment YAML Files

The Kubernetes YAML files needed for deployment are located in the `wildlife-movement-analysis/k8s-Demo` directory. These files include:

1. `tool1-configmap.yaml`: ConfigMap for `tool1` configuration.
2. `tool1-deployment.yaml`: Deployment configuration for `tool1`.
3. `tool1-hpa.yaml`: Horizontal Pod Autoscaler configuration for `tool1`.
4. `tool1-vpa.yaml`: Vertical Pod Autoscaler configuration for `tool1`.
5. `tool2-configmap.yaml`: ConfigMap for `tool2` configuration.
6. `tool2-deployment.yaml`: Deployment configuration for `tool2`.
7. `load-generator-job.yaml`: Job configuration for generating load on `tool1`.

To deploy `tool1` and `tool2` on Kubernetes, navigate to the `k8s-Demo` directory and apply each configuration file using `kubectl apply -f <file-name>.yaml`.

Example:

```sh
kubectl apply -f tool1-configmap.yaml
kubectl apply -f tool1-deployment.yaml
kubectl apply -f tool1-hpa.yaml
kubectl apply -f tool1-vpa.yaml
kubectl apply -f tool2-configmap.yaml
kubectl apply -f tool2-deployment.yaml
kubectl apply -f load-generator-job.yaml
```

This will set up and deploy the applications on your Kubernetes cluster.
