# Kubernetes AI System

A Kubernetes-based AI system for training and serving a Flan-T5 transformer model fine-tuned on the MultiNLI dataset for natural language inference (NLI). This project demonstrates how to deploy scalable AI workloads in a Kubernetes environment, from model training to production inference.

## Overview

This project implements a complete ML pipeline that:
- **Trains** a Flan-T5 model on the MultiNLI dataset for natural language inference
- **Serves** predictions via a FastAPI REST API
- **Orchestrates** everything using Kubernetes for scalability and reliability

The system takes premise-hypothesis pairs as input and predicts whether the hypothesis is entailed by, contradicts, or is neutral with respect to the premise.

## Key Features

- **Transformer Model Training**: Fine-tunes Google's Flan-T5 model on MultiNLI dataset
- **Natural Language Inference**: Predicts entailment, contradiction, or neutral relationships
- **Kubernetes Orchestration**: Scalable deployment using Jobs and Deployments
- **FastAPI Inference Service**: RESTful API for real-time predictions
- **Persistent Storage**: Shared storage for model artifacts between training and inference
- **Health Checks**: Liveness and readiness probes for reliable deployments
- **Horizontal Pod Autoscaling**: Automatic scaling based on demand

## Architecture

### Core Components

1. **Training Job** (`k8s/training-job.yaml`)
   - Kubernetes Job that runs model training
   - Fine-tunes Flan-T5-small on MultiNLI dataset
   - Saves trained model to persistent storage
   - Configurable via environment variables (batch size, epochs, learning rate)

2. **Inference Deployment** (`k8s/inference-deployment.yaml`)
   - Kubernetes Deployment with 2 replicas
   - FastAPI service serving NLI predictions
   - Loads model from shared persistent storage
   - Health checks ensure model is loaded before serving

3. **Storage** (`k8s/storage.yaml`)
   - PersistentVolume and PersistentVolumeClaim
   - Shared storage for model artifacts between training and inference pods

4. **Configuration** (`k8s/configmap.yaml`)
   - ConfigMap for application configuration
   - Environment variables for training and inference

5. **Autoscaling** (`k8s/hpa.yaml`)
   - HorizontalPodAutoscaler for dynamic scaling
   - Scales inference pods based on CPU/memory usage

## Technologies

- **AI/ML**: Python, PyTorch, Hugging Face Transformers
- **Model**: Google Flan-T5 (text-to-text transformer)
- **Dataset**: MultiNLI (Multi-Genre Natural Language Inference)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Storage**: Kubernetes PersistentVolumes

## Prerequisites

- Docker installed and running
- Kubernetes cluster (minikube, kind, or cloud-based)
- kubectl configured for cluster access
- Python 3.8+ for local development (optional)

## Quick Start

### 1. Build Docker Images

```bash
# Build training image
docker build -t ai-training:v2 ./training/

# Build inference image
docker build -t ai-inference:latest ./inference/
```

### 2. Load Images into Kubernetes (for local clusters)

If using minikube or kind:

```bash
# For minikube
minikube image load ai-training:v2
minikube image load ai-inference:latest

# For kind
kind load docker-image ai-training:v2
kind load docker-image ai-inference:latest
```

### 3. Deploy to Kubernetes

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/
```

### 4. Monitor Training

```bash
# Watch training job
kubectl get jobs
kubectl logs -f job/ai-model-training

# Check pod status
kubectl get pods
```

### 5. Access Inference API

```bash
# Port forward to inference service
kubectl port-forward service/ai-inference-service 8080:80

# Test the API
curl http://localhost:8080/health
```

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Model Information

```bash
curl http://localhost:8080/model/info
```

### Natural Language Inference Prediction

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "A man is playing guitar",
    "hypothesis": "A man is making music"
  }'
```

Response:
```json
{
  "success": true,
  "premise": "A man is playing guitar",
  "hypothesis": "A man is making music",
  "prediction": {
    "label": "entailment",
    "confidence": 0.85
  }
}
```

### Example Predictions

**Entailment** (hypothesis follows from premise):
```json
{
  "premise": "The cat sat on the mat",
  "hypothesis": "The cat was on the mat"
}
→ {"label": "entailment"}
```

**Contradiction** (hypothesis contradicts premise):
```json
{
  "premise": "The dog is sleeping",
  "hypothesis": "The dog is running"
}
→ {"label": "contradiction"}
```

**Neutral** (hypothesis is unrelated):
```json
{
  "premise": "The cat sat on the mat",
  "hypothesis": "The cat was hungry"
}
→ {"label": "neutral"}
```

## Project Structure

```
Kubernetes_AI_System/
├── training/                 # Model training components
│   ├── Dockerfile           # Training container image
│   ├── train.py             # Training script (Flan-T5 + MultiNLI)
│   └── requirements.txt     # Python dependencies
├── inference/               # Inference API components
│   ├── Dockerfile           # Inference container image
│   ├── app.py               # FastAPI application
│   ├── model_loader.py      # Model loading and prediction logic
│   └── requirements.txt     # Python dependencies
├── k8s/                     # Kubernetes manifests
│   ├── training-job.yaml    # Training Kubernetes Job
│   ├── inference-deployment.yaml  # Inference Deployment + Service
│   ├── storage.yaml         # PersistentVolume and PVC
│   ├── configmap.yaml       # Configuration
│   └── hpa.yaml             # Horizontal Pod Autoscaler
├── monitoring/              # Monitoring configuration
│   └── prometheus.yaml      # Prometheus config (optional)
├── docs/                    # Documentation
│   └── deployment-guide.md
├── Makefile                 # Build and deployment shortcuts
├── quick-start.sh           # Quick setup script
└── README.md                # This file
```

## Configuration

### Training Configuration

Environment variables (set in `k8s/training-job.yaml`):
- `BATCH_SIZE`: Training batch size (default: 4)
- `EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate for fine-tuning (default: 5e-5)
- `MODEL_NAME`: Hugging Face model name (default: google/flan-t5-small)
- `MAX_LENGTH`: Maximum sequence length (default: 256)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps (default: 2)

### Inference Configuration

Environment variables (set in `k8s/inference-deployment.yaml`):
- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)
- `WORKERS`: Number of Uvicorn workers (default: 1)

## Resource Requirements

### Training Job
- **Memory**: 2-4GB (requests: 2Gi, limits: 4Gi)
- **CPU**: 1-2 cores (requests: 1000m, limits: 2000m)
- **Storage**: PersistentVolume for model artifacts

### Inference Deployment
- **Memory**: 2-4GB per pod (requests: 2Gi, limits: 4Gi)
- **CPU**: 0.5-1 core per pod (requests: 500m, limits: 1000m)
- **Replicas**: 2 (configurable)
- **Storage**: Shared PersistentVolume for model loading

## Model Details

- **Base Model**: Google Flan-T5-small (60M parameters)
- **Task**: Natural Language Inference (NLI)
- **Dataset**: MultiNLI (Multi-Genre Natural Language Inference)
- **Labels**: entailment, contradiction, neutral
- **Input Format**: "premise: {premise} hypothesis: {hypothesis}"
- **Output Format**: Text label (entailment/contradiction/neutral)

## Monitoring

### Check Pod Status

```bash
kubectl get pods -l app=ai-inference
kubectl get pods -l app=ai-training
```

### View Logs

```bash
# Training logs
kubectl logs job/ai-model-training

# Inference logs
kubectl logs -f deployment/ai-inference
```

### Check Services

```bash
kubectl get services
kubectl describe service ai-inference-service
```

## Troubleshooting

### Training Job Fails

1. Check logs: `kubectl logs job/ai-model-training`
2. Verify storage: `kubectl get pvc`
3. Check resource limits: `kubectl describe job ai-model-training`

### Inference Service Not Ready

1. Verify model exists: `kubectl exec -it <pod-name> -- ls -la /app/models/`
2. Check health endpoint: `curl http://localhost:8080/health`
3. Review logs: `kubectl logs deployment/ai-inference`

### Model Not Loading

1. Ensure training job completed successfully
2. Verify PersistentVolume is mounted correctly
3. Check model path in ConfigMap matches actual location

## Development

### Local Development

```bash
# Install dependencies
pip install -r training/requirements.txt
pip install -r inference/requirements.txt

# Run training locally (requires model storage)
python training/train.py

# Run inference API locally
cd inference && python app.py
```

### Rebuilding Images

```bash
# Rebuild and reload
docker build -t ai-training:v2 ./training/
docker build -t ai-inference:latest ./inference/
minikube image load ai-training:v2
minikube image load ai-inference:latest

# Restart deployments
kubectl rollout restart deployment/ai-inference
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for the Transformers library and model hub
- Google for the Flan-T5 model
- The MultiNLI dataset creators
- Kubernetes community for excellent documentation

---

**Built for demonstrating Kubernetes-based AI/ML deployments**
