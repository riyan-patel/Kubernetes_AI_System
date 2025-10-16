# Kube-AI-Pipeline Deployment Guide

This guide will walk you through deploying the complete Kube-AI-Pipeline on your Kubernetes cluster.

## Prerequisites

### Required Software
- Docker installed and running
- Kubernetes cluster (minikube, kind, or cloud-based)
- kubectl configured for your cluster
- NVIDIA GPU with CUDA support (for GPU workloads)

### Cluster Requirements
- At least 2 CPU cores and 4GB RAM available
- GPU-enabled nodes (if using GPU acceleration)
- Persistent storage support

## Quick Start

### 1. Build Docker Images

First, build the Docker images for both training and inference components:

```bash
# Build training image
cd training/
docker build -t ai-training:latest .

# Build inference image
cd ../inference/
docker build -t ai-inference:latest .
```

### 2. Deploy to Kubernetes

Deploy all components to your Kubernetes cluster:

```bash
# Apply storage configuration
kubectl apply -f k8s/storage.yaml

# Apply configuration and secrets
kubectl apply -f k8s/configmap.yaml

# Deploy inference service
kubectl apply -f k8s/inference-deployment.yaml

# Deploy training job
kubectl apply -f k8s/training-job.yaml

# Apply autoscaling configuration
kubectl apply -f k8s/hpa.yaml

# Deploy monitoring (optional)
kubectl apply -f monitoring/prometheus.yaml
```

### 3. Verify Deployment

Check that all components are running:

```bash
# Check pods
kubectl get pods

# Check services
kubectl get services

# Check persistent volumes
kubectl get pv,pvc

# Check training job status
kubectl get jobs
```

### 4. Access the Inference API

Port-forward to access the inference API locally:

```bash
# Port forward to inference service
kubectl port-forward service/ai-inference-service 8080:80

# Test the API
curl http://localhost:8080/health
```

## Detailed Configuration

### Environment Variables

The following environment variables can be configured:

#### Training Component
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)
- `LEARNING_RATE`: Learning rate for optimizer (default: 0.001)

#### Inference Component
- `HOST`: API host address (default: 0.0.0.0)
- `PORT`: API port (default: 8000)
- `WORKERS`: Number of worker processes (default: 1)

### Resource Management

#### Training Job Resources
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

#### Inference Service Resources
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Storage Configuration

The pipeline uses persistent storage to share model artifacts between training and inference:

- **Storage Class**: `manual`
- **Capacity**: 10Gi
- **Access Mode**: ReadWriteOnce
- **Host Path**: `/tmp/kube-ai-models`

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus metrics:

```bash
# Port forward to Prometheus
kubectl port-forward service/prometheus-service 9090:9090

# Access Prometheus UI
open http://localhost:9090
```

### Application Logs

View application logs:

```bash
# Training job logs
kubectl logs -l app=ai-training

# Inference service logs
kubectl logs -l app=ai-inference

# Follow logs in real-time
kubectl logs -f deployment/ai-inference
```

### Health Checks

The inference API provides health check endpoints:

- `GET /health`: Basic health status
- `GET /model/info`: Model information
- `GET /metrics`: API metrics

## Scaling and Performance

### Horizontal Pod Autoscaling

The inference service automatically scales based on CPU and memory usage:

- **Min Replicas**: 2
- **Max Replicas**: 10
- **CPU Target**: 70% utilization
- **Memory Target**: 80% utilization

### Manual Scaling

Manually scale the inference service:

```bash
# Scale to 5 replicas
kubectl scale deployment ai-inference --replicas=5

# Check scaling status
kubectl get deployment ai-inference
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

#### 2. Model Loading Issues
```bash
# Check if model file exists
kubectl exec -it <inference-pod> -- ls -la /app/models/

# Check model loading logs
kubectl logs <inference-pod> | grep -i model
```

#### 3. GPU Resource Issues
```bash
# Check GPU availability
kubectl describe nodes | grep -i gpu

# Check GPU device plugin
kubectl get pods -n kube-system | grep nvidia
```

#### 4. Storage Issues
```bash
# Check persistent volume status
kubectl get pv,pvc

# Check storage mount
kubectl exec -it <pod> -- df -h
```

### Debug Commands

```bash
# Get all resources
kubectl get all

# Describe specific resource
kubectl describe <resource-type> <resource-name>

# Check resource usage
kubectl top pods
kubectl top nodes

# View cluster events
kubectl get events --all-namespaces
```

## Production Considerations

### Security
- Use Kubernetes secrets for sensitive data
- Implement network policies
- Enable RBAC for service accounts
- Use non-root containers

### Performance
- Tune resource requests and limits
- Use node affinity for GPU workloads
- Implement proper health checks
- Monitor resource utilization

### Reliability
- Use multiple replicas for inference service
- Implement proper backup strategies
- Use persistent storage for model artifacts
- Monitor application health

## Cleanup

To remove all deployed resources:

```bash
# Delete all resources
kubectl delete -f k8s/
kubectl delete -f monitoring/

# Clean up persistent volumes
kubectl delete pv model-pv
```

## Next Steps

1. **Customize the Model**: Modify `training/train.py` for your specific use case
2. **Add More Metrics**: Extend monitoring with custom metrics
3. **Implement CI/CD**: Set up automated deployment pipelines
4. **Add More Models**: Support multiple model types
5. **Enhance Security**: Implement authentication and authorization

