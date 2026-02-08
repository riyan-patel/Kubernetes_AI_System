# Setup Instructions for Kubernetes AI System

## Current Status ✅
- ✅ Docker is installed (v28.5.1)
- ✅ kubectl is installed (v1.34.1)
- ⚠️ Docker Desktop needs to be started
- ❌ minikube needs to be installed

## Step-by-Step Setup

### Step 1: Start Docker Desktop
1. Open Docker Desktop from Applications (or Spotlight search "Docker")
2. Wait for Docker to fully start (whale icon in menu bar should be steady)
3. Verify it's running:
   ```bash
   docker ps
   ```
   Should show an empty list (not an error)

### Step 2: Install Minikube
Run this command in your terminal:
```bash
brew install minikube
```

### Step 3: Start Minikube Cluster
```bash
# Start minikube with recommended resources
minikube start --memory=4096 --cpus=2 --driver=docker

# Verify cluster is running
minikube status
kubectl get nodes
```

### Step 4: Build Docker Images
Navigate to your project directory:
```bash
cd /Users/riyanpatel/Documents/Kubernetes_AI_System

# Build training image (takes 10-20 minutes first time)
docker build -t ai-training:latest ./training/

# Build inference image
docker build -t ai-inference:latest ./inference/
```

### Step 5: Load Images into Minikube
Minikube uses its own Docker daemon, so you need to load images:
```bash
# Option 1: Load existing images
minikube image load ai-training:latest
minikube image load ai-inference:latest

# Option 2: Build directly in minikube's Docker
eval $(minikube docker-env)
docker build -t ai-training:latest ./training/
docker build -t ai-inference:latest ./inference/
eval $(minikube docker-env -u)
```

### Step 6: Deploy to Kubernetes
```bash
# Deploy all components
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/training-job.yaml
kubectl apply -f k8s/hpa.yaml

# Or use the Makefile
make deploy
```

### Step 7: Check Status
```bash
# Check pods
kubectl get pods

# Check services
kubectl get services

# Check jobs
kubectl get jobs

# Watch pods (Ctrl+C to exit)
kubectl get pods -w
```

### Step 8: Wait for Training to Complete
Training takes 10-30 minutes depending on your CPU:
```bash
# Watch training logs
kubectl logs -f -l app=ai-training

# Check job status
kubectl get jobs
# Wait until STATUS shows "Completed"
```

### Step 9: Access the API
Once inference pods are ready:
```bash
# Port forward to API
kubectl port-forward service/ai-inference-service 8080:80

# In another terminal, test the API
curl http://localhost:8080/health
curl http://localhost:8080/model/info
```

## Quick Commands Reference

### Check Everything
```bash
# All resources
kubectl get all

# Pods with details
kubectl get pods -o wide

# Logs
kubectl logs -l app=ai-training
kubectl logs -l app=ai-inference
```

### Troubleshooting
```bash
# If pods are stuck
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Restart deployment
kubectl rollout restart deployment/ai-inference

# Delete and redeploy
kubectl delete -f k8s/
kubectl apply -f k8s/
```

### Cleanup
```bash
# Stop minikube
minikube stop

# Delete cluster
minikube delete

# Remove all resources
kubectl delete -f k8s/
```

## Expected Timeline
- Docker startup: 1-2 minutes
- Minikube startup: 2-3 minutes
- Building images: 10-20 minutes (first time)
- Training: 10-30 minutes (CPU) or 2-5 minutes (GPU)
- Total setup time: ~30-60 minutes

## Next Steps After Setup
1. Test the API with an image:
   ```bash
   curl -X POST "http://localhost:8080/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
   ```

2. Monitor autoscaling:
   ```bash
   kubectl get hpa -w
   ```

3. View metrics (if Prometheus is deployed):
   ```bash
   kubectl port-forward service/prometheus-service 9090:9090
   # Open http://localhost:9090 in browser
   ```
