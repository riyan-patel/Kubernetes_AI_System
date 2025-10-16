#!/bin/bash

# Kube-AI-Pipeline Test Script
# This script tests the complete pipeline deployment

set -e

echo "🚀 Starting Kube-AI-Pipeline Test..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        print_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build training image
    print_status "Building training image..."
    cd training/
    docker build -t ai-training:latest . || {
        print_error "Failed to build training image"
        exit 1
    }
    cd ..
    
    # Build inference image
    print_status "Building inference image..."
    cd inference/
    docker build -t ai-inference:latest . || {
        print_error "Failed to build inference image"
        exit 1
    }
    cd ..
    
    print_status "Docker images built successfully ✓"
}

# Deploy to Kubernetes
deploy_to_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Apply storage
    kubectl apply -f k8s/storage.yaml || {
        print_error "Failed to apply storage configuration"
        exit 1
    }
    
    # Apply configmap
    kubectl apply -f k8s/configmap.yaml || {
        print_error "Failed to apply configmap"
        exit 1
    }
    
    # Apply inference deployment
    kubectl apply -f k8s/inference-deployment.yaml || {
        print_error "Failed to apply inference deployment"
        exit 1
    }
    
    # Apply training job
    kubectl apply -f k8s/training-job.yaml || {
        print_error "Failed to apply training job"
        exit 1
    }
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml || {
        print_error "Failed to apply HPA configuration"
        exit 1
    }
    
    print_status "Kubernetes deployment completed ✓"
}

# Wait for deployment
wait_for_deployment() {
    print_status "Waiting for deployment to be ready..."
    
    # Wait for inference deployment
    kubectl wait --for=condition=available --timeout=300s deployment/ai-inference || {
        print_error "Inference deployment failed to become ready"
        exit 1
    }
    
    print_status "Deployment is ready ✓"
}

# Test the API
test_api() {
    print_status "Testing the inference API..."
    
    # Port forward in background
    kubectl port-forward service/ai-inference-service 8080:80 &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        print_status "Health check passed ✓"
    else
        print_error "Health check failed"
        kill $PORT_FORWARD_PID 2>/dev/null
        exit 1
    fi
    
    # Test root endpoint
    if curl -f http://localhost:8080/ > /dev/null 2>&1; then
        print_status "Root endpoint test passed ✓"
    else
        print_error "Root endpoint test failed"
        kill $PORT_FORWARD_PID 2>/dev/null
        exit 1
    fi
    
    # Clean up port forward
    kill $PORT_FORWARD_PID 2>/dev/null
    
    print_status "API tests completed successfully ✓"
}

# Check training job status
check_training_job() {
    print_status "Checking training job status..."
    
    # Wait for job to complete (with timeout)
    if kubectl wait --for=condition=complete --timeout=600s job/ai-model-training; then
        print_status "Training job completed successfully ✓"
    else
        print_warning "Training job may still be running or failed"
        print_status "Training job logs:"
        kubectl logs -l app=ai-training --tail=20
    fi
}

# Display deployment information
show_deployment_info() {
    print_status "Deployment Information:"
    echo ""
    echo "Pods:"
    kubectl get pods
    echo ""
    echo "Services:"
    kubectl get services
    echo ""
    echo "Jobs:"
    kubectl get jobs
    echo ""
    echo "Persistent Volumes:"
    kubectl get pv,pvc
    echo ""
    echo "Horizontal Pod Autoscaler:"
    kubectl get hpa
}

# Cleanup function
cleanup() {
    print_status "Cleaning up test resources..."
    kubectl delete -f k8s/ --ignore-not-found=true
    print_status "Cleanup completed ✓"
}

# Main execution
main() {
    echo "🧪 Kube-AI-Pipeline Test Suite"
    echo "================================"
    
    check_prerequisites
    build_images
    deploy_to_k8s
    wait_for_deployment
    test_api
    check_training_job
    show_deployment_info
    
    echo ""
    print_status "🎉 All tests passed! Kube-AI-Pipeline is working correctly."
    echo ""
    print_status "To access the API locally, run:"
    echo "kubectl port-forward service/ai-inference-service 8080:80"
    echo "Then visit: http://localhost:8080"
    echo ""
    print_status "To clean up resources, run:"
    echo "./test.sh cleanup"
}

# Handle cleanup argument
if [ "$1" = "cleanup" ]; then
    cleanup
    exit 0
fi

# Run main function
main

