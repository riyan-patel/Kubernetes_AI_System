# makefile for the kube-ai-pipeline
# just some handy commands to build and deploy everything

.PHONY: help build deploy test clean status logs

# show help by default
help:
	@echo "kube-ai-pipeline commands:"
	@echo ""
	@echo "build stuff:"
	@echo "  build-training    build the training docker image"
	@echo "  build-inference   build the inference docker image"
	@echo "  build-all        build both images"
	@echo ""
	@echo "deploy stuff:"
	@echo "  deploy           deploy everything to kubernetes"
	@echo "  deploy-training  just deploy the training job"
	@echo "  deploy-inference just deploy the inference service"
	@echo "  deploy-monitoring deploy prometheus monitoring"
	@echo ""
	@echo "check stuff:"
	@echo "  status           see what's running"
	@echo "  logs             see the logs"
	@echo "  test             run tests"
	@echo "  clean            delete everything"
	@echo ""
	@echo "dev stuff:"
	@echo "  dev-setup        install python deps locally"
	@echo "  port-forward     forward api to localhost"

# build the docker images
build-training:
	@echo "building training image..."
	docker build -t ai-training:latest ./training/

build-inference:
	@echo "building inference image..."
	docker build -t ai-inference:latest ./inference/

build-all: build-training build-inference
	@echo "all images built!"

# deploy to kubernetes
deploy:
	@echo "deploying everything..."
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/inference-deployment.yaml
	kubectl apply -f k8s/training-job.yaml
	kubectl apply -f k8s/hpa.yaml
	@echo "deployment done!"

deploy-training:
	@echo "deploying training job..."
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/training-job.yaml

deploy-inference:
	@echo "deploying inference service..."
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/inference-deployment.yaml
	kubectl apply -f k8s/hpa.yaml

deploy-monitoring:
	@echo "deploying monitoring..."
	kubectl apply -f monitoring/prometheus.yaml

# check what's happening
status:
	@echo "=== pods ==="
	kubectl get pods
	@echo ""
	@echo "=== services ==="
	kubectl get services
	@echo ""
	@echo "=== jobs ==="
	kubectl get jobs
	@echo ""
	@echo "=== hpa ==="
	kubectl get hpa

logs:
	@echo "=== training logs ==="
	kubectl logs -l app=ai-training --tail=20
	@echo ""
	@echo "=== inference logs ==="
	kubectl logs -l app=ai-inference --tail=20

test:
	@echo "running tests..."
	@if [ -f "./test.sh" ]; then \
		chmod +x ./test.sh && ./test.sh; \
	else \
		echo "test script not found"; \
	fi

clean:
	@echo "cleaning up..."
	kubectl delete -f k8s/ --ignore-not-found=true
	kubectl delete -f monitoring/ --ignore-not-found=true
	kubectl delete pv model-pv --ignore-not-found=true
	@echo "cleanup done!"

# dev stuff
dev-setup:
	@echo "setting up dev environment..."
	pip install -r training/requirements.txt
	pip install -r inference/requirements.txt
	@echo "dev setup done!"

port-forward:
	@echo "port forwarding to api..."
	@echo "api will be at: http://localhost:8080"
	kubectl port-forward service/ai-inference-service 8080:80

# test the api
api-test:
	@echo "testing api..."
	@echo "health check:"
	curl -s http://localhost:8080/health || echo "api not accessible. run 'make port-forward' first."
	@echo ""
	@echo "model info:"
	curl -s http://localhost:8080/model/info || echo "api not accessible. run 'make port-forward' first."

# monitoring
monitor-logs:
	@echo "following inference logs..."
	kubectl logs -f deployment/ai-inference

monitor-metrics:
	@echo "port forwarding to prometheus..."
	kubectl port-forward service/prometheus-service 9090:9090

