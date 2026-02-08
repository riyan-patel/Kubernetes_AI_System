# Changes: Flan-T5 + MultiNLI Implementation

## What Changed

### Model & Task
- **Before**: CNN for CIFAR-10 image classification
- **After**: Flan-T5 for MultiNLI natural language inference

### Dataset
- **Before**: CIFAR-10 (images)
- **After**: MultiNLI (text pairs - premise + hypothesis)

### API Changes
- **Before**: Image upload → Class prediction
- **After**: Text input (premise + hypothesis) → Entailment/Contradiction/Neutral

## New API Usage

### Health Check
```bash
curl http://localhost:8080/health
```

### Model Info
```bash
curl http://localhost:8080/model/info
```

### Prediction (Natural Language Inference)
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "A man is playing guitar",
    "hypothesis": "A man is making music"
  }'
```

**Response:**
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

**Entailment:**
```json
{
  "premise": "The cat sat on the mat",
  "hypothesis": "The cat was on the mat"
}
→ {"label": "entailment"}
```

**Contradiction:**
```json
{
  "premise": "The dog is sleeping",
  "hypothesis": "The dog is running"
}
→ {"label": "contradiction"}
```

**Neutral:**
```json
{
  "premise": "The cat sat on the mat",
  "hypothesis": "The cat was hungry"
}
→ {"label": "neutral"}
```

## Training Configuration

- **Model**: `google/flan-t5-base`
- **Batch Size**: 8 (smaller for transformers)
- **Epochs**: 3 (fine-tuning needs fewer epochs)
- **Learning Rate**: 5e-5 (standard for fine-tuning)
- **Max Length**: 512 tokens

## Resource Requirements

### Training
- Memory: 4-8GB (increased for transformers)
- CPU: 1-2 cores

### Inference
- Memory: 2-4GB (increased for transformers)
- CPU: 0.5-1 core

## Model Output

The trained model will be saved to:
```
/app/models/flan_t5_multinli/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.json
└── ...
```

## Next Steps

1. Rebuild Docker images:
   ```bash
   docker build -t ai-training:latest ./training/
   docker build -t ai-inference:latest ./inference/
   ```

2. Load into minikube:
   ```bash
   minikube image load ai-training:latest
   minikube image load ai-inference:latest
   ```

3. Redeploy:
   ```bash
   kubectl delete -f k8s/
   kubectl apply -f k8s/
   ```

4. Wait for training to complete (will take longer - transformers are bigger)

5. Test the API with text inputs!
