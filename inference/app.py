#!/usr/bin/env python3
"""
NLP Inference API for Natural Language Inference using Flan-T5
Takes premise + hypothesis pairs and returns entailment/contradiction/neutral predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import os
from model_loader import model_loader
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the API app
app = FastAPI(
    title="Kube-AI-Pipeline NLP Inference API",
    description="Natural Language Inference API using Flan-T5 fine-tuned on MultiNLI",
    version="2.0.0"
)

# Request model for prediction
class NLIRequest(BaseModel):
    premise: str
    hypothesis: str

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts up"""
    logger.info("Starting Kube-AI-Pipeline NLP Inference API...")
    
    # Try to load the model
    if not model_loader.load_model():
        logger.error("Failed to load model on startup")
        raise RuntimeError("Model loading failed")
    
    logger.info("Model loaded successfully. API ready to serve requests.")

@app.get("/")
async def root():
    """Basic info about the API"""
    return {
        "message": "Kube-AI-Pipeline NLP Inference API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model_loader.is_loaded(),
        "task": "Natural Language Inference (MultiNLI)",
        "model": "Flan-T5"
    }

@app.get("/health")
async def health_check():
    """Check if everything is working"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "device": str(model_loader.device)
    }

@app.post("/predict")
async def predict_nli(request: NLIRequest):
    """
    Predict natural language inference for a premise-hypothesis pair
    
    Returns: entailment, contradiction, or neutral
    """
    try:
        # Validate input
        if not request.premise or not request.hypothesis:
            raise HTTPException(status_code=400, detail="Both premise and hypothesis are required")
        
        if len(request.premise.strip()) == 0 or len(request.hypothesis.strip()) == 0:
            raise HTTPException(status_code=400, detail="Premise and hypothesis cannot be empty")
        
        # Get prediction
        result = model_loader.predict(request.premise, request.hypothesis)
        
        logger.info(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
        
        return JSONResponse(content={
            "success": True,
            "premise": request.premise,
            "hypothesis": request.hypothesis,
            "prediction": {
                "label": result['label'],
                "confidence": result['confidence']
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the model"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": model_loader.model_path,
        "device": str(model_loader.device),
        "model_type": "Flan-T5",
        "task": "Natural Language Inference",
        "dataset": "MultiNLI",
        "labels": model_loader.label_names,
        "num_labels": len(model_loader.label_names),
        "model_loaded": True
    }

@app.get("/metrics")
async def get_metrics():
    """Basic metrics about the API"""
    return {
        "api_version": "2.0.0",
        "model_loaded": model_loader.is_loaded(),
        "device": str(model_loader.device),
        "status": "operational",
        "task": "Natural Language Inference"
    }

if __name__ == "__main__":
    # Get config from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
