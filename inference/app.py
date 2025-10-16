#!/usr/bin/env python3
"""
simple api server that takes images and tells you what's in them
uses the trained model to make predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import os
from model_loader import model_loader
import uvicorn

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# create the api app
app = FastAPI(
    title="Kube-AI-Pipeline Inference API",
    description="AI Model Inference API for Kube-AI-Pipeline",
    version="1.0.0"
)

# how we preprocess images before feeding them to the model
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

@app.on_event("startup")
async def startup_event():
    """load the model when the server starts up"""
    logger.info("Starting Kube-AI-Pipeline Inference API...")
    
    # try to load the model
    if not model_loader.load_model():
        logger.error("Failed to load model on startup")
        raise RuntimeError("Model loading failed")
    
    logger.info("Model loaded successfully. API ready to serve requests.")

@app.get("/")
async def root():
    """basic info about the api"""
    return {
        "message": "Kube-AI-Pipeline Inference API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loader.is_loaded()
    }

@app.get("/health")
async def health_check():
    """check if everything is working"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "device": str(model_loader.device)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    upload an image and get a prediction back
    
    just send us any image file and we'll tell you what it is
    """
    try:
        # make sure it's actually an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # read the image data
        image_data = await file.read()
        
        # load the image
        image = Image.open(io.BytesIO(image_data))
        
        # convert to rgb if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # preprocess for the model
        image_tensor = transform(image).unsqueeze(0)  # add batch dimension
        
        # get the prediction
        result = model_loader.predict(image_tensor)
        
        logger.info(f"Prediction made for {file.filename}: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "prediction": result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """get some info about the model we're using"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": model_loader.model_path,
        "device": str(model_loader.device),
        "num_classes": len(model_loader.class_names),
        "class_names": model_loader.class_names,
        "model_loaded": True
    }

@app.get("/metrics")
async def get_metrics():
    """basic metrics about the api"""
    return {
        "api_version": "1.0.0",
        "model_loaded": model_loader.is_loaded(),
        "device": str(model_loader.device),
        "status": "operational"
    }

if __name__ == "__main__":
    # get config from environment or use defaults
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
