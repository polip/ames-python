from google.cloud import storage
import logging
import joblib
import numpy as np
import pandas as pd
import os
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_CLOUD_PROJECT = 'ames-housing-472418'
GOOGLE_CLOUD_REGION = 'europe-west12'
BUCKET_NAME = 'scikit-models'
MODEL_PATH = 'ames-rf-model.pkl'
LOCAL_MODEL_DIR = 'model'

# Global variables for model and features
model = None
feature_names = None

def load_model_from_gcs(bucket_name=BUCKET_NAME, model_path=MODEL_PATH):
    """Load model directly from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        
        # Load model directly from bytes
        model_data = blob.download_as_bytes()
        model = joblib.load(io.BytesIO(model_data))
        
        logger.info("Model loaded directly from GCS")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from GCS: {e}")
        # Fallback to local model
        try:
            local_path = os.path.join(LOCAL_MODEL_DIR, model_path)
            if os.path.exists(local_path):
                model = joblib.load(local_path)
                logger.info("Model loaded from local fallback")
                return model
        except Exception as local_e:
            logger.error(f"Local fallback failed: {local_e}")
        
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    global model, feature_names
    
    # Startup
    logger.info("🚀 Starting Ames Housing FastAPI application...")
    logger.info(f"   - Project: {GOOGLE_CLOUD_PROJECT}")
    logger.info(f"   - Bucket: {BUCKET_NAME}")
    logger.info(f"   - Model Path: {MODEL_PATH}")
    
    # Load model (remove invalid await)
    model = load_model_from_gcs()
    
    if model is None:
        logger.error("Failed to load model!")
        raise RuntimeError("Could not load model")
    
    # Set feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        # Default feature names based on your training
        feature_names = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
            'TotalBsmtSF', 'FirstFlrSF', 'YearBuilt', 'YearRemodAdd', 
            'FullBath', 'TotRmsAbvGrd'
        ]
    
    logger.info(f"✅ Model loaded successfully with {len(feature_names)} features")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="🏠 Ames Housing Price Prediction API",
    description="""
    ## Ames Housing Price Prediction API
    
    This API uses a Random Forest model to predict house prices in Ames, Iowa.
    
    ### Configuration:
    - **Bucket**: `scikit-models`
    - **Model Path**: `ames-rf-model.pkl`
    - **Local Fallback**: `model/` directory
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

## Pydantic models
class HouseFeatures(BaseModel):
    """Input features for house price prediction"""
    OverallQual: int = Field(7, description="Overall material and finish quality (1-10)")
    GrLivArea: int = Field(1500, description="Above ground living area square feet")
    GarageCars: float = Field(2.0, description="Size of garage in car capacity")
    GarageArea: float = Field(500.0, description="Size of garage in square feet")
    TotalBsmtSF: float = Field(1000.0, description="Total square feet of basement area")
    FirstFlrSF: int = Field(1000, description="First Floor square feet")
    YearBuilt: int = Field(1990, description="Original construction date")
    YearRemodAdd: int = Field(1990, description="Remodel date")
    FullBath: int = Field(2, description="Full bathrooms above grade")
    TotRmsAbvGrd: int = Field(7, description="Total rooms above grade")

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 8,
                "GrLivArea": 2000,
                "GarageCars": 2.0,
                "GarageArea": 600.0,
                "TotalBsmtSF": 1200.0,
                "FirstFlrSF": 1200,
                "YearBuilt": 2000,
                "YearRemodAdd": 2000,
                "FullBath": 3,
                "TotRmsAbvGrd": 8
            }
        }

class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price: float
    confidence_interval: Optional[dict] = None
    feature_importance: Optional[dict] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    features_count: Optional[int] = None

# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Ames Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        features_count=len(feature_names) if feature_names else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on input features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([features.model_dump()])
        
        # Ensure columns are in the right order (important!)
        if feature_names:
            input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Calculate confidence interval (using tree variance)
        confidence_interval = None
        if hasattr(model, 'estimators_'):
            # Get predictions from all trees for confidence estimation
            tree_predictions = [tree.predict(input_data)[0] for tree in model.estimators_]
            std_prediction = np.std(tree_predictions)
            confidence_interval = {
                "lower_bound": float(predicted_price - 1.96 * std_prediction),
                "upper_bound": float(predicted_price + 1.96 * std_prediction),
                "confidence_level": 0.95
            }
        
        # Get feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_') and feature_names:
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance (top 5)
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True)[:5])
            feature_importance = sorted_importance
        
        return PredictionResponse(
            predicted_price=float(predicted_price),
            confidence_interval=confidence_interval,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":

    ## Get port from environment variable or default to 8080
    
    port=int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


