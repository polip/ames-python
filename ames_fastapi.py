import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_CLOUD_PROJECT = 'ames-housing-472418'
GOOGLE_CLOUD_REGION = 'europe-west12'
BUCKET_NAME = 'scikit-models'
MODEL_PATH = 'ames-rf-model.pkl'
MODEL_META_PATH = 'ames-rf-model.meta.json'
LOCAL_MODEL_DIR = 'model'

# Global variables for model, feature schema and serving contract
model = None
feature_names = None
model_meta: dict | None = None
feature_defaults: dict | None = None


def load_model_from_gcs(bucket_name=BUCKET_NAME, model_path=MODEL_PATH):
    """Load model directly from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Load model directly from bytes
        model_data = blob.download_as_bytes()
        model = joblib.load(io.BytesIO(model_data))

        logger.info('Model loaded directly from GCS')
        return model

    except Exception as e:
        logger.error(f'Error loading model from GCS: {e}')
        # Fallback to local model
        try:
            local_path = os.path.join(LOCAL_MODEL_DIR, model_path)
            if os.path.exists(local_path):
                model = joblib.load(local_path)
                logger.info('Model loaded from local fallback')
                return model
        except Exception as local_e:
            logger.error(f'Local fallback failed: {local_e}')

        return None


def load_model_meta_from_gcs(bucket_name=BUCKET_NAME, meta_path=MODEL_META_PATH):
    """Load the serving contract (feature schema + target transform) from GCS.

    Contains: `features`, `target_transform` (`identity` | `log10`),
    `feature_defaults`. Missing meta means a legacy log10-target model.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(meta_path)
        meta = json.loads(blob.download_as_bytes())
        logger.info('Model meta loaded directly from GCS')
        return meta
    except Exception as e:
        logger.warning(f'Could not load model meta from GCS: {e}')
        try:
            local_path = os.path.join(LOCAL_MODEL_DIR, meta_path)
            if os.path.exists(local_path):
                meta = json.loads(Path(local_path).read_text())
                logger.info('Model meta loaded from local fallback')
                return meta
        except Exception as local_e:
            logger.error(f'Local meta fallback failed: {local_e}')
        return None


def price_from_raw(raw: float) -> float:
    """Convert a raw model output to dollars, respecting the trained target."""
    if model_meta is not None and model_meta.get('target_transform') == 'identity':
        return float(raw)  # pipeline already inverts log10 internally
    return float(10**raw)  # legacy models trained on log10(SalePrice)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    global model, feature_names, model_meta, feature_defaults

    # Startup
    logger.info('🚀 Starting Ames Housing FastAPI application...')
    logger.info(f'   - Project: {GOOGLE_CLOUD_PROJECT}')
    logger.info(f'   - Bucket: {BUCKET_NAME}')
    logger.info(f'   - Model Path: {MODEL_PATH}')

    # Load training-serving contract first, then the model itself
    model_meta = load_model_meta_from_gcs()
    if model_meta is not None:
        feature_names = model_meta.get('features')
        feature_defaults = model_meta.get('feature_defaults')
        logger.info(
            f'🧾 Serving contract: {model_meta.get("model_class")}, '
            f'target_transform={model_meta.get("target_transform")}'
        )
    else:
        logger.warning('No model meta found — assuming legacy log10-target model')

    # Load model (remove invalid await)
    model = load_model_from_gcs()

    if model is None:
        logger.error('Failed to load model!')
        raise RuntimeError('Could not load model')

    # Set feature names from schema, then from the model, then from defaults
    if not feature_names:
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            # Default feature names based on your training
            feature_names = [
                'OverallQual',
                'GrLivArea',
                'GarageCars',
                'GarageArea',
                'TotalBsmtSF',
                'FirstFlrSF',
                'YearBuilt',
                'YearRemodAdd',
                'FullBath',
                'TotRmsAbvGrd',
            ]

    logger.info(f'✅ Model loaded successfully with {len(feature_names)} features')

    yield  # Application runs here

    # Shutdown
    logger.info('🛑 Application shutdown complete')


# Create FastAPI app
app = FastAPI(
    title='🏠 Ames Housing Price Prediction API',
    description="""
    ## Ames Housing Price Prediction API

    This API uses a Random Forest model to predict house prices in Ames, Iowa.

    ### Configuration:
    - **Bucket**: `scikit-models`
    - **Model Path**: `ames-rf-model.pkl`
    - **Local Fallback**: `model/` directory
    """,
    version='1.0.0',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
)


## Pydantic models
class HouseFeatures(BaseModel):
    """Input features for house price prediction"""

    OverallQual: int = Field(7, description='Overall material and finish quality (1-10)')
    GrLivArea: int = Field(1500, description='Above ground living area square feet')
    GarageCars: float = Field(2.0, description='Size of garage in car capacity')
    GarageArea: float = Field(500.0, description='Size of garage in square feet')
    TotalBsmtSF: float = Field(1000.0, description='Total square feet of basement area')
    FirstFlrSF: int = Field(1000, description='First Floor square feet')
    YearBuilt: int = Field(1990, description='Original construction date')
    YearRemodAdd: int = Field(1990, description='Remodel date')
    FullBath: int = Field(2, description='Full bathrooms above grade')
    TotRmsAbvGrd: int = Field(7, description='Total rooms above grade')

    model_config = {
        'json_schema_extra': {
            'example': {
                'OverallQual': 8,
                'GrLivArea': 2000,
                'GarageCars': 2.0,
                'GarageArea': 600.0,
                'TotalBsmtSF': 1200.0,
                'FirstFlrSF': 1200,
                'YearBuilt': 2000,
                'YearRemodAdd': 2000,
                'FullBath': 3,
                'TotRmsAbvGrd': 8,
            },
        },
    }


class PredictionResponse(BaseModel):
    """Response model for price prediction"""

    predicted_price: float
    confidence_interval: dict | None = None
    feature_importance: dict | None = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    features_count: int | None = None


# API endpoints
@app.get('/', response_model=dict)
async def root():
    """Root endpoint"""
    return {
        'message': 'Ames Housing Price Prediction API',
        'version': '1.0.0',
        'endpoints': {'predict': '/predict', 'health': '/health', 'docs': '/docs'},
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status='healthy' if model is not None else 'unhealthy',
        model_loaded=model is not None,
        features_count=len(feature_names) if feature_names else None,
    )


@app.post('/predict', response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on input features"""
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([features.model_dump()])

        # Resolve the full feature vector: user-provided values first, training
        # defaults (from the model meta) for anything the client did not send.
        provided = features.model_dump()
        defaults = feature_defaults if feature_defaults is not None else {}
        if feature_names:
            resolved = {}
            for col in feature_names:
                if col in provided:
                    resolved[col] = provided[col]
                elif col in defaults:
                    resolved[col] = defaults[col]
                else:
                    resolved[col] = 0  # schema drift: previous behaviour
            input_data = pd.DataFrame([resolved])

        # Make prediction. `model_meta` decides whether the output is already in
        # dollars (identity target transform) or legacy log10(SalePrice).
        predicted_price = price_from_raw(model.predict(input_data)[0])

        # Calculate confidence interval (using tree variance)
        confidence_interval = None
        if hasattr(model, 'estimators_'):
            # Get predictions from all trees for confidence estimation
            tree_predictions = [tree.predict(input_data)[0] for tree in model.estimators_]
            std_prediction = np.std(tree_predictions)
            if model_meta is not None and model_meta.get('target_transform') == 'identity':
                lower_bound, upper_bound = (
                    predicted_price - 1.96 * std_prediction,
                    predicted_price + 1.96 * std_prediction,
                )
            else:
                lower_bound = 10 ** (predicted_price - 1.96 * std_prediction)
                upper_bound = 10 ** (predicted_price + 1.96 * std_prediction)
            confidence_interval = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': 0.95,
            }

        # Get feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_') and feature_names:
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance (top 5)
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            )
            feature_importance = sorted_importance

        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
        )

    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(status_code=400, detail=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    ## Get port from environment variable or default to 8080

    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
