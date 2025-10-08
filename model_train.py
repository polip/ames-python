import pandas as pd
from sklearn.model_selection import cross_val_score # Added for cross-validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import mlflow as mlflow
import mlflow.sklearn


def preprocess_data_and_train():
    """
    Load data, preprocess it, train a Random Forest model, and evaluate performance.
    """
    # Load the training data
    train_data = pd.read_csv('input/train.csv')
    print(f"Loaded training data: {train_data.shape}")
    
    train_data['LogSalePrice'] = np.log1p(train_data['SalePrice'])

    # Enable MLflow autologging
    mlflow.set_experiment("ames-housing-prices")
    mlflow.start_run()
    print("🔬 MLflow tracking enabled")

    # Identify numeric and categorical columns
    num_cols = train_data.select_dtypes(include=[np.number])
    cat_cols = train_data.select_dtypes(include=['object'])
    
    print(f"Numeric columns: {len(num_cols.columns)}")
    print(f"Categorical columns: {len(cat_cols.columns)}")

     # Numeric preprocessing: median imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: handle missing values + encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols.columns.drop(['SalePrice', 'Id', 'LogSalePrice'])),
            ('cat', categorical_transformer, cat_cols.columns)
        ])
    
    mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "preprocessing": "ColumnTransformer"
        })
    
    """Train a Random Forest Regressor."""
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    # Prepare features and target
    X = train_data.drop(['SalePrice', 'LogSalePrice', 'Id'], axis=1)
    y = train_data['LogSalePrice']

    mlflow.log_params({
            "n_samples": len(X),
            "n_features": len(X.columns)
        })
    
    # Split data for training and validation
    
    # Train the model
    model_pipeline.fit(X=X, y=y)
    
    cross_val_metrics = {
        "mse": -np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')),
        "rmse": np.sqrt(-np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error'))),
        "r2": np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='r2')),
        "mae": -np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error'))
    }
        
    # Log metrics to MLflow
    mlflow.log_metric("mse", cross_val_metrics["mse"]) 
    mlflow.log_metric("rmse", cross_val_metrics["rmse"])
    mlflow.log_metric("r2_score", cross_val_metrics["r2"])
    mlflow.log_metric("mae", cross_val_metrics["mae"])
        
     # Print results
    print(f"Model Performance:")
    print(f"MSE: {cross_val_metrics['mse']:.4f}")
    print(f"RMSE: {cross_val_metrics['rmse']:.4f}")
    print(f"R² Score: {cross_val_metrics['r2']:.4f}")
    print(f"MAE: {cross_val_metrics['mae']:.4f}")
  
        
     # Save the model
    joblib.dump(model_pipeline, 'trained_model.pkl')
    print("Model saved as 'trained_model.pkl'")
    mlflow.end_run()

    return model_pipeline

if __name__ == "__main__":
    preprocess_data_and_train()
