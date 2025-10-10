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


def create_preprocessor():
    """
    Create a ColumnTransformer for preprocessing numeric and categorical features.
    """
    # Numeric preprocessing: median imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: handle missing values + encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, []),  # Placeholder, will set columns later
            ('cat', categorical_transformer, [])  # Placeholder, will set columns later
        ],
        remainder='passthrough'  # Keep any remaining columns as-is
    )
    
    return preprocessor

def model_train():
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

    
    # Define selected predictors first
    predictors = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    
    # Prepare features and target
    X = train_data[predictors]  # Use only selected predictors
    y = train_data['LogSalePrice']
    
    # Identify numeric and categorical columns from selected predictors
    num_predictors = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_predictors = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Selected numeric predictors: {num_predictors}")
    print(f"Selected categorical predictors: {cat_predictors}")
    
  
    mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "preprocessing": "ColumnTransformer"
        })
    
    """Train a Random Forest Regressor."""
    model_pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    mlflow.log_params({
            "n_samples": len(X),
            "n_features": len(X.columns),
            "predictors": predictors,
            "target": "LogSalePrice"
        })
    
   
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
    
    if cross_val_metrics['r2'] > 0.8:  # Log only if accuracy is reasonable
            mlflow.sklearn.log_model(
                model_pipeline, 
                "model",
                registered_model_name="ames-housing-random-forest"
         )
    # Get feature names after preprocessing
    try:
        # Get feature names from the preprocessor
        feature_names = (num_predictors + 
                        [f"cat_{i}" for i in range(len(model_pipeline.named_steps['regressor'].feature_importances_) - len(num_predictors))])
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(model_pipeline.named_steps['regressor'].feature_importances_)],
            'Importance': model_pipeline.named_steps['regressor'].feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
    except Exception as e:
        print(f"Could not display feature importance: {e}")
        print(f"Number of features after preprocessing: {len(model_pipeline.named_steps['regressor'].feature_importances_)}")

    # Save the model
    joblib.dump(model_pipeline, 'trained_model.pkl')
    print("Model saved as 'trained_model.pkl'")
    mlflow.end_run()

    return model_pipeline

if __name__ == "__main__":
    create_preprocessor()
    model_train()
