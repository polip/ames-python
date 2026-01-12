import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

def create_preprocessor(numeric_features, categorical_features):
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
    
    # Combine both preprocessors with actual column names
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor

def model_train():
    """
    Load data, preprocess it, train a Random Forest model with feature selection.
    """
    # Load the training data
    train_data = pd.read_csv('input/train.csv')
    print(f"Loaded training data: {train_data.shape}")
    
    # Transform the target variable with log10
    train_data['LogSalePrice'] = np.log10(train_data['SalePrice'])

    # Enable MLflow
    mlflow.set_experiment("ames-housing-prices")
    mlflow.start_run()
    print("🔬 MLflow tracking enabled")

            
    # Prepare features and target
    X = train_data.drop(columns=['LogSalePrice', 'SalePrice','Id']) # drop target and ID columns
    y = train_data['LogSalePrice']
    
    # Identify numeric and categorical columns from selected predictors
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Log parameters
    mlflow.log_params({
        "n_samples": len(X),
        "n_features": len(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "LogSalePrice"
    })
    
    # Model parameters
    model_params = {
        'n_estimators': 100,  # Increased for better performance
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create the pipeline with proper feature selection
    model_pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor(numeric_features, categorical_features)),
        ('feature_selector', SelectFromModel(
            RandomForestRegressor(n_estimators=50, random_state=42),  # ✅ Fixed: Use Regressor
            threshold='median',  # Select features above median importance
            max_features=10  # Limit to top 10 features
        )),
        ('regressor', RandomForestRegressor(**model_params))
    ])
    
    # Log model parameters
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "preprocessing": "ColumnTransformer + SelectFromModel",
        "feature_selection": "SelectFromModel with median threshold, only top 10 features",
        **model_params
    })
    
        
    
    # Cross-validation metrics
    cross_val_metrics = {
            "mse": -np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')),
            "rmse": np.sqrt(-np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error'))),
            "r2": np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='r2')),
            "mae": -np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error'))
        }
    
    mlflow.log_metric("mse", cross_val_metrics["mse"]) 
    mlflow.log_metric("rmse", cross_val_metrics["rmse"])
    mlflow.log_metric("r2_score", cross_val_metrics["r2"])
    mlflow.log_metric("mae", cross_val_metrics["mae"])

    # Train the model
    print("🏃 Training model...")
    model_pipeline.fit(X, y)
    print("✅ Model training complete.")
    
    # Feature selection analysis
    selected_features_mask = model_pipeline.named_steps['feature_selector'].get_support()
    
    # Get feature names after preprocessing
    try:
        # Get feature names from preprocessor
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Get transformed feature names
        numeric_names = numeric_features
        categorical_names = []
        
        if categorical_features:
            # Get categorical feature names after one-hot encoding
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            categorical_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        
        all_feature_names = numeric_names + categorical_names
        selected_feature_names = [name for name, selected in zip(all_feature_names, selected_features_mask) if selected]
        
        print(f"\n🎯 Selected Features ({len(selected_feature_names)} out of {len(all_feature_names)}):")
        for feature in selected_feature_names:
            print(f"  ✅ {feature}")
            
        # Log selected features
        mlflow.log_param("selected_features", selected_feature_names)
        mlflow.log_param("n_selected_features", len(selected_feature_names))
        
        # Feature importance of final model
        if hasattr(model_pipeline.named_steps['regressor'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': selected_feature_names,
                'Importance': model_pipeline.named_steps['regressor'].feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            print(f"\n🔝 Top Feature Importance (Selected Features):")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
            plt.title('Top 10 Feature Importances (After Selection)')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_selected.png', dpi=300, bbox_inches='tight')
            
            # Log plot to MLflow
            mlflow.log_artifact('feature_importance_selected.png')
            plt.show()
            
    except Exception as e:
        print(f"⚠️ Could not analyze feature selection: {e}")

    # Create input example for MLflow
    input_example = X.iloc[:5]  # First 5 rows as example
    
    # Save model if performance is good
    if  cross_val_metrics['r2'] > 0.85:
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="ames-housing-random-forest"
        )
        
        # Save locally
        import os
        os.makedirs('model', exist_ok=True)
        joblib.dump(model_pipeline, 'model/ames_rf_model_selected.pkl')
        joblib.dump(selected_feature_names, 'model/selected_feature_names.pkl')
        print("💾 Model saved locally")
        
        # Upload to GCS
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket('scikit-models')
            blob = bucket.blob('ames-rf-model.pkl')
            blob.upload_from_filename('model/ames_rf_model_selected.pkl')
            print("☁️ Model uploaded to Google Cloud Storage")
        except Exception as e:
            print(f"⚠️ Could not upload to GCS: {e}")
    else:
        print(f"⚠️ Model performance too low (R² = {cross_val_score['r2']:.4f}), not saving")
    
    # End MLflow run
    mlflow.end_run()
    
    return model_pipeline

if __name__ == "__main__":
    model_pipeline = model_train()