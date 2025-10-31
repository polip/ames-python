import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import joblib
import os
import sklearn


st.set_page_config(
    page_title="Ames Housing Price Predictor", 
    layout="wide")


def load_model():
    # Load the pre-trained model
    model_path = 'model/ames_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    
def load_feature_names():
    # Load the feature names used in the model
    feature_names_path = 'model/feature_names.pkl'
    if os.path.exists(feature_names_path):
        feature_names = joblib.load(feature_names_path)
        return feature_names


def main():
    st.title("Ames Housing Price Predictor")
    
    model = load_model()
    feature_names = load_feature_names()
    
    st.sidebar.header("Input Features")
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.sidebar.number_input(f"Enter value for {feature}:", value=0.0)
    
    input_df = pd.DataFrame([input_data])
    
    if st.button("Predict Price"):
        prediction = model.predict(input_df)
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()