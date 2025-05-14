import pickle
import os
import streamlit as st

## LOADING MODEL & BOXCOX LAMBDA VALUES FROM PICKLE

@st.cache_resource
def load_models():
    with open(os.path.join('notebooks', 'models.pkl'), 'rb') as f:
        models = pickle.load(f)
    return models

@st.cache_data
def load_lambda_values():
    with open(os.path.join('notebooks', 'lambda_values.pkl'), 'rb') as f:
        lambda_values = pickle.load(f)
    return lambda_values

def run_prediction(model, features):
    return model.predict(features)