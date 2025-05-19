import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Transaction types used for one-hot encoding
TRANSACTION_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']

# Paths to save/load scaler and feature names
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

def preprocess_training(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the full dataset for training.
    - Drops ID columns
    - One-hot encodes 'type'
    - Adds engineered features
    - Fits scaler and saves feature names + scaler
    """
    df = df.copy()
    
    # Drop string ID columns
    df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore', inplace=True)

    # One-hot encode 'type'
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df['type'] == t).astype(int)
    df.drop('type', axis=1, inplace=True)

    # Add engineered features
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Define consistent column order
    feature_names = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
        'balanceDiffOrig', 'balanceDiffDest'
    ]

    # Save column names
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    # Select features and scale
    X = df[feature_names].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)

    return X_scaled

def preprocess_input(raw_dict: dict) -> np.ndarray:
    """
    Preprocess a single transaction dict for prediction.
    - Loads feature names and scaler
    - One-hot encodes and engineers features
    - Returns a scaled input array
    """
    df = pd.DataFrame([raw_dict])
    df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore', inplace=True)

    # One-hot encode
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df['type'] == t).astype(int)
    df.drop('type', axis=1, inplace=True)

    # Engineer features
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Load feature order and scaler
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Align features
    df = df.reindex(columns=feature_names, fill_value=0)

    return scaler.transform(df.astype(float))
