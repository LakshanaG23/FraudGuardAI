import pandas as pd
import numpy as np
import joblib
import os

# Load scaler and full feature names (used during training & scaling)
scaler = joblib.load("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/scaler.pkl")
feature_names = joblib.load("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/feature_names.pkl")

def preprocess_input(raw_dict: dict) -> np.ndarray:
    df = pd.DataFrame([raw_dict])

    # Drop string columns
    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

    # One-hot encode all 5 transaction types (for scaler compatibility)
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    for t in transaction_types:
        df[f"type_{t}"] = int(df['type'].iloc[0] == t)
    df.drop(['type'], axis=1, inplace=True)

    # Feature engineering
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Align DataFrame to match scaler input format
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale features
    X_scaled = scaler.transform(df)

    # Select only the 11 features used by models
    model_features = [
        'step',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'type_CASH_OUT',
        'type_TRANSFER',
        'type_PAYMENT',
        'balanceDiffOrig',
        'balanceDiffDest'
    ]

    # Slice to model input format
    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    X_model_input = scaled_df[model_features].to_numpy()

    return X_model_input
