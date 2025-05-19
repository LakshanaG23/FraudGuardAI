import pandas as pd
import numpy as np

transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']

def preprocess_input(raw_dict: dict) -> np.ndarray:
    df = pd.DataFrame([raw_dict])

    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

    # One-hot encode 'type'
    for t in transaction_types:
        df[f"type_{t}"] = (df['type'] == t).astype(int)
    df.drop(['type'], axis=1, inplace=True)

    # Add engineered features
    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    ordered_features = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
        'balanceDiffOrig', 'balanceDiffDest'
    ]

    return df[ordered_features].astype(float).values
