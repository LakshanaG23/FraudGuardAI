from tensorflow.keras.models import load_model
import pickle

def load_models():
    autoencoder = load_model("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/models/autoencoder_fraud_detection_model.h5",compile=False)
    with open("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/models/xgb_fraud_detection_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    import numpy as np
    for features in [11, 13]:
        try:
            dummy_input = np.random.rand(1, features)
            xgb_model.predict(dummy_input)
            print(f"XGBoost model accepts input shape: (1, {features})")
            break
        except Exception as e:
            print(f"(1, {features}) failed:", e)

    

    return autoencoder, xgb_model

