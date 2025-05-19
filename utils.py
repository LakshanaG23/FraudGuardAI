from tensorflow.keras.models import load_model
import pickle

def load_models():
    autoencoder = load_model("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/models/autoencoder_fraud_detection_model.h5",compile=False)
    with open("/Users/lakshanagopu/Desktop/FraudGuardAI/FraudGuardAI/models/xgb_fraud_detection_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    return autoencoder, xgb_model
