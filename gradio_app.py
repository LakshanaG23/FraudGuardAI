print("🧪 Starting script...")

import gradio as gr
print("✅ Gradio imported")

import numpy as np
print("✅ NumPy imported")

from preprocessing import preprocess_input
print("✅ Preprocessing imported")

from utils import load_models
print("✅ Utils imported")

print("📦 Loading models...")
autoencoder, xgb_model = load_models()
print("✅ Models loaded")


def predict_fraud(step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                  nameDest, oldbalanceDest, newbalanceDest):
    # Step 1: Construct dictionary
    data = {
        "step": step,
        "type": type,
        "amount": amount,
        "nameOrig": nameOrig,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": 0,              # dummy for schema match
        "isFlaggedFraud": 0        # dummy for schema match
    }

    # Step 2: Preprocess input
    X = preprocess_input(data)

    # Step 3: Get predictions
    reconstructed = autoencoder.predict(X, verbose=0)
    recon_error = np.mean(np.square(X - reconstructed), axis=1)[0]
    xgb_proba = xgb_model.predict_proba(X)[0][1]

    auto_flag = recon_error > 0.01
    xgb_flag = xgb_proba > 0.5

    verdict = "🚨 FRAUD DETECTED" if (auto_flag or xgb_flag) else "✅ Legitimate Transaction"

    return {
        "Autoencoder Loss": round(recon_error, 5),
        "XGBoost Fraud Probability": round(xgb_proba, 5),
        "Fraud by Autoencoder": auto_flag,
        "Fraud by XGBoost": xgb_flag,
        "Final Verdict": verdict
    }


# Step 4: Gradio UI setup
inputs = [
    gr.Number(label="Step"),
    gr.Dropdown(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], label="Transaction Type"),
    gr.Number(label="Amount"),
    gr.Textbox(label="Origin Account ID"),
    gr.Number(label="Old Balance (Origin)"),
    gr.Number(label="New Balance (Origin)"),
    gr.Textbox(label="Destination Account ID"),
    gr.Number(label="Old Balance (Dest)"),
    gr.Number(label="New Balance (Dest)")
]

iface = gr.Interface(
    fn=predict_fraud,
    inputs=inputs,
    outputs="json",
    title="🛡️ Fraud Detection MVP",
    description="Detect fraud using Autoencoder (anomaly detection) and XGBoost (supervised learning)"
)

# Step 5: Launch
if __name__ == "__main__":
    print("🚀 Launching Gradio app...")
    iface.launch(share=True)
