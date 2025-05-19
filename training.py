from datasets import load_dataset
from preprocessing import preprocess_training

ds = load_dataset("purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection")
df = ds["train"].to_pandas()

# Preprocess and save scaler + feature_names
X_scaled = preprocess_training(df)
y = df["isFraud"]
