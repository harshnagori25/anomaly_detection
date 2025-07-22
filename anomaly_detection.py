import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os

# Load your dataset
df = pd.read_csv("unlabeled_cleaned.csv")  # change filename if needed

# Encode categorical columns (if any)
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ---------- Isolation Forest ----------
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_if = iso_forest.fit_predict(X_scaled)
y_pred_if = [1 if x == -1 else 0 for x in y_pred_if]

# Add predictions to dataframe
df['anomaly_if'] = y_pred_if

# ---------- Autoencoder ----------
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=256, validation_split=0.2)

# Get reconstruction error
X_reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
threshold = np.percentile(mse, 90)  # Top 10% anomalies
y_pred_ae = [1 if e > threshold else 0 for e in mse]

# Add to DataFrame
df['anomaly_ae'] = y_pred_ae

# Save output and model
os.makedirs("model", exist_ok=True)
autoencoder.save("model/autoencoder_model.h5")
df.to_csv("anomaly_detected_output.csv", index=False)

print("✅ Done. Anomaly predictions saved in 'anomaly_detected_output.csv'")
