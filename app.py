# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model

st.title("🔍 Network Traffic Anomaly Detection")

uploaded_file = st.file_uploader("Upload your CSV network traffic file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data Preview", df.head())

    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('target', axis=1, errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_option = st.selectbox("Choose model", ['Isolation Forest', 'Autoencoder'])

    if model_option == 'Isolation Forest':
        clf = IsolationForest(contamination=0.1)
        preds = clf.fit_predict(X_scaled)
        preds = [1 if i == -1 else 0 for i in preds]
    else:
        model = load_model("model/autoencoder_model.h5")
        recon = model.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - recon), axis=1)
        threshold = np.percentile(errors, 90)
        preds = [1 if e > threshold else 0 for e in errors]

    df['Anomaly'] = preds
    st.write("📊 Prediction Results", df['Anomaly'].value_counts())
    st.download_button("📥 Download Results", df.to_csv(index=False), file_name='predictions.csv')
