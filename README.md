# 🚨 Anomaly Detection in Network Traffic

This project uses unsupervised machine learning techniques like Isolation Forest and Autoencoder to detect anomalous patterns in network traffic using the KDD Cup 1999 dataset.

## 📁 Folder Structure
- `notebooks/` – Exploratory analysis and model training
- `app/` – Streamlit dashboard
- `report/` – PDF report for submission
- `model/` – Trained models

## 📊 Techniques Used
- Isolation Forest
- Autoencoder Neural Network

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 📚 Dataset
[KDD Cup 1999 - Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

## ✨ Output
- Prediction of normal vs anomalous traffic
- Visual analysis of reconstruction error

## 📑 Report
See [report/anomaly_detection_report.pdf](report/anomaly_detection_report.pdf)
