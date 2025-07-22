# 🧠 Anomaly Detection in Network Traffic using Unsupervised Learning

This project applies unsupervised machine learning techniques — **Isolation Forests** and **Autoencoders** — to detect anomalies in network traffic data. These anomalies can indicate **potential security breaches**, **intrusions**, or **system malfunctions**.

---

## 📁 Dataset

We use the https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data

- **Training data:** `KDDTrain+.txt`
- **Test data:** `new_unlabeled.csv`, `full_unlabeled.csv` (custom datasets)

---

## 🔍 Techniques Used

### 🔸 Isolation Forest (Scikit-learn)
- Detects anomalies based on how isolated a point is in the feature space.
- Unsupervised and fast for high-dimensional data.

### 🔸 Autoencoder (TensorFlow / Keras)
- A neural network trained to reconstruct normal data.
- High reconstruction error → anomaly.
- Trained without labels (unsupervised).

---

## 🧪 Features

- Categorical encoding with `LabelEncoder`
- Data standardization using `StandardScaler`
- Model saving and reloading
- Thresholding based on 90th percentile of reconstruction error
- Works on **new unlabeled network traffic data**

---

## 📦 Requirements

```bash
pip install -r requirements.txt
