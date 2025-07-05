import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load MNIST data
train_df = pd.read_csv("../../datasets/mnist/mnist_train.csv")
test_df = pd.read_csv("../../datasets/mnist/mnist_test.csv")

# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Baseline (No PCA) --------
clf_no_pca = LogisticRegression(max_iter=1000)
clf_no_pca.fit(X_train_scaled, y_train)
y_pred_no_pca = clf_no_pca.predict(X_test_scaled)
acc_no_pca = accuracy_score(y_test, y_pred_no_pca)

# -------- PCA (retain 95% variance) --------
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clf_pca = LogisticRegression(max_iter=1000)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

# -------- Results --------
print(f"Accuracy without PCA: {acc_no_pca:.4f}")
print(f"Accuracy with PCA: {acc_pca:.4f}")
print(f"Original dimensions: {X_train.shape[1]}, after PCA: {X_train_pca.shape[1]}")
