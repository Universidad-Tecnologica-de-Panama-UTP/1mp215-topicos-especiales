import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import collections

# Load MNIST test data
test_csv_path = "../../datasets/mnist/mnist_test.csv"
mnist_test_df = pd.read_csv(test_csv_path)

# Separate labels and features
if mnist_test_df.shape[1] == 785:
    X_test = mnist_test_df.iloc[:, 1:].values
else:
    X_test = mnist_test_df.values

# Normalize data
X_scaled = StandardScaler().fit_transform(X_test)

# Reduce dimensionality
pca = PCA(n_components=30, random_state=42)
X_reduced = pca.fit_transform(X_scaled)

# Apply DBSCAN
dbscan = DBSCAN(eps=8.5, min_samples=5, n_jobs=-1)
cluster_labels = dbscan.fit_predict(X_reduced)

# Print cluster and noise stats
label_counts = collections.Counter(cluster_labels)
print("Conteo de etiquetas (incluyendo ruido):", label_counts)

unique_labels = sorted(set(cluster_labels))
n_clusters = len([label for label in unique_labels if label != -1])
print(f"Detected clusters (excluding noise): {n_clusters}")

# Visualize clusters (up to 10 clusters, 10 samples each)
n_display_clusters = min(n_clusters, 10)
if n_display_clusters > 0:
    fig, axes = plt.subplots(nrows=n_display_clusters, ncols=10, figsize=(10, n_display_clusters))
    if n_display_clusters == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle("DBSCAN Clusters (up to 10) - MNIST Test Set", fontsize=14)
    displayed = 0
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        if displayed >= 10:
            break
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        selected_indices = cluster_indices[:10]
        for i, sample_idx in enumerate(selected_indices):
            ax = axes[displayed, i]
            ax.imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
        displayed += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
else:
    print("No clusters found to display.")

# Visualize up to 25 outliers
outlier_indices = np.where(cluster_labels == -1)[0]
n_outliers_to_show = min(25, len(outlier_indices))

if n_outliers_to_show > 0:
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    fig.suptitle("Outliers detectados por DBSCAN", fontsize=14)

    for i in range(n_outliers_to_show):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[outlier_indices[i]].reshape(28, 28), cmap='gray')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("No hay outliers detectados.")
