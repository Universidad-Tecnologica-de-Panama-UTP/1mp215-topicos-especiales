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
dbscan = DBSCAN(eps=3.0, min_samples=5, n_jobs=-1)
cluster_labels = dbscan.fit_predict(X_reduced)

# Print cluster and noise stats
label_counts = collections.Counter(cluster_labels)
print("Conteo de etiquetas (incluyendo ruido):", label_counts)

unique_labels = sorted(set(cluster_labels))
n_clusters = len([label for label in unique_labels if label != -1])
print(f"Detected clusters (excluding noise): {n_clusters}")

# Visualize 100 samples from the first detected cluster (excluding noise)
cluster_ids = [label for label in unique_labels if label != -1]
if len(cluster_ids) == 0:
    print("No clusters found.")
else:
    selected_cluster = cluster_ids[0]  # you can change this if needed
    cluster_indices = np.where(cluster_labels == selected_cluster)[0]
    n_to_show = min(100, len(cluster_indices))

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle(f"100 samples from DBSCAN cluster {selected_cluster}", fontsize=14)

    for i in range(n_to_show):
        ax = axes[i // 10, i % 10]
        ax.imshow(X_test[cluster_indices[i]].reshape(28, 28), cmap='gray')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
