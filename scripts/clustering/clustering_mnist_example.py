import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load test data
test_csv_path = "../../datasets/mnist/mnist_test.csv"
mnist_test_df = pd.read_csv(test_csv_path)

# Separate labels and features (in case the first column is label)
if mnist_test_df.shape[1] == 785:
    X_test = mnist_test_df.iloc[:, 1:].values  # pixel values
else:
    X_test = mnist_test_df.values

# Fit KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_test)

# Create figure
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
fig.suptitle("10 Clusters - 10 Samples per Cluster (MNIST Test Set)", fontsize=14)

# For each cluster, plot 10 images
for cluster_id in range(10):
    # Get indices of samples in this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    # Take the first 10
    selected_indices = cluster_indices[:10]
    
    for i, sample_idx in enumerate(selected_indices):
        ax = axes[cluster_id, i]
        ax.imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
