import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load the MNIST dataset
df = pd.read_csv('../../datasets/mnist/mnist_train.csv')

# Separate features and label
X = df.drop('label', axis=1)
y = df['label']

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Plot the 3D PCA
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='tab10', alpha=0.7)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D PCA of MNIST Dataset')

legend1 = ax.legend(*scatter.legend_elements(), title="Digit")
ax.add_artist(legend1)

plt.show()
