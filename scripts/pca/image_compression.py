import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar los datos de prueba de MNIST
df = pd.read_csv("../../datasets/mnist/mnist_test.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA conservando el 95% de la varianza
pca = PCA(n_components=0.99, random_state=42)
X_compressed = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_compressed)
print(f"Dimensión original: {X.shape}")
print(f"Dimensión comprimida (PCA): {X_compressed.shape}")

# Desnormalizar para visualizar
X_reconstructed_original = scaler.inverse_transform(X_reconstructed)

# Graficar 25 imágenes originales y reconstruidas
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(12, 6))
fig.suptitle("MNIST - Originales vs Reconstruidas (PCA 95%)", fontsize=16)

for i in range(25):
    # Imagen original
    ax1 = axes[i // 5, (i % 5) * 2]
    ax1.imshow(X[i].reshape(28, 28), cmap='gray')
    ax1.set_title(f"Original")
    ax1.axis('off')

    # Imagen reconstruida
    ax2 = axes[i // 5, (i % 5) * 2 + 1]
    ax2.imshow(X_reconstructed_original[i].reshape(28, 28), cmap='gray')
    ax2.set_title(f"Reconstruida")
    ax2.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
