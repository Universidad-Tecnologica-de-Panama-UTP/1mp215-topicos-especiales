import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Cargar el dataset
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# 2. Eliminar la columna 'Outcome' si existe (no se usa para clustering)
X = df.drop(columns=['Outcome']) if 'Outcome' in df.columns else df

# 3. Aplicar KMeans con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

# 4. Reducir dimensiones a 2D con PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 5. Graficar Clusters vs Etiquetas Reales
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Clusters de KMeans
scatter0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', edgecolor='k', alpha=0.7)
axes[0].set_title('Clusters KMeans (2D PCA)')
axes[0].set_xlabel('Componente principal 1')
axes[0].set_ylabel('Componente principal 2')
axes[0].grid(True)
legend1 = axes[0].legend(*scatter0.legend_elements(), title="Cluster")
axes[0].add_artist(legend1)

# Gráfico 2: Etiquetas reales (Outcome)
if 'Outcome' in df.columns:
    scatter1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Outcome'], cmap='coolwarm', edgecolor='k', alpha=0.7)
    axes[1].set_title('Etiquetas reales de diabetes (2D PCA)')
    axes[1].set_xlabel('Componente principal 1')
    axes[1].set_ylabel('Componente principal 2')
    axes[1].grid(True)
    legend2 = axes[1].legend(*scatter1.legend_elements(), title="Diabetes")
    axes[1].add_artist(legend2)
else:
    axes[1].text(0.5, 0.5, "No se encontró la columna 'Outcome'", ha='center')

plt.tight_layout()
plt.show()

# 6. Mostrar distribución de casos de diabetes por cluster
if 'Outcome' in df.columns:
    conteo = pd.crosstab(df['Cluster'], df['Outcome'], rownames=['Cluster'], colnames=['Diabetes (Outcome)'])
    print("\nDistribución de casos de diabetes en cada cluster:\n")
    print(conteo)
