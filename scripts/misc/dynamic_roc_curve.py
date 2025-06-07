import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.widgets import Slider
import numpy as np

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir el conjunto de datos en características y objetivo
X = df.drop(columns=['Outcome'])  # Reemplazar 'target' con el nombre real de la columna objetivo
y = df['Outcome']  # Reemplazar 'target' with the actual name of the target column

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predecir probabilidades para el conjunto de prueba
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Convertir thresholds a un array de NumPy para operaciones matemáticas
thresholds = np.array([i / 100 for i in range(0, 101, 5)])

# Calcular métricas para diferentes umbrales
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    accuracy_scores.append(accuracy_score(y_test, y_pred_threshold))
    precision_scores.append(precision_score(y_test, y_pred_threshold))
    recall_scores.append(recall_score(y_test, y_pred_threshold))
    f1_scores.append(f1_score(y_test, y_pred_threshold))

# Calcular especificidad y sensibilidad para diferentes umbrales
specificity_scores = [1 - fpr[i] for i in range(len(fpr))]
sensitivity_scores = tpr

# Graficar la curva ROC
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

# Graficar curva ROC
roc_line, = ax.plot(fpr, tpr, color='blue', label='Curva ROC')

# Agregar línea vertical para el deslizador
vertical_line = ax.axvline(x=0, color='gray', linestyle='--')

# Configurar gráfico
ax.set_xlabel('1 - Especificidad (FPR)')
ax.set_ylabel('Sensibilidad (TPR)')
ax.set_title('Curva ROC con Umbral Ajustable')
ax.legend(loc='lower right')
ax.grid()

# Ajustar el rango del deslizador al rango completo de valores de fpr
slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
threshold_slider = Slider(slider_ax, 'Umbral', fpr[0], fpr[-1], valinit=fpr[0], valstep=(fpr[-1] - fpr[0]) / len(fpr))

# Función de actualización para el deslizador
def update(val):
    threshold = threshold_slider.val
    
    # Encontrar el índice más cercano para el umbral actual
    idx = (np.abs(fpr - threshold)).argmin()
    
    # Actualizar la línea vertical
    vertical_line.set_xdata(fpr[idx])
    
    # Actualizar la leyenda con especificidad y sensibilidad en el umbral actual
    ax.legend([
        f'Especificidad: {specificity_scores[idx]:.3f}',
        f'Sensibilidad: {sensitivity_scores[idx]:.3f}'
    ], loc='lower right')
    
    fig.canvas.draw_idle()

threshold_slider.on_changed(update)

plt.show()

