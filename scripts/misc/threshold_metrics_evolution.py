import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.widgets import Slider

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir el conjunto de datos en características y objetivo
X = df.drop(columns=['Outcome'])  # Reemplazar 'target' con el nombre real de la columna objetivo
y = df['Outcome']  # Reemplazar 'target' con el nombre real de la columna objetivo

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predecir probabilidades para el conjunto de prueba
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calcular métricas para diferentes umbrales
thresholds = [i / 100 for i in range(0, 101, 5)]
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

# Graficar la evolución de las métricas en un solo gráfico
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

# Graficar métricas
line_accuracy, = ax.plot(thresholds, accuracy_scores, color='blue', label='Exactitud')
line_precision, = ax.plot(thresholds, precision_scores, color='green', label='Precisión')
line_recall, = ax.plot(thresholds, recall_scores, color='orange', label='Sensibilidad')
line_f1, = ax.plot(thresholds, f1_scores, color='red', label='Puntaje F1')

# Agregar línea vertical para el deslizador
vertical_line = ax.axvline(x=0, color='gray', linestyle='--')

# Configurar gráfico
ax.set_xlabel('Umbral')
ax.set_ylabel('Valor de la Métrica')
ax.set_title('Evolución de las Métricas en Función del Umbral')
ax.legend(loc='lower right')
ax.grid()

# Agregar deslizador
slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
threshold_slider = Slider(slider_ax, 'Umbral', 0, 1, valinit=0, valstep=0.01)

# Función de actualización para el deslizador
def update(val):
    threshold = threshold_slider.val
    vertical_line.set_xdata(threshold)
    
    # Encontrar el índice más cercano para el umbral actual
    idx = thresholds.index(round(threshold, 2))
    
    # Actualizar la leyenda con las métricas en el umbral actual
    ax.legend([
        f'Exactitud: {accuracy_scores[idx]:.3f}',
        f'Precisión: {precision_scores[idx]:.3f}',
        f'Sensibilidad: {recall_scores[idx]:.3f}',
        f'Puntaje F1: {f1_scores[idx]:.3f}'
    ], loc='lower right')
    
    fig.canvas.draw_idle()

threshold_slider.on_changed(update)

plt.show()

