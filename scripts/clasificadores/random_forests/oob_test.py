import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Separar features y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# ==== MODELO 1: 80/20 SPLIT + TEST SET EVALUATION ====

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar Random Forest sin OOB
rf_split = RandomForestClassifier(
    n_estimators=100,
    oob_score=False,
    random_state=42,
    bootstrap=True
)
rf_split.fit(X_train, y_train)

# Evaluación en test set
y_pred = rf_split.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"🔹 Accuracy con 80/20 split (evaluación en test set): {test_acc:.4f}")

# ==== MODELO 2: ENTRENAMIENTO COMPLETO + VALIDACIÓN OOB ====

rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    bootstrap=True
)
rf_oob.fit(X, y)

# Mostrar puntuación OOB (accuracy estimado internamente)
print(f"🔸 Accuracy con OOB (evaluación interna): {rf_oob.oob_score_:.4f}")

# ==== ESTIMACIÓN DE CUÁNTAS MUESTRAS FUERON USADAS EN OOB ====

n_total = X.shape[0]
# Probabilidad de que una muestra NO esté en una bolsa (OOB) para un árbol = (1 - 1/n)^n_estimators
# Aproximación del número de muestras OOB utilizadas al menos una vez
prob_oob = 1 - (1 - 1/n_total)**rf_oob.n_estimators
n_oob_estimate = int(n_total * prob_oob)
print(f"Estimación aproximada de muestras con predicción OOB: {n_oob_estimate} de {n_total}")
