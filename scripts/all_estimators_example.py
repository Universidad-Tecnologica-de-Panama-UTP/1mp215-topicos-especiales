import pandas as pd
import time
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import FitFailedWarning
import warnings

# --- Cargar dataset Pima Indians Diabetes ---
url = "../datasets/pima_indian_diabetes_dataset/full_dataset.csv"
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, header=None, names=column_names)

# --- Eliminar filas con valores no numéricos (posible encabezado duplicado) ---
df = df[pd.to_numeric(df["Outcome"], errors='coerce').notna()]
df = df.astype(float)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# --- Verificar que cada clase tenga al menos 2 muestras ---
class_counts = y.value_counts()
if (class_counts < 2).any():
    raise ValueError(f"Cada clase debe tener al menos 2 muestras para stratify. Conteos actuales:\n{class_counts}")

# --- Obtener clasificadores ---
classifiers = all_estimators(type_filter='classifier')

# --- Suprimir warnings esperados ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Modificar proceso para incluir validación cruzada de 5 pliegues ---
print(f"Evaluando {len(classifiers)} clasificadores en Pima Indians (métrica: AUC, validación cruzada de 5 pliegues)...\n")

results = []

for name, Classifier in classifiers:
    print(f"Evaluando: {name:<30} ... ", end="")
    try:
        model = Classifier()
    except TypeError as e:
        print(f"Error: {type(e).__name__} (constructor)")
        continue
    try:
        start = time.time()
        if hasattr(model, "predict_proba"):
            auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        elif hasattr(model, "decision_function"):
            auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        else:
            print("Error: No predict_proba/decision_function")
            continue
        avg_auc = auc_scores.mean()
        duration = time.time() - start
        print(f"AUC Promedio: {avg_auc:.4f} | Tiempo: {duration:.2f}s")
        results.append((name, avg_auc, duration))
    except Exception as e:
        print(f"Error: {type(e).__name__}")

# --- Mostrar resultados ordenados ---
results_df = pd.DataFrame(results, columns=["Model", "AUC Promedio (5 pliegues)", "Time (s)"])
results_df = results_df.sort_values(by="AUC Promedio (5 pliegues)", ascending=False).reset_index(drop=True)

print("\n=== Clasificadores ordenados por AUC Promedio (5 pliegues) ===")
print(results_df.to_string(index=False))
