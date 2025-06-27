import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluar BaggingClassifier con distintos valores de n_estimators
n_estimators_range = range(1, 31)
auc_scores = []

for n in n_estimators_range:
    bagging_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n,
        random_state=42
    )
    auc = cross_val_score(bagging_clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
    auc_scores.append(auc)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, auc_scores, marker='o')
plt.title('AUC promedio vs. Número de Árboles (BaggingClassifier)')
plt.xlabel('Número de Árboles (n_estimators)')
plt.ylabel('AUC promedio (5-CV)')
plt.grid(True)
plt.tight_layout()
plt.show()
