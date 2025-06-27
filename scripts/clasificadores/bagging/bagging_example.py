import pandas as pd
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

# Validación cruzada (5 pliegues) para DecisionTreeClassifier usando AUC
clf = DecisionTreeClassifier(random_state=42)
cv_scores_clf = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC promedio (5-CV) - DecisionTreeClassifier: {cv_scores_clf.mean():.4f}")

# Validación cruzada (5 pliegues) para BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)
cv_scores_bagging = cross_val_score(bagging_clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC promedio (5-CV) - BaggingClassifier: {cv_scores_bagging.mean():.4f}")
