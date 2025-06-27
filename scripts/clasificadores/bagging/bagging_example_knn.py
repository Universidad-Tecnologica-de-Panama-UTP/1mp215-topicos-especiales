import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluación 1: KNeighborsClassifier solo (5-CV)
knn = KNeighborsClassifier()
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC promedio (5-CV) - KNeighborsClassifier: {cv_scores_knn.mean():.4f}")

# Evaluación 2: BaggingClassifier con KNN (5-CV)
bagging_clf = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=20,
    random_state=42
)
cv_scores_bagging = cross_val_score(bagging_clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC promedio (5-CV) - Bagging con KNN: {cv_scores_bagging.mean():.4f}")
