import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

# Cargar datos
train_csv_path = "../../../datasets/mnist/mnist_train.csv"
mnist_train_df = pd.read_csv(train_csv_path)

test_csv_path = "../../../datasets/mnist/mnist_test.csv"
mnist_test_df = pd.read_csv(test_csv_path)

# Separar características y etiquetas
X_train = mnist_train_df.iloc[:, 1:]
y_train = mnist_train_df.iloc[:, 0]
X_test = mnist_test_df.iloc[:, 1:]
y_test = mnist_test_df.iloc[:, 0]

# Modelo XGBoost con CPU
clf = XGBClassifier(tree_method='hist', eval_metric='mlogloss')

# Tiempo de entrenamiento
start_train_time = time.time()
clf.fit(X_train, y_train)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print("Training time (CPU, seconds):", train_time)

# Tiempo de predicción
start_pred_time = time.time()
y_pred = clf.predict(X_test)
end_pred_time = time.time()
pred_time = end_pred_time - start_pred_time
print("Prediction time (CPU, seconds):", pred_time)

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy (CPU):", accuracy)
