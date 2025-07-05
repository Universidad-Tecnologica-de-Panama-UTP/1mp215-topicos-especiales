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

# Modelo XGBoost con configuración moderna para GPU
clf = XGBClassifier(
    tree_method='hist',    # hist + device=cuda reemplaza gpu_hist
    device='cuda',
    eval_metric='mlogloss'
)

# Tiempo de entrenamiento
start_train_time = time.time()
clf.fit(X_train, y_train)
end_train_time = time.time()
print("Training time (GPU, seconds):", end_train_time - start_train_time)

# Tiempo de predicción
start_pred_time = time.time()
y_pred = clf.predict(X_test)
end_pred_time = time.time()
print("Prediction time (GPU, seconds):", end_pred_time - start_pred_time)

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy (GPU):", accuracy)
