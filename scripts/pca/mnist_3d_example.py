import pandas as pd
import numpy as np
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import PCA

# Load MNIST test CSV
test_csv_path = "../../datasets/mnist/mnist_test.csv"
mnist_test_df = pd.read_csv(test_csv_path)

# Separate labels and features
if mnist_test_df.shape[1] == 785:
    y_test = mnist_test_df.iloc[:, 0].values  # labels
    X_test = mnist_test_df.iloc[:, 1:].values  # pixels
else:
    raise ValueError("Expected first column to be labels and 784 pixels")

# Reduce to 3 components with PCA
pca = PCA(n_components=3, random_state=42)
X_reduced = pca.fit_transform(X_test)

# Create DataFrame for plotly
df_plot = pd.DataFrame(X_reduced, columns=["PC1", "PC2", "PC3"])
df_plot["label"] = y_test

# Create checkboxes for digits 0-9
checkboxes = [widgets.Checkbox(value=True, description=str(i)) for i in range(10)]
checkbox_box = widgets.HBox(checkboxes)
display(checkbox_box)

# Interactive update function
def update_plot(*args):
    selected_digits = [i for i, cb in enumerate(checkboxes) if cb.value]
    filtered_df = df_plot[df_plot["label"].isin(selected_digits)]

    fig = px.scatter_3d(
        filtered_df,
        x="PC1", y="PC2", z="PC3",
        color=filtered_df["label"].astype(str),
        title="MNIST Test Set - PCA 3D Visualization",
        opacity=0.7
    )
    fig.show()

# Connect checkboxes to update function
for cb in checkboxes:
    cb.observe(update_plot, "value")

# Show initial plot
update_plot()
