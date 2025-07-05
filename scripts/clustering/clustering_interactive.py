import os
import shutil
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from PIL import Image

# === Paths ===
test_csv_path = "../../datasets/mnist/mnist_test.csv"
unclassified_dir = "interactive_example/unclassified"
classified_base = "interactive_example/classified_digits"

# === Prepare and clean directories ===
if os.path.exists(unclassified_dir):
    shutil.rmtree(unclassified_dir)
os.makedirs(unclassified_dir)

if os.path.exists(classified_base):
    shutil.rmtree(classified_base)
os.makedirs(classified_base)

for i in range(10):
    os.makedirs(os.path.join(classified_base, f"digit_{i}"), exist_ok=True)

# === Load Data ===
df = pd.read_csv(test_csv_path)
original_indices = df.index.to_numpy()
X = df.iloc[:, 1:].values if df.shape[1] == 785 else df.values

# === KMeans Clustering ===
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)
cluster_to_original_indices = {
    cluster_id: list(original_indices[cluster_labels == cluster_id])
    for cluster_id in range(10)
}

current_display_rows = {i: set() for i in range(10)}
used_rows = {i: set() for i in range(10)}
position_matrix = [[None for _ in range(10)] for _ in range(10)]

# === Helper ===
def save_image(flat_pixels, path):
    img_array = flat_pixels.reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(path)

# === Tkinter GUI Setup ===
root = tk.Tk()
root.title("MNIST Cluster Labeling Tool")

frame = tk.Frame(root)
frame.pack(side=tk.TOP)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)

# === Initial display ===
for cluster_id in range(10):
    row_indices = cluster_to_original_indices[cluster_id]
    for i in range(10):
        original_row = row_indices[i]
        current_display_rows[cluster_id].add(original_row)
        used_rows[cluster_id].add(original_row)
        sample_idx = original_row
        position_matrix[cluster_id][i] = (sample_idx, original_row)
        ax = axes[cluster_id, i]
        ax.imshow(X[sample_idx].reshape(28, 28), cmap='gray')
        ax.axis('off')

# === Save removed image to unclassified ===
def save_removed_image(original_row):
    save_path = os.path.join(unclassified_dir, f"row_{original_row}.png")
    save_image(X[original_row], save_path)
    print(f"Saved removed image to {save_path}")

# === On right-click replace ===
def on_click(event):
    if event.button != 3:
        return

    for row in range(10):
        for col in range(10):
            ax = axes[row, col]
            if event.inaxes == ax:
                cluster_id = row
                current_idx, current_row = position_matrix[row][col]

                candidates = [
                    r for r in cluster_to_original_indices[cluster_id]
                    if r not in used_rows[cluster_id]
                ]
                if not candidates:
                    print(f"No more unused samples in cluster {cluster_id}")
                    return

                save_removed_image(current_row)

                new_row = candidates[0]
                new_idx = new_row
                current_display_rows[cluster_id].remove(current_row)
                current_display_rows[cluster_id].add(new_row)
                used_rows[cluster_id].add(new_row)
                position_matrix[row][col] = (new_idx, new_row)

                ax.clear()
                ax.imshow(X[new_idx].reshape(28, 28), cmap='gray')
                ax.axis('off')
                canvas.draw()
                return

# === Classify a single row ===
def classify_row(row):
    label = combo_vars[row].get()
    if label not in [str(i) for i in range(10)]:
        print(f"Row {row}: No valid digit selected.")
        return
    label = int(label)
    folder = os.path.join(classified_base, f"digit_{label}")

    for col in range(10):
        idx, original_row = position_matrix[row][col]
        save_path = os.path.join(folder, f"row_{original_row}.png")
        save_image(X[idx], save_path)

    candidates = [
        r for r in cluster_to_original_indices[row]
        if r not in used_rows[row]
    ]
    if len(candidates) < 10:
        print(f"Not enough new samples to refill row {row}")
        return

    for col in range(10):
        new_row = candidates[col]
        new_idx = new_row
        used_rows[row].add(new_row)
        current_display_rows[row].add(new_row)
        _, old_row = position_matrix[row][col]
        current_display_rows[row].discard(old_row)
        position_matrix[row][col] = (new_idx, new_row)

        ax = axes[row][col]
        ax.clear()
        ax.imshow(X[new_idx].reshape(28, 28), cmap='gray')
        ax.axis('off')

    canvas.draw()
    print(f"Row {row} classified to digit_{label}/ and refreshed.")

# === Classify all selected rows ===
def classify_all_rows():
    for row in range(10):
        label = combo_vars[row].get()
        if label in [str(i) for i in range(10)]:
            classify_row(row)

# === Combo boxes and a single classify button ===
combo_vars = []
for i in range(10):
    row_frame = tk.Frame(frame)
    row_frame.pack(side=tk.TOP, anchor='w')

    label = tk.Label(row_frame, text=f"Cluster {i}: ")
    label.pack(side=tk.LEFT)

    var = tk.StringVar()
    combo = ttk.Combobox(row_frame, textvariable=var, values=[""] + [str(x) for x in range(10)], width=5)
    combo.pack(side=tk.LEFT)
    combo_vars.append(var)

    if i == 0:
        button = tk.Button(row_frame, text="Classify All", command=classify_all_rows)
        button.pack(side=tk.LEFT, padx=10)

# === Embed matplotlib in tkinter ===
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.mpl_connect('button_press_event', on_click)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# === Launch GUI ===
root.mainloop()
