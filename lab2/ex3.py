import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

X, y_true = make_blobs(
    n_samples=[200, 100],
    n_features=2,
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    random_state=10
)

contamination = 0.07
n_neighbors = 10  # can change to see different behavior

knn_model = KNN(contamination=contamination, n_neighbors=n_neighbors)
knn_model.fit(X)
y_knn = knn_model.predict(X)

lof_model = LOF(contamination=contamination, n_neighbors=n_neighbors)
lof_model.fit(X)
y_lof = lof_model.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = [f"KNN (n_neighbors={n_neighbors})", f"LOF (n_neighbors={n_neighbors})"]
predictions = [y_knn, y_lof]

for ax, title, y_pred in zip(axes, titles, predictions):
    inliers = X[y_pred == 0]
    outliers = X[y_pred == 1]
    ax.scatter(inliers[:, 0], inliers[:, 1], color='blue', s=20, alpha=0.6, label='Inliers')
    ax.scatter(outliers[:, 0], outliers[:, 1], color='red', s=20, alpha=0.7, label='Outliers')
    ax.set_title(title)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("ex3.svg")
