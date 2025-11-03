import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1, random_state=10
)

neighbor_values = [3, 5, 7, 9]
best_nn = 0
best_ba = 0

for nn in neighbor_values:
    model = KNN(contamination=0.1, n_neighbors=nn)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)
    print(f"n_neighbors = {nn}  →  Balanced Accuracy = {ba:.4f}")

    if ba >= best_ba:
        best_ba = ba
        best_nn = nn

print(f"\nBest n_neighbors = {best_nn} with balanced accuracy = {best_ba:.4f}")

model = KNN(contamination=0.1, n_neighbors=best_nn)
model.fit(X_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = [
    "Ground truth (Train)",
    "Predicted labels (Train)",
    "Ground truth (Test)",
    "Predicted labels (Test)"
]
datasets = [
    (X_train, y_train),
    (X_train, y_train_pred),
    (X_test, y_test),
    (X_test, y_test_pred)
]

for ax, (X, y), title in zip(axes.ravel(), datasets, titles):
    inliers = X[y == 0]
    outliers = X[y == 1]
    ax.scatter(inliers[:, 0], inliers[:, 1], color="blue", s=20, alpha=0.6, label="Inliers")
    ax.scatter(outliers[:, 0], outliers[:, 1], color="red", s=20, alpha=0.6, label="Outliers")
    ax.set_title(title)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("ex2.svg")
